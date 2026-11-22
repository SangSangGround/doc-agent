import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from docagent.home.templates import load_template, template_digest, validate_manifest


def write_bundle(tmp_path, *, purpose="demo", approved=False):
    image = np.full((840, 600, 3), 255, np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    blob = encoded.tobytes()
    (tmp_path / "reference.png").write_bytes(blob)
    manifest = {
        "schema_version": 1, "purpose": purpose, "document_id": "consent",
        "document_version": "1", "title": "검증용 동의서", "page_size_mm": [210, 297],
        "reference_image": "reference.png", "reference_sha256": hashlib.sha256(blob).hexdigest(),
        "source": {"reference": "synthetic:test"},
        "pages": [{"page": 1, "source_text": "  수집·이용 동의\n동의함 / 동의하지 않음\n"}],
        "groups": [{"id": "consent", "source_text": "  수집·이용 동의\n", "min_selected": 1,
                    "max_selected": 1, "options": [
                        {"id": "yes", "label": "동의함", "box_mm": [20, 80, 5, 5]},
                        {"id": "no", "label": "동의하지 않음", "box_mm": [50, 80, 5, 5]},
                    ]}],
        "approval": {"status": "draft"},
    }
    if approved:
        manifest["approval"] = {"status": "approved", "reviewed_by": "test reviewer",
                                "reviewed_at": "2026-11-04", "content_sha256": template_digest(manifest)}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path, manifest


def test_demo_requires_opt_in_and_preserves_original(tmp_path):
    path, manifest = write_bundle(tmp_path)
    with pytest.raises(ValueError, match="데모"):
        load_template(path)
    loaded = load_template(path, demo=True)
    assert loaded.manifest == manifest
    assert loaded.reference.shape == (840, 600, 3)


def test_production_requires_review_and_detects_text_change(tmp_path):
    path, manifest = write_bundle(tmp_path, purpose="production")
    with pytest.raises(ValueError, match="승인"):
        load_template(path, demo=True)
    path, manifest = write_bundle(tmp_path, purpose="production", approved=True)
    load_template(path)
    manifest["pages"][0]["source_text"] += "추가된 원문"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="변경"):
        load_template(path)


def test_reference_image_change_invalidates_bundle(tmp_path):
    path, _ = write_bundle(tmp_path, purpose="production", approved=True)
    (tmp_path / "reference.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="기준 이미지가 변경"):
        load_template(path)


@pytest.mark.parametrize("change", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(page_size_mm=[float("nan"), 297]),
    lambda d: d.update(page_size_mm=[True, 297]),
    lambda d: d["pages"].append({"page": 2, "source_text": "unsupported"}),
    lambda d: d["pages"][0].update(source_text=" "),
    lambda d: d["groups"].append(copy.deepcopy(d["groups"][0])),
    lambda d: d["groups"][0].update(source_text="원문에 없는 안내"),
    lambda d: d["groups"][0].update(min_selected=2, max_selected=1),
    lambda d: d["groups"][0].update(min_selected=True),
    lambda d: d["groups"][0]["options"][0].update(box_mm=[-1, 10, 5, 5]),
    lambda d: d["groups"][0]["options"][0].update(box_mm=[209, 10, 5, 5]),
    lambda d: d["groups"][0]["options"][0].update(box_mm=[50, 80, 5, 5]),
    lambda d: d["groups"][0]["options"][0].update(id="no"),
    lambda d: d["source"].update(reference=""),
])
def test_invalid_manifests_rejected(tmp_path, change):
    _, manifest = write_bundle(tmp_path)
    change(manifest)
    with pytest.raises(ValueError):
        validate_manifest(manifest, demo=True)


@pytest.mark.parametrize("path_kind", ["absolute", "traversal", "symlink"])
def test_reference_cannot_escape_bundle(tmp_path, path_kind):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    path, manifest = write_bundle(bundle)
    outside = tmp_path / "outside.png"
    outside.write_bytes((bundle / "reference.png").read_bytes())
    if path_kind == "symlink":
        link = bundle / "link.png"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlink 권한이 없는 환경")
        manifest["reference_image"] = "link.png"
    elif path_kind == "absolute":
        manifest["reference_image"] = str(outside)
    else:
        manifest["reference_image"] = "../outside.png"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="묶음"):
        load_template(path, demo=True)
