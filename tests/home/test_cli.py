"""실제 양식은 명시적인 검토를 거쳐야 하며 실패한 승인은 원본을 보존한다."""
import json

import cv2
import numpy as np
import pytest

from docagent.home.__main__ import main
from docagent.home.templates import load_template


@pytest.fixture
def draft_bundle(tmp_path):
    reference = tmp_path / "blank.png"
    ok, encoded = cv2.imencode(".png", np.full((600, 400, 3), 255, np.uint8))
    assert ok
    reference.write_bytes(encoded.tobytes())
    source = "  동의 항목\n□ 동의함  □ 동의하지 않음\n"
    (tmp_path / "original.txt").write_text(source, encoding="utf-8")
    (tmp_path / "groups.json").write_text(json.dumps([{
        "id": "consent", "source_text": source, "min_selected": 1, "max_selected": 1,
        "options": [
            {"id": "yes", "label": "동의함", "box_mm": [20, 80, 5, 5]},
            {"id": "no", "label": "동의하지 않음", "box_mm": [50, 80, 5, 5]},
        ],
    }], ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "bundle"
    args = [
        "draft-template", "--reference", str(reference),
        "--text", str(tmp_path / "original.txt"), "--groups", str(tmp_path / "groups.json"),
        "--output", str(output), "--document-id", "consent",
        "--title", "동의서", "--source", "test fixture only",
    ]
    assert main(args) == 0
    return output / "template.json", source, args


def test_draft_is_not_usable_even_in_demo_mode(draft_bundle):
    path, source, args = draft_bundle
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pages"][0]["source_text"] == source
    assert data["purpose"] == "production"
    assert data["approval"] == {"status": "draft"}
    with pytest.raises(ValueError, match="승인"):
        load_template(path, demo=True)
    original = path.read_bytes()
    assert main(args) == 2  # Existing bundle must not be overwritten.
    assert path.read_bytes() == original


def test_review_requires_explicit_flag_and_binds_image_and_text(draft_bundle):
    path, source, _ = draft_bundle
    command = ["approve-template", "--template", str(path), "--reviewer", "테스트 검토자"]
    with pytest.raises(SystemExit) as exc:
        main(command)
    assert exc.value.code == 2
    assert main(command + ["--confirm-reviewed"]) == 0
    template = load_template(path)
    assert template.manifest["pages"][0]["source_text"] == source
    assert template.manifest["approval"]["reviewed_by"] == "테스트 검토자"
    assert not list(path.parent.glob(".home-*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pages"][0]["source_text"] += "승인 후 추가한 문구"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="변경"):
        load_template(path)


def test_failed_image_validation_preserves_draft(draft_bundle):
    path, _, _ = draft_bundle
    before = path.read_bytes()
    (path.parent / "reference.png").write_bytes(b"changed image")
    assert main(["approve-template", "--template", str(path),
                 "--reviewer", "검토자", "--confirm-reviewed"]) == 2
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".home-*.json"))


@pytest.mark.parametrize("data", [[], None, "invalid", {"purpose": "demo"}])
def test_invalid_approval_input_fails_without_overwriting(tmp_path, data):
    path = tmp_path / "template.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_bytes()
    assert main(["approve-template", "--template", str(path),
                 "--reviewer", "검토자", "--confirm-reviewed"]) == 2
    assert path.read_bytes() == before
