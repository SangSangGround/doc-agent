"""Exercise the approved-image relay and live pixel refinements without uploading any document."""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import threading

import pytest
from PIL import Image, ImageDraw

from docagent import cloud_providers, writing_fields
from docagent.cloud_providers import RelayError, json_bytes
from docagent.cloud_relay import provider_settings
from docagent.writing_fields import MAX_REQUEST, WritingFieldDetector
pytest_plugins = ("tests.test_cloud_relay",)


def page(checked=False, orientation=1):
    image = Image.new("RGB", (1000, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([100, 100, 125, 125], outline="black", width=2)
    draw.rectangle([600, 600, 820, 690], outline="black", width=3)
    draw.rectangle([100, 600, 260, 690], outline="black", width=3)
    if checked:
        draw.rectangle([105, 105, 120, 120], fill="black")
    exif = Image.Exif()
    exif[274] = orientation
    exif[315] = "metadata which must stay on the Mac"
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92, exif=exif)
    return out.getvalue()


def request(image):
    return {
        "allowImageProcessing": True,
        "imageJPEG": base64.b64encode(image).decode(),
        "imageSHA256": hashlib.sha256(image).hexdigest(),
    }


def proposal(kind, label, bounds):
    return {"class_name": kind, "label": label, "box_2d": bounds}


BOXES = {
    "boxes": [
        proposal("check_box", "개인정보 수집 동의함", [100, 100, 125, 125]),
        proposal("signature_field", "신청인 서명", [600, 600, 690, 820]),
        proposal("input_field", "성명", [600, 100, 690, 260]),
        proposal("signature_field", "출력된 서명 문구만 있음", [200, 310, 230, 415]),
        proposal("input_field", "접수담당", [295, 100, 330, 210]),
    ]
}


def google_result(boxes):
    return {
        "status": "completed",
        "steps": [
            {"type": "thought", "content": [{"type": "text", "text": "not a result"}]},
            {"type": "model_output", "content": [{"type": "text", "text": json.dumps(boxes)}]},
        ],
    }


def test_writing_gate_cache_and_image_binding_over_https(relay, monkeypatch):
    server, send = relay
    calls = []

    def fake(host, path, headers, body):
        assert (host, path) == ("generativelanguage.googleapis.com", "/v1beta/interactions")
        assert headers["x-goog-api-key"] == "sample"
        data = json.loads(body)
        assert (
            data["model"] == "gemini-4-hydrogen" and data["store"] is False and len(data["input"]) == 2
        )
        outgoing = Image.open(io.BytesIO(base64.b64decode(data["input"][1]["data"])))
        assert outgoing.format == "JPEG" and outgoing.size == (1000, 1000) and not outgoing.getexif()
        calls.append(outgoing.size)
        return google_result(BOXES)

    monkeypatch.setattr(cloud_providers, "post", fake)
    body = request(page())
    assert send("/v1/fields", json_bytes(body))[0] == 503
    server.writing = WritingFieldDetector("sample", "gemini-4-hydrogen")
    assert send("/v1/fields", json_bytes(body), token="bad")[0] == 401
    for changed in (
        {"allowImageProcessing": False},
        {"allowImageProcessing": 1},
        {"imageJPEG": "bad"},
        {"imageSHA256": "0" * 64},
        {"extra": "audio"},
    ):
        assert send("/v1/fields", json_bytes({**body, **changed}))[0] == 400
    assert send("/v1/fields", b"", extra={"Content-Length": str(MAX_REQUEST + 1)})[0] == 413
    assert send("/v1/fields", b"", "image/jpeg")[0] == 415
    assert send("/v1/review", json_bytes(body))[0] == 400
    assert not calls
    first = send("/v1/fields", json_bytes(body))
    assert first[0] == 200
    assert first[1]["imageSHA256"] == body["imageSHA256"]
    assert first[1]["coordinateSpace"] == "upright-top-left-normalized"
    assert [(f["id"], f["kind"]) for f in first[1]["fields"]] == [
        ("F001", "checkbox"),
        ("F002", "signature"),
        ("F003", "input"),
    ]
    assert first[1]["fields"][0]["bounds"] == pytest.approx([0.10, 0.10, 0.026, 0.026])
    assert send("/v1/fields", json_bytes(body)) == first
    assert send("/v1/fields", json_bytes({**body, "allowImageProcessing": False}))[0] == 400
    assert len(calls) == 1
    second = send("/v1/fields", json_bytes(request(page(True))))
    assert second[0] == 200 and first[1]["imageSHA256"] != second[1]["imageSHA256"]
    # The same physical square still exists when ink is added. The phone reads the current mark separately.
    assert second[1]["fields"][0]["bounds"] == pytest.approx(first[1]["fields"][0]["bounds"])
    assert len(calls) == 2


def test_unclosed_colliding_and_oversize_model_boxes_never_become_a_hinted_target():
    with writing_fields.decode_page(page()) as image:
        bad = [
            *([BOXES["boxes"][0]] * 3),
            BOXES["boxes"][3],
            proposal("signature_field", "공무원 확인", [600, 600, 690, 820]),
            proposal("check_box", "잘못된 크기", [0, 0, 1000, 1000]),
            proposal("check_box", "잘못된 좌표", [1010, 20, 1000, 50]),
            proposal("check_box", "중첩", [None, 20, 1000, 50]),
        ]
        assert writing_fields.field_regions(image, {"boxes": bad}) == []


def test_orientation_and_nonjpeg_are_refused_before_contacting_gemini(monkeypatch):
    detector = WritingFieldDetector("test", "test")
    monkeypatch.setattr(cloud_providers, "post", lambda *args: pytest.fail("bad image forwarded"))
    for data in (page(orientation=6), b"fake jpeg"):
        with pytest.raises(RelayError) as error:
            detector.fields(json_bytes(request(data)))
        assert error.value.status == 400


def test_interrupted_response_can_be_retried_and_concurrent_requests_coalesce(monkeypatch):
    detector = WritingFieldDetector("test", "test")
    item = json_bytes(request(page()))
    monkeypatch.setattr(cloud_providers, "post", lambda *args: {"status": "in_progress", "steps": []})
    with pytest.raises(RelayError) as error:
        detector.fields(item)
    assert error.value.status == 502
    table = []
    in_progress, release = threading.Event(), threading.Event()

    def slow(*args):
        table.append(1)
        in_progress.set()
        assert release.wait(4)
        return google_result(BOXES)

    monkeypatch.setattr(cloud_providers, "post", slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(detector.fields, item)
        assert in_progress.wait(4)
        duplicate = pool.submit(detector.fields, item)
        release.set()
        assert first.result() == duplicate.result()
    assert len(table) == 1


def test_gemini_settings_independent_of_openai(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "sample-secret")
    path = tmp_path / "settings"
    path.write_text("GEMINI_API_KEY=old\nGEMINI_MODEL=testing-model\n")
    result = provider_settings(path)
    assert (result.gemini_key, result.gemini_model) == ("sample-secret", "testing-model")
    assert "sample-secret" not in repr(result)


def test_signature_needs_printed_anchor_room_and_a_rule(monkeypatch):
    """Open signature space above a label is usable. Moving the source onto the label removes it."""
    import numpy as np
    from docagent import writing_anchors

    gray = np.full((400, 600), 255, np.uint8)
    gray[118:121, 120:420] = 0
    # Rectangles represent OCR word boxes, (서명 and 인), on a printed line below the writing strip.
    words = [("서명", np.array([135, 156, 180, 178])), ("인", np.array([225, 156, 239, 178]))]
    monkeypatch.setattr(writing_anchors, "printed_words", lambda _: words)
    anchors = writing_anchors.WritingAnchors(gray)
    region = anchors.region("signature", "신청인 서명", np.array([130, 120, 280, 180]), [])
    assert region is not None and region[1] > 120 and region[3] < 156
    for src, obstacles in [
        ([130, 147, 280, 180], []),  # would point at the "(서명 또는 인)" label
        (
            [130, 120, 280, 180],
            [np.array([120, 115, 290, 158])],
        ),  # another proposed field occupies the space
    ]:
        assert anchors.region("signature", "신청인 서명", np.array(src), obstacles) is None
    blank = writing_anchors.WritingAnchors(np.full_like(gray, 255))
    assert blank.region("signature", "신청인 서명", np.array([130, 120, 280, 180]), []) is None


def test_input_recovery_rejects_wrong_printed_label_and_nonblank_or_occupied_region(monkeypatch):
    import numpy as np
    from docagent import writing_anchors

    gray = np.full((400, 600), 255, np.uint8)
    gray[220:223, 40:400] = 0
    words = [("주민등록번호", np.array([40, 181, 148, 205]))]
    monkeypatch.setattr(writing_anchors, "printed_words", lambda _: words)
    source = np.array([150, 175, 380, 220])
    anchors = writing_anchors.WritingAnchors(gray)
    assert anchors.region("input", "신청인 주민등록번호", source, []) is not None
    assert anchors.region("input", "신청인 주소", source, []) is None
    assert anchors.region("input", "주민등록번호", source, [np.array([150, 175, 378, 219])]) is None
    gray[182:205, 149:375:3] = 0
    assert writing_anchors.WritingAnchors(gray).region("input", "주민등록번호", source, []) is None
