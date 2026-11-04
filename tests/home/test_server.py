"""합성 양식으로 실제 정합·세션·HTTP 처리기를 함께 검증한다. 사용자 성능 측정은 아니다."""
import base64
import io
import json
from email.message import Message
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from docagent.home.demo import create_demo
from docagent.home import server as server_module
from docagent.home.server import HomeApplication, decode_image, encode_image, make_server
from docagent.home.templates import load_template


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    directory = tmp_path_factory.mktemp("home-server")
    template_path = create_demo(directory / "demo")
    return template_path.parent, load_template(template_path, demo=True)


def capture(app, bundle, name):
    directory, _ = bundle
    blob = (directory / f"{name}.png").read_bytes()
    return app.handle("capture", {"image": base64.b64encode(blob).decode(), "mode": "scan"})


def choose_all(app):
    for group in app.template.manifest["groups"]:
        app.handle("select", {"group_id": group["id"], "option_ids": ["disagree"]})
        app.handle("confirm", {"group_id": group["id"]})


@pytest.mark.parametrize("name,expected", [
    ("reference", ["unmarked", "unmarked"]),
    ("correct", ["matched", "matched"]),
    ("opposite", ["mismatch", "matched"]),
    ("missing", ["unmarked", "matched"]),
    ("duplicate", ["multiple", "matched"]),
])
def test_real_capture_choice_and_verification(bundle, name, expected):
    _, template = bundle
    app = HomeApplication(template)
    assert capture(app, bundle, "reference")["document_ready"]
    assert app.handle("read", {"scope": "page"})["instruction"] == template.manifest["pages"][0]["source_text"]
    assert not app.handle("verify", {})["session"]["complete"]
    for group in template.manifest["groups"]:
        app.handle("select", {"group_id": group["id"], "option_ids": ["disagree"]})
        assert not app.state()["can_verify"]
        with pytest.raises(ValueError):
            app.handle("verify", {})
        app.handle("confirm", {"group_id": group["id"]})
    assert not app.state()["can_verify"]
    assert capture(app, bundle, name)["can_verify"]
    result = app.handle("verify", {})["session"]
    assert [group["verification"]["status"] for group in result["groups"]] == expected
    assert result["complete"] == (name == "correct")


@pytest.mark.parametrize("recapture", ["blurred", "invalid", "reset"])
def test_new_or_failed_capture_clears_completion(bundle, recapture):
    _, template = bundle
    app = HomeApplication(template)
    capture(app, bundle, "reference")
    choose_all(app)
    capture(app, bundle, "correct")
    assert app.handle("verify", {})["session"]["complete"]
    if recapture == "invalid":
        with pytest.raises(ValueError):
            app.handle("capture", {"image": "not an image", "mode": "scan"})
    elif recapture == "reset":
        app.handle("reset_capture", {})
    else:
        assert not capture(app, bundle, "blurred")["document_ready"]
    assert not app.state()["session"]["complete"]
    with pytest.raises(ValueError):
        app.handle("read", {"scope": "page"})


def test_unsupported_document_never_reads_template(bundle):
    _, template = bundle
    app = HomeApplication(template)
    wrong = np.random.default_rng(123).integers(0, 256, (1782, 1260, 3), dtype=np.uint8)
    result = app.handle("capture", {"image": encode_image(wrong), "mode": "scan"})
    assert not result["document_ready"]
    with pytest.raises(ValueError):
        app.handle("read", {})
    with pytest.raises(ValueError):
        app.handle("select", {"group_id": "consent_1", "option_ids": ["agree"]})


def test_faint_mark_allows_original_reading_but_never_completes(bundle):
    _, template = bundle
    app = HomeApplication(template)
    capture(app, bundle, "reference")
    choose_all(app)
    image = template.reference.copy()
    scale_x = image.shape[1] / template.manifest["page_size_mm"][0]
    scale_y = image.shape[0] / template.manifest["page_size_mm"][1]
    x, y, w, h = template.manifest["groups"][0]["options"][1]["box_mm"]
    cv2.line(image, (round((x+1)*scale_x), round((y+1)*scale_y)),
             (round((x+w-1)*scale_x), round((y+h-1)*scale_y)), (220,220,220), 2)
    result = app.handle("capture", {"image": encode_image(image), "mode": "scan"})
    assert result["observation"]["alignment"]["reason"] == "registered"
    assert result["observation"]["status"] == "uncertain"
    assert result["document_ready"] and result["can_verify"]
    assert "일부 체크 표시" in result["instruction"]
    assert app.handle("read", {"scope": "page"})["instruction"] == template.manifest["pages"][0]["source_text"]
    assert not app.handle("verify", {})["session"]["complete"]
    app.handle("select", {"group_id": "consent_1", "option_ids": ["agree"]})
    assert not app.state()["can_verify"]


def test_obscured_original_text_blocks_template_reading(bundle):
    _, template = bundle
    app = HomeApplication(template)
    image = template.reference.copy()
    # 제목과 안내 본문을 지운 사진을 원본 양식으로 읽어주면 안 된다.
    image[75:410, 90:1180] = 255
    result = app.handle("capture", {"image": encode_image(image), "mode": "scan"})
    assert not result["document_ready"]
    assert result["observation"]["alignment"].get("reason") != "registered"
    with pytest.raises(ValueError):
        app.handle("read", {"scope": "page"})


def test_restore_keeps_preferences_choices_but_requires_recapture(bundle, tmp_path):
    _, template = bundle
    path = tmp_path / "private-state.json"
    app = HomeApplication(template, session_path=path)
    capture(app, bundle, "reference")
    choose_all(app)
    capture(app, bundle, "correct")
    app.handle("preferences", {"speech_rate": 1.5})
    assert app.handle("verify", {})["session"]["complete"]
    restored = HomeApplication(template, session_path=path)
    state = restored.state()
    assert state["session"]["preferences"]["speech_rate"] == 1.5
    assert all(group["confirmed"] for group in state["session"]["groups"])
    assert not state["document_ready"] and not state["has_capture"] and not state["session"]["complete"]
    with pytest.raises(ValueError):
        restored.handle("verify", {})
    with pytest.raises(ValueError):
        restored.handle("read", {})


@pytest.mark.parametrize("action", ["select", "confirm", "cancel"])
def test_selection_mutation_requires_fresh_capture(bundle, action):
    _, template = bundle
    app = HomeApplication(template)
    capture(app, bundle, "reference")
    choose_all(app)
    capture(app, bundle, "correct")
    assert app.handle("verify", {})["session"]["complete"]
    if action == "select":
        app.handle("select", {"group_id": "consent_1", "option_ids": ["disagree"]})
    else:
        app.handle("select", {"group_id": "consent_1", "option_ids": ["disagree"]})
        capture(app, bundle, "correct")
        app.handle(action, {"group_id": "consent_1"})
    state = app.state()
    assert not state["can_verify"] and not state["session"]["complete"]
    assert state["document_ready"] and state["has_capture"]
    assert app.handle("read", {"scope": "page"})["instruction"] == template.manifest["pages"][0]["source_text"]
    with pytest.raises(ValueError):
        app.handle("verify", {})


def test_recapture_keeps_pending_selection_and_requires_confirm_then_recapture(bundle):
    _, template = bundle
    app = HomeApplication(template)
    capture(app, bundle, "reference")
    choose_all(app)
    app.handle("select", {"group_id": "consent_1", "option_ids": ["disagree"]})
    capture(app, bundle, "correct")
    state = app.handle("verify", {})["session"]
    assert not state["complete"]
    assert state["groups"][0]["pending_option_ids"] == ["disagree"]
    assert state["groups"][0]["verification"]["status"] == "unconfirmed"
    app.handle("confirm", {"group_id": "consent_1"})
    with pytest.raises(ValueError):
        app.handle("verify", {})
    capture(app, bundle, "correct")
    assert app.handle("verify", {})["session"]["complete"]


@pytest.mark.parametrize("value", [None, [], 42, "not base64", base64.b64encode(b"not image").decode()])
def test_invalid_uploads_rejected(value):
    with pytest.raises(ValueError):
        decode_image(value)


def test_oversized_pixels_rejected_before_opencv_decode(monkeypatch):
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (4001, 4000)).save(data, format="PNG")
    monkeypatch.setattr(cv2, "imdecode", lambda *args: pytest.fail("큰 이미지를 디코딩하면 안 됩니다"))
    with pytest.raises(ValueError):
        decode_image(base64.b64encode(data.getvalue()).decode())


@pytest.fixture
def request_handler(bundle, monkeypatch):
    """실제 Handler를 메모리 HTTP 스트림에 연결해 소켓 권한 없이 요청 경계를 검사한다."""
    _, template = bundle
    app = HomeApplication(template)

    def fake_server(address, handler):
        assert address[0] == "127.0.0.1"
        return SimpleNamespace(server_port=address[1], handler=handler)

    monkeypatch.setattr(server_module, "ThreadingHTTPServer", fake_server)
    server = make_server(app, port=8765)

    def request(method, path, payload=None, **overrides):
        handler = server.handler.__new__(server.handler)
        handler.server, handler.path = server, path
        handler.headers = Message()
        body = json.dumps(payload if payload is not None else {}).encode()
        headers = {"Host": "127.0.0.1:8765", "X-Session-Token": app.token,
                   "Content-Type": "application/json", "Content-Length": str(len(body))}
        headers.update(overrides)
        for key, value in headers.items():
            if value is not None:
                handler.headers[key] = value
        response = {"headers": {}}
        handler.send_response = lambda code: response.update(status=code)
        handler.send_header = lambda key, value: response["headers"].update({key: value})
        handler.end_headers = lambda: None
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.connection = SimpleNamespace(settimeout=lambda _: None)
        getattr(handler, "do_" + method)()
        response["body"] = handler.wfile.getvalue()
        return response

    return request


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example:8765"},
    {"Origin": "https://evil.example"},
    {"X-Session-Token": "wrong"},
    {"X-Session-Token": "비정상토큰"},
    {"X-Session-Token": None},
])
def test_post_authorization_boundary(request_handler, headers):
    response = request_handler("POST", "/api/reset_capture", **headers)
    assert response["status"] == 403


def test_get_state_needs_token_and_html_has_protection_headers(request_handler):
    assert request_handler("GET", "/api/state", **{"X-Session-Token": None})["status"] == 404
    assert request_handler("GET", "/api/state")["status"] == 200
    page = request_handler("GET", "/")
    assert page["status"] == 200
    assert page["headers"]["Cache-Control"] == "no-store"
    assert "frame-ancestors 'none'" in page["headers"]["Content-Security-Policy"]
    assert request_handler("GET", "/", **{"Host": "evil.example"})["status"] == 403
    assert request_handler("GET", "/../templates.py")["status"] == 404


@pytest.mark.parametrize("path,payload,headers,code", [
    ("/api/reset_capture", {}, {"Content-Type": "text/plain"}, 415),
    ("/api/reset_capture", {}, {"Content-Length": "999999999"}, 413),
    ("/api/reset_capture", [], {}, 400),
    ("/api/unsupported", {}, {}, 400),
    ("/api/capture", {"image": None}, {}, 400),
    ("/api/preferences", {"speech_rate": float("nan")}, {}, 400),
    ("/api/confirm", {"group_id": ["consent_1"]}, {}, 400),
])
def test_malformed_requests_are_bounded(request_handler, path, payload, headers, code):
    assert request_handler("POST", path, payload, **headers)["status"] == code
