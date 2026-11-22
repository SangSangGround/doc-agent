"""아이폰 연결은 HTTPS와 실행별 토큰으로 한정하며 웹 토큰 배포를 열지 않는다."""
import io
import json
from email.message import Message
from pathlib import Path
import shutil
import ssl
from types import SimpleNamespace

import pytest

from docagent.home.mobile import private_ipv4, connection_files, connection_payload, write_pairing
from docagent.home import server as server_module
from docagent.home.server import HomeApplication, make_server
from tests.home.test_server import bundle


@pytest.mark.parametrize("host", ["10.1.2.3", "192.168.0.2", "172.16.1.2", "127.0.0.1"])
def test_private_pc_addresses(host):
    assert private_ipv4(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "172.32.1.2", "169.254.1.2", "example.com", "::1"])
def test_public_wildcard_or_ambiguous_address_rejected(host):
    with pytest.raises(ValueError):
        private_ipv4(host)


def test_nonlocal_listener_needs_tls(bundle):
    _, template = bundle
    with pytest.raises(ValueError, match="TLS"):
        make_server(HomeApplication(template), mobile_host="192.168.0.2")


@pytest.fixture
def mobile_request(bundle, monkeypatch):
    _, template = bundle
    app = HomeApplication(template)
    wrapped = []
    def factory(address, handler):
        assert address == ("192.168.0.2", 8766)
        return SimpleNamespace(server_port=8766, handler=handler, socket="socket")
    def wrap(sock, server_side):
        wrapped.append((sock, server_side)); return "tls-socket"
    monkeypatch.setattr(server_module, "ThreadingHTTPServer", factory)
    server = make_server(app, 8766, mobile_host="192.168.0.2", tls_context=SimpleNamespace(wrap_socket=wrap))
    assert wrapped == [("socket", True)] and server.socket == "tls-socket"
    def request(method="GET", path="/api/state", token=True, **headers):
        handler = server.handler.__new__(server.handler)
        handler.server, handler.path = server, path
        handler.headers = Message()
        fields = {"Host":"192.168.0.2:8766", "Content-Length":"2", "Content-Type":"application/json"}
        if token: fields["X-Session-Token"] = app.token
        fields.update(headers)
        for key, value in fields.items(): handler.headers[key] = value
        result = {}
        handler.send_response = lambda code: result.update(status=code)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        handler.rfile, handler.wfile = io.BytesIO(b"{}"), io.BytesIO()
        handler.connection = SimpleNamespace(settimeout=lambda _:None)
        getattr(handler, "do_"+method)()
        result["body"] = handler.wfile.getvalue()
        assert app.token.encode() not in result["body"]
        return result
    return request


def test_mobile_has_no_web_token_bootstrap(mobile_request):
    for path in ["/", "/app.js", "/style.css"]:
        assert mobile_request(path=path)["status"] == 404
    assert mobile_request()["status"] == 200
    assert mobile_request(token=False)["status"] == 404


@pytest.mark.parametrize("headers", [
    {"Host":"evil.example:8766"}, {"Host":"127.0.0.1:8766"},
    {"Origin":"https://192.168.0.2:8766"}, {"Origin":"null"}, {"X-Session-Token":"wrong"},
])
def test_mobile_blocks_browsers_and_wrong_peers(mobile_request, headers):
    assert mobile_request("POST", "/api/reset_capture", **headers)["status"] == 403


@pytest.mark.skipif(shutil.which("openssl") is None, reason="개발 인증서 생성은 OpenSSL 필요")
def test_pairing_binds_certificate_and_saves_secret_privately(tmp_path):
    cert, key = connection_files(tmp_path, "192.168.0.2")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    payload = connection_payload(cert, "192.168.0.2", 8766, "test-token-only")
    assert payload["server_url"] == "https://192.168.0.2:8766"
    assert len(payload["certificate_sha256"]) == 64
    pairing = tmp_path / "pairing.json"
    write_pairing(pairing, payload)
    assert json.loads(pairing.read_text())["token"] == "test-token-only"
    import os
    if os.name != "nt":
        assert pairing.stat().st_mode & 0o777 == 0o600
        assert key.stat().st_mode & 0o777 == 0o600
    before = cert.read_bytes()
    connection_files(tmp_path, "192.168.0.2")
    assert cert.read_bytes() == before


def test_incomplete_certificate_pair_is_not_overwritten(tmp_path):
    key = tmp_path / "server-key.pem"
    key.write_text("existing content")
    with pytest.raises(ValueError): connection_files(tmp_path, "192.168.0.2")
    assert key.read_text() == "existing content"
