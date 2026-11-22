"""Exercise the actual authenticated HTTPS relay without sending data to OpenAI."""
import http.client
import json
import socket
import ssl
import threading

import pytest

from docagent import cloud_providers
from docagent.cloud_providers import Providers
from docagent.cloud_relay import RelayServer, provider_settings
from docagent.home.mobile import connection_files


@pytest.fixture
def relay(tmp_path, offline_relay_ports):
    cert, key = connection_files(tmp_path, "127.0.0.1")
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert, key)
    server = RelayServer("127.0.0.1", 0, tls, "iphone-test-pairing", Providers("fake-openai"))
    offline_relay_ports.add(server.server_port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()

    def send(path, body=None, content_type="application/json", token="iphone-test-pairing", extra=None, *, timeout=5):
        headers = {"Authorization": f"Bearer {token}", "Content-Type": content_type, **(extra or {})}
        client = http.client.HTTPSConnection("127.0.0.1", server.server_port, timeout=timeout,
                                            context=ssl.create_default_context(cafile=str(cert)))
        try:
            client.request("GET" if body is None else "POST", path, body=body, headers=headers)
            result = client.getresponse()
            assert result.getheader("Cache-Control") == "no-store"
            return result.status, json.loads(result.read())
        finally:
            client.close()
    try:
        yield server, send
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_relay_only_allows_its_own_loopback_port(relay, monkeypatch):
    server, send = relay
    assert send("/v1/status")[0] == 200
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args: pytest.fail("unexpected DNS lookup"))
    with pytest.raises(AssertionError, match="등록되지 않은 주소"):
        socket.create_connection(("external.invalid", server.server_port))
    client = socket.socket()
    client.close()  # A broken guard must not make an outbound connection during the test.
    with pytest.raises(AssertionError, match="등록되지 않은 주소"):
        client.connect(("127.0.0.1", server.server_port - 1))
    with pytest.raises(AssertionError, match="등록되지 않은 주소"):
        client.connect_ex(("192.0.2.1", server.server_port))


def qa_data():
    return {"question": "그중 두 번째는 언제 내?", "lines": [
        {"id": 2, "text": "신청서 제출은 12월 2일까지", "uncertain": True, "bounds": [0.1, 0.3, 0.4, 0.1]},
        {"id": 1, "text": "제출 기한 12월 2일", "uncertain": False}],
        "history": [{"question": "준비물은?", "answer": "신분증과 신청서예요."}],
        "references": [{"id": "term1", "kind": "term", "text": "서명은 이름을 쓰는 뜻이에요."}],
        "focusLineIDs": [2]}


def qa_reply():
    return {"kind": "document", "text": "신청서는 12월 2일까지예요.", "choices": [],
            "evidence": [{"id": 2, "quote": "12월 2일까지"}],
            "lineContexts": [{"id": 2, "score": 0.7, "supportID": 1, "supportQuote": "제출 기한 12월 2일"}],
            "referenceID": None}


def model_reply(data):
    return {"status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(data)}]}]}


def test_tls_auth_and_config_status(relay, monkeypatch):
    server, send = relay
    monkeypatch.setattr(cloud_providers, "post", lambda *args: pytest.fail("unexpected provider call"))
    assert send("/v1/status") == (200, {"answersReady": True, "fieldsReady": False})
    assert send("/v1/status", token="bad")[0] == 401
    assert send("/v1/answer", b"{}", extra={"Origin": "https://unknown.example"})[0] == 403
    assert send("/v1/status", extra={"Host": "wrong.example"})[0] == 403
    assert send("/v1/review", b"{}")[0] == 400
    server.providers = Providers("")
    assert send("/v1/status") == (200, {"answersReady": False, "fieldsReady": False})
    assert send("/v1/answer", json.dumps(qa_data()).encode())[0] == 503
    assert send("/v1/review", json.dumps({"lines": qa_data()["lines"]}).encode())[0] == 503


def test_relay_refuses_question_audio_before_reading_or_forwarding(relay, monkeypatch):
    _, send = relay
    monkeypatch.setattr(cloud_providers, "post", lambda *args: pytest.fail("unexpected provider call"))
    # Declare a body and withhold it. Reading audio before refusing would stall the request.
    header = {"Content-Length": "32044"}
    assert send("/v1/transcribe", b"", "audio/wav", extra=header)[0] == 404
    assert send("/v1/answer", b"", "audio/wav", extra=header)[0] == 415


def test_grounded_followup_uses_ocr_history_geometry_and_real_ids(relay, monkeypatch):
    _, send = relay
    data, output = qa_data(), qa_reply()

    def fake(host, path, headers, body):
        assert host == "api.openai.com" and path == "/v1/responses"
        assert headers["Authorization"] == "Bearer fake-openai"
        args = json.loads(body)
        assert args["model"] == "gpt-5.6-luna" and not args["store"]
        assert args["text"]["format"]["strict"]
        assert json.loads(args["input"]) == data
        return model_reply(output)
    monkeypatch.setattr(cloud_providers, "post", fake)
    status, result = send("/v1/answer", json.dumps(data).encode())
    assert status == 200
    assert (result["text"], result["lineIDs"], result["lineContexts"]) == (output["text"], [2], [{"id": 2, "score": 0.7}])


def test_proactive_review_corroborates_without_a_user_question_or_images(relay, monkeypatch):
    _, send = relay
    page = {"lines": qa_data()["lines"]}
    def fake(host, path, headers, body):
        args = json.loads(body)
        assert host == "api.openai.com" and path == "/v1/responses"
        assert args["model"] == "gpt-5.6-luna" and not args["store"]
        assert args["text"]["format"]["name"] == "ocr_review"
        assert json.loads(args["input"]) == page
        return model_reply({"lineContexts": qa_reply()["lineContexts"]})
    monkeypatch.setattr(cloud_providers, "post", fake)
    assert send("/v1/review", json.dumps(page).encode()) == (200, {"lineContexts": [{"id": 2, "score": 0.7}]})
    assert send("/v1/review", json.dumps({**page, "photo": "image"}).encode())[0] == 400


@pytest.mark.parametrize("citation", [
    {"supportID": 2, "supportQuote": "12월 2일"},  # Repeating the target cannot verify it.
    {"supportID": 77}, {"supportQuote": "12월 3일"}, {"supportQuote": "제출 기한"},
])
def test_review_discards_scores_without_verifiable_context(citation, monkeypatch):
    provider = Providers("offline-test-key")
    cited = {**qa_reply()["lineContexts"][0], **citation}
    monkeypatch.setattr(cloud_providers, "post", lambda *args: model_reply({"lineContexts": [cited]}))
    assert provider.review(json.dumps({"lines": qa_data()["lines"]}).encode()) == {"lineContexts": []}


def test_review_skips_cloud_call_when_there_is_nothing_to_corroborate(monkeypatch):
    provider = Providers("")
    monkeypatch.setattr(cloud_providers, "post", lambda *args: pytest.fail("unexpected provider call"))
    for lines in ([qa_data()["lines"][0]], [{**line, "uncertain": False} for line in qa_data()["lines"]]):
        assert provider.review(json.dumps({"lines": lines}).encode()) == {"lineContexts": []}


@pytest.mark.parametrize("bad", [
    {"evidence": []}, {"evidence": [{"id": 2, "quote": "12월 3일"}]},
    {"evidence": [{"id": 3, "quote": "12월 2일까지"}]},
    {"kind": "example", "evidence": [], "lineContexts": [], "referenceID": "term1"},
])
def test_invalid_grounding_does_not_reach_speech(bad):
    answer = cloud_providers.checked_answer({**qa_reply(), **bad}, qa_data())
    assert answer == cloud_providers.missing()


@pytest.mark.parametrize("scores", [
    [{**qa_reply()["lineContexts"][0], "id": 4}], [{**qa_reply()["lineContexts"][0], "score": float("nan")}],
    qa_reply()["lineContexts"] * 2, [{"id": 2, "score": 0.7}],
])
def test_invalid_confidence_keeps_the_warning_and_the_valid_answer(scores):
    answer = cloud_providers.checked_answer({**qa_reply(), "lineContexts": scores}, qa_data())
    assert answer["text"] == qa_reply()["text"] and answer["lineIDs"] == [2] and answer["lineContexts"] == []


def test_no_photo_or_unbounded_payload_forwards(relay, monkeypatch):
    _, send = relay
    monkeypatch.setattr(cloud_providers, "post", lambda *args: pytest.fail("unexpected provider call"))
    bad_inputs = [{"image": "abcd"}, {**qa_data(), "image": "abcd"},
                  {**qa_data(), "focusLineIDs": [2, 2]}, {**qa_data(), "lines": []},
                  {**qa_data(), "question": "\ud800"}]
    for bad in bad_inputs:
        assert send("/v1/answer", json.dumps(bad).encode())[0] == 400
    # The relay rejects oversized headers before the client has to upload any content.
    assert send("/v1/answer", b"", extra={"Content-Length": str(512 * 1024 + 1)})[0] == 413


def test_question_budget_counts_content_scalars_without_json_keys():
    lines = [{"id": i, "text": "글" * 1999, "uncertain": False} for i in range(50)]
    data = {"question": "문" * 50, "lines": lines}
    assert cloud_providers.check_question(data)["lines"] == lines
    with pytest.raises(cloud_providers.RelayError):
        cloud_providers.check_question({**data, "question": data["question"] + "장"})


def test_provider_failures_do_not_leak_errors(relay, monkeypatch):
    _, send = relay
    def failing(*args):
        raise RuntimeError("sensitive private debug material")
    monkeypatch.setattr(cloud_providers, "post", failing)
    status, result = send("/v1/answer", json.dumps(qa_data()).encode())
    assert status == 500 and "private" not in result["error"]


def test_env_file_is_literal_and_environment_overrides(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        monkeypatch.delenv(key, raising=False)
    path = tmp_path / ".env.cloud"
    path.write_text("OPENAI_API_KEY='fake'\nIGNORED_NAME=hello\n")
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    providers = provider_settings(path)
    assert providers == Providers("from-environment")
    assert "from-environment" not in repr(providers)
