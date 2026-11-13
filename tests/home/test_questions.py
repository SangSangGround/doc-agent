import copy
import json
import io
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from docagent.home.demo import create_demo
from docagent.home.document import document_guide
from docagent.home.questions import OllamaQuestions, QuestionUnavailable, resolve_selection
from docagent.home.server import HomeApplication, encode_image
from docagent.home.templates import load_template, validate_manifest


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return load_template(create_demo(tmp_path_factory.mktemp("spoken-demo") / "demo"), demo=True)


def test_signature_guide_is_backed_by_source_and_coordinates(template):
    guide = document_guide(template, "test")
    assert "신청인 서명란이 1개" in guide["summary"]
    assert "종이 아래쪽 오른쪽" in guide["summary"]
    for item in guide["references"]:
        assert item["text"] in template.manifest["pages"][0]["source_text"]
    assert len({item["id"] for item in guide["references"]}) == len(guide["references"])
    for group in template.manifest["groups"]:
        assert group["source_text"] in guide["summary"]


def test_missing_signature_registration_does_not_mean_zero(template):
    manifest = copy.deepcopy(template.manifest)
    del manifest["signatures"]
    guide = document_guide(replace(template, manifest=manifest), "test")
    assert not guide["signatures_reviewed"]
    assert "아직 등록되지" in guide["summary"]
    manifest["signatures"] = []
    assert "서명란이 없습니다" in document_guide(replace(template, manifest=manifest), "test")["summary"]


@pytest.mark.parametrize("change", [
    {"source_text": "원문에 없는 신청인 서명"}, {"role": "staff"},
    {"box_mm": [200, 280, 50, 10]}, {"before_signing_group_ids": ["missing"]},
])
def test_signature_metadata_must_reference_valid_template_evidence(template, change):
    manifest = copy.deepcopy(template.manifest)
    manifest["signatures"][0].update(change)
    with pytest.raises(ValueError):
        validate_manifest(manifest, demo=True)


@pytest.mark.parametrize("selection", [
    {"status": "found", "source_ids": ["invented"]},
    {"status": "found", "source_ids": []},
    {"status": "found", "source_ids": ["fee", "fee"]},
    {"status": "not_found", "source_ids": ["fee"]},
    {"status": "found", "source_ids": ["fee"], "answer": "수수료는 무료입니다."},
    {"status": "found", "source_ids": [["fee"]]},
    {"status": "found", "source_ids": [True]},
])
def test_model_cannot_add_or_invent_spoken_evidence(selection):
    with pytest.raises(ValueError):
        resolve_selection(selection, [{"id": "fee", "text": "수수료는 12,000원입니다.", "page": 1}])


def test_amount_is_quoted_exactly_and_absence_abstains():
    source = {"id": "fee", "text": "수수료는 12,000원이며 환불되지 않습니다.", "page": 1}
    answer = resolve_selection({"status": "found", "source_ids": ["fee"]}, [source])
    assert answer["answer"] == "원문에서 다음 내용을 확인했습니다.\n" + source["text"]
    assert answer["sources"] == [source]
    absent = resolve_selection({"status": "not_found", "source_ids": []}, [source])
    assert absent["sources"] == [] and "확인할 수 없습니다" in absent["answer"]


class FakeQuestions:
    model = "test-only"

    def __init__(self):
        self.calls = []

    def answer(self, question, references, history):
        self.calls.append((question, copy.deepcopy(references), copy.deepcopy(history)))
        return resolve_selection({"status": "found", "source_ids": [references[0]["id"]]}, references)


class FakeSpeech:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(audio)
        return "서명 전에 무엇을 확인해야 하나요?"


def test_questions_require_current_capture_and_history_is_cleared(template):
    questions, speech = FakeQuestions(), FakeSpeech()
    app = HomeApplication(template, questions=questions, speech=speech)
    with pytest.raises(ValueError):
        app.handle("ask", {"question": "질문", "capture_id": None})
    captured = app.handle("capture", {"image": encode_image(template.reference), "mode": "scan"})
    capture_id = captured["capture_id"]
    assert captured["document_ready"] and captured["document"]
    app.handle("ask", {"question": "확인할 내용?", "capture_id": capture_id})
    app.handle("ask", {"question": "그 항목은?", "capture_id": capture_id})
    assert questions.calls[1][2][0]["question"] == "확인할 내용?"
    assert app.handle("transcribe", {"audio": "test-audio", "capture_id": capture_id})["text"]
    assert speech.calls == ["test-audio"]
    app.handle("reset_capture", {})
    assert app.question_history == [] and app.state()["document"] is None
    app.handle("capture", {"image": encode_image(template.reference), "mode": "scan"})
    for action, values in [("ask", {"question": "이전 문서?"}), ("transcribe", {"audio": "stale"})]:
        with pytest.raises(ValueError):
            app.handle(action, {**values, "capture_id": capture_id})
    assert len(questions.calls) == 2 and speech.calls == ["test-audio"]


def test_failed_capture_clears_signature_and_question_access(template):
    app = HomeApplication(template, questions=FakeQuestions())
    app.handle("capture", {"image": encode_image(template.reference), "mode": "scan"})
    with pytest.raises(ValueError):
        app.handle("capture", {"image": "bad"})
    assert app.state()["document"] is None and app.state()["capture_id"] is None
    with pytest.raises(ValueError):
        app.handle("guide", {})


def test_unavailable_model_does_not_return_plausible_answer(monkeypatch):
    import urllib.request

    class Broken:
        def open(self, *args, **kwargs):
            raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Broken())
    with pytest.raises(QuestionUnavailable):
        OllamaQuestions().answer("수수료는?", [{"id": "a", "text": "원문", "page": 1}], [])


def test_oversize_document_is_not_silently_truncated():
    with pytest.raises(QuestionUnavailable, match="길이"):
        OllamaQuestions().answer("질문", [{"id": "a", "text": "원" * 10000, "page": 1}], [])


@pytest.mark.parametrize("response,accepted", [
    ({"status": "found", "source_ids": ["fee"]}, True),
    ({"status": "found", "source_ids": ["fee"], "answer": "무료"}, False),
])
def test_mlx_fallback_still_validates_source_only_json(monkeypatch, response, accepted):
    requests = []

    class LocalEngine:
        def open(self, request, **kwargs):
            payload = json.loads(request.data)
            requests.append(payload)
            if "format" in payload:
                raise urllib.error.HTTPError(request.full_url, 501, "unsupported", {},
                    io.BytesIO(b'{"error":"structured output is unavailable"}'))
            return io.BytesIO(json.dumps({"message": {"content": json.dumps(response)}}).encode())

    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: LocalEngine())
    model = OllamaQuestions()
    source = [{"id": "fee", "page": 1, "text": "수수료는 12,000원입니다."}]
    if accepted:
        assert source[0]["text"] in model.answer("수수료는?", source, [])["answer"]
        model.answer("환불은?", source, [])
    else:
        with pytest.raises(QuestionUnavailable):
            model.answer("수수료는?", source, [])
    assert "format" in requests[0]
    assert all("format" not in item for item in requests[1:])
