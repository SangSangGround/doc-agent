"""원문 대조: 한 자리 금액 변경과 새 조건을 정합 성공만으로 통과시키지 않는다."""
import copy
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from docagent.home.content_check import compare_content, normalized_text, numeric_tokens, read_printed_content
from docagent.home.demo import create_demo
from docagent.home.server import HomeApplication, encode_image
from docagent.home.templates import load_template
from docagent.testing.fonts import load_font


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    return load_template(create_demo(tmp_path_factory.mktemp("content-check") / "demo"), demo=True)


def ocr(text):
    return {"status": "read", "text": text}


def test_spacing_and_decorative_symbols_do_not_change_printed_words():
    result = compare_content("1. 문서·체크 확인\n□ 동의함 / □ 동의하지 않음\n신청인 (서명)",
                             ocr("1 문 서 : 체크 확 인\n동의 함 / 동의하지 않음\n신청인(서명)"))
    assert result["status"] == "matched"


@pytest.mark.parametrize("before,after", [
    ("수수료: 100,000원", "수수료: 180,000원"),
    ("수수료: 1,000원", "수수료: 1.000원"),
    ("수수료: -100원", "수수료: 100원"),
    ("금리: 10%", "금리: 10"),
    ("금리: .5%", "금리: 5%"),
    ("허용 차이: ±100원", "허용 차이: 100원"),
    ("나이 ≤ 18", "나이 ≥ 18"),
    ("환불됩니다.", "환불되지 않습니다."),
    ("신청인 (서명)", "신청인 (서명)\n아래 서명란은 사용하지 않습니다."),
    ("서명 전에 확인하세요.", "서명 전에 확인하세요. 수수료: 200,000원"),
])
def test_changed_amount_condition_or_added_text_is_not_accepted(before, after):
    assert compare_content(before, ocr(after))["status"] == "mismatch"


def test_numeric_notation_is_checked_even_when_letters_and_digits_match():
    assert normalized_text("수수료 1.000원") == normalized_text("수수료 1,000원")
    assert numeric_tokens("수수료 1.000원") != numeric_tokens("수수료 1,000원")
    assert compare_content("수수료 1.000원", ocr("수수료 1,000원"))["status"] == "mismatch"


@pytest.mark.parametrize("result", [{"status": "unavailable", "text": ""}, {"status": "empty", "text": ""},
                                    {"status": "read", "text": ""}])
def test_missing_ocr_is_not_confirmation(result):
    assert compare_content("수수료는 100,000원입니다.", result)["status"] == "unavailable"


def test_unreadable_additional_ink_line_cannot_be_silently_dropped(monkeypatch):
    image = np.full((1800, 1273), 255, np.uint8)
    image[100:113, 100:145] = 0
    image[300:313, 100:145] = 0
    words = iter([[SimpleNamespace(text="읽은문구", box_mm=SimpleNamespace(x_mm=10))], []])
    monkeypatch.setattr("docagent.home.content_check.TesseractOcr.read", lambda *_: next(words))
    result = read_printed_content(image, {"page_size_mm": [210, 297], "groups": []})
    assert result["text"] == "읽은문구" and result["unreadable_lines"] == 1
    assert compare_content("읽은문구", result)["status"] == "unavailable"


def test_disabled_ocr_cannot_bypass_real_web_content_gate(template):
    app = HomeApplication(template, read_ocr=False, require_content_match=True)
    captured = app.handle("capture", {"image": encode_image(template.reference), "mode": "scan"})
    assert captured["observation"]["alignment"]["reason"] == "registered"
    assert not captured["document_ready"] and not captured["can_verify"]
    assert captured["document"] is None and captured["capture_id"] is None
    assert captured["content_check"]["status"] == "disabled"
    for action, data in [("read", {}), ("guide", {}), ("ask", {"question": "수수료는?", "capture_id": None}), ("verify", {})]:
        with pytest.raises(ValueError):
            app.handle(action, data)


def test_unavailable_ocr_revokes_previously_readable_document(template, monkeypatch):
    app = HomeApplication(template, read_ocr=True)
    monkeypatch.setattr(app, "_read_capture", lambda *_: ocr(template.manifest["pages"][0]["source_text"]))
    image = encode_image(template.reference)
    assert app.handle("capture", {"image": image, "mode": "scan"})["document_ready"]
    monkeypatch.setattr(app, "_read_capture", lambda *_: {"status": "unavailable", "text": ""})
    blocked = app.handle("capture", {"image": image, "mode": "scan"})
    assert not blocked["document_ready"] and blocked["document"] is None
    assert blocked["capture_id"] is None and not blocked["can_verify"]
    assert blocked["content_check"]["status"] == "unavailable"


def require_tesseract():
    if os.environ.get("DOCAGENT_TEST_OFFLINE") or not shutil.which("tesseract"):
        pytest.skip("실제 로컬 Tesseract 한국어 통합 검사는 별도 실행")
    pytesseract = pytest.importorskip("pytesseract")
    if "kor" not in pytesseract.get_languages():
        pytest.skip("Tesseract 한국어 데이터가 필요합니다")


def with_fee(template, fee):
    image = Image.fromarray(cv2.cvtColor(template.reference, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((108, 1380), f"수수료: {fee}원", font=load_font(size_px=23), fill="black")
    manifest = copy.deepcopy(template.manifest)
    original = manifest["pages"][0]["source_text"]
    manifest["pages"][0]["source_text"] = original.replace("서명 전에", f"수수료: {fee}원\n서명 전에", 1)
    return replace(template, reference=cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), manifest=manifest)


def test_real_ocr_rejects_single_digit_amount_change_after_alignment(template):
    require_tesseract()
    reference = with_fee(template, "100,000")
    modified = with_fee(template, "180,000")
    app = HomeApplication(reference, read_ocr=True)
    normal = app.handle("capture", {"image": encode_image(reference.reference), "mode": "scan"})
    assert normal["document_ready"] and normal["content_check"]["status"] == "matched"
    changed = app.handle("capture", {"image": encode_image(modified.reference), "mode": "scan"})
    assert changed["observation"]["alignment"]["reason"] == "registered"
    assert not changed["document_ready"] and changed["content_check"]["status"] == "mismatch"
    assert not changed["content_check"]["numbers_match"]
    with pytest.raises(ValueError):
        app.handle("read", {"scope": "page"})


@pytest.mark.parametrize("ink", [0, 190])
def test_real_ocr_rejects_added_signature_cancellation(template, ink):
    require_tesseract()
    image = Image.fromarray(cv2.cvtColor(template.reference, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(image).text((550, 1480), "아래 서명란은 사용하지 않습니다.", font=load_font(size_px=23), fill=(ink, ink, ink))
    modified = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    app = HomeApplication(template, read_ocr=True)
    changed = app.handle("capture", {"image": encode_image(modified), "mode": "scan"})
    assert changed["observation"]["alignment"]["reason"] == "registered"
    assert not changed["document_ready"] and changed["content_check"]["status"] == "mismatch"
    assert "아래서명란은사용하지않습니다" in normalized_text(changed["capture_ocr"]["text"])
    with pytest.raises(ValueError):
        app.handle("guide", {})


def test_real_ocr_allows_marked_checkboxes_without_changing_original(template):
    require_tesseract()
    marked = cv2.imread(str(template.reference_path.with_name("correct.png")))
    app = HomeApplication(template, read_ocr=True)
    result = app.handle("capture", {"image": encode_image(marked), "mode": "scan"})
    assert result["document_ready"] and result["content_check"]["status"] == "matched"


def test_real_ocr_handles_page_border_after_camera_normalization(template):
    require_tesseract()
    image = cv2.resize(template.reference, (round(template.reference.shape[1] * 1200 / template.reference.shape[0]), 1200))
    image = cv2.copyMakeBorder(image, 80, 80, 100, 100, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    app = HomeApplication(template, read_ocr=True)
    captured = app.handle("capture", {"image": encode_image(image), "mode": "photo"})
    assert captured["document_ready"] and captured["content_check"]["status"] == "matched"
