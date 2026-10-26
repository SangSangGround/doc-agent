"""선택 전처리가 원본·좌표·저신뢰 OCR 증거를 보존하는지 검증한다."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from docagent.contracts import BoxMm, Detection, FieldType
from docagent.errors import VisionError
from docagent.vision.ocr import TesseractOcr
from docagent.vision.ocr_preprocessing import remove_form_lines


def test_removes_long_rules_but_preserves_checkbox_and_short_strokes():
    original = np.full((300, 210), 255, dtype=np.uint8)
    cv2.line(original, (20, 30), (180, 30), 0, 1)
    cv2.line(original, (20, 50), (20, 250), 0, 1)
    cv2.rectangle(original, (55, 80), (59, 84), 0, 1)
    cv2.line(original, (80, 100), (84, 100), 0, 1)
    before = original.copy()
    result = remove_form_lines(original, page_size_mm=(210, 300))
    assert np.array_equal(original, before)
    assert result.shape == original.shape
    assert np.all(result[30, 20:181] == 255)
    assert np.all(result[50:251, 20] == 255)
    assert np.array_equal(result[79:86, 54:61], original[79:86, 54:61])
    assert np.array_equal(result[99:102, 79:86], original[99:102, 79:86])


def test_does_not_remove_large_dark_area_or_change_color_outside_lines():
    original = np.full((300, 210, 3), (200, 230, 255), dtype=np.uint8)
    original[90:190, 70:170] = 0
    result = remove_form_lines(original, page_size_mm=(210, 300))
    assert np.array_equal(result, original)


@pytest.mark.parametrize("bad", [np.zeros((0, 4), np.uint8), np.zeros((4, 4, 4), np.uint8),
                                 np.zeros((4, 4), np.float32)])
def test_bad_preprocessing_input_fails_clearly(bad):
    with pytest.raises(VisionError):
        remove_form_lines(bad, page_size_mm=(210, 297))


def fake_sdk(monkeypatch, responses):
    calls = []

    def image_to_data(image, **kwargs):
        calls.append(image.copy())
        text, confidence = responses[len(calls) - 1]
        return dict(text=[text], conf=[confidence], left=[20], top=[30], width=[10], height=[5])

    sdk = SimpleNamespace(image_to_data=image_to_data, Output=SimpleNamespace(DICT="dict"))
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(lambda: sdk))
    return calls


def test_default_reads_once_and_retains_filtered_out_raw_confidence(monkeypatch):
    calls = fake_sdk(monkeypatch, [("원문", 17)])
    engine = TesseractOcr(min_confidence=0.9)
    assert engine.read(np.full((297, 210), 255, np.uint8)) == []
    assert len(calls) == 1
    assert engine.last_result.baseline_words[0].text == "원문"
    assert engine.last_result.baseline_words[0].confidence == 0.17
    assert engine.last_result.words == engine.last_result.baseline_words


def test_form_lines_does_not_choose_high_confidence_result_or_shift_coordinates(monkeypatch):
    calls = fake_sdk(monkeypatch, [("원본", 98), ("처리본", 31)])
    original = np.full((297, 210, 3), 255, np.uint8)
    original[45, 10:200] = 0
    engine = TesseractOcr(preprocess="form_lines")
    words = engine.read(original)
    result = engine.last_result
    assert len(calls) == 2
    assert calls[0].shape == calls[1].shape == original.shape
    assert calls[0][45, 100, 0] == 0
    assert calls[1][45, 100, 0] == 255
    assert original[45, 100, 0] == 0
    assert [word.text for word in words] == ["처리본"]
    assert words[0].confidence == 0.31
    assert result.baseline_words[0].confidence == 0.98
    assert words[0].box_mm == result.baseline_words[0].box_mm == BoxMm(20, 30, 10, 5)
    assert result.to_dict()["baseline_words"][0]["text"] == "원본"


def test_region_evidence_restores_both_variants_to_page_coordinates(monkeypatch):
    fake_sdk(monkeypatch, [("원본", 70), ("처리본", 50)])
    detection = Detection(FieldType.CHECKBOX, BoxMm(60, 100, 5, 5), 0.9, "test")
    engine = TesseractOcr(preprocess="form_lines")
    result = engine.read_regions(np.full((2970, 2100), 255, np.uint8), [detection])
    assert result[0].box_mm == BoxMm(17, 88, 1, 0.5)
    assert engine.last_result.baseline_words[0].box_mm == result[0].box_mm
    assert engine.last_result.words[0].confidence == 0.5
    assert engine.last_result.baseline_words[0].confidence == 0.7


def test_failed_read_cannot_leave_previous_page_evidence(monkeypatch):
    fake_sdk(monkeypatch, [("원본", 91)])
    engine = TesseractOcr()
    engine.read(np.full((297, 210), 255, np.uint8))
    with pytest.raises(VisionError):
        engine.read(np.zeros((0, 10), np.uint8))
    assert engine.last_result is None


def test_invalid_preprocess_is_rejected():
    with pytest.raises(ValueError, match="preprocess"):
        TesseractOcr(preprocess="auto_pick_best")
