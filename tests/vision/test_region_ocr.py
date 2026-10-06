"""Crop 좌표, 중복 영역, 로컬 OCR 및 실제 구조화 연결을 검증한다."""
from types import SimpleNamespace

import numpy as np
import pytest

from docagent.contracts import BoxMm, Detection, FieldRole, FieldType, OcrWord
from docagent.errors import VisionError
from docagent.vision.ocr import TesseractOcr
from docagent.vision.regions import detection_regions
from docagent.vision.structuring import build_structure


def detect(x=60, y=100, w=5, h=5, kind=FieldType.CHECKBOX):
    return Detection(kind, BoxMm(x, y, w, h), 0.99, "test")


def word(text, x, y=100, w=10, confidence=0.98):
    return OcrWord(text, BoxMm(x, y, w, 4), confidence)


@pytest.mark.parametrize("scale", [2, 10])
def test_crop_clamps_all_page_edges(scale):
    regions = detection_regions(
        [detect(0, 0), detect(205, 292)], (297 * scale, 210 * scale, 3)
    )
    assert regions[0].box_mm == BoxMm(0, 0, 50, 13)
    assert regions[1].box_mm == BoxMm(160, 277, 50, 20)
    assert regions[0].box_px.x == 0
    last = regions[1].box_px
    assert last.x + last.w == 210 * scale
    assert last.y + last.h == 297 * scale


def test_overlapping_crops_are_read_once():
    boxes = [detect(60), detect(110), detect(150)]
    regions = detection_regions(boxes, (2970, 2100, 3))
    assert len(regions) == 1
    assert regions[0].box_mm == BoxMm(15, 85, 185, 28)
    assert detection_regions(list(reversed(boxes)), (2970, 2100, 3)) == regions
    assert detection_regions([], (2970, 2100, 3)) == []


@pytest.mark.parametrize("box", [detect(220), detect(w=0), detect(x=float('nan'))])
def test_invalid_detection_is_not_silently_skipped(box):
    with pytest.raises(VisionError):
        detection_regions([box], (2970, 2100, 3))


def test_crop_ocr_restores_page_coordinates_and_confidence(monkeypatch):
    calls = []

    def image_to_data(image, **kwargs):
        calls.append(image)
        # Union crop is x=15, y=85, w=145, h=28 mm at 10 px/mm.
        return dict(text=["동의함", "동의하지 않음"], conf=[97, 41],
                    left=[520, 1020], top=[150, 150], width=[100, 250], height=[40, 40])

    sdk = SimpleNamespace(image_to_data=image_to_data, Output=SimpleNamespace(DICT="dict"))
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(lambda: sdk))
    image = np.full((2970, 2100, 3), (10, 20, 30), dtype=np.uint8)
    engine = TesseractOcr(config="--psm 11")
    detections = [detect(), detect(110)]
    words = engine.read_regions(image, detections)
    assert len(calls) == 1
    assert calls[0].shape == (280, 1450, 3)
    assert calls[0][0, 0].tolist() == [30, 20, 10]  # BGR → RGB
    assert words[0].box_mm == BoxMm(67, 100, 10, 4)
    assert words[1].box_mm == BoxMm(117, 100, 25, 4)
    assert words[1].confidence == 0.41
    assert engine.page_size_mm == (210, 297)
    structure = build_structure(detections, words)
    assert [o.label for o in structure.fields[0].options] == ["동의함", "동의하지 않음"]
    assert structure.fields[0].confidence <= 0.41
    assert structure.warnings


def test_rounding_does_not_stretch_crop_to_a4(monkeypatch):
    def read(self, image):
        # Word fills exactly the crop, exercising both scale axes and offset.
        return [OcrWord("label", BoxMm(0, 0, *self.page_size_mm), 0.9)]
    monkeypatch.setattr(TesseractOcr, "read", read)
    image = np.zeros((842, 595), dtype=np.uint8)
    result = TesseractOcr().read_regions(image, [detect()])[0]
    assert result.box_mm.x_mm == pytest.approx(42 * 210 / 595)
    assert result.box_mm.y_mm == pytest.approx(240 * 297 / 842)
    assert result.box_mm.right_mm == pytest.approx(312 * 210 / 595)
    assert result.box_mm.bottom_mm == pytest.approx(321 * 297 / 842)


def test_empty_detections_do_not_call_ocr(monkeypatch):
    def forbidden():
        pytest.fail("no crop should require OCR")
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(forbidden))
    assert TesseractOcr().read_regions(np.zeros((297, 210)), []) == []


def test_ocr_failure_is_not_treated_as_empty_result(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("OCR failed")
    sdk = SimpleNamespace(image_to_data=fail, Output=SimpleNamespace(DICT="dict"))
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(lambda: sdk))
    with pytest.raises(VisionError, match="OCR 실행"):
        TesseractOcr().read_regions(np.zeros((297, 210)), [detect()])


def test_preflight_reports_missing_korean_language(monkeypatch):
    sdk = SimpleNamespace(get_languages=lambda **kw: ["eng"])
    monkeypatch.setattr(TesseractOcr, "_import_pytesseract", staticmethod(lambda: sdk))
    with pytest.raises(VisionError, match="kor"):
        TesseractOcr().check_available()


@pytest.mark.parametrize("positions", [(67, 117), (48, 98)])
def test_matches_right_and_left_label_layouts(positions):
    structure = build_structure([detect(), detect(110)], [
        word("동의함", positions[0]), word("거부함", positions[1]),
    ])
    assert [o.label for o in structure.fields[0].options] == ["동의함", "거부함"]


def test_missing_option_does_not_reuse_neighbor_text():
    structure = build_structure([detect(), detect(85)], [word("동의함", 67)])
    labels = [o.label for o in structure.fields[0].options]
    assert labels == ["동의함", ""]
    assert any("선택지 문구" in warning for warning in structure.warnings)


def test_equidistant_label_is_ambiguous():
    structure = build_structure([detect(), detect(89)], [word("동의함", 72)])
    assert all(not o.label for o in structure.fields[0].options)
    assert structure.fields[0].confidence < 0.85


def test_signature_label_above_box_and_ocr_confidence():
    structure = build_structure([detect(100, 180, 30, 10, FieldType.SIGNATURE)], [
        word("신청인", 100, 172, confidence=0.4), word("서명", 113, 172),
    ])
    signature = structure.fields[0]
    assert signature.role is FieldRole.APPLICANT
    assert "신청인" in signature.title
    assert signature.confidence <= 0.4


def test_ocr_syllable_spacing_does_not_turn_date_into_signature():
    structure = build_structure([detect(100, 180, 30, 10, FieldType.SIGNATURE)], [
        word("신 청 일 자", 70, 180, w=25),
    ])
    assert structure.fields[0].type is FieldType.DATE
    assert structure.fields[0].title == "신 청 일 자"  # keep original OCR text


def test_signature_role_tolerates_ocr_syllable_spacing():
    structure = build_structure([detect(100, 180, 30, 10, FieldType.SIGNATURE)], [
        word("신 청 인", 70, 180, w=25),
    ])
    assert structure.fields[0].role is FieldRole.APPLICANT
