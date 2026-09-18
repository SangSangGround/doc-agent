"""Verify 단계(:mod:`docagent.vision.verify`) 테스트.

핵심 단언은 **거짓양성 0** 이다. 체크하지 않은 칸, 서명하지 않은 칸, 조명만
바뀐 이미지에서 ``written=True`` 가 하나라도 나오면 사용자는 "다 썼습니다"라는
잘못된 안내를 듣게 된다. 잘못 쓴 것보다 위험한 실패다.

이미지는 ``written_form`` 픽스처(``clean_form.image`` 와 픽셀 정렬이 보장된
기입본)를 쓰며, 해상도·JPEG 열화·조명 기울기를 바꿔 가며 같은 판정이 유지되는지
확인한다. 모듈 scope 픽스처 이미지는 절대 직접 수정하지 않고 ``copy()`` 로 다룬다.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from docagent.contracts import (
    PARTIAL_THRESHOLD,
    BoxMm,
    DocumentStructure,
    Field,
    FieldType,
    VerificationResult,
)
from docagent.errors import VisionError
from docagent.testing.synthetic import SyntheticForm
from docagent.vision.verify import (
    CHECKBOX_INK_DELTA_THRESHOLD,
    SIGNATURE_INK_DELTA_THRESHOLD,
    ink_ratio,
    unwritten_required,
    verify_checkbox,
    verify_field,
    verify_options,
    verify_region,
    verify_required_fields,
    verify_signature,
)


def apply_illumination(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """대각 방향 곱셈 조명 기울기를 입힌 **새 이미지**를 만든다.

    :param image: 원본 이미지(변경하지 않는다).
    :param low: 좌상단 밝기 배율.
    :param high: 우하단 밝기 배율.
    :returns: 같은 크기의 uint8 이미지.
    """
    height, width = image.shape[:2]
    ramp_y = np.linspace(low, high, height, dtype=np.float32)[:, None]
    ramp_x = np.linspace(low, high, width, dtype=np.float32)[None, :]
    field = (ramp_y + ramp_x) / 2.0
    if image.ndim == 3:
        field = field[:, :, None]
    return np.clip(image.astype(np.float32) * field, 0.0, 255.0).astype(np.uint8)


#: ``written_pair`` 픽스처가 돌려주는 묶음 타입.
WrittenPair = tuple[SyntheticForm, np.ndarray, np.ndarray, tuple[float, float]]


@pytest.fixture(scope="module")
def written_pair(
    written_form: tuple[SyntheticForm, np.ndarray],
) -> tuple[SyntheticForm, np.ndarray, np.ndarray, tuple[float, float]]:
    """``(form, 기입 전 이미지, 기입 후 이미지, 페이지 크기 mm)`` 묶음."""
    form, written = written_form
    return (form, form.image, written, form.truth.page_size_mm)


@pytest.fixture(scope="module")
def blank_after(
    clean_form: SyntheticForm, synthetic_module: ModuleType
) -> np.ndarray:
    """아무것도 기입하지 않은 '작성 후' 이미지(원본과 바이트 단위로 동일)."""
    return synthetic_module.render_written(clean_form, checked_option=None, sign=False)


# --------------------------------------------------------------------------
# 체크박스
# --------------------------------------------------------------------------


class TestVerifyCheckbox:
    """체크칸 판정 — 체크한 칸만 True."""

    def test_checked_option_is_detected(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """체크한 칸은 written=True 이고 잉크가 뚜렷하게 늘어난다."""
        form, before, after, coords = written_pair
        box = form.option_box(synthetic_module.AGREE_LABEL)
        result = verify_checkbox(before, after, coords, box, field_id="consent_01:동의함")
        assert result.written is True
        assert result.ink_delta >= CHECKBOX_INK_DELTA_THRESHOLD * 2
        assert result.field_id == "consent_01:동의함"
        assert "체크" in result.reason

    def test_unchecked_option_has_no_false_positive(
        self, written_pair: WrittenPair, synthetic_module: ModuleType
    ) -> None:
        """체크하지 않은 칸은 절대 True 가 되지 않는다(거짓양성 0)."""
        form, before, after, coords = written_pair
        box = form.option_box(synthetic_module.DISAGREE_LABEL)
        result = verify_checkbox(before, after, coords, box)
        assert result.written is False
        assert result.ink_delta < CHECKBOX_INK_DELTA_THRESHOLD

    def test_nothing_written_is_false_for_every_option(
        self, clean_form: SyntheticForm, blank_after: np.ndarray
    ) -> None:
        """아무것도 쓰지 않으면 모든 선택지가 False 다."""
        coords = clean_form.truth.page_size_mm
        for option in clean_form.field("consent_01").options:
            result = verify_checkbox(clean_form.image, blank_after, coords, option.box_mm)
            assert result.written is False
            assert result.ink_delta == pytest.approx(0.0, abs=1e-6)

    def test_illumination_change_alone_is_not_written(
        self, clean_form: SyntheticForm, synthetic_module: ModuleType
    ) -> None:
        """조명만 크게 바뀐 이미지에서 오탐이 나지 않는다."""
        coords = clean_form.truth.page_size_mm
        lit = apply_illumination(clean_form.image, 0.55, 1.15)
        for label in (synthetic_module.AGREE_LABEL, synthetic_module.DISAGREE_LABEL):
            result = verify_checkbox(
                clean_form.image, lit, coords, clean_form.option_box(label)
            )
            assert result.written is False

    def test_illumination_change_does_not_hide_a_real_check(
        self, written_pair: WrittenPair, synthetic_module: ModuleType
    ) -> None:
        """조명이 바뀌어도 실제 체크는 놓치지 않는다(미탐 방지)."""
        form, before, after, coords = written_pair
        lit_after = apply_illumination(after, 0.55, 1.15)
        result = verify_checkbox(
            before, lit_after, coords, form.option_box(synthetic_module.AGREE_LABEL)
        )
        assert result.written is True

    def test_small_misalignment_does_not_create_false_positive(
        self, clean_form: SyntheticForm, synthetic_module: ModuleType
    ) -> None:
        """1~3px(<0.4mm) 정합 오차로는 체크 판정이 뒤집히지 않는다."""
        coords = clean_form.truth.page_size_mm
        for shift in (1, 2, 3):
            rolled = np.roll(np.roll(clean_form.image, shift, axis=0), shift, axis=1)
            for label in (synthetic_module.AGREE_LABEL, synthetic_module.DISAGREE_LABEL):
                result = verify_checkbox(
                    clean_form.image, rolled, coords, clean_form.option_box(label)
                )
                assert result.written is False

    def test_option_box_is_required(self, written_pair: WrittenPair) -> None:
        """체크칸 좌표 없이 호출하면 한국어 ValueError."""
        _, before, after, coords = written_pair
        with pytest.raises(ValueError):
            verify_checkbox(before, after, coords, None)


# --------------------------------------------------------------------------
# 서명
# --------------------------------------------------------------------------


class TestVerifySignature:
    """서명 판정 — 내용은 읽지 않고 획의 존재만 본다."""

    def test_signed_box_is_detected(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """서명 후에는 written=True 이며 근거 문장에 '읽지 않았다'가 명시된다."""
        form, before, after, coords = written_pair
        box = form.field(synthetic_module.APPLICANT_SIGNATURE_FIELD_ID).box_mm
        result = verify_signature(before, after, coords, box, field_id="signature_01")
        assert result.written is True
        assert result.confidence > PARTIAL_THRESHOLD
        assert "읽지 않았습니다" in result.reason

    def test_unsigned_box_is_false(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """서명하지 않은 대리인란은 False 다(밑줄을 서명으로 오인하지 않는다)."""
        form, before, after, coords = written_pair
        box = form.field(synthetic_module.REPRESENTATIVE_SIGNATURE_FIELD_ID).box_mm
        result = verify_signature(before, after, coords, box)
        assert result.written is False

    def test_before_signing_is_false(
        self, clean_form: SyntheticForm, blank_after: np.ndarray, synthetic_module: ModuleType
    ) -> None:
        """서명 전 이미지끼리 비교하면 False 다."""
        coords = clean_form.truth.page_size_mm
        box = clean_form.field(synthetic_module.APPLICANT_SIGNATURE_FIELD_ID).box_mm
        result = verify_signature(clean_form.image, blank_after, coords, box)
        assert result.written is False

    def test_illumination_change_alone_is_not_signature(
        self, clean_form: SyntheticForm, synthetic_module: ModuleType
    ) -> None:
        """조명 변화만으로 서명 판정이 뒤집히지 않는다."""
        coords = clean_form.truth.page_size_mm
        lit = apply_illumination(clean_form.image, 0.55, 1.15)
        box = clean_form.field(synthetic_module.APPLICANT_SIGNATURE_FIELD_ID).box_mm
        assert verify_signature(clean_form.image, lit, coords, box).written is False

    def test_signature_survives_low_resolution_and_jpeg(
        self, synthetic_module: ModuleType
    ) -> None:
        """150dpi + JPEG 열화 + 잡음에서도 체크·서명 판정이 유지된다."""
        spec = synthetic_module.FormSpec(
            document_id="verify_lowres",
            dpi=150,
            jpeg_quality=70,
            noise_sigma=2.5,
            include_representative=True,
        )
        form = synthetic_module.make_application_form(spec)
        after = synthetic_module.render_written(
            form, checked_option=synthetic_module.AGREE_LABEL, sign=True
        )
        coords = form.truth.page_size_mm
        agree = form.option_box(synthetic_module.AGREE_LABEL)
        disagree = form.option_box(synthetic_module.DISAGREE_LABEL)
        signature = form.field(synthetic_module.APPLICANT_SIGNATURE_FIELD_ID).box_mm
        assert verify_checkbox(form.image, after, coords, agree).written is True
        assert verify_checkbox(form.image, after, coords, disagree).written is False
        assert verify_signature(form.image, after, coords, signature).written is True

    def test_signature_box_is_required(self, written_pair: WrittenPair) -> None:
        """서명란 좌표 없이 호출하면 한국어 ValueError."""
        _, before, after, coords = written_pair
        with pytest.raises(ValueError):
            verify_signature(before, after, coords, None)


# --------------------------------------------------------------------------
# 일반 영역 · 잉크 비율
# --------------------------------------------------------------------------


class TestVerifyRegionAndInkRatio:
    """문자·날짜란 검증과 잉크 비율 측정."""

    def test_untouched_text_field_is_false(self, written_pair: WrittenPair) -> None:
        """건드리지 않은 인적사항 칸은 False 다(인쇄된 예시 값에 속지 않는다)."""
        form, before, after, coords = written_pair
        box = form.field("applicant_name").box_mm
        result = verify_region(before, after, coords, box, field_id="text_01")
        assert result.written is False
        assert result.field_id == "text_01"

    def test_ink_ratio_is_stable_under_illumination(self, clean_form: SyntheticForm) -> None:
        """조명이 바뀌어도 잉크 비율이 거의 변하지 않는다(정규화 검증)."""
        coords = clean_form.truth.page_size_mm
        box = clean_form.field("applicant_name").box_mm
        base = ink_ratio(clean_form.image, box, coords)
        lit = ink_ratio(apply_illumination(clean_form.image, 0.55, 1.15), box, coords)
        assert base > 0.0
        assert abs(base - lit) < 0.01

    def test_region_validates_threshold(self, written_pair: WrittenPair) -> None:
        """0 이하 임계값은 거부한다."""
        form, before, after, coords = written_pair
        box = form.field("applicant_name").box_mm
        with pytest.raises(ValueError):
            verify_region(before, after, coords, box, threshold=0.0)
        with pytest.raises(ValueError):
            verify_region(before, after, coords, None)


# --------------------------------------------------------------------------
# 통합 진입점
# --------------------------------------------------------------------------


class TestVerifyField:
    """:func:`verify_field` 가 항목 유형에 맞는 방법을 고르는지 확인한다."""

    def test_choice_field_reports_checked_option(
        self, written_pair: WrittenPair, synthetic_module: ModuleType
    ) -> None:
        """선택형은 체크된 선택지 라벨을 근거 문장에 담는다."""
        form, before, after, coords = written_pair
        result = verify_field(
            form.truth, synthetic_module.CONSENT_FIELD_ID, before, after, coords
        )
        assert result.written is True
        assert result.field_id == synthetic_module.CONSENT_FIELD_ID
        assert synthetic_module.AGREE_LABEL in result.reason

    def test_choice_field_without_marks_is_false(
        self, clean_form: SyntheticForm, blank_after: np.ndarray, synthetic_module: ModuleType
    ) -> None:
        """아무 선택지도 체크되지 않으면 False 이고 그 사실을 문장으로 남긴다."""
        coords = clean_form.truth.page_size_mm
        result = verify_field(
            clean_form.truth,
            synthetic_module.CONSENT_FIELD_ID,
            clean_form.image,
            blank_after,
            coords,
        )
        assert result.written is False
        assert "확인하지 못했습니다" in result.reason

    def test_signature_field_dispatch(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """SIGNATURE 항목은 서명 검증 경로로 간다."""
        form, before, after, coords = written_pair
        signed = verify_field(
            form.truth,
            synthetic_module.APPLICANT_SIGNATURE_FIELD_ID,
            before,
            after,
            coords,
        )
        unsigned = verify_field(
            form.truth,
            synthetic_module.REPRESENTATIVE_SIGNATURE_FIELD_ID,
            before,
            after,
            coords,
        )
        assert signed.written is True
        assert unsigned.written is False

    def test_text_field_dispatch(self, written_pair: WrittenPair) -> None:
        """TEXT_INPUT 항목은 영역 검증 경로로 간다."""
        form, before, after, coords = written_pair
        result = verify_field(form.truth, "applicant_name", before, after, coords)
        assert result.written is False

    def test_verify_options_returns_one_result_per_option(
        self, written_pair: WrittenPair, synthetic_module: ModuleType
    ) -> None:
        """선택지별 결과가 라벨과 함께 하나씩 나온다."""
        form, before, after, coords = written_pair
        results = verify_options(
            form.truth, synthetic_module.CONSENT_FIELD_ID, before, after, coords
        )
        assert len(results) == 2
        assert [r.written for r in results] == [True, False]
        assert results[0].field_id.endswith(synthetic_module.AGREE_LABEL)
        assert results[1].field_id.endswith(synthetic_module.DISAGREE_LABEL)

    def test_required_fields_summary(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """필수 항목 일괄 검증에서 아직 안 쓴 항목만 골라낸다."""
        form, before, after, coords = written_pair
        results = verify_required_fields(form.truth, before, after, coords)
        assert len(results) == len(form.truth.required_fields())
        pending = unwritten_required(results)
        assert synthetic_module.CONSENT_FIELD_ID not in pending
        assert synthetic_module.APPLICANT_SIGNATURE_FIELD_ID not in pending
        assert "applicant_name" in pending

    def test_unknown_field_id_is_rejected(self, written_pair: WrittenPair) -> None:
        """없는 항목 id 는 허용 값을 안내하는 ValueError."""
        form, before, after, coords = written_pair
        with pytest.raises(ValueError) as info:
            verify_field(form.truth, "없는항목", before, after, coords)
        assert "없는항목" in str(info.value)

    def test_field_without_coordinates_raises(self, written_pair: WrittenPair) -> None:
        """좌표가 없는 항목은 조용히 넘어가지 않고 VisionError 를 던진다."""
        _, before, after, coords = written_pair
        structure = DocumentStructure(
            document_id="no_box",
            fields=(Field(id="text_01", type=FieldType.TEXT_INPUT, box_mm=None),),
        )
        with pytest.raises(VisionError):
            verify_field(structure, "text_01", before, after, coords)


class TestCoordinateFrameGuards:
    """좌표계 사전 검증 — 조용한 오판정을 막는다."""

    def test_size_mismatch_raises(self, written_pair: WrittenPair) -> None:
        """전/후 이미지 크기가 다르면 VisionError."""
        form, before, after, coords = written_pair
        cropped = after[:-10, :-10]
        with pytest.raises(VisionError) as info:
            verify_checkbox(before, cropped, coords, form.option_box("동의함"))
        assert "좌표계" in str(info.value)

    def test_invalid_image_raises(self, written_pair: WrittenPair) -> None:
        """1차원 배열 같은 잘못된 입력은 VisionError."""
        form, before, _, coords = written_pair
        with pytest.raises(VisionError):
            verify_checkbox(
                np.zeros((5,), dtype=np.uint8),
                np.zeros((5,), dtype=np.uint8),
                coords,
                form.option_box("동의함"),
            )

    def test_invalid_coords_raise_value_error(self, written_pair: WrittenPair) -> None:
        """페이지 크기가 0 이하면 ValueError."""
        form, before, after, _ = written_pair
        with pytest.raises(ValueError):
            verify_checkbox(before, after, (0.0, 297.0), form.option_box("동의함"))

    def test_out_of_page_box_raises(self, written_pair: WrittenPair) -> None:
        """페이지 밖 영역은 조용히 0 을 돌려주지 않고 VisionError."""
        _, before, after, coords = written_pair
        with pytest.raises(VisionError):
            verify_checkbox(before, after, coords, BoxMm(400.0, 400.0, 6.0, 6.0))


class TestResultContract:
    """:class:`VerificationResult` 계약 준수."""

    def test_result_roundtrip(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """결과 dict 왕복이 무손실이다."""
        form, before, after, coords = written_pair
        result = verify_checkbox(
            before, after, coords, form.option_box(synthetic_module.AGREE_LABEL)
        )
        assert VerificationResult.from_dict(result.to_dict()) == result

    def test_confidence_is_bounded(self, written_pair: WrittenPair, synthetic_module: ModuleType) -> None:
        """모든 판정의 신뢰도가 0.0~1.0 범위 안에 있다."""
        form, before, after, coords = written_pair
        boxes = [
            form.option_box(synthetic_module.AGREE_LABEL),
            form.option_box(synthetic_module.DISAGREE_LABEL),
        ]
        for box in boxes:
            result = verify_checkbox(before, after, coords, box)
            assert 0.0 <= result.confidence <= 1.0

    def test_boundary_confidence_is_low(self) -> None:
        """판정 경계에 걸린 증가량은 신뢰도가 낮게 나온다(재확인 트리거)."""
        from docagent.vision.verify import BOUNDARY_CONFIDENCE, _confidence

        assert _confidence(CHECKBOX_INK_DELTA_THRESHOLD, CHECKBOX_INK_DELTA_THRESHOLD) == (
            pytest.approx(BOUNDARY_CONFIDENCE)
        )
        assert _confidence(0.0, SIGNATURE_INK_DELTA_THRESHOLD) > BOUNDARY_CONFIDENCE
