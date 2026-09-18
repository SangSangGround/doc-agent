"""합성 신청서 생성기 테스트.

이후 모든 Vision 테스트가 이 생성기에 의존하므로, 여기서는
**API 안정성 · 결정론 · 정답 정확성** 세 가지를 집중적으로 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    DocumentStructure,
    FieldRole,
    FieldType,
    Sensitivity,
)
from docagent.testing import fonts
from docagent.testing.synthetic import (
    AGREE_LABEL,
    APPLICANT_SIGNATURE_FIELD_ID,
    CONSENT_FIELD_ID,
    DATE_FIELD_ID,
    DEFAULT_SEED,
    DISAGREE_LABEL,
    REPRESENTATIVE_SIGNATURE_FIELD_ID,
    FormSpec,
    build_layout,
    build_truth,
    expected_written_truth,
    iter_spec_grid,
    make_application_form,
    mm_to_px,
    page_size_px,
    px_per_mm,
    render_written,
)


def _ink_ratio(image: np.ndarray, box) -> float:
    """픽셀 사각형 안의 잉크(어두운 픽셀) 비율을 잰다.

    :param image: ``(H, W, 3)`` uint8 BGR 이미지.
    :param box: :class:`~docagent.contracts.BoxPx`.
    :returns: 0.0~1.0 비율.
    """
    patch = image[box.y : box.y + box.h, box.x : box.x + box.w]
    gray = patch.mean(axis=2)
    return float((gray < 128).mean())


# --------------------------------------------------------------------------
# 단위 변환
# --------------------------------------------------------------------------


class TestUnits:
    """mm ↔ px 환산 헬퍼."""

    def test_px_per_mm_matches_inch_definition(self) -> None:
        assert px_per_mm(254) == pytest.approx(10.0)

    def test_mm_to_px_rounds(self) -> None:
        assert mm_to_px(25.4, 300) == 300

    def test_page_size_px_matches_a4(self) -> None:
        width, height = page_size_px(300)
        assert width == mm_to_px(A4_WIDTH_MM, 300)
        assert height == mm_to_px(A4_HEIGHT_MM, 300)

    def test_px_per_mm_rejects_invalid_dpi(self) -> None:
        with pytest.raises(ValueError, match="dpi"):
            px_per_mm(0)


# --------------------------------------------------------------------------
# 사양 검증
# --------------------------------------------------------------------------


class TestFormSpec:
    """:class:`FormSpec` 파라미터 검증 — 조용한 실패 금지."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dpi": 10},
            {"dpi": 5000},
            {"rotation_deg": 90.0},
            {"perspective_strength": 1.5},
            {"margin_px": -1},
            {"background_gray": 300},
            {"noise_sigma": -0.5},
            {"blur_ksize": 4},
            {"blur_ksize": 0},
            {"jpeg_quality": 0},
            {"jpeg_quality": 101},
            {"illumination_gradient": -0.1},
            {"document_id": ""},
        ],
    )
    def test_invalid_parameters_raise_value_error(self, overrides: dict) -> None:
        with pytest.raises(ValueError):
            FormSpec(**overrides)

    def test_error_message_is_korean(self) -> None:
        with pytest.raises(ValueError) as info:
            FormSpec(blur_ksize=4)
        assert "홀수" in str(info.value)

    def test_with_returns_new_instance(self) -> None:
        base = FormSpec()
        derived = base.with_(dpi=300)
        assert base.dpi != derived.dpi
        assert derived.seed == base.seed

    def test_has_geometry_change_flag(self) -> None:
        assert FormSpec().has_geometry_change is False
        assert FormSpec(rotation_deg=5.0).has_geometry_change is True
        assert FormSpec(margin_px=10).has_geometry_change is True
        assert FormSpec(perspective_strength=0.1).has_geometry_change is True
        # 화질 저하만으로는 기하가 바뀌지 않는다.
        assert FormSpec(noise_sigma=5.0, jpeg_quality=50).has_geometry_change is False


# --------------------------------------------------------------------------
# 이미지 기본 속성
# --------------------------------------------------------------------------


class TestImageBasics:
    """이미지 형태·dtype·크기."""

    def test_clean_form_shape_and_dtype(self, clean_form) -> None:
        assert clean_form.image.dtype == np.uint8
        assert clean_form.image.ndim == 3
        assert clean_form.image.shape[2] == 3

    def test_clean_form_size_matches_dpi(self, clean_form) -> None:
        width, height = page_size_px(clean_form.dpi)
        assert clean_form.image_size_px == (width, height)
        assert clean_form.page_px == (width, height)

    def test_page_is_mostly_white_paper(self, clean_form) -> None:
        # 변형이 없으면 이미지 전체가 문서이므로 대부분이 흰 종이다.
        assert float((clean_form.image.mean(axis=2) > 200).mean()) > 0.8

    def test_lowres_form_is_smaller_than_default(self, lowres_form, clean_form) -> None:
        assert lowres_form.dpi == 150
        assert lowres_form.image_size_px[0] < clean_form.image_size_px[0]

    def test_document_actually_has_ink(self, clean_form) -> None:
        table = clean_form.layout.table_box_mm
        assert _ink_ratio(clean_form.image, clean_form.flat_box_px(table)) > 0.005


# --------------------------------------------------------------------------
# 기하 변형
# --------------------------------------------------------------------------


class TestGeometry:
    """회전·원근·여백 변형."""

    def test_margin_creates_background_border(self) -> None:
        spec = FormSpec(document_id="margin", dpi=150, margin_px=40, background_gray=180)
        form = make_application_form(spec)
        width, height = page_size_px(150)
        assert form.image_size_px == (width + 80, height + 80)
        # 네 모서리는 배경색이어야 한다(윤곽선 검출의 전제).
        for y, x in ((5, 5), (5, -5), (-5, 5), (-5, -5)):
            assert form.image[y, x, 0] == pytest.approx(180, abs=2)

    def test_rotation_expands_canvas(self, skewed_form) -> None:
        base_w, base_h = page_size_px(skewed_form.dpi)
        # 회전 + 여백이므로 원본보다 확실히 커진다.
        assert skewed_form.image_size_px[0] > base_w
        assert skewed_form.image_size_px[1] > base_h

    def test_rotation_changes_pixels(self, skewed_form, clean_form) -> None:
        assert skewed_form.image_size_px != clean_form.image_size_px

    def test_perspective_keeps_canvas_size(self, perspective_form) -> None:
        width, height = page_size_px(perspective_form.dpi)
        margin = perspective_form.spec.margin_px
        assert perspective_form.image_size_px == (width + 2 * margin, height + 2 * margin)

    def test_perspective_actually_warps(self) -> None:
        base = FormSpec(document_id="warp", dpi=150, margin_px=30)
        flat = make_application_form(base)
        warped = make_application_form(base.with_(perspective_strength=0.3))
        assert flat.image.shape == warped.image.shape
        assert not np.array_equal(flat.image, warped.image)

    def test_illumination_darkens_one_side(self) -> None:
        spec = FormSpec(document_id="illum", dpi=150, illumination_gradient=0.6)
        form = make_application_form(spec)
        gray = form.image.mean(axis=2)
        top_left = float(gray[:100, :100].mean())
        bottom_right = float(gray[-100:, -100:].mean())
        assert top_left > bottom_right

    def test_jpeg_quality_produces_artifacts(self) -> None:
        base = FormSpec(document_id="jpeg", dpi=150)
        clean = make_application_form(base)
        lossy = make_application_form(base.with_(jpeg_quality=40))
        assert clean.image.shape == lossy.image.shape
        assert not np.array_equal(clean.image, lossy.image)

    def test_blur_reduces_edge_energy(self) -> None:
        base = FormSpec(document_id="blur", dpi=150)
        sharp = make_application_form(base).image.astype(np.float32)
        blurred = make_application_form(base.with_(blur_ksize=9)).image.astype(np.float32)
        sharp_energy = float(np.abs(np.diff(sharp[:, :, 0], axis=1)).mean())
        blurred_energy = float(np.abs(np.diff(blurred[:, :, 0], axis=1)).mean())
        assert blurred_energy < sharp_energy


# --------------------------------------------------------------------------
# 결정론
# --------------------------------------------------------------------------


class TestDeterminism:
    """같은 사양은 항상 바이트 단위로 같은 이미지를 만든다."""

    def test_same_spec_gives_identical_bytes(self) -> None:
        spec = FormSpec(
            document_id="det",
            dpi=150,
            rotation_deg=10.0,
            perspective_strength=0.2,
            margin_px=30,
            noise_sigma=4.0,
            jpeg_quality=75,
            illumination_gradient=0.3,
        )
        first = make_application_form(spec)
        second = make_application_form(spec)
        assert np.array_equal(first.image, second.image)
        assert first.truth == second.truth

    def test_different_seed_changes_noise(self) -> None:
        base = FormSpec(document_id="seedcmp", dpi=150, noise_sigma=6.0)
        first = make_application_form(base.with_(seed=1))
        second = make_application_form(base.with_(seed=2))
        assert not np.array_equal(first.image, second.image)

    def test_render_written_is_deterministic(self, clean_form) -> None:
        first = render_written(clean_form)
        second = render_written(clean_form)
        assert np.array_equal(first, second)

    def test_geometry_identical_across_written_variants(self) -> None:
        """작성 전/후 이미지의 기하 변형이 완전히 동일해야 verify 비교가 성립한다."""
        spec = FormSpec(
            document_id="align",
            dpi=150,
            rotation_deg=-12.0,
            perspective_strength=0.2,
            margin_px=40,
        )
        form = make_application_form(spec)
        written = render_written(form)
        assert written.shape == form.image.shape
        # 아무것도 기입하지 않으면 결과가 원본과 완전히 같아야 한다(정렬 증명).
        untouched = render_written(form, checked_option=None, sign=False)
        assert np.array_equal(untouched, form.image)


# --------------------------------------------------------------------------
# 정답 구조
# --------------------------------------------------------------------------


class TestTruth:
    """정답 :class:`DocumentStructure` 의 내용과 좌표."""

    def test_truth_roundtrips_through_json(self, clean_form) -> None:
        restored = DocumentStructure.from_json(clean_form.truth.to_json())
        assert restored == clean_form.truth

    def test_build_truth_matches_rendered_truth(self, sample_form_spec, clean_form) -> None:
        assert build_truth(sample_form_spec) == clean_form.truth

    def test_page_size_is_a4(self, clean_form) -> None:
        assert clean_form.truth.page_size_mm == (A4_WIDTH_MM, A4_HEIGHT_MM)

    def test_all_boxes_inside_page(self, clean_form) -> None:
        for item in clean_form.truth.fields:
            boxes: list[BoxMm] = []
            if item.box_mm is not None:
                boxes.append(item.box_mm)
            boxes.extend(option.box_mm for option in item.options)
            for box in boxes:
                assert box.x_mm >= 0.0
                assert box.y_mm >= 0.0
                assert box.right_mm <= A4_WIDTH_MM
                assert box.bottom_mm <= A4_HEIGHT_MM

    def test_field_orders_are_unique_and_sequential(self, clean_form) -> None:
        orders = [item.order for item in clean_form.truth.fields]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)

    def test_person_fields_are_private_text_inputs(self, clean_form) -> None:
        for field_id in ("applicant_name", "applicant_rrn", "applicant_phone"):
            item = clean_form.field(field_id)
            assert item.type is FieldType.TEXT_INPUT
            assert item.sensitivity is Sensitivity.PRIVATE
            assert item.role is FieldRole.APPLICANT

    def test_consent_field_is_public_choice_with_two_options(self, clean_form) -> None:
        consent = clean_form.field(CONSENT_FIELD_ID)
        assert consent.type is FieldType.CHOICE
        assert consent.required is True
        assert consent.sensitivity is Sensitivity.PUBLIC
        assert consent.clause_text
        assert [option.label for option in consent.options] == [AGREE_LABEL, DISAGREE_LABEL]
        assert all(option.checked is None for option in consent.options)

    def test_signature_field_roles(self) -> None:
        form = make_application_form(
            FormSpec(document_id="roles", dpi=150, include_representative=True)
        )
        assert form.field(APPLICANT_SIGNATURE_FIELD_ID).role is FieldRole.APPLICANT
        assert form.field(APPLICANT_SIGNATURE_FIELD_ID).type is FieldType.SIGNATURE
        assert (
            form.field(REPRESENTATIVE_SIGNATURE_FIELD_ID).role is FieldRole.REPRESENTATIVE
        )

    def test_representative_excluded_by_default(self) -> None:
        form = make_application_form(FormSpec(document_id="norep", dpi=150))
        assert form.truth.field_by_id(REPRESENTATIVE_SIGNATURE_FIELD_ID) is None

    def test_date_field_present(self, clean_form) -> None:
        assert clean_form.field(DATE_FIELD_ID).type is FieldType.DATE

    def test_required_fields_include_consent_and_signature(self, clean_form) -> None:
        required_ids = {item.id for item in clean_form.truth.required_fields()}
        assert CONSENT_FIELD_ID in required_ids
        assert APPLICANT_SIGNATURE_FIELD_ID in required_ids

    def test_public_payload_redacts_private_fields(self, clean_form) -> None:
        payload = clean_form.truth.public_payload()
        assert "applicant_rrn" in payload["redacted_field_ids"]
        assert CONSENT_FIELD_ID not in payload["redacted_field_ids"]
        redacted = [item for item in payload["fields"] if item.get("redacted")]
        assert redacted
        for item in redacted:
            # 강등된 항목에는 구조 정보만 남고 내용·좌표는 사라져야 한다.
            assert set(item) == {
                "id",
                "type",
                "role",
                "required",
                "order",
                "sensitivity",
                "redacted",
            }

    def test_public_payload_keeps_printed_clause(self, clean_form) -> None:
        """약관 본문은 문서에 인쇄된 공개 정보이므로 payload 에 남아야 한다."""
        payload = clean_form.truth.public_payload()
        consent = next(
            item for item in payload["fields"] if item["id"] == CONSENT_FIELD_ID
        )
        assert consent.get("redacted") is None
        assert "개인정보" in consent["clause_text"]

    def test_public_payload_never_leaks_example_values(self, clean_form) -> None:
        """표에 인쇄된 가상 예시 값은 어느 경로로도 payload 에 들어가지 않는다."""
        serialized = json.dumps(clean_form.truth.public_payload(), ensure_ascii=False)
        for example in ("900101-1234567", "010-1234-5678", "홍길동"):
            assert example not in serialized

    def test_unknown_field_id_raises(self, clean_form) -> None:
        with pytest.raises(ValueError, match="정답에 없는 항목"):
            clean_form.field("존재하지_않는_항목")


# --------------------------------------------------------------------------
# 좌표 정확성
# --------------------------------------------------------------------------


class TestCoordinates:
    """정답 mm 좌표가 실제로 그려진 위치와 맞는지."""

    def test_option_boxes_have_ink_borders(self, clean_form) -> None:
        for label in (AGREE_LABEL, DISAGREE_LABEL):
            box_px = clean_form.flat_box_px(clean_form.option_box(label))
            # 사각 테두리를 실제로 그렸으므로 상당한 잉크가 있어야 한다.
            assert _ink_ratio(clean_form.image, box_px) > 0.1

    def test_option_box_is_square(self, clean_form) -> None:
        box = clean_form.option_box(AGREE_LABEL)
        assert box.w_mm == pytest.approx(box.h_mm)

    def test_option_boxes_do_not_overlap(self, clean_form) -> None:
        agree = clean_form.option_box(AGREE_LABEL)
        disagree = clean_form.option_box(DISAGREE_LABEL)
        assert agree.right_mm < disagree.x_mm

    def test_unknown_option_label_raises(self, clean_form) -> None:
        with pytest.raises(ValueError, match="정의되지 않은 선택지"):
            clean_form.option_box("아마도아님")

    def test_signature_box_center_is_actuator_target(self, clean_form) -> None:
        signature = clean_form.field(APPLICANT_SIGNATURE_FIELD_ID)
        target = signature.target_point()
        assert target is not None
        assert signature.box_mm is not None
        assert signature.box_mm.contains(target)

    def test_layout_shared_between_drawing_and_truth(self, sample_form_spec) -> None:
        layout = build_layout(sample_form_spec)
        truth = build_truth(sample_form_spec, layout=layout)
        consent = truth.field_by_id(CONSENT_FIELD_ID)
        assert consent is not None
        assert consent.options[0].box_mm == layout.option_box(AGREE_LABEL)


# --------------------------------------------------------------------------
# 작성(기입) 렌더링
# --------------------------------------------------------------------------


class TestRenderWritten:
    """체크·서명 기입 렌더링과 전/후 비교."""

    def test_written_image_matches_original_shape(self, written_form) -> None:
        form, written = written_form
        assert written.shape == form.image.shape
        assert written.dtype == np.uint8

    def test_checked_option_gains_ink(self, written_form) -> None:
        form, written = written_form
        box_px = form.flat_box_px(form.option_box(AGREE_LABEL))
        before = _ink_ratio(form.image, box_px)
        after = _ink_ratio(written, box_px)
        assert after > before + 0.05

    def test_unchecked_option_ink_unchanged(self, written_form) -> None:
        form, written = written_form
        box_px = form.flat_box_px(form.option_box(DISAGREE_LABEL))
        assert _ink_ratio(written, box_px) == pytest.approx(
            _ink_ratio(form.image, box_px), abs=1e-6
        )

    def test_signature_area_gains_ink(self, written_form) -> None:
        form, written = written_form
        signature = form.field(APPLICANT_SIGNATURE_FIELD_ID)
        assert signature.box_mm is not None
        box_px = form.flat_box_px(signature.box_mm)
        assert _ink_ratio(written, box_px) > _ink_ratio(form.image, box_px) + 0.01

    def test_disagree_option_can_be_checked(self, clean_form) -> None:
        written = render_written(clean_form, checked_option=DISAGREE_LABEL, sign=False)
        box_px = clean_form.flat_box_px(clean_form.option_box(DISAGREE_LABEL))
        assert _ink_ratio(written, box_px) > _ink_ratio(clean_form.image, box_px) + 0.05

    def test_no_marks_equals_original(self, clean_form) -> None:
        untouched = render_written(clean_form, checked_option=None, sign=False)
        assert np.array_equal(untouched, clean_form.image)

    def test_unknown_option_label_raises(self, clean_form) -> None:
        with pytest.raises(ValueError, match="정의되지 않은 선택지"):
            render_written(clean_form, checked_option="그런선택지없음")

    def test_missing_signature_slot_raises(self) -> None:
        form = make_application_form(FormSpec(document_id="norep2", dpi=150))
        with pytest.raises(ValueError, match="정의되지 않은 서명란"):
            render_written(form, sign_representative=True)

    def test_representative_signature_renders(self) -> None:
        form = make_application_form(
            FormSpec(document_id="repsign", dpi=150, include_representative=True)
        )
        written = render_written(form, sign=False, sign_representative=True)
        slot = form.field(REPRESENTATIVE_SIGNATURE_FIELD_ID)
        assert slot.box_mm is not None
        box_px = form.flat_box_px(slot.box_mm)
        assert _ink_ratio(written, box_px) > _ink_ratio(form.image, box_px) + 0.01

    def test_expected_written_truth_marks_checked(self, clean_form) -> None:
        truth = expected_written_truth(clean_form, checked_option=AGREE_LABEL)
        consent = truth.field_by_id(CONSENT_FIELD_ID)
        assert consent is not None
        checked = {option.label: option.checked for option in consent.options}
        assert checked == {AGREE_LABEL: True, DISAGREE_LABEL: False}

    def test_expected_written_truth_none_marks_all_false(self, clean_form) -> None:
        truth = expected_written_truth(clean_form, checked_option=None, sign=False)
        consent = truth.field_by_id(CONSENT_FIELD_ID)
        assert consent is not None
        assert all(option.checked is False for option in consent.options)

    def test_expected_written_truth_rejects_unknown_label(self, clean_form) -> None:
        with pytest.raises(ValueError, match="정의되지 않은 선택지"):
            expected_written_truth(clean_form, checked_option="없는라벨")


# --------------------------------------------------------------------------
# 사양 그리드
# --------------------------------------------------------------------------


class TestSpecGrid:
    """데이터셋 생성용 사양 조합."""

    def test_grid_is_deterministic(self) -> None:
        first = list(iter_spec_grid(count=6, seed=7))
        second = list(iter_spec_grid(count=6, seed=7))
        assert first == second

    def test_grid_cycles_angles_and_dpis(self) -> None:
        specs = list(iter_spec_grid(angles=(0.0, 10.0), dpis=(150, 300), count=4, seed=3))
        assert [spec.rotation_deg for spec in specs] == [0.0, 10.0, 0.0, 10.0]
        assert [spec.dpi for spec in specs] == [150, 300, 150, 300]

    def test_grid_ids_are_unique(self) -> None:
        specs = list(iter_spec_grid(count=12, seed=DEFAULT_SEED))
        assert len({spec.document_id for spec in specs}) == 12

    def test_grid_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="1 이상"):
            list(iter_spec_grid(count=0))
        with pytest.raises(ValueError, match="비어"):
            list(iter_spec_grid(angles=(), count=1))


# --------------------------------------------------------------------------
# 폰트
# --------------------------------------------------------------------------


class TestFonts:
    """폰트 탐색·폴백 — 폰트가 없어도 예외를 던지지 않는다."""

    def test_load_font_never_raises(self) -> None:
        font = fonts.load_font(24)
        assert font is not None

    def test_load_font_bold_variant(self) -> None:
        assert fonts.load_font(18, bold=True) is not None

    def test_load_font_rejects_zero_size(self) -> None:
        with pytest.raises(ValueError, match="1px 이상"):
            fonts.load_font(0)

    def test_find_font_path_returns_path_or_none(self) -> None:
        path = fonts.find_font_path()
        assert path is None or isinstance(path, Path)

    def test_diagnostics_shape(self) -> None:
        diag = fonts.font_diagnostics()
        assert set(diag) == {"available", "regular", "bold", "env_var", "env_value"}
        assert isinstance(diag["available"], bool)

    @pytest.mark.filterwarnings(
        "ignore::docagent.testing.fonts.MissingKoreanFontWarning"
    )
    def test_rendering_works_without_korean_font(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """폰트를 찾지 못해도 도형 기반 검증은 그대로 유효해야 한다."""
        monkeypatch.setattr(fonts, "find_font_path", lambda *, bold=False: None)
        form = make_application_form(FormSpec(document_id="nofont", dpi=150))
        box_px = form.flat_box_px(form.option_box(AGREE_LABEL))
        # 체크박스 테두리는 폰트와 무관하게 그려진다.
        assert _ink_ratio(form.image, box_px) > 0.1
        assert form.truth.field_by_id(CONSENT_FIELD_ID) is not None


# --------------------------------------------------------------------------
# 공용 fixture 자체 검증
# --------------------------------------------------------------------------


class TestSharedFixtures:
    """후속 Vision·Agent 에이전트가 그대로 쓰는 fixture 계약."""

    def test_sample_structure_has_required_choice(self, sample_structure) -> None:
        choices = [
            item
            for item in sample_structure.fields
            if item.type is FieldType.CHOICE and item.required
        ]
        assert len(choices) == 1
        labels = [option.label for option in choices[0].options]
        assert labels == [AGREE_LABEL, DISAGREE_LABEL]
        assert choices[0].clause_text

    def test_sample_structure_has_applicant_signature(self, sample_structure) -> None:
        signatures = [
            item
            for item in sample_structure.fields
            if item.type is FieldType.SIGNATURE and item.role is FieldRole.APPLICANT
        ]
        assert len(signatures) == 1

    def test_sample_structure_has_two_private_text_inputs(self, sample_structure) -> None:
        private_inputs = [
            item
            for item in sample_structure.fields
            if item.type is FieldType.TEXT_INPUT
            and item.sensitivity is Sensitivity.PRIVATE
        ]
        titles = {item.title for item in private_inputs}
        assert len(private_inputs) >= 2
        assert {"성명", "주민등록번호"} <= titles

    def test_sample_structure_needs_no_image(self, sample_structure) -> None:
        assert sample_structure.source_image is None

    def test_perspective_fixture_has_warp(self, perspective_form) -> None:
        assert perspective_form.spec.perspective_strength > 0.0

    def test_skewed_fixture_angle(self, skewed_form) -> None:
        assert skewed_form.spec.rotation_deg == pytest.approx(15.0)

    def test_lowres_fixture_dpi(self, lowres_form) -> None:
        assert lowres_form.dpi == 150

    def test_written_fixture_is_tuple(self, written_form) -> None:
        form, written = written_form
        assert isinstance(written, np.ndarray)
        assert form.truth.document_id == "fixture_clean"

    def test_synthetic_module_fixture(self, synthetic_module) -> None:
        assert synthetic_module.AGREE_LABEL == AGREE_LABEL
