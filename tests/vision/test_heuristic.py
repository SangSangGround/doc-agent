"""규칙 기반 탐지기(:mod:`docagent.vision.heuristic`) 테스트.

로드맵 KPI 를 **실제로 측정**한다. 합성 신청서의 정답(:class:`DocumentStructure`)과
탐지 결과를 IoU >= 0.3 기준으로 매칭해 클래스별 재현율(recall)·정밀도(precision)를
계산하고, 다음을 단언한다.

* 체크박스 재현율 >= 0.90
* 서명란 재현율 >= 0.90
* 전체 정밀도 >= 0.60 (오검출률 <= 0.40)

탐지기 선택 팩토리(:mod:`docagent.vision.detector_factory`), YOLO 어댑터의
미설치 동작(:mod:`docagent.vision.yolo_adapter`), 학습 스크립트
(``scripts/train_yolo.py``)의 인자·판정 로직도 함께 검증한다. 선택적 패키지는
하나도 설치하지 않은 상태에서 전부 통과해야 한다.

알려진 오검출
-------------
합성 서식의 **신청일자 기입선**은 서명란 밑줄과 기하학적으로 구분되지 않는다
(길이 60mm, 두께 0.4mm, 위 공간 비어 있음). 이 탐지기는 좌표와 유형만 산출하고
역할·의미 판별은 구조화 단계가 하므로, 이 항목은 의도된 한계로 두고
정밀도 목표(0.60)를 그 위에서 만족시킨다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pytest

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    Detection,
    DocumentStructure,
    FieldType,
)
from docagent.errors import VisionError
from docagent.interfaces import Detector
from docagent.testing.synthetic import (
    AGREE_LABEL,
    DEFAULT_SEED,
    FormSpec,
    SyntheticForm,
    make_application_form,
    render_written,
)
from docagent.vision.geometry import A4CoordinateSystem
from docagent.vision.heuristic import (
    SOURCE_CHECKBOX,
    SOURCE_UNDERLINE,
    HeuristicDetector,
    HeuristicParams,
    coordinate_system_from_image,
    iou_mm,
)

#: KPI 측정에 쓰는 IoU 매칭 임계값.
MATCH_IOU: float = 0.30
#: 로드맵 KPI — 클래스별 최소 재현율.
MIN_RECALL: float = 0.90
#: 로드맵 KPI — 최소 정밀도.
MIN_PRECISION: float = 0.60

#: 인적사항 표가 차지하는 세로 구간(mm). 표 테두리 오검출 억제 검증에 쓴다.
TABLE_BAND_MM: tuple[float, float] = (46.0, 90.0)


# --------------------------------------------------------------------------
# KPI 측정 헬퍼
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchStats:
    """정답 대비 탐지 결과의 매칭 통계.

    :param label: 통계 이름(예: ``"checkbox"``).
    :param truth_count: 정답 상자 개수.
    :param predicted_count: 탐지 상자 개수.
    :param matched: 정답과 1:1 매칭에 성공한 개수.
    """

    label: str
    truth_count: int
    predicted_count: int
    matched: int

    @property
    def recall(self) -> float:
        """재현율. 정답이 없으면 1.0 으로 본다."""
        return 1.0 if self.truth_count == 0 else self.matched / self.truth_count

    @property
    def precision(self) -> float:
        """정밀도. 탐지 결과가 없으면 1.0 으로 본다."""
        return 1.0 if self.predicted_count == 0 else self.matched / self.predicted_count

    @property
    def false_positive_rate(self) -> float:
        """오검출률 = 1 - 정밀도."""
        return 1.0 - self.precision

    def format_report(self) -> str:
        """사람이 읽는 한 줄 리포트."""
        return (
            f"{self.label}: 정답 {self.truth_count}건 / 탐지 {self.predicted_count}건 / "
            f"매칭 {self.matched}건 → recall={self.recall:.3f}, "
            f"precision={self.precision:.3f}, 오검출률={self.false_positive_rate:.3f}"
        )


def truth_boxes(structure: DocumentStructure, field_type: FieldType) -> list[BoxMm]:
    """정답 구조에서 유형별 상자 목록을 뽑는다.

    체크박스는 선택형 항목의 ``options[].box_mm`` 이 정답이다(항목 전체 상자가 아니라
    실제로 표시를 넣는 네모 칸).

    :param structure: 정답 문서 구조.
    :param field_type: :attr:`FieldType.CHECKBOX` 또는 :attr:`FieldType.SIGNATURE`.
    :returns: mm 상자 목록.
    :raises ValueError: 지원하지 않는 유형인 경우.
    """
    if field_type is FieldType.SIGNATURE:
        return [
            item.box_mm
            for item in structure.fields
            if item.type is FieldType.SIGNATURE and item.box_mm is not None
        ]
    if field_type is FieldType.CHECKBOX:
        boxes = [option.box_mm for item in structure.fields for option in item.options]
        boxes.extend(
            item.box_mm
            for item in structure.fields
            if item.type is FieldType.CHECKBOX and item.box_mm is not None
        )
        return boxes
    raise ValueError(f"지원하지 않는 정답 유형입니다: {field_type}")


def match_stats(
    label: str,
    truths: list[BoxMm],
    predictions: list[Detection],
    *,
    iou_threshold: float = MATCH_IOU,
) -> MatchStats:
    """IoU 기준 1:1 탐욕 매칭으로 재현율·정밀도를 계산한다.

    IoU 가 큰 쌍부터 짝지어 한 정답이 여러 탐지에 중복 매칭되지 않게 한다.

    :param label: 통계 이름.
    :param truths: 정답 상자 목록.
    :param predictions: 탐지 결과 목록.
    :param iou_threshold: 매칭으로 인정할 최소 IoU.
    :returns: :class:`MatchStats`.
    """
    pairs: list[tuple[float, int, int]] = []
    for truth_index, truth in enumerate(truths):
        for pred_index, prediction in enumerate(predictions):
            score = iou_mm(truth, prediction.box_mm)
            if score >= iou_threshold:
                pairs.append((score, truth_index, pred_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_truth: set[int] = set()
    used_pred: set[int] = set()
    for _score, truth_index, pred_index in pairs:
        if truth_index in used_truth or pred_index in used_pred:
            continue
        used_truth.add(truth_index)
        used_pred.add(pred_index)
    return MatchStats(
        label=label,
        truth_count=len(truths),
        predicted_count=len(predictions),
        matched=len(used_truth),
    )


def evaluate(form: SyntheticForm, image: np.ndarray | None = None) -> dict[str, MatchStats]:
    """합성 문서 한 장에 대해 클래스별·전체 통계를 만든다.

    :param form: 합성 신청서(정답 포함).
    :param image: 평가할 이미지. ``None`` 이면 ``form.image``.
    :returns: ``{"checkbox": ..., "signature": ..., "전체": ...}``.
    """
    target = form.image if image is None else image
    detections = HeuristicDetector().detect(target)
    stats: dict[str, MatchStats] = {}
    total_truth = 0
    total_pred = 0
    total_matched = 0
    for name, field_type in (("checkbox", FieldType.CHECKBOX), ("signature", FieldType.SIGNATURE)):
        truths = truth_boxes(form.truth, field_type)
        predictions = [item for item in detections if item.type is field_type]
        result = match_stats(name, truths, predictions)
        stats[name] = result
        total_truth += result.truth_count
        total_pred += result.predicted_count
        total_matched += result.matched
    stats["전체"] = MatchStats("전체", total_truth, total_pred, total_matched)
    return stats


# --------------------------------------------------------------------------
# 좌표계
# --------------------------------------------------------------------------


class TestCoordinateSystemFromImage:
    """:func:`coordinate_system_from_image` 단위 테스트.

    좌표계 자체는 :mod:`docagent.vision.geometry` 의 공용 구현을 재사용하고,
    여기서는 "이미지 배열 → 좌표계" 어댑팅만 검증한다.
    """

    def test_returns_shared_geometry_type(self, clean_form: SyntheticForm) -> None:
        """중복 정의 없이 공용 A4CoordinateSystem 을 돌려준다."""
        assert isinstance(
            coordinate_system_from_image(clean_form.image), A4CoordinateSystem
        )

    def test_px_per_mm_matches_dpi(self, clean_form: SyntheticForm) -> None:
        """이미지 크기에서 유도한 px/mm 가 렌더링 dpi 와 일치한다."""
        coords = coordinate_system_from_image(clean_form.image)
        assert coords.px_per_mm_x == pytest.approx(clean_form.px_per_mm, rel=1e-3)
        assert coords.px_per_mm_y == pytest.approx(clean_form.px_per_mm, rel=1e-3)
        assert coords.dpi == pytest.approx(clean_form.dpi, rel=1e-3)

    def test_box_round_trip(self, clean_form: SyntheticForm) -> None:
        """mm → px → mm 왕복 오차가 1픽셀 이내다."""
        coords = coordinate_system_from_image(clean_form.image)
        original = BoxMm(26.0, 139.0, 6.0, 6.0)
        restored = coords.to_mm(coords.to_px(original))
        tolerance = 1.0 / coords.px_per_mm_x
        assert restored.x_mm == pytest.approx(original.x_mm, abs=tolerance)
        assert restored.y_mm == pytest.approx(original.y_mm, abs=tolerance)
        assert restored.w_mm == pytest.approx(original.w_mm, abs=tolerance)
        assert restored.h_mm == pytest.approx(original.h_mm, abs=tolerance)

    def test_clamp_keeps_box_inside_page(self, clean_form: SyntheticForm) -> None:
        """페이지 밖으로 나간 상자는 잘려 들어온다."""
        coords = coordinate_system_from_image(clean_form.image)
        clamped = coords.clamp_box(BoxMm(-10.0, 290.0, 40.0, 40.0), warn=False)
        assert clamped.x_mm == 0.0
        assert clamped.right_mm <= A4_WIDTH_MM
        assert clamped.bottom_mm <= A4_HEIGHT_MM

    def test_rejects_non_array(self) -> None:
        """배열이 아닌 입력은 VisionError 로 막는다."""
        with pytest.raises(VisionError, match="numpy 배열"):
            coordinate_system_from_image("이미지가 아님")  # type: ignore[arg-type]

    def test_rejects_wrong_dimensions(self) -> None:
        """1차원 배열은 VisionError."""
        with pytest.raises(VisionError, match="형태"):
            coordinate_system_from_image(np.zeros(10, dtype=np.uint8))

    def test_rejects_empty_image(self) -> None:
        """크기 0 이미지는 VisionError."""
        with pytest.raises(VisionError, match="크기가 0"):
            coordinate_system_from_image(np.zeros((0, 5), dtype=np.uint8))

    def test_warns_on_non_a4_aspect(self, caplog: pytest.LogCaptureFixture) -> None:
        """A4 종횡비를 벗어나면 경고 로그를 남기되 진행한다."""
        with caplog.at_level(logging.WARNING, logger="docagent.vision.heuristic"):
            coords = coordinate_system_from_image(np.zeros((100, 100), dtype=np.uint8))
        assert coords.width_px == 100
        assert any("종횡비" in record.message for record in caplog.records)


class TestIouHelper:
    """:func:`iou_mm` 단위 테스트."""

    def test_identical_boxes(self) -> None:
        """같은 상자의 IoU 는 1.0."""
        box = BoxMm(10.0, 10.0, 20.0, 20.0)
        assert iou_mm(box, box) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        """겹치지 않으면 0.0."""
        assert iou_mm(BoxMm(0, 0, 5, 5), BoxMm(10, 10, 5, 5)) == 0.0

    def test_half_overlap(self) -> None:
        """절반 겹치면 IoU = 1/3."""
        assert iou_mm(BoxMm(0, 0, 10, 10), BoxMm(5, 0, 10, 10)) == pytest.approx(1 / 3)

    def test_zero_area_box(self) -> None:
        """넓이 0 상자는 0.0 을 돌려준다(0 나눗셈 없음)."""
        assert iou_mm(BoxMm(0, 0, 0, 0), BoxMm(0, 0, 10, 10)) == 0.0


# --------------------------------------------------------------------------
# 파라미터 검증
# --------------------------------------------------------------------------


class TestHeuristicParams:
    """:class:`HeuristicParams` 검증 규칙."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"checkbox_min_side_mm": 0.0}, "변 길이"),
            ({"checkbox_max_side_mm": 1.0}, "변 길이"),
            ({"checkbox_max_aspect": 0.5}, "checkbox_max_aspect"),
            ({"underline_max_length_mm": 5.0}, "밑줄 길이"),
            ({"underline_max_thickness_mm": 0.0}, "underline_max_thickness_mm"),
            ({"signature_height_mm": 0.0}, "signature_height_mm"),
            ({"min_confidence": 1.5}, "min_confidence"),
            ({"checkbox_nms_iou": -0.1}, "checkbox_nms_iou"),
        ],
    )
    def test_invalid_params_raise_korean_value_error(
        self, kwargs: dict[str, float], message: str
    ) -> None:
        """허용 범위를 벗어난 파라미터는 한국어 ValueError 를 던진다."""
        with pytest.raises(ValueError, match=message):
            HeuristicParams(**kwargs)

    def test_defaults_are_valid(self) -> None:
        """기본값은 항상 유효하다."""
        params = HeuristicParams()
        assert params.checkbox_min_side_mm < params.checkbox_max_side_mm
        assert params.underline_min_length_mm < params.underline_max_length_mm


# --------------------------------------------------------------------------
# 계약 준수
# --------------------------------------------------------------------------


class TestDetectorContract:
    """탐지기가 계약(Protocol·타입·좌표 범위)을 지키는지 확인한다."""

    def test_satisfies_detector_protocol(self) -> None:
        """:class:`Detector` 프로토콜을 만족한다."""
        assert isinstance(HeuristicDetector(), Detector)

    def test_all_outputs_are_mm_boxes(self, clean_form: SyntheticForm) -> None:
        """모든 결과가 mm 좌표이며 픽셀 상자가 새어 나오지 않는다."""
        detections = HeuristicDetector().detect(clean_form.image)
        assert detections, "정상 문서에서 탐지 결과가 하나도 없습니다."
        for item in detections:
            assert isinstance(item, Detection)
            assert isinstance(item.box_mm, BoxMm)
            assert not isinstance(item.box_mm, BoxPx)
            assert 0.0 <= item.confidence <= 1.0
            assert 0.0 <= item.box_mm.x_mm <= A4_WIDTH_MM
            assert 0.0 <= item.box_mm.y_mm <= A4_HEIGHT_MM
            assert item.box_mm.right_mm <= A4_WIDTH_MM + 1e-6
            assert item.box_mm.bottom_mm <= A4_HEIGHT_MM + 1e-6

    def test_only_supported_types(self, clean_form: SyntheticForm) -> None:
        """SIGNATURE / CHECKBOX 외의 유형은 산출하지 않는다(역할 분류는 다른 단계)."""
        detections = HeuristicDetector().detect(clean_form.image)
        assert {item.type for item in detections} <= {
            FieldType.SIGNATURE,
            FieldType.CHECKBOX,
        }

    def test_source_labels(self, clean_form: SyntheticForm) -> None:
        """출처 문자열이 유형별로 고정되어 있다."""
        for item in HeuristicDetector().detect(clean_form.image):
            expected = (
                SOURCE_CHECKBOX if item.type is FieldType.CHECKBOX else SOURCE_UNDERLINE
            )
            assert item.source == expected

    def test_deterministic(self, clean_form: SyntheticForm) -> None:
        """같은 입력은 항상 같은 결과를 낸다(결정론)."""
        detector = HeuristicDetector()
        first = detector.detect(clean_form.image)
        second = HeuristicDetector().detect(clean_form.image)
        assert [item.to_dict() for item in first] == [item.to_dict() for item in second]

    def test_grayscale_input_supported(self, clean_form: SyntheticForm) -> None:
        """그레이스케일 입력도 BGR 과 같은 개수를 탐지한다."""
        import cv2

        gray = cv2.cvtColor(clean_form.image, cv2.COLOR_BGR2GRAY)
        assert len(HeuristicDetector().detect(gray)) == len(
            HeuristicDetector().detect(clean_form.image)
        )

    def test_blank_page_returns_empty_list(self) -> None:
        """아무것도 없는 백지는 예외 없이 빈 리스트를 돌려준다."""
        blank = np.full((2339, 1654, 3), 255, dtype=np.uint8)
        assert HeuristicDetector().detect(blank) == []

    def test_rejects_invalid_input(self) -> None:
        """잘못된 입력은 조용히 넘어가지 않고 VisionError 를 던진다."""
        with pytest.raises(VisionError, match="numpy 배열"):
            HeuristicDetector().detect([[0, 0], [0, 0]])  # type: ignore[arg-type]
        with pytest.raises(VisionError, match="비어 있습니다"):
            HeuristicDetector().detect(np.zeros((0, 0), dtype=np.uint8))
        with pytest.raises(VisionError, match="지원하지 않는"):
            HeuristicDetector().detect(np.zeros((10, 10, 5), dtype=np.uint8))

    def test_toggles_disable_branches(self, clean_form: SyntheticForm) -> None:
        """유형별 탐지를 개별적으로 끌 수 있다."""
        only_boxes = HeuristicDetector(detect_signatures=False).detect(clean_form.image)
        only_lines = HeuristicDetector(detect_checkboxes=False).detect(clean_form.image)
        assert only_boxes and all(item.type is FieldType.CHECKBOX for item in only_boxes)
        assert only_lines and all(item.type is FieldType.SIGNATURE for item in only_lines)


# --------------------------------------------------------------------------
# 정확도 KPI
# --------------------------------------------------------------------------


class TestDetectionKpi:
    """로드맵 KPI(재현율 90%, 정밀도 60%)를 실제 측정으로 단언한다."""

    def test_clean_form_meets_kpi(self, clean_form: SyntheticForm) -> None:
        """변형 없는 정상 문서에서 KPI 를 만족한다."""
        stats = evaluate(clean_form)
        report = "\n".join(item.format_report() for item in stats.values())
        print("\n[정상 문서 KPI]\n" + report)
        assert stats["checkbox"].recall >= MIN_RECALL, report
        assert stats["signature"].recall >= MIN_RECALL, report
        assert stats["전체"].precision >= MIN_PRECISION, report

    def test_lowres_form_meets_kpi(self, lowres_form: SyntheticForm) -> None:
        """150dpi + JPEG 열화 + 잡음 상황에서도 KPI 를 만족한다."""
        stats = evaluate(lowres_form)
        report = "\n".join(item.format_report() for item in stats.values())
        print("\n[저해상도 문서 KPI]\n" + report)
        assert stats["checkbox"].recall >= MIN_RECALL, report
        assert stats["signature"].recall >= MIN_RECALL, report
        assert stats["전체"].precision >= MIN_PRECISION, report

    @pytest.mark.parametrize(
        ("name", "overrides"),
        [
            ("300dpi", {"dpi": 300}),
            ("조명 불균일", {"illumination_gradient": 0.35}),
            ("블러", {"blur_ksize": 5}),
            ("잡음", {"noise_sigma": 6.0}),
            ("JPEG 60", {"jpeg_quality": 60}),
            ("예시값 없음", {"fill_example_values": False}),
        ],
    )
    def test_degradations_keep_recall(self, name: str, overrides: dict) -> None:
        """화질 저하 조건에서도 재현율 KPI 가 유지된다."""
        spec = FormSpec(
            document_id=f"kpi_{name}",
            include_representative=True,
            seed=DEFAULT_SEED,
            **overrides,
        )
        stats = evaluate(make_application_form(spec))
        report = f"[{name}]\n" + "\n".join(item.format_report() for item in stats.values())
        print("\n" + report)
        assert stats["checkbox"].recall >= MIN_RECALL, report
        assert stats["signature"].recall >= MIN_RECALL, report
        assert stats["전체"].precision >= MIN_PRECISION, report

    def test_written_form_still_detected(self, clean_form: SyntheticForm) -> None:
        """체크·서명이 이미 기입된 문서에서도 재현율 KPI 가 유지된다."""
        written = render_written(
            clean_form, checked_option=AGREE_LABEL, sign=True, sign_representative=True
        )
        stats = evaluate(clean_form, written)
        report = "\n".join(item.format_report() for item in stats.values())
        print("\n[기입 완료 문서 KPI]\n" + report)
        assert stats["checkbox"].recall >= MIN_RECALL, report
        assert stats["signature"].recall >= MIN_RECALL, report

    def test_localization_error_is_small(self, clean_form: SyntheticForm) -> None:
        """매칭된 상자의 중심 좌표 오차가 2mm 이내다(액추에이터 목표 좌표 품질)."""
        detections = HeuristicDetector().detect(clean_form.image)
        for field_type in (FieldType.CHECKBOX, FieldType.SIGNATURE):
            predictions = [item for item in detections if item.type is field_type]
            for truth in truth_boxes(clean_form.truth, field_type):
                best = max(
                    predictions, key=lambda item: iou_mm(truth, item.box_mm), default=None
                )
                assert best is not None
                if iou_mm(truth, best.box_mm) < MATCH_IOU:
                    continue
                assert best.box_mm.center().x_mm == pytest.approx(
                    truth.center().x_mm, abs=2.0
                )
                assert best.box_mm.center().y_mm == pytest.approx(
                    truth.center().y_mm, abs=2.0
                )


# --------------------------------------------------------------------------
# 오검출 억제
# --------------------------------------------------------------------------


class TestFalsePositiveSuppression:
    """표 격자·글자에서 생기는 오검출이 억제되는지 확인한다."""

    def test_table_borders_are_not_signature_lines(self, clean_form: SyntheticForm) -> None:
        """인적사항 표의 가로 테두리를 서명란으로 잡지 않는다."""
        top, bottom = TABLE_BAND_MM
        offenders = [
            item
            for item in HeuristicDetector().detect(clean_form.image)
            if item.type is FieldType.SIGNATURE
            and item.box_mm.y_mm < bottom
            and item.box_mm.bottom_mm > top
        ]
        assert not offenders, (
            "표 테두리가 서명란으로 오검출되었습니다: "
            f"{[item.box_mm.to_tuple() for item in offenders]}"
        )

    def test_table_cells_are_not_checkboxes(self, clean_form: SyntheticForm) -> None:
        """표 칸(40x11mm, 110x11mm)을 체크박스로 잡지 않는다."""
        top, bottom = TABLE_BAND_MM
        offenders = [
            item
            for item in HeuristicDetector().detect(clean_form.image)
            if item.type is FieldType.CHECKBOX
            and top <= item.box_mm.center().y_mm <= bottom
        ]
        assert not offenders, (
            "표 칸이 체크박스로 오검출되었습니다: "
            f"{[item.box_mm.to_tuple() for item in offenders]}"
        )

    def test_title_underline_is_not_signature(self, clean_form: SyntheticForm) -> None:
        """제목 밑줄(위에 글자가 있음)을 서명란으로 잡지 않는다."""
        offenders = [
            item
            for item in HeuristicDetector().detect(clean_form.image)
            if item.type is FieldType.SIGNATURE and item.box_mm.bottom_mm < 40.0
        ]
        assert not offenders, (
            f"제목 밑줄이 오검출되었습니다: {[item.box_mm.to_tuple() for item in offenders]}"
        )

    def test_text_glyphs_are_not_checkboxes(self, clean_form: SyntheticForm) -> None:
        """약관 본문 영역(글자만 있는 구간)에서 체크박스를 잡지 않는다."""
        offenders = [
            item
            for item in HeuristicDetector().detect(clean_form.image)
            if item.type is FieldType.CHECKBOX
            and 100.0 <= item.box_mm.center().y_mm <= 135.0
        ]
        assert not offenders, (
            f"본문 글자가 체크박스로 오검출되었습니다: {[item.box_mm.to_tuple() for item in offenders]}"
        )

    def test_nms_removes_nested_duplicates(self, clean_form: SyntheticForm) -> None:
        """한 체크박스에서 안팎 윤곽선이 각각 잡혀도 하나만 남는다."""
        checkboxes = [
            item
            for item in HeuristicDetector().detect(clean_form.image)
            if item.type is FieldType.CHECKBOX
        ]
        assert len(checkboxes) == 2
        assert iou_mm(checkboxes[0].box_mm, checkboxes[1].box_mm) == 0.0


# --------------------------------------------------------------------------
# 신뢰도 산출
# --------------------------------------------------------------------------


class TestConfidenceScoring:
    """신뢰도가 조건 만족 정도를 실제로 반영하는지 확인한다."""

    def test_ideal_shapes_score_high(self, clean_form: SyntheticForm) -> None:
        """정확히 그려진 도형은 Vision 신뢰 임계값을 넘는다."""
        from docagent.contracts import VISION_TRUST_THRESHOLD

        for item in HeuristicDetector().detect(clean_form.image):
            assert item.confidence >= VISION_TRUST_THRESHOLD, item.to_dict()

    def test_confidence_drops_for_distorted_checkbox(self) -> None:
        """직사각형으로 찌그러진 칸은 정사각형보다 신뢰도가 낮다."""
        import cv2

        def draw(width_mm: float) -> float:
            image = np.full((2339, 1654, 3), 255, dtype=np.uint8)
            scale = 1654 / A4_WIDTH_MM
            x0, y0 = int(30 * scale), int(100 * scale)
            x1 = int((30 + width_mm) * scale)
            y1 = int((100 + 6.0) * scale)
            cv2.rectangle(image, (x0, y0), (x1, y1), (20, 20, 24), 4)
            found = [
                item
                for item in HeuristicDetector().detect(image)
                if item.type is FieldType.CHECKBOX
            ]
            assert found, f"폭 {width_mm}mm 칸을 탐지하지 못했습니다."
            return found[0].confidence

        assert draw(6.0) > draw(7.6)

    def test_low_confidence_candidates_are_dropped(self, clean_form: SyntheticForm) -> None:
        """min_confidence 를 1.0 로 올리면 결과가 모두 걸러진다."""
        strict = HeuristicDetector(HeuristicParams(min_confidence=1.0))
        assert all(item.confidence >= 1.0 for item in strict.detect(clean_form.image))


# --------------------------------------------------------------------------
# 선택적 어댑터 · 팩토리 · 학습 스크립트
# --------------------------------------------------------------------------


def _load_train_script():
    """``scripts/train_yolo.py`` 를 파일 경로로 직접 불러온다.

    ``scripts`` 는 패키지가 아니므로 importlib 로 로드한다.

    :returns: 로드된 모듈 객체.
    """
    import importlib.util
    from pathlib import Path as _Path

    path = _Path(__file__).resolve().parents[2] / "scripts" / "train_yolo.py"
    spec = importlib.util.spec_from_file_location("train_yolo_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestYoloAdapterWithoutUltralytics:
    """ultralytics 미설치 환경에서의 어댑터 동작(이 저장소의 기본 상태)."""

    def test_import_never_fails(self) -> None:
        """모듈 import 만으로는 절대 실패하지 않는다(최상단 지연 import 규약)."""
        import docagent.vision.yolo_adapter as adapter

        assert adapter.SOURCE_YOLO == "yolo"
        assert adapter.CLASS_ID_TO_FIELD_TYPE[1] is FieldType.CHECKBOX

    def test_availability_probe_is_quiet(self) -> None:
        """가용성 확인 함수는 예외를 던지지 않는다."""
        from docagent.vision.yolo_adapter import ultralytics_available

        assert ultralytics_available() is False

    def test_require_raises_korean_install_guide(self) -> None:
        """실제 사용 경로는 한국어 설치 안내를 담은 AdapterUnavailable 을 던진다."""
        from docagent.errors import AdapterUnavailable
        from docagent.vision.yolo_adapter import require_yolo_class

        with pytest.raises(AdapterUnavailable) as info:
            require_yolo_class()
        message = str(info.value)
        assert "ultralytics" in message
        assert "설치" in message
        assert ".venv" in message

    def test_detector_construction_raises(self, tmp_path) -> None:
        """가중치가 있어도 ultralytics 가 없으면 생성자에서 막힌다."""
        from docagent.errors import AdapterUnavailable
        from docagent.vision.yolo_adapter import YoloDetector

        weights = tmp_path / "best.pt"
        weights.write_bytes(b"dummy")
        with pytest.raises(AdapterUnavailable):
            YoloDetector(weights)

    def test_constructor_validates_parameters_first(self) -> None:
        """파라미터 범위 위반은 패키지 유무와 무관하게 ValueError."""
        from docagent.vision.yolo_adapter import YoloDetector

        for kwargs, message in (
            ({"conf": 1.5}, "conf"),
            ({"iou": -0.1}, "iou"),
            ({"imgsz": 8}, "imgsz"),
        ):
            with pytest.raises(ValueError, match=message):
                YoloDetector("best.pt", **kwargs)

    def test_export_onnx_raises(self, tmp_path) -> None:
        """ONNX 내보내기도 같은 예외로 막힌다."""
        from docagent.errors import AdapterUnavailable
        from docagent.vision.yolo_adapter import export_onnx

        with pytest.raises(AdapterUnavailable):
            export_onnx(tmp_path / "best.pt")

    def test_export_onnx_validates_imgsz_first(self) -> None:
        """잘못된 imgsz 는 패키지 유무와 무관하게 ValueError."""
        from docagent.vision.yolo_adapter import export_onnx

        with pytest.raises(ValueError, match="imgsz"):
            export_onnx("best.pt", imgsz=16)


class TestDetectorFactory:
    """:func:`docagent.vision.detector_factory.build_detector` 선택 규칙."""

    def test_default_is_heuristic(self) -> None:
        """가중치가 없으면 규칙 기반 탐지기를 준다."""
        from docagent.vision.detector_factory import build_detector

        assert isinstance(build_detector(), HeuristicDetector)

    def test_explicit_heuristic(self) -> None:
        """``prefer='heuristic'`` 은 항상 규칙 기반."""
        from docagent.vision.detector_factory import build_detector

        assert isinstance(build_detector("best.pt", prefer="heuristic"), HeuristicDetector)

    def test_auto_falls_back_and_logs(
        self, caplog: pytest.LogCaptureFixture, tmp_path
    ) -> None:
        """자동 모드는 폴백 사실을 로그로 남긴다(조용한 폴백 금지)."""
        from docagent.vision.detector_factory import build_detector

        with caplog.at_level(logging.INFO, logger="docagent.vision.detector_factory"):
            detector = build_detector(tmp_path / "없는가중치.pt")
        assert isinstance(detector, HeuristicDetector)
        assert any("폴백" in record.message for record in caplog.records)

    def test_auto_falls_back_when_ultralytics_missing(
        self, caplog: pytest.LogCaptureFixture, tmp_path
    ) -> None:
        """가중치가 있어도 ultralytics 가 없으면 규칙 기반으로 폴백한다."""
        from docagent.vision.detector_factory import build_detector

        weights = tmp_path / "best.pt"
        weights.write_bytes(b"dummy")
        with caplog.at_level(logging.INFO, logger="docagent.vision.detector_factory"):
            detector = build_detector(weights)
        assert isinstance(detector, HeuristicDetector)
        assert any("ultralytics" in record.message for record in caplog.records)

    def test_prefer_yolo_does_not_fall_back(self, tmp_path) -> None:
        """``prefer='yolo'`` 는 폴백하지 않고 실패를 드러낸다."""
        from docagent.errors import AdapterUnavailable
        from docagent.vision.detector_factory import build_detector

        weights = tmp_path / "best.pt"
        weights.write_bytes(b"dummy")
        with pytest.raises(AdapterUnavailable):
            build_detector(weights, prefer="yolo")

    def test_prefer_yolo_without_weights(self) -> None:
        """``prefer='yolo'`` 인데 가중치가 없으면 FileNotFoundError."""
        from docagent.vision.detector_factory import build_detector

        with pytest.raises(FileNotFoundError, match="가중치 경로"):
            build_detector(prefer="yolo")

    def test_invalid_preference(self) -> None:
        """허용되지 않은 prefer 값은 한국어 ValueError."""
        from docagent.vision.detector_factory import build_detector

        with pytest.raises(ValueError, match="prefer"):
            build_detector(prefer="mystery")  # type: ignore[arg-type]

    def test_params_are_passed_through(self) -> None:
        """규칙 기반 임계값을 그대로 전달한다."""
        from docagent.vision.detector_factory import build_detector

        params = HeuristicParams(min_confidence=0.9)
        detector = build_detector(params=params)
        assert isinstance(detector, HeuristicDetector)
        assert detector.params.min_confidence == 0.9

    def test_describe_detector(self) -> None:
        """설명 문자열이 한국어로 무엇이 도는지 밝힌다."""
        from docagent.vision.detector_factory import build_detector, describe_detector

        assert "규칙 기반" in describe_detector(build_detector())

    def test_factory_output_satisfies_protocol(self) -> None:
        """팩토리 결과는 Detector 프로토콜을 만족한다."""
        from docagent.vision.detector_factory import build_detector

        assert isinstance(build_detector(), Detector)

    def test_factory_detector_finds_fields(self, clean_form: SyntheticForm) -> None:
        """팩토리로 만든 기본 탐지기가 실제로 기입란을 찾는다(데모 경로)."""
        from docagent.vision.detector_factory import build_detector

        detections = build_detector().detect(clean_form.image)
        assert sum(1 for item in detections if item.type is FieldType.CHECKBOX) == 2
        assert sum(1 for item in detections if item.type is FieldType.SIGNATURE) >= 2


class TestTrainScript:
    """``scripts/train_yolo.py`` 의 인자 처리와 목표 판정."""

    def test_parser_defaults_match_roadmap(self) -> None:
        """기본값이 로드맵 계획(yolov8m, 100 epoch, imgsz 640, batch 16, patience 20)과 같다."""
        module = _load_train_script()
        args = module.build_parser().parse_args(["--data", "data.yaml"])
        assert args.model == "yolov8m.pt"
        assert (args.epochs, args.imgsz, args.batch, args.patience) == (100, 640, 16, 20)

    def test_data_argument_required(self) -> None:
        """``--data`` 없이는 실행할 수 없다."""
        module = _load_train_script()
        with pytest.raises(SystemExit):
            module.build_parser().parse_args([])

    def test_targets_match_roadmap(self) -> None:
        """목표치는 mAP50 0.85 / mAP50-95 0.75 다."""
        module = _load_train_script()
        assert (module.TARGET_MAP50, module.TARGET_MAP50_95) == (0.85, 0.75)

    def test_judge_pass_and_fail(self) -> None:
        """목표 달성·미달을 정확히 판정한다."""
        module = _load_train_script()
        passed, lines = module.judge_metrics({"map50": 0.90, "map50_95": 0.80})
        assert passed and all("달성" in line for line in lines)
        failed, lines = module.judge_metrics({"map50": 0.50, "map50_95": 0.80})
        assert not failed and any("미달" in line for line in lines)

    def test_judge_handles_missing_metric(self) -> None:
        """지표를 못 읽으면 달성으로 처리하지 않는다."""
        module = _load_train_script()
        passed, lines = module.judge_metrics({})
        assert not passed
        assert any("확인할 수 없습니다" in line for line in lines)

    def test_extract_metrics_from_results_dict(self) -> None:
        """ultralytics 결과 dict 에서 지표를 뽑는다."""
        module = _load_train_script()

        class _Fake:
            results_dict = {"metrics/mAP50(B)": 0.91, "metrics/mAP50-95(B)": 0.77}

        values = module.extract_metrics(_Fake())
        assert values["map50"] == pytest.approx(0.91)
        assert values["map50_95"] == pytest.approx(0.77)

    def test_extract_metrics_returns_nan_when_absent(self) -> None:
        """지표가 없으면 NaN 을 돌려준다(0.0 으로 위장하지 않는다)."""
        module = _load_train_script()
        values = module.extract_metrics(object())
        assert values["map50"] != values["map50"]

    def test_main_exits_when_data_missing(self, tmp_path, capsys) -> None:
        """데이터셋 정의가 없으면 종료 코드 1 과 한국어 안내를 낸다."""
        module = _load_train_script()
        code = module.main(["--data", str(tmp_path / "없음.yaml")])
        assert code == 1
        assert "찾을 수 없습니다" in capsys.readouterr().out

    def test_main_exits_when_ultralytics_missing(self, tmp_path, capsys) -> None:
        """ultralytics 가 없으면 설치 안내를 출력하고 종료 코드 1."""
        from docagent.vision.dataset import write_data_yaml

        module = _load_train_script()
        data = write_data_yaml(tmp_path / "data.yaml", tmp_path)
        code = module.main(["--data", str(data)])
        assert code == 1
        output = capsys.readouterr().out
        assert "ultralytics" in output
        assert "pip install" in output
