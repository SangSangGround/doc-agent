"""``docagent.vision.normalize`` 테스트 — 좌표 정확도를 실제로 측정한다.

핵심은 "정규화 후 좌표가 정답 mm 좌표와 몇 mm 차이 나는가"를 **추정이 아니라
실측**하는 것이다. 측정 방법은 다음과 같다.

1. 합성 문서 생성기가 갖고 있는 변형 이전 페이지(``form.flat_image``)를
   정립 이미지 크기로 리사이즈해 **이상적 참조 이미지**를 만든다.
   이 참조 이미지에서 정답 mm 좌표는 곧 정확한 픽셀 좌표다.
2. 정답 영역을 참조 이미지에서 잘라 템플릿으로 삼고,
   정규화 결과 이미지의 같은 위치 주변을 탐색해 정합(``cv2.matchTemplate``)한다.
3. 정합 위치와 기대 위치의 차이를 mm 로 환산한 값이 곧 **좌표 오차**다.

로드맵 KPI(물리 이동 오차 ±3mm) 대비 판정 기준:

============== =====================
픽스처          허용 오차
============== =====================
``clean_form``       3mm
``perspective_form`` 10mm
``skewed_form``      10mm (15도 회전)
``lowres_form``      15mm (150dpi)
============== =====================
"""

from __future__ import annotations

import json
import math
import warnings

import cv2
import numpy as np
import pytest

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    VISION_TRUST_THRESHOLD,
    BoxMm,
    BoxPx,
)
from docagent.errors import DocumentNotFoundError, LowConfidenceError, VisionError
from docagent.testing.synthetic import SyntheticForm
from docagent.vision.geometry import A4CoordinateSystem, order_quad
from docagent.vision.normalize import (
    DEFAULT_NORMALIZE_DPI,
    ERROR_MM_LOWRES,
    ERROR_MM_NORMAL,
    ERROR_MM_SEVERE,
    ERROR_MM_SKEWED,
    NormalizedDocument,
    detect_document_quad,
    detect_document_quad_detailed,
    expected_error_mm_for,
    normalize_boxes,
    normalize_document,
)

#: 정확도 측정에 쓸 정답 영역(라벨, 접근자) — 잉크가 충분해 정합이 안정적인 곳만 고른다.
MEASURE_SEARCH_MM: float = 8.0
MEASURE_PAD_MM: float = 1.5


def _to_gray(image: np.ndarray) -> np.ndarray:
    """BGR 또는 그레이스케일 이미지를 그레이스케일로 통일한다.

    :param image: 입력 이미지.
    :returns: ``(H, W)`` uint8 배열.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ideal_reference(form: SyntheticForm, size_px: tuple[int, int]) -> np.ndarray:
    """변형 이전 페이지를 정립 이미지 크기로 맞춘 이상적 참조 이미지.

    :param form: 합성 문서.
    :param size_px: ``(width_px, height_px)``.
    :returns: ``(H, W)`` uint8 그레이스케일 배열.
    """
    resized = cv2.resize(form.flat_image, size_px, interpolation=cv2.INTER_AREA)
    return _to_gray(resized)


def measure_box_error_mm(
    form: SyntheticForm,
    doc: NormalizedDocument,
    box_mm: BoxMm,
    *,
    search_mm: float = MEASURE_SEARCH_MM,
) -> tuple[float, float]:
    """정답 영역이 정규화 이미지에서 실제로 놓인 위치의 오차(mm)를 잰다.

    :param form: 합성 문서(정답 보유).
    :param doc: 정규화 결과.
    :param box_mm: 정답 mm 사각형.
    :param search_mm: 탐색 여유(mm).
    :returns: ``(오차_mm, 정합 점수)``. 정합 점수는 -1.0~1.0.
    :raises AssertionError: 템플릿이나 탐색 창을 만들 수 없는 경우.
    """
    coords = doc.coords
    reference = ideal_reference(form, coords.size_px)
    target = _to_gray(doc.image)

    scale_x = coords.px_per_mm_x
    scale_y = coords.px_per_mm_y
    x0 = int(round((box_mm.x_mm - MEASURE_PAD_MM) * scale_x))
    y0 = int(round((box_mm.y_mm - MEASURE_PAD_MM) * scale_y))
    x1 = int(round((box_mm.right_mm + MEASURE_PAD_MM) * scale_x))
    y1 = int(round((box_mm.bottom_mm + MEASURE_PAD_MM) * scale_y))
    template = reference[y0:y1, x0:x1]
    assert template.size > 0, "정답 영역에서 템플릿을 만들지 못했습니다."

    margin_x = int(round(search_mm * scale_x))
    margin_y = int(round(search_mm * scale_y))
    sx0 = max(0, x0 - margin_x)
    sy0 = max(0, y0 - margin_y)
    sx1 = min(target.shape[1], x1 + margin_x)
    sy1 = min(target.shape[0], y1 + margin_y)
    window = target[sy0:sy1, sx0:sx1]
    assert (
        window.shape[0] >= template.shape[0] and window.shape[1] >= template.shape[1]
    ), "탐색 창이 템플릿보다 작습니다."

    scores = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, best_score, _, best_loc = cv2.minMaxLoc(scores)
    dx_px = (sx0 + best_loc[0]) - x0
    dy_px = (sy0 + best_loc[1]) - y0
    error_mm = math.hypot(dx_px / scale_x, dy_px / scale_y)
    return (float(error_mm), float(best_score))


def measure_worst_error_mm(form: SyntheticForm, doc: NormalizedDocument) -> float:
    """문서 곳곳의 정답 영역에서 잰 오차 중 최댓값(mm)을 반환한다.

    한 지점만 재면 평행이동 오차만 보게 되므로, 페이지 위·중간·아래에
    흩어진 영역을 함께 재서 회전·축척 잔차까지 잡는다.

    :param form: 합성 문서.
    :param doc: 정규화 결과.
    :returns: 최대 오차(mm).
    """
    layout = form.layout
    boxes: list[BoxMm] = [layout.title_box_mm, layout.clause_box_mm]
    boxes.extend(row.label_box_mm for row in layout.person_rows)
    boxes.extend(option.box_mm for option in layout.options)

    worst = 0.0
    for box in boxes:
        error_mm, score = measure_box_error_mm(form, doc, box)
        assert score > 0.5, f"정합 점수가 너무 낮아 측정을 신뢰할 수 없습니다: {score:.3f}"
        worst = max(worst, error_mm)
    return worst


def synthetic_page_on_desk(
    page_size: tuple[int, int] = (424, 600),
    offset: tuple[int, int] = (138, 150),
    canvas: tuple[int, int] = (900, 700),
    background: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """픽스처 없이 만든 "어두운 책상 위 흰 A4 문서" 이미지.

    :param page_size: 문서 크기 ``(width_px, height_px)``.
    :param offset: 문서 좌상단 위치 ``(x_px, y_px)``.
    :param canvas: 전체 이미지 크기 ``(height_px, width_px)``.
    :param background: 배경 밝기.
    :returns: ``(이미지, 정답 quad)``. quad 는 TL, TR, BR, BL 순.
    """
    height, width = canvas
    image = np.full((height, width, 3), background, np.uint8)
    page_w, page_h = page_size
    x0, y0 = offset
    page = np.full((page_h, page_w, 3), 250, np.uint8)
    page[80:130, 40:page_w - 40] = 30  # 제목 줄(잉크)
    page[300:340, 40:page_w - 120] = 30  # 본문 줄(잉크)
    image[y0:y0 + page_h, x0:x0 + page_w] = page
    quad = np.array(
        [
            [x0, y0],
            [x0 + page_w - 1, y0],
            [x0 + page_w - 1, y0 + page_h - 1],
            [x0, y0 + page_h - 1],
        ],
        dtype=np.float64,
    )
    return (image, quad)


class TestQuadDetection:
    """문서 사각형 검출."""

    def test_detects_page_on_dark_desk(self) -> None:
        """합성한 흰 문서의 네 꼭짓점을 5px 이내로 찾는다."""
        image, truth = synthetic_page_on_desk()
        quad, confidence = detect_document_quad(image)
        assert quad.shape == (4, 2)
        assert confidence > VISION_TRUST_THRESHOLD
        assert np.max(np.abs(order_quad(quad) - truth)) <= 5.0

    def test_detects_skewed_fixture(self, skewed_form: SyntheticForm) -> None:
        """15도 기울어진 문서를 대체 경로 없이 검출한다."""
        detection = detect_document_quad_detailed(skewed_form.image)
        assert detection.fallback is False
        assert detection.confidence > 0.8
        assert detection.warnings == ()
        assert 1.0 < detection.aspect < 2.0

    def test_detects_perspective_fixture(self, perspective_form: SyntheticForm) -> None:
        """원근 왜곡 + 조명 불균일 문서도 검출한다(적응형 경로 포함)."""
        detection = detect_document_quad_detailed(perspective_form.image)
        assert detection.fallback is False
        assert detection.confidence > 0.8
        assert 0.2 <= detection.area_ratio <= 0.995

    def test_falls_back_to_image_bounds(self, clean_form: SyntheticForm) -> None:
        """여백 없이 문서로 가득 찬 이미지는 대체 경로로 처리하고 사유를 남긴다."""
        detection = detect_document_quad_detailed(clean_form.image)
        assert detection.fallback is True
        assert detection.method == "image_bounds"
        assert detection.confidence < VISION_TRUST_THRESHOLD
        assert detection.warnings and "이미지 전체를" in detection.warnings[0]
        height, width = clean_form.image.shape[:2]
        assert order_quad(detection.quad).tolist() == [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ]

    def test_fallback_confidence_drops_for_odd_aspect(self) -> None:
        """A4 비율과 동떨어진 이미지는 대체 경로 신뢰도를 더 낮춘다."""
        blank = np.full((300, 900, 3), 128, np.uint8)
        detection = detect_document_quad_detailed(blank)
        assert detection.fallback is True
        assert detection.confidence <= 0.3

    def test_accepts_grayscale_input(self) -> None:
        """그레이스케일 입력도 그대로 처리한다."""
        image, truth = synthetic_page_on_desk()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        quad, _ = detect_document_quad(gray)
        assert np.max(np.abs(order_quad(quad) - truth)) <= 5.0

    def test_is_deterministic(self, skewed_form: SyntheticForm) -> None:
        """같은 입력이면 같은 사각형이 나온다."""
        first, first_conf = detect_document_quad(skewed_form.image)
        second, second_conf = detect_document_quad(skewed_form.image)
        assert np.array_equal(first, second)
        assert first_conf == second_conf

    @pytest.mark.parametrize(
        "bad, expected",
        [
            (None, DocumentNotFoundError),
            (np.zeros((0, 0, 3), np.uint8), DocumentNotFoundError),
            (np.zeros((4, 4, 3), np.uint8), VisionError),
            (np.zeros((10, 10, 5), np.uint8), VisionError),
            (np.zeros((2, 3, 4, 5), np.uint8), VisionError),
        ],
    )
    def test_rejects_invalid_input(self, bad: object, expected: type) -> None:
        """잘못된 입력은 도메인 예외로 감싸 올린다(조용한 실패 금지)."""
        with pytest.raises(expected):
            detect_document_quad(bad)


class TestNormalizeAccuracy:
    """정규화 후 좌표 정확도 실측."""

    def test_clean_document_within_3mm(self, clean_form: SyntheticForm) -> None:
        """정상 문서: 좌표 오차 3mm 이내(로드맵 KPI ±3mm)."""
        doc = normalize_document(clean_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        worst = measure_worst_error_mm(clean_form, doc)
        assert worst <= 3.0, f"정상 문서 좌표 오차가 큽니다: {worst:.2f}mm"

    def test_skewed_document_within_10mm(self, skewed_form: SyntheticForm) -> None:
        """15도 기울어진 문서: 좌표 오차 10mm 이내."""
        doc = normalize_document(skewed_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        assert abs(doc.skew_deg) == pytest.approx(15.0, abs=1.5)
        worst = measure_worst_error_mm(skewed_form, doc)
        assert worst <= 10.0, f"기울어진 문서 좌표 오차가 큽니다: {worst:.2f}mm"

    def test_lowres_document_within_15mm(self, lowres_form: SyntheticForm) -> None:
        """150dpi 저해상도 문서: 오차 15mm 이내이고 등급도 15.0 으로 보고한다."""
        doc = normalize_document(lowres_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        assert doc.dpi_estimate == pytest.approx(150.0, abs=5.0)
        assert doc.expected_error_mm == ERROR_MM_LOWRES
        worst = measure_worst_error_mm(lowres_form, doc)
        assert worst <= 15.0, f"저해상도 문서 좌표 오차가 큽니다: {worst:.2f}mm"

    def test_perspective_document_within_10mm(
        self, perspective_form: SyntheticForm
    ) -> None:
        """원근 왜곡 문서: 정립 후 좌표 오차 10mm 이내."""
        doc = normalize_document(perspective_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        worst = measure_worst_error_mm(perspective_form, doc)
        assert worst <= 10.0, f"원근 왜곡 문서 좌표 오차가 큽니다: {worst:.2f}mm"

    def test_checkbox_center_maps_to_actuator_target(
        self, clean_form: SyntheticForm
    ) -> None:
        """체크박스 중심(액추에이터 목표점)이 실제 잉크 위에 떨어진다."""
        doc = normalize_document(clean_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        box = clean_form.option_box("동의함")
        error_mm, _ = measure_box_error_mm(clean_form, doc, box)
        assert error_mm <= 3.0

        center_px = doc.coords.point_to_px(box.center())
        half = int(round(box.w_mm / 2.0 * doc.coords.px_per_mm_x))
        patch = _to_gray(doc.image)[
            center_px[1] - half : center_px[1] + half + 1,
            center_px[0] - half : center_px[0] + half + 1,
        ]
        assert patch.size > 0
        # 체크박스는 테두리만 그려져 있으므로 사각형 테두리의 잉크가 잡혀야 한다.
        assert float(np.min(patch)) < 128.0


class TestNormalizedDocument:
    """정규화 결과 객체의 계약."""

    def test_output_is_exact_a4_canvas(self, clean_form: SyntheticForm) -> None:
        """정립 이미지 크기는 요청 dpi 의 A4 크기와 정확히 같다."""
        doc = normalize_document(clean_form.image, dpi=300)
        assert doc.size_px == (2480, 3508)
        assert doc.coords.size_px == doc.size_px
        assert doc.coords.dpi == 300.0
        assert doc.image.dtype == np.uint8
        assert doc.image.ndim == 3

    def test_grayscale_input_keeps_two_dimensions(
        self, clean_form: SyntheticForm
    ) -> None:
        """그레이스케일 입력은 그레이스케일로 정립된다."""
        gray = cv2.cvtColor(clean_form.image, cv2.COLOR_BGR2GRAY)
        doc = normalize_document(gray, dpi=150)
        assert doc.image.ndim == 2
        assert doc.size_px == doc.coords.size_px

    def test_is_deterministic(self, skewed_form: SyntheticForm) -> None:
        """같은 입력이면 픽셀 단위로 같은 결과가 나온다."""
        first = normalize_document(skewed_form.image, dpi=150)
        second = normalize_document(skewed_form.image, dpi=150)
        assert np.array_equal(first.image, second.image)
        assert first.confidence == second.confidence
        assert first.expected_error_mm == second.expected_error_mm

    def test_low_confidence_flags_instead_of_raising(
        self, clean_form: SyntheticForm
    ) -> None:
        """신뢰도 미달이어도 예외를 던지지 않고 플래그와 경고만 남긴다."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 경고를 예외로 승격시켜도 통과해야 한다.
            doc = normalize_document(clean_form.image, dpi=200)
        assert doc.low_confidence is True
        assert doc.confidence < VISION_TRUST_THRESHOLD
        assert any("신뢰도" in message for message in doc.warnings)

    def test_low_confidence_does_not_raise_low_confidence_error(
        self, clean_form: SyntheticForm
    ) -> None:
        """LowConfidenceError 는 Vision 이 아니라 Agent 계층의 판단 재료다."""
        try:
            normalize_document(clean_form.image, dpi=200)
        except LowConfidenceError as exc:  # pragma: no cover - 회귀 방지용
            pytest.fail(f"정규화가 LowConfidenceError 를 던졌습니다: {exc}")

    def test_trusted_document_has_no_confidence_warning(
        self, skewed_form: SyntheticForm
    ) -> None:
        """문서 경계를 실제로 찾았고 등급이 좋으면 신뢰 임계값을 넘는다."""
        doc = normalize_document(skewed_form.image, dpi=DEFAULT_NORMALIZE_DPI)
        assert doc.detection_confidence > 0.85
        assert doc.method != "image_bounds"

    def test_describe_is_json_serializable_without_image(
        self, clean_form: SyntheticForm
    ) -> None:
        """요약 정보는 이미지 없이 JSON 으로 남길 수 있다(로그용)."""
        doc = normalize_document(clean_form.image, dpi=200)
        payload = doc.describe()
        text = json.dumps(payload, ensure_ascii=False)
        assert "image" not in payload
        assert "expected_error_mm" in payload
        assert "dpi_estimate" in text

    def test_source_to_mm_maps_quad_corners_to_page_corners(
        self, perspective_form: SyntheticForm
    ) -> None:
        """원본 이미지의 문서 꼭짓점은 페이지 꼭짓점 mm 로 환산된다."""
        doc = normalize_document(perspective_form.image, dpi=200)
        corners = order_quad(doc.quad)
        top_left = doc.source_to_mm(float(corners[0][0]), float(corners[0][1]))
        bottom_right = doc.source_to_mm(float(corners[2][0]), float(corners[2][1]))
        assert top_left.x_mm == pytest.approx(0.0, abs=0.5)
        assert top_left.y_mm == pytest.approx(0.0, abs=0.5)
        assert bottom_right.x_mm == pytest.approx(A4_WIDTH_MM, abs=0.5)
        assert bottom_right.y_mm == pytest.approx(A4_HEIGHT_MM, abs=0.5)

    def test_to_mm_delegates_to_coordinate_system(
        self, clean_form: SyntheticForm
    ) -> None:
        """``to_mm`` 은 동봉된 좌표계와 같은 결과를 준다."""
        doc = normalize_document(clean_form.image, dpi=200)
        box_px = BoxPx(x=100, y=200, w=50, h=60)
        assert doc.to_mm(box_px) == doc.coords.to_mm(box_px)

    def test_rejects_invalid_dpi(self, clean_form: SyntheticForm) -> None:
        """dpi 0 은 거부한다."""
        with pytest.raises(ValueError, match="dpi"):
            normalize_document(clean_form.image, dpi=0)

    def test_rejects_missing_image(self) -> None:
        """입력이 없으면 문서 미검출 예외를 던진다."""
        with pytest.raises(DocumentNotFoundError):
            normalize_document(None)


class TestExpectedErrorGrade:
    """기대 오차 등급 규칙."""

    @pytest.mark.parametrize(
        "skew, dpi, expected",
        [
            (0.0, 300.0, ERROR_MM_NORMAL),
            (4.9, 250.0, ERROR_MM_NORMAL),
            (-4.9, 600.0, ERROR_MM_NORMAL),
            (0.0, 200.0, ERROR_MM_SKEWED),  # 해상도가 250 미만이면 정상 등급이 아니다.
            (15.0, 300.0, ERROR_MM_SKEWED),
            (-24.9, 300.0, ERROR_MM_SKEWED),
            (0.0, 150.0, ERROR_MM_LOWRES),  # 저해상도는 등급을 끌어올린다.
            (20.0, 150.0, ERROR_MM_LOWRES),
            (30.0, 300.0, ERROR_MM_SEVERE),
            (30.0, 150.0, ERROR_MM_SEVERE),  # 겹치면 더 큰 값
        ],
    )
    def test_grade_table(self, skew: float, dpi: float, expected: float) -> None:
        """등급표가 문서화된 규칙 그대로 동작한다."""
        assert expected_error_mm_for(skew, dpi) == expected

    def test_dpi_is_rounded_before_comparison(self) -> None:
        """199.94dpi 는 반올림되어 저해상도로 강등되지 않는다."""
        assert expected_error_mm_for(0.0, 199.94) == ERROR_MM_SKEWED
        assert expected_error_mm_for(0.0, 199.4) == ERROR_MM_LOWRES


class TestNormalizeBoxes:
    """픽셀 사각형 → mm 사각형 일괄 변환."""

    def test_converts_in_order(self) -> None:
        """입력 순서를 유지하며 mm 로 바꾼다."""
        coords = A4CoordinateSystem.from_dpi(300)
        boxes = [BoxPx(0, 0, 100, 200), BoxPx(500, 600, 50, 50)]
        result = normalize_boxes(boxes, coords)
        assert len(result) == 2
        assert all(isinstance(box, BoxMm) for box in result)
        assert result[0].x_mm == pytest.approx(0.0)
        assert result[1].x_mm == pytest.approx(500 / coords.px_per_mm_x)

    def test_clamps_out_of_page_boxes(self) -> None:
        """페이지 밖으로 나간 사각형은 잘라 낸다(기본값)."""
        coords = A4CoordinateSystem.from_dpi(300)
        huge = BoxPx(0, 0, coords.width_px * 2, coords.height_px * 2)
        with pytest.warns(Warning):
            result = normalize_boxes([huge], coords)
        assert result[0].w_mm == pytest.approx(A4_WIDTH_MM)
        assert result[0].h_mm == pytest.approx(A4_HEIGHT_MM)

    def test_rejects_wrong_type(self) -> None:
        """BoxMm 을 잘못 넣으면 즉시 실패한다."""
        coords = A4CoordinateSystem.from_dpi(300)
        with pytest.raises(TypeError, match="BoxPx"):
            normalize_boxes([BoxMm(0.0, 0.0, 10.0, 10.0)], coords)  # type: ignore[list-item]

    def test_empty_input(self) -> None:
        """빈 목록은 빈 목록을 돌려준다."""
        assert normalize_boxes([], A4CoordinateSystem.from_dpi(300)) == []


@pytest.mark.parametrize("grayscale", [False, True])
def test_a4_scan_keeps_signature_outside_large_table(grayscale):
    """An interior A4-shaped table must not crop the page's title or signature."""
    image = np.full((849, 600, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (80, 90), (515, 705), (0, 0, 0), 3)
    cv2.putText(image, "TITLE", (190, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(image, (420, 790), (475, 810), (0, 0, 0), -1)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detection = detect_document_quad_detailed(image)
    assert detection.method == "image_bounds"
    assert detection.quad.tolist() == [[0.0, 0.0], [599.0, 0.0], [599.0, 848.0], [0.0, 848.0]]
    assert detection.confidence < VISION_TRUST_THRESHOLD
    doc = normalize_document(image, dpi=100)
    # The bottom signature remains near its original proportional position.
    assert abs(doc.skew_deg) < 0.01
    x, y = round(450 * doc.image.shape[1] / 600), round(800 * doc.image.shape[0] / 849)
    assert float(np.mean(doc.image[y-2:y+3, x-2:x+3])) < 20
