"""Verify 단계 — 기입 전/후 이미지를 비교해 **실제로 써졌는지**만 판정한다.

이 모듈은 **필기 내용을 읽지 않는다.** 무엇이 써졌는지가 아니라 *써졌는지 여부*만
본다. 이유는 두 가지다.

1. 손글씨 인식은 실패율이 높고, 잘못 읽은 값을 사용자에게 낭독하면
   "대필 없이 스스로 확인한다"는 목표가 오히려 훼손된다.
2. 기입 흔적(잉크 증가)은 조명·해상도가 나빠도 안정적으로 측정된다.

판정 원리
---------
관심 영역(ROI)을 이진화해 **잉크 픽셀 비율**을 재고, 기입 전후 증가량으로
판정한다. 조명 변화에 흔들리지 않도록 ROI 주변 고리(ring) 영역의 밝은 쪽
분위수를 "종이 밝기"로 잡고, 그 값에 :data:`INK_RELATIVE_THRESHOLD` 를 곱한
값보다 어두운 픽셀만 잉크로 센다. 곱셈 조명 변화는 종이 밝기와 임계값에
같은 비율로 반영되므로 잉크 비율이 거의 변하지 않는다.

임계값 근거
-----------
* :data:`CHECKBOX_INK_DELTA_THRESHOLD` = 0.04
  한 변 6mm 체크칸 안쪽(테두리 제외)에 그은 체크 표시는 ROI 면적의 10% 이상을
  덮는다. 반대로 아무것도 쓰지 않으면 증가량은 0 에 수렴한다. 그 사이인 4% 를
  기준으로 잡아 오탐(거짓양성)과 미탐 양쪽에 여유를 둔다.
* :data:`SIGNATURE_INK_DELTA_THRESHOLD` = 0.012
  서명은 넓은 영역에 가는 획으로 퍼지므로 체크보다 낮게 잡는다. 대신
  **새로 생긴 잉크**의 연결 성분을 세어 가로 밑줄(인쇄선 잔상)을 제외한다.
* :data:`ALIGNMENT_PAD_MM` = 0.6
  정합 오차 보정용 ROI 여유. 이보다 큰 오차는 좌표계 자체가 다른 것으로 보고
  상위 단계에서 재정합해야 한다.

신뢰도
------
판정 경계에서 멀수록 높다. 증가량이 임계값과 같으면 :data:`BOUNDARY_CONFIDENCE`,
임계값의 두 배 이상 떨어져 있으면 :data:`MAX_CONFIDENCE` 에 도달한다.
:data:`docagent.contracts.PARTIAL_THRESHOLD` 미만이면 에이전트가 재확인하거나
직원 연결로 넘겨야 한다.

좌표 규약
---------
``coords`` 는 입력 이미지가 담고 있는 페이지 크기 ``(가로_mm, 세로_mm)`` 다.
mm → px 환산은 ``이미지_px / 페이지_mm`` 비율로 하며, 따라서 입력 이미지는
정합(deskew·crop)이 끝나 페이지 전체를 꽉 채운 상태여야 한다.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from docagent.contracts import (
    A4_PAGE_SIZE_MM,
    BoxMm,
    DocumentStructure,
    Field,
    FieldType,
    VerificationResult,
)
from docagent.errors import VisionError
from docagent.interfaces import ImageArray

__all__ = [
    "INK_RELATIVE_THRESHOLD",
    "PAPER_PERCENTILE",
    "RING_MM",
    "ALIGNMENT_PAD_MM",
    "CHECKBOX_INSET_RATIO",
    "CHECKBOX_INK_DELTA_THRESHOLD",
    "SIGNATURE_INK_DELTA_THRESHOLD",
    "REGION_INK_DELTA_THRESHOLD",
    "MIN_STROKE_HEIGHT_MM",
    "MIN_COMPONENT_AREA_MM2",
    "UNDERLINE_MIN_WIDTH_MM",
    "BOUNDARY_CONFIDENCE",
    "MAX_CONFIDENCE",
    "ink_ratio",
    "verify_checkbox",
    "verify_signature",
    "verify_region",
    "verify_options",
    "verify_field",
    "verify_required_fields",
    "unwritten_required",
]


#: 종이 밝기 대비 이 비율보다 어두우면 잉크로 센다.
INK_RELATIVE_THRESHOLD: float = 0.72
#: 주변 고리에서 "종이 밝기"로 삼을 분위수(%).
PAPER_PERCENTILE: float = 80.0
#: 종이 밝기를 재기 위해 ROI 바깥으로 넓히는 폭(mm).
RING_MM: float = 3.0
#: 정합 오차 보정용 ROI 여유(mm).
ALIGNMENT_PAD_MM: float = 0.6
#: 체크칸 테두리를 ROI 에서 빼기 위한 안쪽 축소 비율(짧은 변 기준).
CHECKBOX_INSET_RATIO: float = 0.18
#: 체크 판정 잉크 증가량 임계값.
CHECKBOX_INK_DELTA_THRESHOLD: float = 0.04
#: 서명 판정 잉크 증가량 임계값(밑줄 제외 후).
SIGNATURE_INK_DELTA_THRESHOLD: float = 0.012
#: 일반 영역(문자·날짜) 판정 잉크 증가량 임계값.
REGION_INK_DELTA_THRESHOLD: float = 0.012
#: 서명 획으로 인정할 최소 세로 길이(mm). 이보다 납작하면 밑줄로 본다.
MIN_STROKE_HEIGHT_MM: float = 1.2
#: 잡음으로 버릴 연결 성분 최소 면적(mm^2).
MIN_COMPONENT_AREA_MM2: float = 0.3
#: 밑줄로 의심할 최소 가로 길이(mm).
UNDERLINE_MIN_WIDTH_MM: float = 8.0
#: 판정 경계에서의 신뢰도.
BOUNDARY_CONFIDENCE: float = 0.60
#: 도달 가능한 최대 신뢰도(항상 1.0 미만으로 두어 과신을 막는다).
MAX_CONFIDENCE: float = 0.99


# --------------------------------------------------------------------------
# 이미지 헬퍼
# --------------------------------------------------------------------------


def _to_gray(image: ImageArray, *, name: str) -> np.ndarray:
    """입력 이미지를 uint8 그레이스케일 배열로 만든다.

    :param image: ``(H, W, 3)`` BGR 또는 ``(H, W)`` 그레이스케일 배열.
    :param name: 오류 메시지에 쓸 인자 이름.
    :returns: ``(H, W)`` uint8 배열.
    :raises docagent.errors.VisionError: 배열이 아니거나 차원·채널이 맞지 않는 경우.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] not in (3, 4):
            raise VisionError(
                f"{name} 이미지의 채널 수가 올바르지 않습니다: {array.shape!r}"
            )
        array = cv2.cvtColor(
            array[:, :, :3].astype(np.uint8, copy=False), cv2.COLOR_BGR2GRAY
        )
    elif array.ndim == 2:
        array = array.astype(np.uint8, copy=False)
    else:
        raise VisionError(
            f"{name} 이미지는 (H, W) 또는 (H, W, 3) 배열이어야 합니다: {array.shape!r}"
        )
    if array.size == 0:
        raise VisionError(f"{name} 이미지가 비어 있습니다.")
    return array


def _check_same_frame(before: np.ndarray, after: np.ndarray) -> None:
    """두 이미지가 같은 좌표계(같은 크기)인지 검증한다.

    :param before: 기입 전 그레이스케일 이미지.
    :param after: 기입 후 그레이스케일 이미지.
    :returns: ``None``.
    :raises docagent.errors.VisionError: 크기가 다른 경우(정합이 선행되어야 한다).
    """
    if before.shape != after.shape:
        raise VisionError(
            "기입 전/후 이미지의 좌표계가 다릅니다. 같은 크기로 정합한 뒤 비교해야 합니다. "
            f"(전={before.shape!r}, 후={after.shape!r})"
        )


def _scale(image: np.ndarray, coords: tuple[float, float]) -> tuple[float, float]:
    """페이지 크기와 이미지 크기로부터 ``(px/mm_x, px/mm_y)`` 를 구한다.

    :param image: 그레이스케일 이미지.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :returns: ``(px_per_mm_x, px_per_mm_y)``.
    :raises ValueError: ``coords`` 가 올바르지 않은 경우.
    """
    values = tuple(float(v) for v in coords)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"coords 는 0 보다 큰 (가로_mm, 세로_mm) 이어야 합니다: {coords!r}")
    height, width = image.shape[:2]
    return (width / values[0], height / values[1])


def _rect_px(
    box: BoxMm, image: np.ndarray, scale: tuple[float, float]
) -> tuple[int, int, int, int]:
    """mm 사각형을 이미지 안으로 자른 ``(x0, y0, x1, y1)`` 픽셀 사각형으로 바꾼다.

    :param box: mm 사각형.
    :param image: 대상 이미지(경계 클리핑용).
    :param scale: ``(px/mm_x, px/mm_y)``.
    :returns: 픽셀 사각형(끝 좌표 제외).
    :raises docagent.errors.VisionError: 영역이 이미지 밖이라 유효 픽셀이 없는 경우.
    """
    height, width = image.shape[:2]
    x0 = int(round(box.x_mm * scale[0]))
    y0 = int(round(box.y_mm * scale[1]))
    x1 = int(round(box.right_mm * scale[0]))
    y1 = int(round(box.bottom_mm * scale[1]))
    x0, x1 = max(0, min(x0, width)), max(0, min(x1, width))
    y0, y1 = max(0, min(y0, height)), max(0, min(y1, height))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise VisionError(
            f"검증 영역이 이미지 밖이거나 너무 작습니다: {box.to_dict()!r} "
            f"(이미지 {width}x{height}px)"
        )
    return (x0, y0, x1, y1)


def _pad(box: BoxMm, pad_mm: float) -> BoxMm:
    """사각형을 사방으로 ``pad_mm`` 만큼 넓힌다(음수면 줄인다)."""
    width = max(0.1, box.w_mm + 2 * pad_mm)
    height = max(0.1, box.h_mm + 2 * pad_mm)
    return BoxMm(box.x_mm - pad_mm, box.y_mm - pad_mm, width, height)


def _inset(box: BoxMm, ratio: float) -> BoxMm:
    """짧은 변 기준 비율만큼 사각형을 안쪽으로 줄인다(테두리 제외용)."""
    inset = min(box.w_mm, box.h_mm) * ratio
    width = max(0.1, box.w_mm - 2 * inset)
    height = max(0.1, box.h_mm - 2 * inset)
    return BoxMm(box.x_mm + inset, box.y_mm + inset, width, height)


def _paper_level(
    gray: np.ndarray, rect: tuple[int, int, int, int], scale: tuple[float, float]
) -> float:
    """ROI 주변 고리에서 종이 밝기를 추정한다.

    :param gray: 그레이스케일 이미지.
    :param rect: ROI 픽셀 사각형.
    :param scale: ``(px/mm_x, px/mm_y)``.
    :returns: 종이 밝기(0~255). 주변을 쓸 수 없으면 ROI 자체의 밝은 쪽 분위수.
    """
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = rect
    ring_x = max(1, int(round(RING_MM * scale[0])))
    ring_y = max(1, int(round(RING_MM * scale[1])))
    ox0, oy0 = max(0, x0 - ring_x), max(0, y0 - ring_y)
    ox1, oy1 = min(width, x1 + ring_x), min(height, y1 + ring_y)
    outer = gray[oy0:oy1, ox0:ox1]
    mask = np.ones(outer.shape, dtype=bool)
    mask[y0 - oy0 : y1 - oy0, x0 - ox0 : x1 - ox0] = False
    ring = outer[mask]
    sample = ring if ring.size >= 16 else gray[y0:y1, x0:x1].ravel()
    if sample.size == 0:
        return 255.0
    return float(np.percentile(sample.astype(np.float32), PAPER_PERCENTILE))


def _ink_mask(
    gray: np.ndarray, rect: tuple[int, int, int, int], paper: float
) -> np.ndarray:
    """ROI 안에서 잉크로 판정된 픽셀의 불리언 마스크를 만든다.

    :param gray: 그레이스케일 이미지.
    :param rect: ROI 픽셀 사각형.
    :param paper: 종이 밝기(0~255).
    :returns: ``(h, w)`` bool 배열.
    """
    x0, y0, x1, y1 = rect
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    threshold = paper * INK_RELATIVE_THRESHOLD if paper > 1.0 else 128.0
    return roi < threshold


def ink_ratio(
    image: ImageArray, box_mm: BoxMm, coords: tuple[float, float] = A4_PAGE_SIZE_MM
) -> float:
    """영역의 잉크 픽셀 비율(0.0~1.0)을 잰다.

    주변 밝기로 정규화하므로 조명이 달라져도 값이 크게 흔들리지 않는다.

    :param image: 대상 이미지.
    :param box_mm: 측정할 mm 영역.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :returns: 잉크 픽셀 비율.
    :raises docagent.errors.VisionError: 이미지·영역이 유효하지 않은 경우.
    """
    gray = _to_gray(image, name="입력")
    scale = _scale(gray, coords)
    rect = _rect_px(box_mm, gray, scale)
    paper = _paper_level(gray, rect, scale)
    return float(_ink_mask(gray, rect, paper).mean())


def _confidence(delta: float, threshold: float) -> float:
    """판정 경계로부터의 거리로 신뢰도를 계산한다.

    :param delta: 잉크 비율 증가량.
    :param threshold: 판정 임계값.
    :returns: :data:`BOUNDARY_CONFIDENCE` ~ :data:`MAX_CONFIDENCE` 사이 값.
    """
    if threshold <= 0:
        return BOUNDARY_CONFIDENCE
    margin = min(1.0, abs(delta - threshold) / threshold)
    return round(
        BOUNDARY_CONFIDENCE + (MAX_CONFIDENCE - BOUNDARY_CONFIDENCE) * margin, 4
    )


# --------------------------------------------------------------------------
# 개별 검증
# --------------------------------------------------------------------------


def verify_checkbox(
    before_img: ImageArray,
    after_img: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
    option_box: BoxMm | None = None,
    *,
    field_id: str = "checkbox",
) -> VerificationResult:
    """체크칸 하나가 실제로 체크되었는지 판정한다.

    체크칸 **안쪽**(:data:`CHECKBOX_INSET_RATIO` 만큼 축소)만 본다. 인쇄된
    테두리는 전후 모두에 있어 상쇄되지만, 정합 오차가 있을 때 굵은 테두리가
    ROI 안팎을 들락거리며 오탐을 만들기 때문이다.

    :param before_img: 기입 전 이미지.
    :param after_img: 기입 후 이미지. ``before_img`` 와 같은 크기여야 한다.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :param option_box: 체크칸 mm 좌표. ``None`` 이면 :class:`ValueError`.
    :param field_id: 결과에 기록할 항목 식별자.
    :returns: :class:`VerificationResult`. ``written`` 은 잉크 증가량이
        :data:`CHECKBOX_INK_DELTA_THRESHOLD` 이상일 때 True.
    :raises ValueError: ``option_box`` 가 ``None`` 인 경우.
    :raises docagent.errors.VisionError: 두 이미지의 좌표계가 다르거나 영역이 유효하지 않은 경우.
    """
    if option_box is None:
        raise ValueError("verify_checkbox 에는 option_box(체크칸 mm 좌표)가 필요합니다.")
    before = _to_gray(before_img, name="기입 전")
    after = _to_gray(after_img, name="기입 후")
    _check_same_frame(before, after)

    scale = _scale(before, coords)
    roi_box = _inset(option_box, CHECKBOX_INSET_RATIO)
    rect = _rect_px(roi_box, before, scale)

    ratio_before = float(_ink_mask(before, rect, _paper_level(before, rect, scale)).mean())
    ratio_after = float(_ink_mask(after, rect, _paper_level(after, rect, scale)).mean())
    delta = ratio_after - ratio_before
    written = delta >= CHECKBOX_INK_DELTA_THRESHOLD
    confidence = _confidence(delta, CHECKBOX_INK_DELTA_THRESHOLD)

    if written:
        reason = (
            f"체크칸 안쪽 잉크 비율이 {ratio_before:.3f} 에서 {ratio_after:.3f} 로 "
            f"{delta:.3f} 늘어 체크된 것으로 판정했습니다."
        )
    else:
        reason = (
            f"체크칸 안쪽 잉크 비율 증가량이 {delta:.3f} 로 기준"
            f"({CHECKBOX_INK_DELTA_THRESHOLD:.3f})에 못 미쳐 체크되지 않은 것으로 판정했습니다."
        )
    return VerificationResult(
        field_id=field_id,
        written=written,
        ink_ratio_before=ratio_before,
        ink_ratio_after=ratio_after,
        confidence=confidence,
        reason=reason,
    )


def verify_signature(
    before_img: ImageArray,
    after_img: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
    signature_box: BoxMm | None = None,
    *,
    field_id: str = "signature",
) -> VerificationResult:
    """서명란에 실제로 서명 획이 그어졌는지 판정한다(내용은 읽지 않는다).

    **새로 생긴 잉크**(기입 후에만 잉크인 픽셀)의 연결 성분을 세고,
    가로로 길면서 납작한 성분(인쇄 밑줄의 정합 잔상)은 제외한다.
    남은 획의 면적 비율과 세로 퍼짐으로 서명 여부를 판정한다.

    :param before_img: 기입 전 이미지.
    :param after_img: 기입 후 이미지. ``before_img`` 와 같은 크기여야 한다.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :param signature_box: 서명란 mm 좌표. ``None`` 이면 :class:`ValueError`.
    :param field_id: 결과에 기록할 항목 식별자.
    :returns: :class:`VerificationResult`. ``ink_ratio_*`` 는 ROI 전체 잉크 비율,
        판정은 밑줄을 제외한 **새 잉크** 비율로 한다.
    :raises ValueError: ``signature_box`` 가 ``None`` 인 경우.
    :raises docagent.errors.VisionError: 두 이미지의 좌표계가 다르거나 영역이 유효하지 않은 경우.
    """
    if signature_box is None:
        raise ValueError("verify_signature 에는 signature_box(서명란 mm 좌표)가 필요합니다.")
    before = _to_gray(before_img, name="기입 전")
    after = _to_gray(after_img, name="기입 후")
    _check_same_frame(before, after)

    scale = _scale(before, coords)
    roi_box = _pad(signature_box, ALIGNMENT_PAD_MM)
    rect = _rect_px(roi_box, before, scale)

    mask_before = _ink_mask(before, rect, _paper_level(before, rect, scale))
    mask_after = _ink_mask(after, rect, _paper_level(after, rect, scale))
    ratio_before = float(mask_before.mean())
    ratio_after = float(mask_after.mean())

    fresh = np.logical_and(mask_after, np.logical_not(mask_before)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fresh, connectivity=8)

    min_area_px = max(1.0, MIN_COMPONENT_AREA_MM2 * scale[0] * scale[1])
    min_height_px = max(1.0, MIN_STROKE_HEIGHT_MM * scale[1])
    underline_width_px = max(1.0, UNDERLINE_MIN_WIDTH_MM * scale[0])

    stroke_area = 0
    stroke_count = 0
    stroke_height_px = 0
    for index in range(1, count):
        width_px = int(stats[index, cv2.CC_STAT_WIDTH])
        height_px = int(stats[index, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[index, cv2.CC_STAT_AREA])
        if area_px < min_area_px:
            continue
        if height_px < min_height_px and width_px >= underline_width_px:
            continue  # 가로로 길고 납작한 성분 = 인쇄 밑줄 잔상
        stroke_area += area_px
        stroke_count += 1
        stroke_height_px = max(stroke_height_px, height_px)

    roi_area = float(fresh.size)
    stroke_ratio = stroke_area / roi_area if roi_area else 0.0
    written = (
        stroke_ratio >= SIGNATURE_INK_DELTA_THRESHOLD
        and stroke_count >= 1
        and stroke_height_px >= min_height_px
    )
    confidence = _confidence(stroke_ratio, SIGNATURE_INK_DELTA_THRESHOLD)

    if written:
        reason = (
            f"서명란에서 밑줄을 제외한 새 획 {stroke_count}개(면적 비율 {stroke_ratio:.3f})를 "
            "확인해 서명된 것으로 판정했습니다. 서명 내용은 읽지 않았습니다."
        )
    else:
        reason = (
            f"서명란의 새 획 면적 비율이 {stroke_ratio:.3f} 로 기준"
            f"({SIGNATURE_INK_DELTA_THRESHOLD:.3f})에 못 미쳐 서명되지 않은 것으로 판정했습니다."
        )
    return VerificationResult(
        field_id=field_id,
        written=written,
        ink_ratio_before=ratio_before,
        ink_ratio_after=ratio_after,
        confidence=confidence,
        reason=reason,
    )


def verify_region(
    before_img: ImageArray,
    after_img: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
    box_mm: BoxMm | None = None,
    *,
    field_id: str = "region",
    threshold: float = REGION_INK_DELTA_THRESHOLD,
) -> VerificationResult:
    """임의 영역(문자 입력란·날짜란)에 기입 흔적이 생겼는지 판정한다.

    :param before_img: 기입 전 이미지.
    :param after_img: 기입 후 이미지.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :param box_mm: 검사할 mm 영역. ``None`` 이면 :class:`ValueError`.
    :param field_id: 결과에 기록할 항목 식별자.
    :param threshold: 판정 임계값(잉크 비율 증가량).
    :returns: :class:`VerificationResult`.
    :raises ValueError: ``box_mm`` 이 ``None`` 이거나 ``threshold`` 가 0 이하인 경우.
    :raises docagent.errors.VisionError: 두 이미지의 좌표계가 다른 경우.
    """
    if box_mm is None:
        raise ValueError("verify_region 에는 box_mm(검사 영역 mm 좌표)이 필요합니다.")
    if threshold <= 0:
        raise ValueError(f"threshold 는 0 보다 커야 합니다: {threshold}")
    before = _to_gray(before_img, name="기입 전")
    after = _to_gray(after_img, name="기입 후")
    _check_same_frame(before, after)

    scale = _scale(before, coords)
    rect = _rect_px(_pad(box_mm, ALIGNMENT_PAD_MM), before, scale)
    ratio_before = float(_ink_mask(before, rect, _paper_level(before, rect, scale)).mean())
    ratio_after = float(_ink_mask(after, rect, _paper_level(after, rect, scale)).mean())
    delta = ratio_after - ratio_before
    written = delta >= threshold

    reason = (
        f"기입 영역의 잉크 비율이 {delta:+.3f} 변해 "
        f"{'기입된' if written else '기입되지 않은'} 것으로 판정했습니다."
    )
    return VerificationResult(
        field_id=field_id,
        written=written,
        ink_ratio_before=ratio_before,
        ink_ratio_after=ratio_after,
        confidence=_confidence(delta, threshold),
        reason=reason,
    )


# --------------------------------------------------------------------------
# 통합 진입점
# --------------------------------------------------------------------------


def _require_field(structure: DocumentStructure, field_id: str) -> Field:
    """구조에서 항목을 꺼내되 없으면 한국어 오류를 던진다.

    :param structure: 문서 구조.
    :param field_id: 항목 id.
    :returns: :class:`~docagent.contracts.Field`.
    :raises ValueError: 해당 항목이 없는 경우.
    :raises docagent.errors.VisionError: 항목에 좌표가 없어 검증할 수 없는 경우.
    """
    field = structure.field_by_id(field_id)
    if field is None:
        allowed = ", ".join(item.id for item in structure.fields) or "(없음)"
        raise ValueError(f"문서 구조에 없는 항목 id 입니다: {field_id!r}. 허용 값: {allowed}")
    return field


def verify_options(
    structure: DocumentStructure,
    field_id: str,
    before_img: ImageArray,
    after_img: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
) -> tuple[VerificationResult, ...]:
    """선택형 항목의 **선택지별** 체크 여부를 각각 판정한다.

    :param structure: 문서 구조.
    :param field_id: 선택형 항목 id.
    :param before_img: 기입 전 이미지.
    :param after_img: 기입 후 이미지.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :returns: 선택지 순서대로의 :class:`VerificationResult` 튜플.
        각 결과의 ``field_id`` 는 ``"<항목id>:<선택지라벨>"`` 형식이다.
    :raises ValueError: 항목이 없거나 선택지가 없는 경우.
    :raises docagent.errors.VisionError: 이미지 좌표계가 다른 경우.
    """
    field = _require_field(structure, field_id)
    if not field.options:
        raise ValueError(f"선택지가 없는 항목입니다: {field_id!r}")
    return tuple(
        verify_checkbox(
            before_img,
            after_img,
            coords,
            option.box_mm,
            field_id=f"{field.id}:{option.label}",
        )
        for option in field.options
    )


def verify_field(
    structure: DocumentStructure,
    field_id: str,
    before: ImageArray,
    after: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
) -> VerificationResult:
    """항목 하나의 기입 여부를 유형에 맞는 방법으로 판정한다(통합 진입점).

    * 선택형·체크박스 — 선택지별로 :func:`verify_checkbox` 를 돌리고
      **증가량이 가장 큰 선택지**를 대표로 삼는다. 어느 것도 기준을 넘지 못하면
      ``written=False``.
    * 서명 — :func:`verify_signature`.
    * 그 밖(문자·날짜·미분류) — :func:`verify_region`.

    :param structure: 문서 구조(:func:`~docagent.vision.structuring.build_structure` 산출물).
    :param field_id: 검증할 항목 id.
    :param before: 기입 전 이미지.
    :param after: 기입 후 이미지.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :returns: :class:`VerificationResult`. ``field_id`` 는 인자로 받은 항목 id 그대로다.
    :raises ValueError: 항목이 없는 경우.
    :raises docagent.errors.VisionError: 항목에 좌표가 없거나 이미지 좌표계가 다른 경우.
    """
    field = _require_field(structure, field_id)

    if field.type in (FieldType.CHOICE, FieldType.CHECKBOX) and field.options:
        results = verify_options(structure, field_id, before, after, coords)
        best = max(results, key=lambda item: item.ink_delta)
        label = best.field_id.split(":", 1)[-1]
        if best.written:
            reason = f"선택지 '{label}' 에 체크 표시를 확인했습니다. {best.reason}"
        else:
            reason = (
                "어느 선택지에서도 체크 표시를 확인하지 못했습니다. "
                f"가장 변화가 큰 선택지는 '{label}' 였습니다."
            )
        return VerificationResult(
            field_id=field.id,
            written=best.written,
            ink_ratio_before=best.ink_ratio_before,
            ink_ratio_after=best.ink_ratio_after,
            confidence=best.confidence,
            reason=reason,
        )

    if field.box_mm is None:
        raise VisionError(
            f"항목에 좌표가 없어 기입 여부를 검증할 수 없습니다: {field_id!r}"
        )

    if field.type is FieldType.SIGNATURE:
        return verify_signature(before, after, coords, field.box_mm, field_id=field.id)
    return verify_region(before, after, coords, field.box_mm, field_id=field.id)


def verify_required_fields(
    structure: DocumentStructure,
    before: ImageArray,
    after: ImageArray,
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
) -> tuple[VerificationResult, ...]:
    """필수 항목 전체를 ``order`` 순으로 검증한다.

    :param structure: 문서 구조.
    :param before: 기입 전 이미지.
    :param after: 기입 후 이미지.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``.
    :returns: 필수 항목별 :class:`VerificationResult` 튜플.
    :raises docagent.errors.VisionError: 이미지 좌표계가 다르거나 좌표가 없는 항목이 있는 경우.
    """
    results: list[VerificationResult] = []
    for field in structure.required_fields():
        results.append(verify_field(structure, field.id, before, after, coords))
    return tuple(results)


def unwritten_required(results: Sequence[VerificationResult]) -> tuple[str, ...]:
    """검증 결과 중 아직 기입되지 않은 항목 id 를 모은다.

    :param results: :func:`verify_required_fields` 등의 결과.
    :returns: ``written`` 이 False 인 항목 id 튜플(입력 순서 유지).
    """
    return tuple(item.field_id for item in results if not item.written)
