"""문서 정규화(See 단계) — 촬영 이미지를 정립된 A4 페이지로 펴 준다.

파이프라인::

    입력 이미지
      → 문서 사각형 검출(detect_document_quad)
      → 호모그래피로 A4 정립(warpPerspective)
      → A4CoordinateSystem 동봉(NormalizedDocument)

이 단계가 끝나면 이미지의 픽셀 좌표는 **결정론적으로 mm 로 환산**된다.
이후의 탐지·OCR·검증은 모두 이 정립 이미지 위에서 이루어지고,
액추에이터가 실제로 펜을 옮길 목표점도 여기서 나온 mm 좌표다.

정확도 등급(로드맵 Phase 1 Week 3-4 실측 등급을 코드로 명문화)
--------------------------------------------------------------
============================== ===================
조건                            expected_error_mm
============================== ===================
|기울기| < 5도 이고 dpi ≥ 250    5.0
|기울기| < 25도                  10.0
dpi < 200                        15.0
|기울기| ≥ 25도                  20.0
============================== ===================

조건이 겹치면 **더 큰 값**을 쓴다. 이 값은 "이 이미지에서 뽑은 좌표로 펜을
움직이면 이 정도 오차를 각오해야 한다"는 상한이며, 상위 계층(Agent)이
사용자에게 재확인을 요구할지 판단하는 근거가 된다.

신뢰도가 :data:`~docagent.contracts.VISION_TRUST_THRESHOLD` 미만이어도
이 모듈은 **예외를 던지지 않는다.** :attr:`NormalizedDocument.low_confidence`
플래그와 한국어 경고만 남긴다. 직원 연결(handoff) 판단은 Agent 계층의 책임이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    Point,
    VISION_TRUST_THRESHOLD,
)
from docagent.errors import DocumentNotFoundError, VisionError
from docagent.vision.geometry import (
    A4_ASPECT_RATIO,
    MM_PER_INCH,
    A4CoordinateSystem,
    apply_homography,
    estimate_skew_deg,
    order_quad,
    quad_area_ratio,
    quad_edge_lengths,
    solve_homography,
)

__all__ = [
    "DEFAULT_NORMALIZE_DPI",
    "MIN_QUAD_AREA_RATIO",
    "MAX_QUAD_AREA_RATIO",
    "ASPECT_TOLERANCE",
    "SKEW_NORMAL_DEG",
    "SKEW_MAX_DEG",
    "DPI_NORMAL",
    "DPI_LOW",
    "ERROR_MM_NORMAL",
    "ERROR_MM_SKEWED",
    "ERROR_MM_LOWRES",
    "ERROR_MM_SEVERE",
    "QuadDetection",
    "NormalizedDocument",
    "detect_document_quad",
    "detect_document_quad_detailed",
    "expected_error_mm_for",
    "normalize_document",
    "normalize_boxes",
]


# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------

#: 정립 이미지의 기본 해상도(dpi). 300dpi 면 1px ≈ 0.085mm 로 ±3mm KPI 에 여유가 있다.
DEFAULT_NORMALIZE_DPI: int = 300

#: 문서 후보가 이미지에서 차지해야 하는 최소 면적비.
MIN_QUAD_AREA_RATIO: float = 0.20
#: 최대 면적비. 이보다 크면 "이미지 테두리 그 자체"를 잡은 것으로 보고 후보에서 뺀다.
MAX_QUAD_AREA_RATIO: float = 0.995
#: A4 종횡비(≈1.414)에서 허용하는 편차.
ASPECT_TOLERANCE: float = 0.45
#: 문서 후보의 최소 변 길이(px).
MIN_EDGE_PX: float = 20.0

#: 이 각도 미만이면 "똑바로 놓인" 것으로 본다(도).
SKEW_NORMAL_DEG: float = 5.0
#: 이 각도 이상이면 "심하게 기울어진" 것으로 본다(도).
SKEW_MAX_DEG: float = 25.0
#: 이 해상도 이상이면 "충분한 해상도"로 본다(dpi).
DPI_NORMAL: float = 250.0
#: 이 해상도 미만이면 "저해상도"로 본다(dpi).
DPI_LOW: float = 200.0

#: 정상 등급 기대 오차(mm).
ERROR_MM_NORMAL: float = 5.0
#: 기울어짐 등급 기대 오차(mm).
ERROR_MM_SKEWED: float = 10.0
#: 저해상도 등급 기대 오차(mm).
ERROR_MM_LOWRES: float = 15.0
#: 심한 기울기 등급 기대 오차(mm).
ERROR_MM_SEVERE: float = 20.0

#: 등급별 신뢰도 계수. 기대 오차가 클수록 최종 신뢰도를 깎는다.
_GRADE_QUALITY: dict[float, float] = {
    ERROR_MM_NORMAL: 1.00,
    ERROR_MM_SKEWED: 0.90,
    ERROR_MM_LOWRES: 0.70,
    ERROR_MM_SEVERE: 0.50,
}

#: 검출 방법별 신뢰도 계수(윤곽선 > 적응형 이진화 > 최소 외접 사각형).
_METHOD_FACTOR: dict[str, float] = {
    "canny_contour": 1.00,
    "adaptive_threshold": 0.95,
    "min_area_rect": 0.85,
}

#: 문서 검출에 실패해 이미지 전체를 문서로 간주할 때의 신뢰도(기본).
FALLBACK_CONFIDENCE: float = 0.30
#: 대체 경로이지만 이미지 종횡비가 A4 에 가까울 때의 신뢰도.
FALLBACK_CONFIDENCE_A4: float = 0.55
#: 대체 경로이지만 A4 비율 + 테두리까지 문서로 가득 찬 것으로 보일 때의 신뢰도.
FALLBACK_CONFIDENCE_FULL_PAGE: float = 0.70


# --------------------------------------------------------------------------
# 결과 타입
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class QuadDetection:
    """문서 사각형 검출 결과와 그 근거.

    :param quad: ``(4, 2)`` float64 배열. 좌상 → 우상 → 우하 → 좌하 순.
    :param confidence: 검출 신뢰도(0.0~1.0).
    :param method: 검출 방법 식별자
        (``"canny_contour"`` / ``"adaptive_threshold"`` / ``"min_area_rect"`` /
        ``"image_bounds"``).
    :param fallback: 문서를 찾지 못해 이미지 전체 경계를 쓴 경우 True.
    :param area_ratio: 검출 사각형이 이미지에서 차지하는 면적비.
    :param aspect: 검출 사각형의 세로/가로 비율.
    :param warnings: 한국어 경고 메시지 목록.
    """

    quad: np.ndarray
    confidence: float
    method: str
    fallback: bool = False
    area_ratio: float = 0.0
    aspect: float = 0.0
    warnings: tuple[str, ...] = ()

    def as_tuple(self) -> tuple[np.ndarray, float]:
        """``(quad, confidence)`` 2-튜플로 축약한다.

        :returns: :func:`detect_document_quad` 가 돌려주는 형태.
        """
        return (self.quad, self.confidence)


@dataclass(frozen=True, eq=False)
class NormalizedDocument:
    """정립이 끝난 문서 이미지와 그 좌표계.

    :param image: A4 비율로 정립된 이미지. 입력이 컬러면 ``(H, W, 3)`` uint8 BGR,
        그레이스케일이면 ``(H, W)`` uint8.
    :param coords: 이 이미지 전용 px ↔ mm 환산기.
    :param quad: **원본 이미지에서** 검출한 문서 사각형 ``(4, 2)``(TL, TR, BR, BL).
    :param skew_deg: 원본 이미지에서의 기울기(도). 양수는 시계 방향.
    :param dpi_estimate: 원본 이미지의 추정 해상도(dpi). 등급 판정의 기준이다.
    :param confidence: 정규화 결과 종합 신뢰도(0.0~1.0).
    :param expected_error_mm: 이 이미지에서 뽑은 좌표의 기대 오차 상한(mm).
    :param warnings: 한국어 경고 메시지 목록.
    :param homography: 원본 → 정립 이미지 호모그래피 ``(3, 3)``.
    :param source_size_px: 원본 이미지 크기 ``(width_px, height_px)``.
    :param detection_confidence: 사각형 검출 단계만의 신뢰도.
    :param method: 사각형 검출 방법 식별자.
    """

    image: np.ndarray
    coords: A4CoordinateSystem
    quad: np.ndarray
    skew_deg: float
    dpi_estimate: float
    confidence: float
    expected_error_mm: float
    warnings: tuple[str, ...] = ()
    homography: np.ndarray = dc_field(default_factory=lambda: np.eye(3))
    source_size_px: tuple[int, int] = (0, 0)
    detection_confidence: float = 0.0
    method: str = ""

    @property
    def low_confidence(self) -> bool:
        """신뢰도가 :data:`~docagent.contracts.VISION_TRUST_THRESHOLD` 미만이면 True.

        이 플래그가 서더라도 이 모듈은 예외를 던지지 않는다. 사용자에게
        재확인을 요구할지, 직원에게 연결할지는 Agent 계층이 결정한다.
        """
        return self.confidence < VISION_TRUST_THRESHOLD

    @property
    def size_px(self) -> tuple[int, int]:
        """정립 이미지 크기 ``(width_px, height_px)``."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))

    def to_mm(self, box_px: BoxPx, *, clamp: bool = False) -> BoxMm:
        """정립 이미지의 픽셀 사각형을 mm 사각형으로 바꾼다.

        :param box_px: 정립 이미지 기준 픽셀 사각형.
        :param clamp: True 면 페이지 범위로 잘라 낸다.
        :returns: :class:`~docagent.contracts.BoxMm`.
        """
        return self.coords.to_mm(box_px, clamp=clamp)

    def source_to_mm(self, x_px: float, y_px: float, *, clamp: bool = False) -> Point:
        """**원본 이미지**의 픽셀 좌표를 문서 mm 좌표로 바꾼다.

        정립 이미지가 아니라 촬영 원본 위에서 탐지한 결과를 도메인 좌표로
        옮길 때 쓴다. 내부적으로 호모그래피를 적용해 정립 이미지 좌표로 옮긴 뒤
        mm 로 환산한다.

        :param x_px: 원본 이미지 픽셀 x.
        :param y_px: 원본 이미지 픽셀 y.
        :param clamp: True 면 결과를 페이지 범위로 잘라 낸다.
        :returns: :class:`~docagent.contracts.Point`.
        :raises ValueError: 투영 결과가 발산하는 경우.
        """
        projected = apply_homography(self.homography, [[float(x_px), float(y_px)]])
        return self.coords.point_to_mm(
            float(projected[0, 0]), float(projected[0, 1]), clamp=clamp
        )

    def describe(self) -> dict[str, Any]:
        """이미지를 뺀 요약 정보를 dict 로 반환한다(로그용).

        :returns: 스칼라 값만 담은 dict. 개인정보는 포함되지 않는다.
        """
        return {
            "size_px": list(self.size_px),
            "source_size_px": list(self.source_size_px),
            "skew_deg": round(self.skew_deg, 3),
            "dpi_estimate": round(self.dpi_estimate, 2),
            "confidence": round(self.confidence, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "expected_error_mm": self.expected_error_mm,
            "low_confidence": self.low_confidence,
            "method": self.method,
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------------
# 입력 검증 · 전처리
# --------------------------------------------------------------------------


def _ensure_image(image: Any) -> np.ndarray:
    """입력이 다룰 수 있는 이미지인지 검사한다.

    :param image: 검사 대상. ``(H, W)`` 또는 ``(H, W, 3|4)`` ndarray.
    :returns: uint8 ndarray.
    :raises docagent.errors.DocumentNotFoundError: 입력이 ``None`` 이거나 비어 있는 경우.
    :raises docagent.errors.VisionError: 형태·자료형이 지원 범위를 벗어난 경우.
    """
    if image is None:
        raise DocumentNotFoundError("입력 이미지가 없습니다(None).")
    array = np.asarray(image)
    if array.size == 0:
        raise DocumentNotFoundError("입력 이미지가 비어 있습니다(픽셀 0개).")
    if array.ndim not in (2, 3):
        raise VisionError(
            f"이미지는 (H, W) 또는 (H, W, C) 형태여야 합니다: 입력 형태={array.shape}"
        )
    if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
        raise VisionError(
            f"채널 수는 1, 3, 4 중 하나여야 합니다: 입력 형태={array.shape}"
        )
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise VisionError(
            f"이미지가 너무 작아 문서를 찾을 수 없습니다: 입력 형태={array.shape}"
        )
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number):
            raise VisionError(f"이미지 자료형이 숫자가 아닙니다: {array.dtype}")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """이미지를 8비트 그레이스케일로 바꾼다.

    :param image: ``(H, W)`` 또는 ``(H, W, 3|4)`` uint8 배열(BGR 가정).
    :returns: ``(H, W)`` uint8 배열.
    :raises docagent.errors.VisionError: OpenCV 색공간 변환에 실패한 경우.
    """
    if image.ndim == 2:
        return image
    try:
        if image.shape[2] == 1:
            return image[:, :, 0]
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error as exc:  # 조용한 실패 금지
        raise VisionError(f"그레이스케일 변환에 실패했습니다: {exc}") from exc


def _odd(value: int, *, minimum: int = 3) -> int:
    """홀수로 보정한 커널 크기를 반환한다.

    :param value: 원하는 크기.
    :param minimum: 하한(홀수여야 한다).
    :returns: ``minimum`` 이상인 홀수.
    """
    size = max(minimum, int(value))
    return size if size % 2 == 1 else size + 1


# --------------------------------------------------------------------------
# 사각형 후보 생성
# --------------------------------------------------------------------------


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """윤곽선을 4각형으로 근사한다.

    여러 epsilon 을 순서대로 시도하고, 볼록한 4각형이 나오면 그것을 쓴다.

    :param contour: OpenCV 윤곽선 ``(N, 1, 2)``.
    :returns: ``(4, 2)`` float64 배열. 4각형을 얻지 못하면 ``None``.
    """
    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 0.0:
        return None
    for ratio in (0.02, 0.03, 0.04, 0.05, 0.015, 0.01):
        approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float64)
    return None


def _contours_desc(binary: np.ndarray, *, limit: int = 8) -> list[np.ndarray]:
    """이진 이미지에서 면적 내림차순 외곽 윤곽선을 뽑는다.

    :param binary: 0/255 이진 이미지.
    :param limit: 반환할 최대 개수.
    :returns: 윤곽선 목록(면적 내림차순).
    :raises docagent.errors.VisionError: 윤곽선 추출에 실패한 경우.
    """
    try:
        found = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    except cv2.error as exc:  # 조용한 실패 금지
        raise VisionError(f"윤곽선 추출에 실패했습니다: {exc}") from exc
    contours = found[0] if len(found) == 2 else found[1]
    return sorted(contours, key=cv2.contourArea, reverse=True)[:limit]


def _binary_canny(gray: np.ndarray) -> np.ndarray:
    """Canny 경계 + 팽창으로 문서 테두리를 강조한 이진 이미지를 만든다.

    중앙값 기반 자동 임계값을 써서 조명 밝기에 덜 민감하게 한다.

    :param gray: ``(H, W)`` uint8 그레이스케일.
    :returns: 0/255 이진 이미지.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(blurred))
    low = int(max(10.0, 0.66 * median))
    high = int(min(255.0, max(low + 30.0, 1.33 * median)))
    edges = cv2.Canny(blurred, low, high)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=2)


def _binary_adaptive(gray: np.ndarray) -> np.ndarray:
    """적응형 이진화로 조명 불균일에 강한 문서 영역 마스크를 만든다.

    블록 크기를 이미지의 1/4 수준으로 크게 잡으면, 종이 안쪽은 지역 평균보다
    밝아 흰색으로 남고 종이 경계 바깥에는 어두운 띠가 생겨 문서 영역이
    독립된 덩어리로 분리된다. 조명 기울기가 있어도 지역 평균이 함께 움직이므로
    전역 임계값(Otsu)보다 안정적이다.

    :param gray: ``(H, W)`` uint8 그레이스케일.
    :returns: 0/255 이진 이미지(문서 안쪽이 흰색).
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    block = _odd(min(gray.shape[:2]) // 4, minimum=51)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block,
        -5,
    )
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)


def _score_quad(quad: np.ndarray, shape: Sequence[int]) -> tuple[float, float, float] | None:
    """문서 사각형 후보를 검증하고 점수를 매긴다.

    :param quad: ``(4, 2)`` 좌표 배열.
    :param shape: 이미지 ``shape``.
    :returns: ``(점수, 면적비, 종횡비)``. 후보로 부적합하면 ``None``.
    """
    try:
        ordered = order_quad(quad)
    except ValueError:
        return None
    top, right, bottom, left = quad_edge_lengths(ordered)
    if min(top, right, bottom, left) < MIN_EDGE_PX:
        return None
    width = (top + bottom) / 2.0
    height = (left + right) / 2.0
    if width <= 0.0 or height <= 0.0:
        return None
    aspect = height / width
    if abs(aspect - A4_ASPECT_RATIO) > ASPECT_TOLERANCE:
        return None
    area_ratio = quad_area_ratio(ordered, shape)
    if not MIN_QUAD_AREA_RATIO <= area_ratio <= MAX_QUAD_AREA_RATIO:
        return None
    aspect_score = 1.0 - abs(aspect - A4_ASPECT_RATIO) / ASPECT_TOLERANCE
    area_score = min(1.0, (area_ratio - MIN_QUAD_AREA_RATIO) / 0.35)
    score = 0.55 + 0.25 * aspect_score + 0.20 * area_score
    return (float(min(1.0, score)), float(area_ratio), float(aspect))


def _candidates(gray: np.ndarray) -> list[tuple[float, np.ndarray, str, float, float]]:
    """세 가지 경로로 문서 사각형 후보를 모은다.

    경로는 (1) Canny 윤곽선, (2) 적응형 이진화, (3) 최소 외접 사각형 순이며
    방법별 신뢰도 계수를 곱해 하나의 순위로 합친다.

    :param gray: ``(H, W)`` uint8 그레이스케일.
    :returns: ``(신뢰도, quad, method, area_ratio, aspect)`` 목록(신뢰도 내림차순).
    """
    results: list[tuple[float, np.ndarray, str, float, float]] = []
    binaries = (
        ("canny_contour", _binary_canny(gray)),
        ("adaptive_threshold", _binary_adaptive(gray)),
    )
    for method, binary in binaries:
        contours = _contours_desc(binary)
        for contour in contours:
            approx = _approx_quad(contour)
            if approx is not None:
                scored = _score_quad(approx, gray.shape)
                if scored is not None:
                    score, area_ratio, aspect = scored
                    results.append(
                        (
                            score * _METHOD_FACTOR[method],
                            order_quad(approx),
                            method,
                            area_ratio,
                            aspect,
                        )
                    )
            # 근사 4각형이 안 나오면 최소 외접 사각형으로 한 번 더 시도한다.
            box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64)
            scored = _score_quad(box, gray.shape)
            if scored is not None:
                score, area_ratio, aspect = scored
                results.append(
                    (
                        score * _METHOD_FACTOR["min_area_rect"],
                        order_quad(box),
                        "min_area_rect",
                        area_ratio,
                        aspect,
                    )
                )
    results.sort(key=lambda item: item[0], reverse=True)
    return results


def _image_bounds_quad(shape: Sequence[int]) -> np.ndarray:
    """이미지 전체 경계를 사각형으로 만든다(대체 경로).

    :param shape: 이미지 ``shape``.
    :returns: ``(4, 2)`` float64 배열(TL, TR, BR, BL).
    """
    height, width = int(shape[0]), int(shape[1])
    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float64,
    )


def _fallback_confidence(gray: np.ndarray) -> tuple[float, str]:
    """대체 경로(이미지 전체)에 줄 신뢰도와 근거를 정한다.

    이미지 자체가 A4 비율이고 테두리까지 밝으면 "이미 문서만 잘라낸 스캔"일
    가능성이 높으므로 신뢰도를 조금 올려 준다. 다만 문서 경계를 실제로
    확인하지는 못했으므로 **어떤 경우에도 신뢰 임계값을 넘지 않는다.**

    :param gray: ``(H, W)`` uint8 그레이스케일.
    :returns: ``(신뢰도, 한국어 근거 문장)``.
    """
    height, width = gray.shape[:2]
    aspect = height / float(width) if width else 0.0
    if abs(aspect - A4_ASPECT_RATIO) > ASPECT_TOLERANCE:
        return (
            FALLBACK_CONFIDENCE,
            f"이미지 종횡비({aspect:.3f})가 A4 비율과 달라 신뢰도를 크게 낮췄습니다.",
        )
    band = max(2, min(height, width) // 100)
    border = np.concatenate(
        [
            gray[:band, :].reshape(-1),
            gray[-band:, :].reshape(-1),
            gray[:, :band].reshape(-1),
            gray[:, -band:].reshape(-1),
        ]
    )
    if float(np.mean(border)) >= 200.0:
        return (
            FALLBACK_CONFIDENCE_FULL_PAGE,
            "이미지 종횡비가 A4 에 가깝고 테두리도 밝아 문서가 화면을 가득 채운 "
            "스캔으로 보고 처리했습니다(문서 경계는 확인하지 못했습니다).",
        )
    return (
        FALLBACK_CONFIDENCE_A4,
        "이미지 종횡비가 A4 에 가까워 이미지 전체를 문서로 간주했습니다"
        "(문서 경계는 확인하지 못했습니다).",
    )


# --------------------------------------------------------------------------
# 공개 API — 사각형 검출
# --------------------------------------------------------------------------


def detect_document_quad_detailed(image: Any) -> QuadDetection:
    """문서 사각형을 검출하고 근거까지 함께 돌려준다.

    :param image: 촬영 원본 이미지. ``(H, W)`` 또는 ``(H, W, 3|4)`` uint8.
    :returns: :class:`QuadDetection`. 문서를 찾지 못하면 이미지 전체 경계를
        ``fallback=True`` 로 돌려주며 신뢰도를 크게 낮추고 사유를
        :attr:`QuadDetection.warnings` 에 남긴다.
    :raises docagent.errors.DocumentNotFoundError: 이미지가 비었거나 ``None`` 인 경우.
    :raises docagent.errors.VisionError: 이미지 형태가 지원 범위를 벗어나거나
        OpenCV 전처리가 실패한 경우.
    """
    array = _ensure_image(image)
    gray = _to_gray(array)
    candidates = _candidates(gray)
    if candidates:
        confidence, quad, method, area_ratio, aspect = candidates[0]
        return QuadDetection(
            quad=quad,
            confidence=float(min(1.0, max(0.0, confidence))),
            method=method,
            fallback=False,
            area_ratio=area_ratio,
            aspect=aspect,
            warnings=(),
        )

    confidence, reason = _fallback_confidence(gray)
    quad = _image_bounds_quad(gray.shape)
    height, width = gray.shape[:2]
    return QuadDetection(
        quad=quad,
        confidence=confidence,
        method="image_bounds",
        fallback=True,
        area_ratio=1.0,
        aspect=height / float(width),
        warnings=(
            "문서 사각형을 찾지 못해 이미지 전체를 문서로 간주했습니다. " + reason,
        ),
    )


def detect_document_quad(image: Any) -> tuple[np.ndarray, float]:
    """문서 사각형과 검출 신뢰도를 반환한다.

    :param image: 촬영 원본 이미지.
    :returns: ``(quad, confidence)``. ``quad`` 는 ``(4, 2)`` float64 배열로
        좌상 → 우상 → 우하 → 좌하 순이며, ``confidence`` 는 0.0~1.0 이다.
        검출 실패 시에는 이미지 전체 경계와 크게 낮춘 신뢰도를 돌려준다
        (사유는 :func:`detect_document_quad_detailed` 로 확인한다).
    :raises docagent.errors.DocumentNotFoundError: 이미지가 비었거나 ``None`` 인 경우.
    :raises docagent.errors.VisionError: 이미지 형태가 지원 범위를 벗어난 경우.
    """
    return detect_document_quad_detailed(image).as_tuple()


# --------------------------------------------------------------------------
# 공개 API — 등급 판정
# --------------------------------------------------------------------------


def expected_error_mm_for(skew_deg: float, dpi_estimate: float) -> float:
    """기울기와 해상도로 기대 오차 상한(mm)을 정한다.

    해상도는 **정수로 반올림한 뒤** 비교한다. 검출 사각형에서 역산한 dpi 는
    꼭짓점 위치에 따라 ±1dpi 정도 흔들리므로, 200dpi 로 찍은 문서가 199.94 로
    추정되었다는 이유만으로 저해상도 등급이 되는 경계 흔들림을 막는다.

    :param skew_deg: 기울기(도). 부호는 무시하고 절댓값만 본다.
    :param dpi_estimate: 원본 이미지의 추정 해상도(dpi).
    :returns: :data:`ERROR_MM_NORMAL` / :data:`ERROR_MM_SKEWED` /
        :data:`ERROR_MM_LOWRES` / :data:`ERROR_MM_SEVERE` 중 하나.
        조건이 겹치면 더 큰 값을 돌려준다.
    """
    skew = abs(float(skew_deg))
    dpi = float(round(float(dpi_estimate)))
    if skew < SKEW_NORMAL_DEG and dpi >= DPI_NORMAL:
        grade = ERROR_MM_NORMAL
    elif skew < SKEW_MAX_DEG:
        grade = ERROR_MM_SKEWED
    else:
        grade = ERROR_MM_SEVERE
    if dpi < DPI_LOW:
        grade = max(grade, ERROR_MM_LOWRES)
    return float(grade)


def _estimate_source_dpi(quad: np.ndarray) -> float:
    """검출 사각형의 픽셀 크기로 원본 해상도를 추정한다.

    :param quad: ``(4, 2)`` 사각형(TL, TR, BR, BL).
    :returns: 추정 dpi. 가로·세로 추정값의 평균이다.
    """
    top, right, bottom, left = quad_edge_lengths(quad)
    width_px = (top + bottom) / 2.0
    height_px = (left + right) / 2.0
    dpi_from_width = width_px / A4_WIDTH_MM * MM_PER_INCH
    dpi_from_height = height_px / A4_HEIGHT_MM * MM_PER_INCH
    return float((dpi_from_width + dpi_from_height) / 2.0)


# --------------------------------------------------------------------------
# 공개 API — 정규화
# --------------------------------------------------------------------------


def normalize_document(
    image: Any, dpi: int = DEFAULT_NORMALIZE_DPI
) -> NormalizedDocument:
    """촬영 이미지를 정립된 A4 페이지로 펴고 좌표계를 붙여 돌려준다.

    :param image: 촬영 원본 이미지. ``(H, W)`` 또는 ``(H, W, 3|4)`` uint8.
    :param dpi: 정립 이미지의 해상도(72 이상 1200 이하 권장).
    :returns: :class:`NormalizedDocument`.
        :attr:`~NormalizedDocument.confidence` 가
        :data:`~docagent.contracts.VISION_TRUST_THRESHOLD` 미만이어도
        **예외를 던지지 않고** :attr:`~NormalizedDocument.low_confidence`
        플래그와 경고만 남긴다.
    :raises ValueError: ``dpi`` 가 1 미만인 경우.
    :raises docagent.errors.DocumentNotFoundError: 이미지가 비었거나 ``None`` 인 경우.
    :raises docagent.errors.VisionError: 전처리·투영 변환이 실패한 경우.
    """
    if dpi < 1:
        raise ValueError(f"dpi 는 1 이상이어야 합니다: {dpi}")

    array = _ensure_image(image)
    detection = detect_document_quad_detailed(array)
    quad = detection.quad
    messages: list[str] = list(detection.warnings)

    skew_deg = estimate_skew_deg(quad)
    dpi_estimate = _estimate_source_dpi(quad)
    coords = A4CoordinateSystem.from_dpi(dpi)
    width_px, height_px = coords.size_px

    destination = np.array(
        [
            [0.0, 0.0],
            [width_px - 1.0, 0.0],
            [width_px - 1.0, height_px - 1.0],
            [0.0, height_px - 1.0],
        ],
        dtype=np.float64,
    )
    homography = solve_homography(quad, destination)
    try:
        warped = cv2.warpPerspective(
            array,
            homography,
            (width_px, height_px),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    except cv2.error as exc:  # 조용한 실패 금지
        raise VisionError(f"문서 정립(투영 변환)에 실패했습니다: {exc}") from exc

    expected_error = expected_error_mm_for(skew_deg, dpi_estimate)
    quality = _GRADE_QUALITY[expected_error]
    confidence = float(min(1.0, max(0.0, detection.confidence * quality)))

    if abs(skew_deg) >= SKEW_NORMAL_DEG:
        messages.append(
            f"문서가 {abs(skew_deg):.1f}도 기울어져 있어 좌표 오차가 커질 수 있습니다."
        )
    if round(dpi_estimate) < DPI_LOW:
        messages.append(
            f"원본 해상도가 낮습니다(추정 {dpi_estimate:.0f}dpi). "
            "가능하면 더 가까이서 다시 촬영하십시오."
        )
    if detection.aspect < 1.0:
        messages.append(
            "검출된 문서가 가로로 누워 있는 것으로 보입니다. 세로 방향으로 다시 촬영하십시오."
        )
    if confidence < VISION_TRUST_THRESHOLD:
        messages.append(
            f"정규화 신뢰도가 기준({VISION_TRUST_THRESHOLD:.2f})에 미달합니다"
            f"(현재 {confidence:.2f}). 결과를 그대로 신뢰하지 말고 사용자에게 재확인하십시오."
        )

    return NormalizedDocument(
        image=np.ascontiguousarray(warped),
        coords=coords,
        quad=quad,
        skew_deg=float(skew_deg),
        dpi_estimate=float(dpi_estimate),
        confidence=confidence,
        expected_error_mm=expected_error,
        warnings=tuple(messages),
        homography=homography,
        source_size_px=(int(array.shape[1]), int(array.shape[0])),
        detection_confidence=float(detection.confidence),
        method=detection.method,
    )


def normalize_boxes(
    boxes_px: Sequence[BoxPx], coords: A4CoordinateSystem, *, clamp: bool = True
) -> list[BoxMm]:
    """정립 이미지의 픽셀 사각형 목록을 mm 사각형 목록으로 바꾼다.

    Vision 내부에서 픽셀로 탐지한 결과를 **모듈 경계 밖으로 내보내기 직전에**
    호출하는 헬퍼다.

    :param boxes_px: 정립 이미지 기준 픽셀 사각형 목록.
    :param coords: 그 이미지의 좌표계.
    :param clamp: True 면 페이지 범위를 벗어난 사각형을 잘라 낸다.
    :returns: :class:`~docagent.contracts.BoxMm` 목록(입력 순서 유지).
    :raises TypeError: 목록에 :class:`~docagent.contracts.BoxPx` 가 아닌 값이 있는 경우.
    """
    result: list[BoxMm] = []
    for index, box in enumerate(boxes_px):
        if not isinstance(box, BoxPx):
            raise TypeError(
                f"boxes_px[{index}] 는 BoxPx 여야 합니다: {type(box).__name__}"
            )
        result.append(coords.to_mm(box, clamp=clamp))
    return result
