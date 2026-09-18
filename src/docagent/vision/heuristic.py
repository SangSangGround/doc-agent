"""규칙 기반 기입란 탐지기 — 학습 데이터 없이도 동작하는 기본 Detector.

이 모듈은 로드맵 Phase 3(Vision PoC)의 **기본값**이다. 학습된 가중치가 없어도
데모와 CI 가 100% 동작해야 하므로, OpenCV 의 고전적 영상처리(적응형 이진화 ·
윤곽선 · 형태학 연산)만으로 체크박스와 서명란을 찾는다.

산출 범위
---------
이 탐지기는 **좌표와 유형(SIGNATURE / CHECKBOX)만** 산출한다.
역할(신청인/대리인) 부여와 선택지 라벨 매칭은 구조화(structuring) 단계의
책임이며 여기서 다루지 않는다.

입력 가정
---------
입력 이미지는 이미 **A4 로 정립(deskew·crop)된 정규화 이미지**라고 가정한다
(:func:`docagent.vision.normalize.normalize_document` 의 산출물).
따라서 픽셀 ↔ 밀리미터 환산은 이미지 크기만으로 결정되며,
:func:`coordinate_system_from_image` 가 :class:`~docagent.vision.geometry.A4CoordinateSystem`
을 만들어 준다. 종횡비가 A4 와 크게 다르면 경고 로그를 남기고 계속 진행한다
(정합은 상위 파이프라인의 책임이다).

출력은 계약대로 **A4 mm 좌표의** :class:`~docagent.contracts.Detection` 이며,
:class:`~docagent.contracts.BoxPx` 는 이 모듈 밖으로 나가지 않는다.

신뢰도 산출
-----------
:attr:`Detection.confidence` 는 임의 상수가 아니라 **조건 만족 정도의 가중
평균**이다. 각 조건은 "허용 경계에서 0, 이상적 구간에서 1" 이 되는 선형·사다리꼴
함수(:func:`_ramp` · :func:`_band`)로 0~1 점수화되며, 항목별 가중치는
:class:`HeuristicParams` 에 노출되어 있다. 구체적 산식은
:meth:`HeuristicDetector._checkbox_confidence` 와
:meth:`HeuristicDetector._underline_confidence` 독스트링의 표에 적었다.

알려진 한계
-----------
밑줄 위에 빈 공간이 있는 **기입선**은 서명란과 기하학적으로 구분되지 않는다
(예: "신청일자 ______"). 이 탐지기는 유형과 좌표만 내놓고 의미 판별은 구조화
단계가 하므로, 그런 항목은 SIGNATURE 후보로 올라올 수 있다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    Detection,
    FieldType,
)
from docagent.errors import VisionError
from docagent.vision.geometry import MM_PER_INCH, A4CoordinateSystem

__all__ = [
    "SOURCE_CHECKBOX",
    "SOURCE_UNDERLINE",
    "DEFAULT_ASPECT_TOLERANCE",
    "coordinate_system_from_image",
    "HeuristicParams",
    "HeuristicDetector",
    "iou_mm",
]

_LOG = logging.getLogger(__name__)

#: 체크박스 탐지 결과의 :attr:`Detection.source` 값.
SOURCE_CHECKBOX: str = "opencv_contour"
#: 서명란(밑줄) 탐지 결과의 :attr:`Detection.source` 값.
SOURCE_UNDERLINE: str = "opencv_morphology"

#: :func:`coordinate_system_from_image` 이 경고를 내기 시작하는 A4 종횡비 상대 오차.
DEFAULT_ASPECT_TOLERANCE: float = 0.08


# --------------------------------------------------------------------------
# 좌표계
# --------------------------------------------------------------------------


def coordinate_system_from_image(
    image: np.ndarray, *, aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE
) -> A4CoordinateSystem:
    """정립 이미지 배열로부터 A4 좌표계를 만든다.

    이미지 전체가 A4 한 장에 대응한다고 보고, 가로 픽셀 수에서 명목 dpi 를 되돌린다.
    좌표계 자체는 :mod:`docagent.vision.geometry` 의 공용 구현을 그대로 쓴다
    (중복 정의를 만들지 않는다).

    :param image: ``(H, W)`` 또는 ``(H, W, C)`` 형태의 배열.
    :param aspect_tolerance: A4 종횡비(297/210)와의 허용 상대 오차.
        초과하면 **경고 로그만 남기고** 그대로 진행한다(정합은 상위 책임).
    :returns: :class:`~docagent.vision.geometry.A4CoordinateSystem`.
    :raises docagent.errors.VisionError: 배열이 아니거나 형태·크기가 유효하지 않은 경우.
    """
    if not isinstance(image, np.ndarray):
        raise VisionError(
            f"입력 이미지는 numpy 배열이어야 합니다: {type(image).__name__}"
        )
    if image.ndim not in (2, 3):
        raise VisionError(
            f"입력 이미지는 (H, W) 또는 (H, W, C) 형태여야 합니다: shape={image.shape}"
        )
    height, width = int(image.shape[0]), int(image.shape[1])
    if height < 1 or width < 1:
        raise VisionError(f"입력 이미지 크기가 0 입니다: shape={image.shape}")

    expected = A4_HEIGHT_MM / A4_WIDTH_MM
    actual = height / width
    if abs(actual - expected) / expected > aspect_tolerance:
        _LOG.warning(
            "입력 이미지 종횡비(%.3f)가 A4 비율(%.3f)과 다릅니다. "
            "정립(deskew·crop)되지 않은 이미지일 수 있어 mm 좌표가 부정확할 수 있습니다.",
            actual,
            expected,
        )
    dpi = max(1.0, width / A4_WIDTH_MM * MM_PER_INCH)
    return A4CoordinateSystem(dpi=dpi, width_px=width, height_px=height)


# --------------------------------------------------------------------------
# 기하 헬퍼
# --------------------------------------------------------------------------


def iou_mm(a: BoxMm, b: BoxMm) -> float:
    """두 mm 사각형의 IoU(Intersection over Union)를 계산한다.

    :param a: 첫 번째 사각형.
    :param b: 두 번째 사각형.
    :returns: 0.0~1.0 IoU. 겹치지 않거나 합집합 넓이가 0 이면 0.0.
    """
    left = max(a.x_mm, b.x_mm)
    top = max(a.y_mm, b.y_mm)
    right = min(a.right_mm, b.right_mm)
    bottom = min(a.bottom_mm, b.bottom_mm)
    if right <= left or bottom <= top:
        return 0.0
    inter = (right - left) * (bottom - top)
    union = a.area_mm2 + b.area_mm2 - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """``zero_at`` 에서 0, ``one_at`` 에서 1 이 되는 선형 점수(0~1 클램프).

    :param value: 점수화할 측정값.
    :param zero_at: 점수 0 이 되는 경계값.
    :param one_at: 점수 1 이 되는 이상값.
    :returns: 0.0~1.0 점수.
    :raises ValueError: ``zero_at`` 과 ``one_at`` 이 같은 경우(0 나눗셈).
    """
    if zero_at == one_at:
        raise ValueError(f"_ramp 의 zero_at 과 one_at 이 같습니다: {zero_at}")
    score = (value - zero_at) / (one_at - zero_at)
    return float(min(1.0, max(0.0, score)))


def _band(
    value: float, low_zero: float, low_one: float, high_one: float, high_zero: float
) -> float:
    """사다리꼴 점수 — ``[low_one, high_one]`` 구간에서 1, 바깥 경계에서 0.

    :param value: 점수화할 측정값.
    :param low_zero: 아래쪽 0 경계.
    :param low_one: 아래쪽 1 경계.
    :param high_one: 위쪽 1 경계.
    :param high_zero: 위쪽 0 경계.
    :returns: 0.0~1.0 점수.
    """
    if value < low_one:
        return _ramp(value, low_zero, low_one)
    if value > high_one:
        return _ramp(value, high_zero, high_one)
    return 1.0


def _weighted(scores: Sequence[tuple[float, float]]) -> float:
    """``(점수, 가중치)`` 쌍의 가중 평균을 0~1 로 반환한다.

    :param scores: ``(score, weight)`` 시퀀스.
    :returns: 0.0~1.0 가중 평균.
    :raises ValueError: 가중치 합이 0 이하인 경우.
    """
    total = sum(weight for _, weight in scores)
    if total <= 0.0:
        raise ValueError("가중치 합이 0 이하입니다. 신뢰도를 계산할 수 없습니다.")
    value = sum(score * weight for score, weight in scores) / total
    return float(min(1.0, max(0.0, value)))


# --------------------------------------------------------------------------
# 파라미터
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HeuristicParams:
    """규칙 기반 탐지기의 임계값 모음(전부 mm 단위로 표현).

    픽셀이 아니라 mm 로 두었기 때문에 해상도(dpi)가 달라져도 동작이 같다.

    체크박스
        :param checkbox_min_side_mm: 허용 최소 변 길이. 이보다 작으면 글자 획으로 본다.
        :param checkbox_max_side_mm: 허용 최대 변 길이. 이보다 크면 표 칸으로 본다.
        :param checkbox_ideal_side_mm: 신뢰도 1.0 이 되는 변 길이 구간 ``(하한, 상한)``.
        :param checkbox_max_aspect: 허용 최대 종횡비(긴 변/짧은 변). 1.3334 = 4:3.
        :param checkbox_min_rectangularity: 윤곽선 넓이 / 외접 사각형 넓이 최소값.
        :param checkbox_max_fill_ratio: 내부 잉크 비율 허용 상한(빈 칸은 테두리만 있다).
        :param checkbox_ideal_fill_ratio: 신뢰도 1.0 이 되는 내부 잉크 비율 상한.
        :param checkbox_inset_ratio: 내부 잉크 비율을 잴 때 변 길이 대비 안쪽 여유 비율.
        :param checkbox_nms_iou: 중첩 제거 IoU 임계값.
        :param approx_epsilon_ratio: ``approxPolyDP`` 의 둘레 대비 epsilon 비율.

    서명란(밑줄)
        :param underline_min_length_mm: 허용 최소 밑줄 길이.
        :param underline_ideal_length_mm: 신뢰도 1.0 이 되는 밑줄 길이.
        :param underline_max_length_mm: 허용 최대 밑줄 길이. 이보다 길면 표 테두리로 본다.
        :param underline_max_thickness_mm: 허용 최대 두께.
        :param underline_ideal_thickness_mm: 신뢰도 1.0 이 되는 두께 상한.
        :param underline_min_solidity: 외접 사각형 대비 실제 선 픽셀 비율 최소값.
        :param signature_height_mm: 밑줄 위로 확보하는 서명 공간의 높이.
        :param clearance_max_ink_ratio: 서명 공간의 잉크 비율 허용 상한.
            인쇄된 글자가 줄 바로 위에 오는 표 테두리·제목 밑줄을 걸러 내되,
            이미 손글씨가 들어간 서명란은 통과시킬 수 있는 값으로 잡았다.
        :param clearance_ideal_ink_ratio: 신뢰도 1.0 이 되는 서명 공간 잉크 비율 상한.
        :param horizontal_kernel_mm: 수평선 추출 형태학 커널 길이.
        :param vertical_kernel_mm: 수직선 추출 형태학 커널 길이(표 억제용).
        :param vertical_guard_mm: 밑줄 상하로 수직선을 검사할 여유 폭.

    공통
        :param adaptive_block_mm: 적응형 이진화 블록 크기(mm).
        :param adaptive_c: 적응형 이진화 상수 C.
        :param min_confidence: 이 값 미만의 후보는 버린다.

    :raises ValueError: 범위가 뒤집혔거나 허용 구간을 벗어난 경우(한국어 메시지).
    """

    checkbox_min_side_mm: float = 2.5
    checkbox_max_side_mm: float = 8.0
    checkbox_ideal_side_mm: tuple[float, float] = (3.5, 7.0)
    checkbox_max_aspect: float = 4.0 / 3.0
    checkbox_min_rectangularity: float = 0.70
    checkbox_max_fill_ratio: float = 0.55
    checkbox_ideal_fill_ratio: float = 0.10
    checkbox_inset_ratio: float = 0.22
    checkbox_nms_iou: float = 0.30
    approx_epsilon_ratio: float = 0.04

    underline_min_length_mm: float = 20.0
    underline_ideal_length_mm: float = 45.0
    underline_max_length_mm: float = 140.0
    underline_max_thickness_mm: float = 1.2
    underline_ideal_thickness_mm: float = 0.8
    underline_min_solidity: float = 0.60
    signature_height_mm: float = 10.0
    clearance_max_ink_ratio: float = 0.120
    clearance_ideal_ink_ratio: float = 0.010
    horizontal_kernel_mm: float = 15.0
    vertical_kernel_mm: float = 15.0
    vertical_guard_mm: float = 2.5

    adaptive_block_mm: float = 2.5
    adaptive_c: float = 7.0
    min_confidence: float = 0.25

    def __post_init__(self) -> None:
        if (
            self.checkbox_min_side_mm <= 0
            or self.checkbox_max_side_mm <= self.checkbox_min_side_mm
        ):
            raise ValueError(
                "체크박스 변 길이 범위가 올바르지 않습니다: "
                f"{self.checkbox_min_side_mm}~{self.checkbox_max_side_mm}mm"
            )
        if self.checkbox_max_aspect < 1.0:
            raise ValueError(
                f"checkbox_max_aspect 는 1.0 이상이어야 합니다: {self.checkbox_max_aspect}"
            )
        if self.underline_max_length_mm <= self.underline_min_length_mm:
            raise ValueError(
                "밑줄 길이 범위가 올바르지 않습니다: "
                f"{self.underline_min_length_mm}~{self.underline_max_length_mm}mm"
            )
        if self.underline_max_thickness_mm <= 0:
            raise ValueError(
                "underline_max_thickness_mm 는 0보다 커야 합니다: "
                f"{self.underline_max_thickness_mm}"
            )
        if self.signature_height_mm <= 0:
            raise ValueError(
                f"signature_height_mm 는 0보다 커야 합니다: {self.signature_height_mm}"
            )
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence 는 0.0~1.0 이어야 합니다: {self.min_confidence}"
            )
        if not 0.0 <= self.checkbox_nms_iou <= 1.0:
            raise ValueError(
                f"checkbox_nms_iou 는 0.0~1.0 이어야 합니다: {self.checkbox_nms_iou}"
            )


# --------------------------------------------------------------------------
# 탐지기
# --------------------------------------------------------------------------


class HeuristicDetector:
    """OpenCV 규칙 기반 체크박스·서명란 탐지기.

    :class:`docagent.interfaces.Detector` 프로토콜을 만족한다.
    학습 가중치가 필요 없으므로 데모와 CI 의 기본 탐지기로 쓴다.

    :param params: 임계값 모음. ``None`` 이면 기본값을 쓴다.
    :param detect_checkboxes: 체크박스 탐지 수행 여부.
    :param detect_signatures: 서명란 탐지 수행 여부.

    사용 예::

        detector = HeuristicDetector()
        for det in detector.detect(image_bgr):
            print(det.type.value, det.box_mm.to_tuple(), round(det.confidence, 3))
    """

    def __init__(
        self,
        params: HeuristicParams | None = None,
        *,
        detect_checkboxes: bool = True,
        detect_signatures: bool = True,
    ) -> None:
        self.params = params if params is not None else HeuristicParams()
        self.detect_checkboxes = detect_checkboxes
        self.detect_signatures = detect_signatures

    # -- 공개 API --------------------------------------------------------

    def detect(self, image: Any) -> list[Detection]:
        """이미지에서 체크박스·서명란 후보를 탐지한다.

        :param image: A4 로 정립된 문서 이미지.
            ``(H, W, 3)`` uint8 BGR 또는 ``(H, W)`` uint8 그레이스케일.
        :returns: :class:`~docagent.contracts.Detection` 목록.
            좌표는 A4 mm 기준이며, ``(type, y_mm, x_mm)`` 순으로 정렬된다.
            후보가 없으면 빈 리스트를 돌려준다(예외를 던지지 않는다).
        :raises docagent.errors.VisionError: 이미지가 유효하지 않거나 전처리에 실패한 경우.
        """
        gray = self._to_gray(image)
        coords = coordinate_system_from_image(gray)
        ink = self._ink_mask(gray, coords)

        detections: list[Detection] = []
        if self.detect_checkboxes:
            detections.extend(self._detect_checkboxes(ink, coords))
        if self.detect_signatures:
            detections.extend(self._detect_signature_lines(ink, coords))

        kept = [item for item in detections if item.confidence >= self.params.min_confidence]
        kept.sort(
            key=lambda d: (d.type.value, round(d.box_mm.y_mm, 3), round(d.box_mm.x_mm, 3))
        )
        _LOG.debug(
            "규칙 기반 탐지 완료: 후보 %d건 중 %d건 채택(임계값 %.2f).",
            len(detections),
            len(kept),
            self.params.min_confidence,
        )
        return kept

    # -- 전처리 ----------------------------------------------------------

    @staticmethod
    def _to_gray(image: Any) -> np.ndarray:
        """입력을 uint8 그레이스케일 배열로 정규화한다.

        :param image: ``(H, W)`` 또는 ``(H, W, 3/4)`` 배열.
        :returns: ``(H, W)`` uint8 배열.
        :raises docagent.errors.VisionError: 형태·자료형이 지원되지 않는 경우.
        """
        if not isinstance(image, np.ndarray):
            raise VisionError(
                f"입력 이미지는 numpy 배열이어야 합니다: {type(image).__name__}"
            )
        if image.size == 0:
            raise VisionError("입력 이미지가 비어 있습니다.")
        try:
            if image.ndim == 2:
                gray = image
            elif image.ndim == 3 and image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 3 and image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                raise VisionError(
                    "지원하지 않는 이미지 형태입니다. (H, W) 또는 (H, W, 3/4) 이어야 합니다: "
                    f"shape={image.shape}"
                )
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(gray)
        except cv2.error as exc:  # pragma: no cover - OpenCV 내부 오류 방어
            raise VisionError(f"이미지 그레이스케일 변환에 실패했습니다: {exc}") from exc

    def _ink_mask(self, gray: np.ndarray, coords: A4CoordinateSystem) -> np.ndarray:
        """적응형 이진화로 잉크(글자·선) 마스크를 만든다.

        조명 불균일에 강하도록 :func:`cv2.adaptiveThreshold` 를 쓰고,
        저해상도·JPEG 잡음으로 생기는 점 잡음은 3x3 중간값 필터로 제거한다.

        :param gray: ``(H, W)`` uint8 그레이스케일 이미지.
        :param coords: 좌표계(블록 크기를 mm 로 정하기 위해 필요).
        :returns: 잉크가 255, 배경이 0 인 ``(H, W)`` uint8 마스크.
        :raises docagent.errors.VisionError: 이진화에 실패한 경우.
        """
        scale = (coords.px_per_mm_x + coords.px_per_mm_y) / 2.0
        block = int(round(self.params.adaptive_block_mm * scale))
        block = max(3, block | 1)  # 3 이상의 홀수로 강제
        try:
            mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                float(self.params.adaptive_c),
            )
            return cv2.medianBlur(mask, 3)
        except cv2.error as exc:
            raise VisionError(f"적응형 이진화에 실패했습니다: {exc}") from exc

    # -- 체크박스 --------------------------------------------------------

    def _detect_checkboxes(
        self, ink: np.ndarray, coords: A4CoordinateSystem
    ) -> list[Detection]:
        """윤곽선 기반으로 체크박스를 찾는다.

        절차: ``findContours`` → ``approxPolyDP`` 로 볼록 사각형 선별 →
        변 길이(mm) · 종횡비 · 사각형 근사도 · 내부 잉크 비율 검사 → 중첩 제거.

        빈 체크박스는 **테두리만 잉크**이므로 내부 잉크 비율이 0 에 가깝다.
        이 조건이 글자 획이 만든 가짜 사각형을 걸러 내는 핵심이다.

        :param ink: 잉크 마스크.
        :param coords: 좌표계.
        :returns: :attr:`FieldType.CHECKBOX` :class:`Detection` 목록.
        :raises docagent.errors.VisionError: 윤곽선 추출에 실패한 경우.
        """
        params = self.params
        try:
            contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        except cv2.error as exc:  # pragma: no cover - OpenCV 내부 오류 방어
            raise VisionError(f"윤곽선 추출에 실패했습니다: {exc}") from exc

        candidates: list[tuple[Detection, float]] = []
        for contour in contours:
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            approx = cv2.approxPolyDP(
                contour, params.approx_epsilon_ratio * perimeter, True
            )
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            x_px, y_px, w_px, h_px = (int(value) for value in cv2.boundingRect(approx))
            if w_px < 2 or h_px < 2:
                continue
            width_mm = w_px / coords.px_per_mm_x
            height_mm = h_px / coords.px_per_mm_y
            if not (
                params.checkbox_min_side_mm <= width_mm <= params.checkbox_max_side_mm
                and params.checkbox_min_side_mm <= height_mm <= params.checkbox_max_side_mm
            ):
                continue

            aspect = max(width_mm / height_mm, height_mm / width_mm)
            if aspect > params.checkbox_max_aspect:
                continue

            rectangularity = float(cv2.contourArea(approx)) / float(w_px * h_px)
            if rectangularity < params.checkbox_min_rectangularity:
                continue

            fill_ratio = self._interior_fill_ratio(ink, x_px, y_px, w_px, h_px)
            if fill_ratio > params.checkbox_max_fill_ratio:
                continue

            confidence = self._checkbox_confidence(
                width_mm=width_mm,
                height_mm=height_mm,
                aspect=aspect,
                rectangularity=rectangularity,
                fill_ratio=fill_ratio,
            )
            box_mm = coords.to_mm(BoxPx(x_px, y_px, w_px, h_px), clamp=True)
            candidates.append(
                (
                    Detection(
                        type=FieldType.CHECKBOX,
                        box_mm=box_mm,
                        confidence=confidence,
                        source=SOURCE_CHECKBOX,
                    ),
                    box_mm.area_mm2,
                )
            )
        return self._suppress_overlaps(candidates, params.checkbox_nms_iou)

    def _interior_fill_ratio(
        self, ink: np.ndarray, x_px: int, y_px: int, w_px: int, h_px: int
    ) -> float:
        """사각형 내부(테두리를 뺀 영역)의 잉크 비율을 잰다.

        :param ink: 잉크 마스크.
        :param x_px: 사각형 좌상단 x(px).
        :param y_px: 사각형 좌상단 y(px).
        :param w_px: 사각형 폭(px).
        :param h_px: 사각형 높이(px).
        :returns: 0.0~1.0 잉크 비율. 내부 영역을 잡을 수 없으면 1.0(= 실격).
        """
        inset_x = max(1, int(round(w_px * self.params.checkbox_inset_ratio)))
        inset_y = max(1, int(round(h_px * self.params.checkbox_inset_ratio)))
        x0, y0 = x_px + inset_x, y_px + inset_y
        x1, y1 = x_px + w_px - inset_x, y_px + h_px - inset_y
        if x1 <= x0 or y1 <= y0:
            return 1.0
        region = ink[y0:y1, x0:x1]
        if region.size == 0:
            return 1.0
        return float(np.count_nonzero(region)) / float(region.size)

    def _checkbox_confidence(
        self,
        *,
        width_mm: float,
        height_mm: float,
        aspect: float,
        rectangularity: float,
        fill_ratio: float,
    ) -> float:
        """체크박스 후보의 신뢰도를 조건 만족 정도로 산출한다.

        네 가지 조건 점수(각 0~1)의 가중 평균이다.

        =============== ==== ==================================================
        조건            가중 점수 정의
        =============== ==== ==================================================
        크기 적정성      0.25 평균 변 길이가 이상 구간이면 1, 허용 경계에서 0
        정사각형성       0.25 종횡비가 1 이면 1, 허용 상한(4:3)에서 0
        사각형 근사도    0.25 윤곽 넓이/외접 넓이가 1 이면 1, 하한에서 0
        내부 공동성      0.25 내부 잉크 비율이 이상 상한 이하면 1, 허용 상한에서 0
        =============== ==== ==================================================

        :param width_mm: 후보 폭(mm).
        :param height_mm: 후보 높이(mm).
        :param aspect: 종횡비(긴 변/짧은 변, 1 이상).
        :param rectangularity: 윤곽 넓이 / 외접 사각형 넓이.
        :param fill_ratio: 내부 잉크 비율.
        :returns: 0.0~1.0 신뢰도.
        """
        params = self.params
        mean_side = (width_mm + height_mm) / 2.0
        ideal_low, ideal_high = params.checkbox_ideal_side_mm
        score_size = _band(
            mean_side,
            params.checkbox_min_side_mm,
            ideal_low,
            ideal_high,
            params.checkbox_max_side_mm,
        )
        score_aspect = _ramp(aspect, params.checkbox_max_aspect, 1.0)
        score_rect = _ramp(rectangularity, params.checkbox_min_rectangularity, 1.0)
        score_hollow = _ramp(
            fill_ratio, params.checkbox_max_fill_ratio, params.checkbox_ideal_fill_ratio
        )
        return _weighted(
            (
                (score_size, 0.25),
                (score_aspect, 0.25),
                (score_rect, 0.25),
                (score_hollow, 0.25),
            )
        )

    # -- 서명란 ----------------------------------------------------------

    def _detect_signature_lines(
        self, ink: np.ndarray, coords: A4CoordinateSystem
    ) -> list[Detection]:
        """수평 형태학 연산으로 서명란 밑줄을 찾고 서명 영역으로 확장한다.

        절차: ``MORPH_OPEN``(긴 가로 커널)으로 밑줄 후보 추출 →
        길이·두께·선 충실도 검사 → **표 격자 억제**(밑줄 상하 여유 범위에
        수직선이 걸치면 표 테두리로 보고 버린다) → 밑줄 위 서명 공간이
        비어 있는지 검사 → 위 공간을 포함한 :class:`BoxMm` 생성.

        :param ink: 잉크 마스크.
        :param coords: 좌표계.
        :returns: :attr:`FieldType.SIGNATURE` :class:`Detection` 목록.
        :raises docagent.errors.VisionError: 형태학 연산·연결요소 추출에 실패한 경우.
        """
        params = self.params
        try:
            h_len = max(3, int(round(params.horizontal_kernel_mm * coords.px_per_mm_x)))
            v_len = max(3, int(round(params.vertical_kernel_mm * coords.px_per_mm_y)))
            horizontal = cv2.morphologyEx(
                ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
            )
            vertical = cv2.morphologyEx(
                ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
            )
            count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                horizontal, connectivity=8
            )
        except cv2.error as exc:  # pragma: no cover - OpenCV 내부 오류 방어
            raise VisionError(f"수평선 추출에 실패했습니다: {exc}") from exc

        guard_px = max(1, int(round(params.vertical_guard_mm * coords.px_per_mm_y)))
        clearance_h_px = max(
            1, int(round(params.signature_height_mm * coords.px_per_mm_y))
        )
        height_px, width_px = ink.shape[:2]
        candidates: list[tuple[Detection, float]] = []

        for label in range(1, count):
            x_px = int(stats[label, cv2.CC_STAT_LEFT])
            y_px = int(stats[label, cv2.CC_STAT_TOP])
            w_px = int(stats[label, cv2.CC_STAT_WIDTH])
            h_px = int(stats[label, cv2.CC_STAT_HEIGHT])
            area_px = int(stats[label, cv2.CC_STAT_AREA])
            if w_px < 2 or h_px < 1:
                continue

            length_mm = w_px / coords.px_per_mm_x
            thickness_mm = h_px / coords.px_per_mm_y
            if not (
                params.underline_min_length_mm
                <= length_mm
                <= params.underline_max_length_mm
            ):
                continue
            if thickness_mm > params.underline_max_thickness_mm:
                continue

            solidity = float(area_px) / float(w_px * h_px)
            if solidity < params.underline_min_solidity:
                continue

            guard_top = max(0, y_px - guard_px)
            guard_bottom = min(height_px, y_px + h_px + guard_px)
            if np.count_nonzero(vertical[guard_top:guard_bottom, x_px : x_px + w_px]) > 0:
                _LOG.debug(
                    "수직선이 교차하여 표 테두리로 판단, 밑줄 후보 제외: y=%.1fmm, 길이=%.1fmm",
                    y_px / coords.px_per_mm_y,
                    length_mm,
                )
                continue

            clear_top = max(0, y_px - clearance_h_px)
            if clear_top >= y_px:
                continue
            clearance = ink[clear_top:y_px, x_px : min(width_px, x_px + w_px)]
            clearance_ratio = (
                float(np.count_nonzero(clearance)) / float(clearance.size)
                if clearance.size
                else 1.0
            )
            if clearance_ratio > params.clearance_max_ink_ratio:
                continue

            confidence = self._underline_confidence(
                length_mm=length_mm,
                thickness_mm=thickness_mm,
                solidity=solidity,
                clearance_ratio=clearance_ratio,
            )
            box_mm = coords.clamp_box(
                BoxMm(
                    x_mm=x_px / coords.px_per_mm_x,
                    y_mm=y_px / coords.px_per_mm_y - params.signature_height_mm,
                    w_mm=length_mm,
                    h_mm=params.signature_height_mm,
                ),
                warn=False,
            )
            candidates.append(
                (
                    Detection(
                        type=FieldType.SIGNATURE,
                        box_mm=box_mm,
                        confidence=confidence,
                        source=SOURCE_UNDERLINE,
                    ),
                    box_mm.area_mm2,
                )
            )
        return self._suppress_overlaps(candidates, params.checkbox_nms_iou)

    def _underline_confidence(
        self,
        *,
        length_mm: float,
        thickness_mm: float,
        solidity: float,
        clearance_ratio: float,
    ) -> float:
        """서명란 후보의 신뢰도를 조건 만족 정도로 산출한다.

        네 가지 조건 점수(각 0~1)의 가중 평균이다.

        =================== ==== ==============================================
        조건                가중 점수 정의
        =================== ==== ==============================================
        밑줄 길이            0.30 최소 길이에서 0, 이상 길이 이상이면 1
        밑줄 두께            0.25 이상 두께 이하면 1, 허용 상한에서 0
        서명 공간 공백도     0.30 잉크 비율이 이상값 이하면 1, 허용 상한에서 0
        선 충실도            0.15 외접 사각형 대비 선 픽셀 비율(하한에서 0, 1 에서 1)
        =================== ==== ==============================================

        :param length_mm: 밑줄 길이(mm).
        :param thickness_mm: 밑줄 두께(mm).
        :param solidity: 외접 사각형 대비 선 픽셀 비율.
        :param clearance_ratio: 밑줄 위 서명 공간의 잉크 비율.
        :returns: 0.0~1.0 신뢰도.
        """
        params = self.params
        score_length = _ramp(
            length_mm, params.underline_min_length_mm, params.underline_ideal_length_mm
        )
        score_thickness = _ramp(
            thickness_mm,
            params.underline_max_thickness_mm,
            params.underline_ideal_thickness_mm,
        )
        score_clear = _ramp(
            clearance_ratio,
            params.clearance_max_ink_ratio,
            params.clearance_ideal_ink_ratio,
        )
        score_solidity = _ramp(solidity, params.underline_min_solidity, 1.0)
        return _weighted(
            (
                (score_length, 0.30),
                (score_thickness, 0.25),
                (score_clear, 0.30),
                (score_solidity, 0.15),
            )
        )

    # -- 공통 후처리 -----------------------------------------------------

    @staticmethod
    def _suppress_overlaps(
        candidates: Iterable[tuple[Detection, float]], iou_threshold: float
    ) -> list[Detection]:
        """중첩 후보를 제거한다(신뢰도 우선, 동률이면 큰 사각형 우선).

        한 체크박스의 바깥·안쪽 윤곽선이 각각 잡히는 경우를 하나로 합치는 단계다.
        정렬 키에 좌표를 포함하여 **결정론적**으로 동작한다.

        :param candidates: ``(Detection, 넓이_mm2)`` 쌍의 반복자.
        :param iou_threshold: 이 값 이상 겹치면 낮은 쪽을 버린다.
        :returns: 살아남은 :class:`Detection` 목록.
        """
        ordered = sorted(
            candidates,
            key=lambda item: (
                -round(item[0].confidence, 6),
                -round(item[1], 6),
                round(item[0].box_mm.y_mm, 3),
                round(item[0].box_mm.x_mm, 3),
            ),
        )
        kept: list[Detection] = []
        for detection, _area in ordered:
            if any(
                iou_mm(detection.box_mm, other.box_mm) >= iou_threshold for other in kept
            ):
                continue
            kept.append(detection)
        return kept
