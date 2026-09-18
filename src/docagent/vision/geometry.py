"""Vision 기하 변환 — 픽셀 ↔ A4 밀리미터 ↔ 기기(액추에이터) 좌표.

이 모듈은 로드맵의 **Vision-to-Physical Coordinate Transformation** 을 담당한다.
카메라가 본 픽셀을 문서 도메인 좌표(A4 mm)로 옮기고, 다시 펜 액추에이터의
기기 좌표로 옮기는 세 단계 변환을 모두 여기서 정의한다.

좌표계 규약(계약 문서 §1 과 동일)
--------------------------------
* 도메인 좌표는 **A4 밀리미터**. 원점은 좌상단, x 는 오른쪽(+), y 는 아래쪽(+).
* 픽셀 좌표(:class:`~docagent.contracts.BoxPx`)는 Vision 내부 전용이며
  모듈 경계를 넘지 못한다. 밖으로 나갈 때는 반드시 :class:`~docagent.contracts.BoxMm`.
* 픽셀 ↔ mm 환산은 **픽셀 격자 규약**을 쓴다.
  ``px = mm * (width_px / 210.0)`` 이므로 ``mm = 210.0`` 은 ``px = width_px``
  (마지막 픽셀 인덱스보다 1 큰 값)에 대응한다. 합성 문서 생성기
  (:mod:`docagent.testing.synthetic`)의 ``mm_to_px`` 와 같은 규약이다.

각도 부호 규약
--------------
:func:`estimate_skew_deg` 는 이미지 좌표계(y 아래쪽 +) 기준으로
**양수 = 시계 방향으로 기울어짐**(문서의 오른쪽이 아래로 내려감)을 뜻한다.

이 모듈은 numpy 와 OpenCV 만 사용한다. 선택적 패키지는 쓰지 않는다.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.contracts import A4_HEIGHT_MM, A4_WIDTH_MM, BoxMm, BoxPx, Point
from docagent.errors import VisionError

__all__ = [
    "MM_PER_INCH",
    "A4_ASPECT_RATIO",
    "PAGE_TOLERANCE_MM",
    "CoordinateOutOfRangeWarning",
    "MachineRangeError",
    "order_quad",
    "solve_homography",
    "apply_homography",
    "quad_edge_lengths",
    "quad_area",
    "quad_area_ratio",
    "estimate_skew_deg",
    "A4CoordinateSystem",
    "MachineCalibration",
]


#: 1 인치의 밀리미터 길이.
MM_PER_INCH: float = 25.4

#: A4 세로/가로 비율(≈1.4142). 문서 후보 검증의 기준값.
A4_ASPECT_RATIO: float = A4_HEIGHT_MM / A4_WIDTH_MM

#: 페이지 범위 검사에 쓰는 허용 오차(mm). 부동소수 반올림을 흡수한다.
PAGE_TOLERANCE_MM: float = 1e-6


class CoordinateOutOfRangeWarning(UserWarning):
    """좌표가 A4 페이지 범위를 벗어났을 때 발생하는 경고.

    변환 자체는 계속 수행된다(값을 잘라내려면 ``clamp=True`` 를 쓴다).
    페이지 밖 좌표는 대개 문서 검출 실패의 신호이므로 조용히 넘기지 않는다.
    """


class MachineRangeError(ValueError, VisionError):
    """기기 가동범위(0 ≤ x ≤ 210mm, 0 ≤ y ≤ 297mm) 밖 좌표를 요청한 경우.

    :class:`ValueError` 이면서 동시에 :class:`~docagent.errors.VisionError`
    이므로, 호출자는 둘 중 어느 쪽으로 잡아도 된다.
    """


# --------------------------------------------------------------------------
# 내부 헬퍼
# --------------------------------------------------------------------------


def _as_quad(pts: Any, *, name: str) -> np.ndarray:
    """임의 입력을 ``(4, 2)`` float64 배열로 검증·변환한다.

    :param pts: 4개 점. ``(4, 2)`` 또는 OpenCV 윤곽선 형태 ``(4, 1, 2)``.
    :param name: 오류 메시지에 쓸 인자 이름.
    :returns: ``(4, 2)`` float64 배열(순서는 바꾸지 않는다).
    :raises ValueError: 형태가 맞지 않거나 유한하지 않은 값이 있는 경우.
    """
    array = np.asarray(pts, dtype=np.float64)
    if array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == 2:
        array = array.reshape(-1, 2)
    if array.shape != (4, 2):
        raise ValueError(
            f"{name} 은(는) (4, 2) 형태의 좌표 배열이어야 합니다: 입력 형태={array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 에 유한하지 않은 좌표(NaN/Inf)가 있습니다.")
    return array


def _polygon_area(quad: np.ndarray) -> float:
    """신발끈 공식으로 사각형 넓이를 구한다(부호 없음).

    :param quad: ``(4, 2)`` 좌표 배열.
    :returns: 넓이(입력 좌표 단위의 제곱).
    """
    x = quad[:, 0]
    y = quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# --------------------------------------------------------------------------
# 사각형 유틸리티
# --------------------------------------------------------------------------


def order_quad(pts: Any) -> np.ndarray:
    """4개 점을 **좌상 → 우상 → 우하 → 좌하** 순으로 정렬한다.

    좌표 합(``x + y``)이 최소인 점이 좌상단, 최대인 점이 우하단이고,
    차(``y - x``)가 최소인 점이 우상단, 최대인 점이 좌하단이다.
    문서 회전이 ±45도를 넘으면 이 판별이 무너지므로 예외를 던진다.

    :param pts: 4개 점. ``(4, 2)`` 또는 ``(4, 1, 2)`` 배열.
    :returns: ``(4, 2)`` float64 배열. 순서는 TL, TR, BR, BL.
    :raises ValueError: 형태가 맞지 않거나, 네 꼭짓점이 서로 구분되지 않는 경우
        (예: 회전이 45도에 가까워 판별이 모호한 경우).
    """
    array = _as_quad(pts, name="order_quad(pts)")
    total = array[:, 0] + array[:, 1]
    diff = array[:, 1] - array[:, 0]
    indices = (
        int(np.argmin(total)),
        int(np.argmin(diff)),
        int(np.argmax(total)),
        int(np.argmax(diff)),
    )
    if len(set(indices)) != 4:
        raise ValueError(
            "네 꼭짓점을 좌상/우상/우하/좌하로 구분할 수 없습니다. "
            "회전이 45도에 가깝거나 사각형이 퇴화했습니다: "
            f"{array.tolist()}"
        )
    return np.array([array[i] for i in indices], dtype=np.float64)


def quad_edge_lengths(quad: Any) -> tuple[float, float, float, float]:
    """정렬된 사각형의 네 변 길이를 반환한다.

    :param quad: ``(4, 2)`` 좌표 배열(정렬되지 않아도 내부에서 정렬한다).
    :returns: ``(위, 오른쪽, 아래, 왼쪽)`` 변 길이.
    """
    ordered = order_quad(quad)
    top = float(np.linalg.norm(ordered[1] - ordered[0]))
    right = float(np.linalg.norm(ordered[2] - ordered[1]))
    bottom = float(np.linalg.norm(ordered[2] - ordered[3]))
    left = float(np.linalg.norm(ordered[3] - ordered[0]))
    return (top, right, bottom, left)


def quad_area(quad: Any) -> float:
    """사각형 넓이(px^2)를 반환한다.

    :param quad: ``(4, 2)`` 좌표 배열.
    :returns: 넓이.
    """
    return _polygon_area(_as_quad(quad, name="quad_area(quad)"))


def quad_area_ratio(quad: Any, image_shape: Sequence[int]) -> float:
    """사각형이 이미지 전체 면적에서 차지하는 비율을 반환한다.

    :param quad: ``(4, 2)`` 좌표 배열.
    :param image_shape: ``(H, W)`` 또는 ``(H, W, C)``. ndarray 의 ``shape`` 그대로.
    :returns: 0.0~ 비율. 이미지 전체를 덮으면 1.0 에 가깝다.
    :raises ValueError: 이미지 크기가 유효하지 않은 경우.
    """
    shape = tuple(int(v) for v in image_shape[:2])
    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"이미지 크기가 유효하지 않습니다: {tuple(image_shape)}")
    return quad_area(quad) / float(shape[0] * shape[1])


def estimate_skew_deg(quad: Any) -> float:
    """사각형의 기울기(도)를 추정한다.

    위/아래 변의 기울기 평균이며, 이미지 좌표계(y 아래쪽 +) 기준으로
    **양수는 시계 방향**(문서 오른쪽이 아래로 내려간 상태)을 뜻한다.

    :param quad: ``(4, 2)`` 좌표 배열.
    :returns: -90.0 ~ 90.0 범위의 기울기(도).
    :raises ValueError: 사각형이 퇴화하여 변 방향을 정의할 수 없는 경우.
    """
    ordered = order_quad(quad)
    angles: list[float] = []
    for start, end in ((0, 1), (3, 2)):  # 위 변(TL→TR), 아래 변(BL→BR)
        vector = ordered[end] - ordered[start]
        if float(np.linalg.norm(vector)) < 1e-9:
            raise ValueError("사각형의 가로 변 길이가 0 이어서 기울기를 구할 수 없습니다.")
        angles.append(math.degrees(math.atan2(float(vector[1]), float(vector[0]))))
    return float(sum(angles) / len(angles))


# --------------------------------------------------------------------------
# 호모그래피
# --------------------------------------------------------------------------


def solve_homography(src_quad: Any, dst_quad: Any) -> np.ndarray:
    """네 점 대응으로 3x3 투영 변환 행렬을 구한다.

    :param src_quad: 원본 사각형 ``(4, 2)``. 순서는 ``dst_quad`` 와 대응해야 한다.
    :param dst_quad: 목표 사각형 ``(4, 2)``.
    :returns: ``(3, 3)`` float64 호모그래피 행렬 ``H``.
        ``dst ≈ apply_homography(H, src)`` 가 성립한다.
    :raises ValueError: 입력 형태가 잘못되었거나 사각형이 퇴화(넓이 0)한 경우.
    :raises docagent.errors.VisionError: OpenCV 가 행렬 계산에 실패한 경우.
    """
    src = _as_quad(src_quad, name="solve_homography(src_quad)")
    dst = _as_quad(dst_quad, name="solve_homography(dst_quad)")
    for name, quad in (("src_quad", src), ("dst_quad", dst)):
        if _polygon_area(quad) <= 1e-9:
            raise ValueError(
                f"solve_homography({name}) 의 넓이가 0 입니다(퇴화한 사각형): {quad.tolist()}"
            )
    try:
        matrix = cv2.getPerspectiveTransform(
            src.astype(np.float32), dst.astype(np.float32)
        )
    except cv2.error as exc:  # 조용한 실패 금지 — 도메인 예외로 감싼다.
        raise VisionError(f"호모그래피 계산에 실패했습니다: {exc}") from exc
    return np.asarray(matrix, dtype=np.float64)


def apply_homography(H: Any, points: Any) -> np.ndarray:
    """호모그래피로 점들을 투영한다.

    :param H: ``(3, 3)`` 변환 행렬.
    :param points: ``(N, 2)`` 점 배열. 길이 2 의 1차원 배열이면 점 1개로 본다.
    :returns: ``(N, 2)`` float64 투영 결과.
    :raises ValueError: 행렬·점 배열의 형태가 잘못되었거나,
        투영 결과의 동차 좌표 ``w`` 가 0 이어서 발산하는 경우.
    """
    matrix = np.asarray(H, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"H 는 (3, 3) 행렬이어야 합니다: 입력 형태={matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("H 에 유한하지 않은 값(NaN/Inf)이 있습니다.")

    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == 2:
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(
            f"points 는 (N, 2) 형태(N≥1)여야 합니다: 입력 형태={np.asarray(points).shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("points 에 유한하지 않은 좌표(NaN/Inf)가 있습니다.")

    homogeneous = np.hstack([array, np.ones((array.shape[0], 1), dtype=np.float64)])
    projected = homogeneous @ matrix.T
    w = projected[:, 2]
    if np.any(np.abs(w) < 1e-12):
        raise ValueError(
            "투영 결과의 동차 좌표가 0 이라 유한한 점으로 환산할 수 없습니다. "
            "호모그래피 또는 입력 점을 확인하십시오."
        )
    return projected[:, :2] / w[:, None]


# --------------------------------------------------------------------------
# A4 좌표계
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class A4CoordinateSystem:
    """정립된 A4 이미지의 픽셀 ↔ 밀리미터 양방향 환산기.

    :func:`~docagent.vision.normalize.normalize_document` 가 만들어 낸
    **정립 이미지 전용**이다. 정립 이미지는 가로 ``width_px`` 가 정확히
    210mm, 세로 ``height_px`` 가 정확히 297mm 에 대응한다고 본다.

    :param dpi: 정립 이미지의 명목 해상도(dots per inch, 1 이상).
    :param width_px: 정립 이미지 가로 픽셀 수(1 이상).
    :param height_px: 정립 이미지 세로 픽셀 수(1 이상).
    :raises ValueError: 인자가 1 미만인 경우.
    """

    dpi: float
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if self.dpi < 1:
            raise ValueError(f"A4CoordinateSystem.dpi 는 1 이상이어야 합니다: {self.dpi}")
        if self.width_px < 1 or self.height_px < 1:
            raise ValueError(
                "A4CoordinateSystem 의 이미지 크기는 1px 이상이어야 합니다: "
                f"width_px={self.width_px}, height_px={self.height_px}"
            )
        object.__setattr__(self, "dpi", float(self.dpi))
        object.__setattr__(self, "width_px", int(self.width_px))
        object.__setattr__(self, "height_px", int(self.height_px))

    # -- 생성 ---------------------------------------------------------------

    @classmethod
    def from_dpi(cls, dpi: float) -> "A4CoordinateSystem":
        """해상도만으로 A4 전면 좌표계를 만든다.

        :param dpi: 목표 해상도.
        :returns: ``width_px`` · ``height_px`` 가 A4 실제 크기에 맞춰진 좌표계.
        :raises ValueError: ``dpi`` 가 1 미만인 경우.
        """
        if dpi < 1:
            raise ValueError(f"dpi 는 1 이상이어야 합니다: {dpi}")
        scale = float(dpi) / MM_PER_INCH
        return cls(
            dpi=float(dpi),
            width_px=max(1, int(round(A4_WIDTH_MM * scale))),
            height_px=max(1, int(round(A4_HEIGHT_MM * scale))),
        )

    # -- 축척 ---------------------------------------------------------------

    @property
    def px_per_mm_x(self) -> float:
        """가로 방향 픽셀/mm 비율."""
        return self.width_px / A4_WIDTH_MM

    @property
    def px_per_mm_y(self) -> float:
        """세로 방향 픽셀/mm 비율."""
        return self.height_px / A4_HEIGHT_MM

    @property
    def size_px(self) -> tuple[int, int]:
        """``(width_px, height_px)``."""
        return (self.width_px, self.height_px)

    # -- 범위 ---------------------------------------------------------------

    def contains(self, point: Point) -> bool:
        """점이 A4 페이지 범위(경계 포함) 안이면 True.

        :param point: mm 좌표 점.
        :returns: 포함 여부.
        """
        return (
            -PAGE_TOLERANCE_MM <= point.x_mm <= A4_WIDTH_MM + PAGE_TOLERANCE_MM
            and -PAGE_TOLERANCE_MM <= point.y_mm <= A4_HEIGHT_MM + PAGE_TOLERANCE_MM
        )

    def clamp_point(self, point: Point, *, warn: bool = True) -> Point:
        """점을 A4 페이지 범위 안으로 잘라 낸다.

        :param point: mm 좌표 점.
        :param warn: 범위를 벗어났을 때 :class:`CoordinateOutOfRangeWarning` 을
            발생시킬지 여부.
        :returns: 페이지 범위 안으로 조정된 :class:`~docagent.contracts.Point`.
        """
        if self.contains(point):
            return point
        if warn:
            self._warn_out_of_range(point)
        return Point(
            x_mm=min(max(point.x_mm, 0.0), A4_WIDTH_MM),
            y_mm=min(max(point.y_mm, 0.0), A4_HEIGHT_MM),
        )

    def clamp_box(self, box_mm: BoxMm, *, warn: bool = True) -> BoxMm:
        """사각형을 A4 페이지 범위 안으로 잘라 낸다.

        :param box_mm: mm 사각형.
        :param warn: 범위를 벗어났을 때 경고를 발생시킬지 여부.
        :returns: 페이지 안으로 잘린 :class:`~docagent.contracts.BoxMm`.
            잘린 결과의 폭·높이는 0 이상이다.
        """
        left = min(max(box_mm.x_mm, 0.0), A4_WIDTH_MM)
        top = min(max(box_mm.y_mm, 0.0), A4_HEIGHT_MM)
        right = min(max(box_mm.right_mm, 0.0), A4_WIDTH_MM)
        bottom = min(max(box_mm.bottom_mm, 0.0), A4_HEIGHT_MM)
        clamped = BoxMm(left, top, max(0.0, right - left), max(0.0, bottom - top))
        # 부동소수 재계산 오차로 잘못된 경고가 나가지 않도록 입력 자체의 범위 이탈만 본다.
        out_of_range = (
            box_mm.x_mm < -PAGE_TOLERANCE_MM
            or box_mm.y_mm < -PAGE_TOLERANCE_MM
            or box_mm.right_mm > A4_WIDTH_MM + PAGE_TOLERANCE_MM
            or box_mm.bottom_mm > A4_HEIGHT_MM + PAGE_TOLERANCE_MM
        )
        if warn and out_of_range:
            warnings.warn(
                "사각형이 A4 페이지 범위를 벗어나 잘라냈습니다: "
                f"입력={box_mm.to_tuple()}, 결과={clamped.to_tuple()}",
                CoordinateOutOfRangeWarning,
                stacklevel=2,
            )
        return clamped

    def _warn_out_of_range(self, point: Point) -> None:
        """페이지 범위 이탈 경고를 발생시킨다.

        :param point: 문제가 된 mm 좌표.
        :returns: ``None``.
        """
        warnings.warn(
            "좌표가 A4 페이지 범위(0~210mm, 0~297mm)를 벗어났습니다: "
            f"({point.x_mm:.2f}, {point.y_mm:.2f})mm",
            CoordinateOutOfRangeWarning,
            stacklevel=3,
        )

    # -- 변환 ---------------------------------------------------------------

    def point_to_mm(
        self, x_px: float, y_px: float, *, clamp: bool = False
    ) -> Point:
        """픽셀 좌표를 mm 좌표로 바꾼다.

        :param x_px: 픽셀 x.
        :param y_px: 픽셀 y.
        :param clamp: True 면 결과를 페이지 범위로 잘라 낸다.
        :returns: :class:`~docagent.contracts.Point`.
        """
        point = Point(
            x_mm=float(x_px) / self.px_per_mm_x,
            y_mm=float(y_px) / self.px_per_mm_y,
        )
        if not self.contains(point):
            if clamp:
                return self.clamp_point(point)
            self._warn_out_of_range(point)
        return point

    def point_to_px(self, point: Point) -> tuple[int, int]:
        """mm 좌표를 픽셀 좌표로 바꾼다(반올림).

        :param point: mm 좌표 점.
        :returns: ``(x_px, y_px)`` 정수 튜플.
        """
        if not self.contains(point):
            self._warn_out_of_range(point)
        return (
            int(round(point.x_mm * self.px_per_mm_x)),
            int(round(point.y_mm * self.px_per_mm_y)),
        )

    def point_to_px_exact(self, point: Point) -> tuple[float, float]:
        """mm 좌표를 반올림 없이 실수 픽셀 좌표로 바꾼다.

        :param point: mm 좌표 점.
        :returns: ``(x_px, y_px)`` 실수 튜플.
        """
        return (point.x_mm * self.px_per_mm_x, point.y_mm * self.px_per_mm_y)

    def to_mm(self, box_px: BoxPx, *, clamp: bool = False) -> BoxMm:
        """픽셀 사각형을 mm 사각형으로 바꾼다.

        :param box_px: 픽셀 사각형(Vision 내부 좌표).
        :param clamp: True 면 결과를 페이지 범위로 잘라 낸다.
        :returns: :class:`~docagent.contracts.BoxMm`.
        """
        box = BoxMm(
            x_mm=box_px.x / self.px_per_mm_x,
            y_mm=box_px.y / self.px_per_mm_y,
            w_mm=box_px.w / self.px_per_mm_x,
            h_mm=box_px.h / self.px_per_mm_y,
        )
        if clamp:
            return self.clamp_box(box)
        if (
            box.x_mm < -PAGE_TOLERANCE_MM
            or box.y_mm < -PAGE_TOLERANCE_MM
            or box.right_mm > A4_WIDTH_MM + PAGE_TOLERANCE_MM
            or box.bottom_mm > A4_HEIGHT_MM + PAGE_TOLERANCE_MM
        ):
            warnings.warn(
                "사각형이 A4 페이지 범위를 벗어났습니다: " f"{box.to_tuple()}",
                CoordinateOutOfRangeWarning,
                stacklevel=2,
            )
        return box

    def to_px(self, box_mm: BoxMm) -> BoxPx:
        """mm 사각형을 픽셀 사각형으로 바꾼다(반올림).

        :param box_mm: mm 사각형.
        :returns: :class:`~docagent.contracts.BoxPx`.
        """
        left = int(round(box_mm.x_mm * self.px_per_mm_x))
        top = int(round(box_mm.y_mm * self.px_per_mm_y))
        right = int(round(box_mm.right_mm * self.px_per_mm_x))
        bottom = int(round(box_mm.bottom_mm * self.px_per_mm_y))
        return BoxPx(x=left, y=top, w=max(0, right - left), h=max(0, bottom - top))


# --------------------------------------------------------------------------
# 기기 좌표 캘리브레이션
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineCalibration:
    """문서 좌표(A4 mm) → 기기 좌표(mm) 아핀 변환.

    펜 액추에이터의 원점이 종이의 좌상단과 일치하지 않고, 축 방향과 축척도
    다를 수 있으므로 그 차이를 흡수한다.

    변환식(``flip_y`` 가 False 일 때)::

        machine_x = origin_offset_mm.x_mm + scale_x * doc_x
        machine_y = origin_offset_mm.y_mm + scale_y * doc_y

    ``flip_y`` 가 True 면 문서 y 를 ``A4_HEIGHT_MM - doc_y`` 로 뒤집은 뒤 적용한다
    (기기 y 축이 위쪽 + 인 장비용).

    :param origin_offset_mm: 문서 원점이 기기 좌표에서 갖는 위치.
    :param scale_x: x 축 축척(0 이 아니어야 한다).
    :param scale_y: y 축 축척(0 이 아니어야 한다).
    :param flip_y: 기기 y 축이 문서 y 축과 반대이면 True.
    :raises ValueError: 축척이 0 이거나 유한하지 않은 경우.
    """

    origin_offset_mm: Point = Point(0.0, 0.0)
    scale_x: float = 1.0
    scale_y: float = 1.0
    flip_y: bool = False

    def __post_init__(self) -> None:
        for name, value in (("scale_x", self.scale_x), ("scale_y", self.scale_y)):
            if not math.isfinite(value) or abs(value) < 1e-9:
                raise ValueError(
                    f"MachineCalibration.{name} 은(는) 0 이 아닌 유한한 값이어야 합니다: {value}"
                )

    # -- 범위 검사 ----------------------------------------------------------

    @staticmethod
    def _assert_in_work_area(point: Point, *, direction: str) -> None:
        """문서 좌표가 기기 가동범위 안인지 검사한다.

        :param point: 검사할 문서 좌표(mm).
        :param direction: 오류 메시지에 붙일 변환 방향 설명.
        :returns: ``None``.
        :raises MachineRangeError: 가동범위를 벗어난 경우.
        """
        if not (
            -PAGE_TOLERANCE_MM <= point.x_mm <= A4_WIDTH_MM + PAGE_TOLERANCE_MM
            and -PAGE_TOLERANCE_MM <= point.y_mm <= A4_HEIGHT_MM + PAGE_TOLERANCE_MM
        ):
            raise MachineRangeError(
                f"{direction} 좌표가 기기 가동범위를 벗어났습니다: "
                f"({point.x_mm:.2f}, {point.y_mm:.2f})mm. "
                f"허용 범위는 0~{A4_WIDTH_MM:.0f}mm × 0~{A4_HEIGHT_MM:.0f}mm 입니다."
            )

    # -- 변환 ---------------------------------------------------------------

    def to_machine(self, point: Point) -> Point:
        """문서 좌표를 기기 좌표로 바꾼다.

        :param point: 문서 좌표(A4 mm).
        :returns: 기기 좌표 :class:`~docagent.contracts.Point`.
        :raises MachineRangeError: 입력이 기기 가동범위를 벗어난 경우.
        """
        self._assert_in_work_area(point, direction="문서")
        y_source = A4_HEIGHT_MM - point.y_mm if self.flip_y else point.y_mm
        return Point(
            x_mm=self.origin_offset_mm.x_mm + self.scale_x * point.x_mm,
            y_mm=self.origin_offset_mm.y_mm + self.scale_y * y_source,
        )

    def from_machine(self, point: Point) -> Point:
        """기기 좌표를 문서 좌표로 되돌린다.

        :param point: 기기 좌표(mm).
        :returns: 문서 좌표 :class:`~docagent.contracts.Point`.
        :raises MachineRangeError: 되돌린 문서 좌표가 가동범위를 벗어난 경우.
        """
        x_mm = (point.x_mm - self.origin_offset_mm.x_mm) / self.scale_x
        y_source = (point.y_mm - self.origin_offset_mm.y_mm) / self.scale_y
        y_mm = A4_HEIGHT_MM - y_source if self.flip_y else y_source
        result = Point(x_mm=x_mm, y_mm=y_mm)
        self._assert_in_work_area(result, direction="기기 역변환")
        return result

    # -- 추정 ---------------------------------------------------------------

    @classmethod
    def estimate_from_pairs(
        cls,
        doc_points: Sequence[Point],
        machine_points: Sequence[Point],
    ) -> "MachineCalibration":
        """대응점 2~N 쌍에서 최소자승으로 캘리브레이션을 추정한다.

        x 축과 y 축을 독립적으로 1차 회귀한다. y 기울기가 음수이면
        기기 y 축이 뒤집힌 것으로 보고 ``flip_y=True`` 로 환산한다.

        :param doc_points: 문서 좌표(A4 mm) 점 목록. 2개 이상.
        :param machine_points: 같은 순서의 기기 좌표 점 목록.
        :returns: 추정된 :class:`MachineCalibration`.
        :raises ValueError: 점 개수가 2 미만이거나 두 목록의 길이가 다른 경우,
            또는 x 나 y 값이 모두 같아 축척을 결정할 수 없는 경우.
        """
        if len(doc_points) != len(machine_points):
            raise ValueError(
                "대응점 개수가 다릅니다: "
                f"문서={len(doc_points)}개, 기기={len(machine_points)}개"
            )
        if len(doc_points) < 2:
            raise ValueError(
                f"캘리브레이션에는 대응점이 2쌍 이상 필요합니다: {len(doc_points)}쌍"
            )

        doc = np.array([[p.x_mm, p.y_mm] for p in doc_points], dtype=np.float64)
        machine = np.array([[p.x_mm, p.y_mm] for p in machine_points], dtype=np.float64)
        if not np.all(np.isfinite(doc)) or not np.all(np.isfinite(machine)):
            raise ValueError("대응점에 유한하지 않은 좌표(NaN/Inf)가 있습니다.")

        slope_x, intercept_x = cls._fit_line(doc[:, 0], machine[:, 0], axis_name="x")
        slope_y, intercept_y = cls._fit_line(doc[:, 1], machine[:, 1], axis_name="y")

        if slope_y < 0.0:
            flip_y = True
            scale_y = -slope_y
            origin_y = intercept_y - A4_HEIGHT_MM * scale_y
        else:
            flip_y = False
            scale_y = slope_y
            origin_y = intercept_y

        return cls(
            origin_offset_mm=Point(x_mm=float(intercept_x), y_mm=float(origin_y)),
            scale_x=float(slope_x),
            scale_y=float(scale_y),
            flip_y=flip_y,
        )

    @staticmethod
    def _fit_line(
        source: np.ndarray, target: np.ndarray, *, axis_name: str
    ) -> tuple[float, float]:
        """``target ≈ slope * source + intercept`` 를 최소자승으로 푼다.

        :param source: 독립 변수 배열.
        :param target: 종속 변수 배열.
        :param axis_name: 오류 메시지에 쓸 축 이름.
        :returns: ``(slope, intercept)``.
        :raises ValueError: ``source`` 의 값이 모두 같아 기울기를 정할 수 없는 경우.
        """
        if float(np.ptp(source)) < 1e-9:
            raise ValueError(
                f"{axis_name} 축 대응점의 문서 좌표가 모두 같아 축척을 추정할 수 없습니다. "
                f"{axis_name} 값이 서로 다른 점을 포함시키십시오."
            )
        design = np.column_stack([source, np.ones_like(source)])
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
        slope = float(solution[0])
        if abs(slope) < 1e-9:
            raise ValueError(
                f"{axis_name} 축 축척이 0 으로 추정되었습니다. 대응점을 확인하십시오."
            )
        return (slope, float(solution[1]))
