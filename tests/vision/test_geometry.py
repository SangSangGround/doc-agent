"""``docagent.vision.geometry`` 단위 테스트.

픽셀 ↔ mm ↔ 기기 좌표 변환의 왕복 정확도, 경계값, 예외를 실제로 측정한다.
난수를 쓰는 곳은 전부 시드를 고정해 결정론을 보장한다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    Point,
)
from docagent.errors import DocAgentError, VisionError
from docagent.vision.geometry import (
    A4_ASPECT_RATIO,
    MM_PER_INCH,
    A4CoordinateSystem,
    CoordinateOutOfRangeWarning,
    MachineCalibration,
    MachineRangeError,
    apply_homography,
    estimate_skew_deg,
    order_quad,
    quad_area,
    quad_area_ratio,
    quad_edge_lengths,
    solve_homography,
)

#: 테스트 전반에서 쓰는 고정 시드.
SEED = 20260909


def rotated_rect(
    center: tuple[float, float],
    width: float,
    height: float,
    angle_deg: float,
) -> np.ndarray:
    """이미지 좌표계에서 ``angle_deg`` 만큼 시계 방향으로 돌린 사각형을 만든다.

    :param center: 중심 ``(x, y)``.
    :param width: 가로 길이.
    :param height: 세로 길이.
    :param angle_deg: 시계 방향 회전각(도).
    :returns: ``(4, 2)`` 배열(TL, TR, BR, BL 순서로 생성된다).
    """
    radian = math.radians(angle_deg)
    matrix = np.array(
        [[math.cos(radian), -math.sin(radian)], [math.sin(radian), math.cos(radian)]],
        dtype=np.float64,
    )
    half = np.array(
        [
            [-width / 2.0, -height / 2.0],
            [width / 2.0, -height / 2.0],
            [width / 2.0, height / 2.0],
            [-width / 2.0, height / 2.0],
        ],
        dtype=np.float64,
    )
    return (half @ matrix.T) + np.asarray(center, dtype=np.float64)


class TestOrderQuad:
    """:func:`order_quad` 정렬 규칙."""

    def test_orders_shuffled_points(self) -> None:
        """뒤섞인 4점을 좌상/우상/우하/좌하 순으로 되돌린다."""
        shuffled = [[100.0, 200.0], [0.0, 0.0], [100.0, 0.0], [0.0, 200.0]]
        ordered = order_quad(shuffled)
        assert ordered.shape == (4, 2)
        assert ordered.tolist() == [
            [0.0, 0.0],
            [100.0, 0.0],
            [100.0, 200.0],
            [0.0, 200.0],
        ]

    def test_accepts_opencv_contour_shape(self) -> None:
        """OpenCV 윤곽선 형태 ``(4, 1, 2)`` 도 받는다."""
        contour = np.array(
            [[[0, 0]], [[10, 0]], [[10, 20]], [[0, 20]]], dtype=np.int32
        )
        ordered = order_quad(contour)
        assert ordered[0].tolist() == [0.0, 0.0]
        assert ordered[2].tolist() == [10.0, 20.0]

    def test_is_idempotent(self) -> None:
        """이미 정렬된 사각형을 다시 정렬해도 같다."""
        quad = rotated_rect((50.0, 60.0), 40.0, 60.0, 12.0)
        once = order_quad(quad)
        assert np.allclose(order_quad(once), once)

    @pytest.mark.parametrize(
        "bad",
        [
            [[0, 0], [1, 1]],
            [[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]],
            [0, 1, 2, 3],
        ],
    )
    def test_rejects_wrong_shape(self, bad: object) -> None:
        """4점이 아니면 한국어 메시지와 함께 거부한다."""
        with pytest.raises(ValueError, match="4, 2"):
            order_quad(bad)

    def test_rejects_non_finite(self) -> None:
        """NaN 좌표는 조용히 통과시키지 않는다."""
        with pytest.raises(ValueError, match="유한"):
            order_quad([[0.0, 0.0], [1.0, float("nan")], [1.0, 1.0], [0.0, 1.0]])

    def test_rejects_ambiguous_45_degree_quad(self) -> None:
        """45도로 돌아간 마름모는 꼭짓점 구분이 모호하므로 거부한다."""
        with pytest.raises(ValueError, match="45도"):
            order_quad([[5.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]])


class TestQuadMetrics:
    """사각형 계측 함수."""

    def test_edge_lengths(self) -> None:
        """축 정렬 사각형의 네 변 길이."""
        top, right, bottom, left = quad_edge_lengths(
            [[0.0, 0.0], [30.0, 0.0], [30.0, 40.0], [0.0, 40.0]]
        )
        assert (top, bottom) == (30.0, 30.0)
        assert (left, right) == (40.0, 40.0)

    def test_area_and_ratio(self) -> None:
        """넓이와 이미지 대비 면적비."""
        quad = [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]]
        assert quad_area(quad) == pytest.approx(5000.0)
        assert quad_area_ratio(quad, (100, 200, 3)) == pytest.approx(0.25)

    def test_area_ratio_rejects_bad_shape(self) -> None:
        """이미지 크기가 0 이면 거부한다."""
        with pytest.raises(ValueError, match="이미지 크기"):
            quad_area_ratio([[0, 0], [1, 0], [1, 1], [0, 1]], (0, 10))

    @pytest.mark.parametrize("angle", [-20.0, -7.5, 0.0, 3.0, 15.0, 30.0])
    def test_skew_matches_generated_angle(self, angle: float) -> None:
        """생성한 회전각을 그대로 복원한다(양수 = 시계 방향)."""
        quad = rotated_rect((300.0, 400.0), 210.0, 297.0, angle)
        assert estimate_skew_deg(quad) == pytest.approx(angle, abs=1e-6)

    def test_skew_is_zero_for_axis_aligned(self) -> None:
        """축 정렬 사각형의 기울기는 0 이다."""
        assert estimate_skew_deg(
            [[0.0, 0.0], [210.0, 0.0], [210.0, 297.0], [0.0, 297.0]]
        ) == pytest.approx(0.0)


class TestHomography:
    """호모그래피 계산과 적용."""

    def test_round_trip_recovers_source(self) -> None:
        """정방향 → 역방향 투영으로 원본 좌표가 복원된다."""
        src = np.array(
            [[10.0, 12.0], [520.0, 40.0], [500.0, 800.0], [30.0, 770.0]]
        )
        dst = np.array([[0.0, 0.0], [209.0, 0.0], [209.0, 296.0], [0.0, 296.0]])
        forward = solve_homography(src, dst)
        backward = solve_homography(dst, src)
        projected = apply_homography(forward, src)
        assert np.allclose(projected, dst, atol=1e-6)
        assert np.allclose(apply_homography(backward, projected), src, atol=1e-6)

    def test_single_point_input(self) -> None:
        """길이 2 의 1차원 입력은 점 1개로 본다."""
        identity = np.eye(3)
        result = apply_homography(identity, [3.0, 4.0])
        assert result.shape == (1, 2)
        assert result[0].tolist() == [3.0, 4.0]

    def test_accepts_contour_shaped_points(self) -> None:
        """``(N, 1, 2)`` 윤곽선 형태도 받는다."""
        points = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        assert apply_homography(np.eye(3), points).shape == (2, 2)

    def test_rejects_degenerate_quad(self) -> None:
        """넓이가 0 인 사각형은 거부한다."""
        flat = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
        square = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
        with pytest.raises(ValueError, match="퇴화"):
            solve_homography(flat, square)

    def test_rejects_bad_matrix_shape(self) -> None:
        """``(3, 3)`` 이 아닌 행렬은 거부한다."""
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            apply_homography(np.eye(2), [[0.0, 0.0]])

    def test_rejects_diverging_projection(self) -> None:
        """동차 좌표가 0 이 되는 투영은 조용히 넘기지 않는다."""
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="동차 좌표"):
            apply_homography(matrix, [[0.0, 5.0]])

    def test_rejects_empty_points(self) -> None:
        """빈 점 배열은 거부한다."""
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            apply_homography(np.eye(3), np.zeros((0, 2)))


class TestA4CoordinateSystem:
    """A4 픽셀 ↔ mm 환산기."""

    def test_from_dpi_matches_physical_size(self) -> None:
        """300dpi A4 는 2480 x 3508 픽셀이다."""
        coords = A4CoordinateSystem.from_dpi(300)
        assert coords.size_px == (2480, 3508)
        assert coords.px_per_mm_x == pytest.approx(300 / MM_PER_INCH, rel=1e-3)
        assert coords.px_per_mm_y == pytest.approx(300 / MM_PER_INCH, rel=1e-3)

    def test_page_corners_round_trip(self) -> None:
        """(0, 0) 과 (210, 297) 경계값이 왕복 변환에서 보존된다."""
        coords = A4CoordinateSystem.from_dpi(300)
        for point in (Point(0.0, 0.0), Point(A4_WIDTH_MM, A4_HEIGHT_MM)):
            x_px, y_px = coords.point_to_px(point)
            restored = coords.point_to_mm(x_px, y_px)
            assert restored.x_mm == pytest.approx(point.x_mm, abs=0.05)
            assert restored.y_mm == pytest.approx(point.y_mm, abs=0.05)

    def test_box_round_trip(self) -> None:
        """mm 사각형 → px → mm 왕복 오차가 0.1mm 미만이다."""
        coords = A4CoordinateSystem.from_dpi(300)
        box = BoxMm(26.0, 139.0, 6.0, 6.0)
        restored = coords.to_mm(coords.to_px(box))
        assert restored.x_mm == pytest.approx(box.x_mm, abs=0.1)
        assert restored.y_mm == pytest.approx(box.y_mm, abs=0.1)
        assert restored.w_mm == pytest.approx(box.w_mm, abs=0.1)
        assert restored.h_mm == pytest.approx(box.h_mm, abs=0.1)

    def test_center_of_box_is_actuator_target(self) -> None:
        """체크박스 중심이 액추에이터 목표점과 같은 좌표로 환산된다."""
        coords = A4CoordinateSystem.from_dpi(300)
        box = BoxMm(26.0, 139.0, 6.0, 6.0)
        center_px = coords.point_to_px_exact(box.center())
        restored = coords.point_to_mm(*center_px)
        assert restored.x_mm == pytest.approx(29.0, abs=1e-6)
        assert restored.y_mm == pytest.approx(142.0, abs=1e-6)

    def test_out_of_range_point_warns(self) -> None:
        """페이지를 벗어난 좌표는 경고를 남긴다(조용한 실패 금지)."""
        coords = A4CoordinateSystem.from_dpi(300)
        with pytest.warns(CoordinateOutOfRangeWarning):
            coords.point_to_mm(coords.width_px * 2, 0.0)

    def test_clamp_point_limits_to_page(self) -> None:
        """clamp 는 값을 페이지 안으로 잘라 낸다."""
        coords = A4CoordinateSystem.from_dpi(300)
        with pytest.warns(CoordinateOutOfRangeWarning):
            clamped = coords.clamp_point(Point(-10.0, 400.0))
        assert clamped.x_mm == 0.0
        assert clamped.y_mm == A4_HEIGHT_MM

    def test_clamp_box_limits_to_page(self) -> None:
        """페이지 밖으로 나간 사각형은 잘려서 페이지 안에 들어온다."""
        coords = A4CoordinateSystem.from_dpi(300)
        with pytest.warns(CoordinateOutOfRangeWarning):
            clamped = coords.clamp_box(BoxMm(200.0, 290.0, 40.0, 40.0))
        assert clamped.right_mm == pytest.approx(A4_WIDTH_MM)
        assert clamped.bottom_mm == pytest.approx(A4_HEIGHT_MM)
        assert clamped.w_mm == pytest.approx(10.0)

    def test_to_mm_with_clamp_option(self) -> None:
        """``clamp=True`` 면 픽셀 사각형도 페이지 안으로 잘린다."""
        coords = A4CoordinateSystem.from_dpi(300)
        huge = BoxPx(x=0, y=0, w=coords.width_px * 2, h=coords.height_px * 2)
        with pytest.warns(CoordinateOutOfRangeWarning):
            box = coords.to_mm(huge, clamp=True)
        assert box.w_mm == pytest.approx(A4_WIDTH_MM)
        assert box.h_mm == pytest.approx(A4_HEIGHT_MM)

    def test_contains_includes_boundary(self) -> None:
        """페이지 경계는 포함으로 본다."""
        coords = A4CoordinateSystem.from_dpi(200)
        assert coords.contains(Point(0.0, 0.0))
        assert coords.contains(Point(A4_WIDTH_MM, A4_HEIGHT_MM))
        assert not coords.contains(Point(A4_WIDTH_MM + 0.5, 0.0))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dpi": 0, "width_px": 100, "height_px": 100},
            {"dpi": 300, "width_px": 0, "height_px": 100},
            {"dpi": 300, "width_px": 100, "height_px": 0},
        ],
    )
    def test_rejects_invalid_construction(self, kwargs: dict[str, int]) -> None:
        """1 미만 인자는 한국어 메시지로 거부한다."""
        with pytest.raises(ValueError):
            A4CoordinateSystem(**kwargs)

    def test_from_dpi_rejects_zero(self) -> None:
        """dpi 0 은 거부한다."""
        with pytest.raises(ValueError, match="dpi"):
            A4CoordinateSystem.from_dpi(0)

    def test_a4_aspect_constant(self) -> None:
        """A4 종횡비 상수가 실제 규격과 일치한다."""
        assert A4_ASPECT_RATIO == pytest.approx(297.0 / 210.0)


class TestMachineCalibration:
    """문서 좌표 → 기기 좌표 변환."""

    @pytest.mark.parametrize("flip_y", [False, True])
    def test_round_trip(self, flip_y: bool) -> None:
        """to_machine → from_machine 왕복이 원래 좌표를 복원한다."""
        calibration = MachineCalibration(
            origin_offset_mm=Point(12.5, -7.25),
            scale_x=1.02,
            scale_y=0.98,
            flip_y=flip_y,
        )
        for point in (Point(0.0, 0.0), Point(29.0, 142.0), Point(210.0, 297.0)):
            machine = calibration.to_machine(point)
            restored = calibration.from_machine(machine)
            assert restored.x_mm == pytest.approx(point.x_mm, abs=1e-9)
            assert restored.y_mm == pytest.approx(point.y_mm, abs=1e-9)

    def test_identity_default(self) -> None:
        """기본 캘리브레이션은 항등 변환이다."""
        calibration = MachineCalibration()
        assert calibration.to_machine(Point(10.0, 20.0)).to_tuple() == (10.0, 20.0)

    def test_flip_y_inverts_axis(self) -> None:
        """flip_y 는 문서 y 를 뒤집는다."""
        calibration = MachineCalibration(flip_y=True)
        assert calibration.to_machine(Point(0.0, 0.0)).y_mm == pytest.approx(
            A4_HEIGHT_MM
        )
        assert calibration.to_machine(Point(0.0, A4_HEIGHT_MM)).y_mm == pytest.approx(
            0.0
        )

    @pytest.mark.parametrize(
        "point",
        [Point(-0.5, 10.0), Point(211.0, 10.0), Point(10.0, -1.0), Point(10.0, 298.0)],
    )
    def test_out_of_work_area_raises(self, point: Point) -> None:
        """가동범위 밖 좌표는 예외를 던진다."""
        with pytest.raises(MachineRangeError, match="가동범위"):
            MachineCalibration().to_machine(point)

    def test_range_error_is_both_value_and_domain_error(self) -> None:
        """MachineRangeError 는 ValueError 이자 도메인 예외다."""
        with pytest.raises(ValueError):
            MachineCalibration().to_machine(Point(500.0, 0.0))
        with pytest.raises(VisionError):
            MachineCalibration().to_machine(Point(500.0, 0.0))
        assert issubclass(MachineRangeError, DocAgentError)

    def test_from_machine_out_of_range_raises(self) -> None:
        """역변환 결과가 종이 밖이면 예외를 던진다."""
        calibration = MachineCalibration(origin_offset_mm=Point(0.0, 0.0))
        with pytest.raises(MachineRangeError):
            calibration.from_machine(Point(400.0, 10.0))

    @pytest.mark.parametrize("scale", [0.0, float("nan")])
    def test_rejects_invalid_scale(self, scale: float) -> None:
        """축척 0 또는 NaN 은 거부한다."""
        with pytest.raises(ValueError, match="scale"):
            MachineCalibration(scale_x=scale)

    @pytest.mark.parametrize("flip_y", [False, True])
    @pytest.mark.parametrize("count", [2, 3, 5])
    def test_estimate_recovers_known_calibration(
        self, flip_y: bool, count: int
    ) -> None:
        """대응점에서 원래 캘리브레이션을 복원한다."""
        truth = MachineCalibration(
            origin_offset_mm=Point(15.0, -4.0),
            scale_x=1.05,
            scale_y=0.95,
            flip_y=flip_y,
        )
        doc_points = [
            Point(10.0, 20.0),
            Point(200.0, 280.0),
            Point(105.0, 150.0),
            Point(30.0, 250.0),
            Point(180.0, 60.0),
        ][:count]
        machine_points = [truth.to_machine(p) for p in doc_points]
        estimated = MachineCalibration.estimate_from_pairs(doc_points, machine_points)

        assert estimated.flip_y is flip_y
        assert estimated.scale_x == pytest.approx(truth.scale_x, abs=1e-6)
        assert estimated.scale_y == pytest.approx(truth.scale_y, abs=1e-6)
        assert estimated.origin_offset_mm.x_mm == pytest.approx(
            truth.origin_offset_mm.x_mm, abs=1e-6
        )
        assert estimated.origin_offset_mm.y_mm == pytest.approx(
            truth.origin_offset_mm.y_mm, abs=1e-6
        )
        for point in doc_points:
            expected = truth.to_machine(point)
            actual = estimated.to_machine(point)
            assert actual.x_mm == pytest.approx(expected.x_mm, abs=1e-6)
            assert actual.y_mm == pytest.approx(expected.y_mm, abs=1e-6)

    def test_estimate_is_robust_to_small_noise(self) -> None:
        """측정 잡음이 섞여도 최소자승으로 0.5mm 이내로 복원한다."""
        truth = MachineCalibration(
            origin_offset_mm=Point(8.0, 3.0), scale_x=1.0, scale_y=1.0
        )
        rng = np.random.default_rng(SEED)
        doc_points = [
            Point(20.0, 30.0),
            Point(190.0, 40.0),
            Point(20.0, 270.0),
            Point(190.0, 270.0),
            Point(105.0, 150.0),
        ]
        machine_points = []
        for point in doc_points:
            exact = truth.to_machine(point)
            jitter = rng.normal(0.0, 0.2, size=2)
            machine_points.append(
                Point(exact.x_mm + float(jitter[0]), exact.y_mm + float(jitter[1]))
            )
        estimated = MachineCalibration.estimate_from_pairs(doc_points, machine_points)
        probe = Point(105.0, 150.0)
        expected = truth.to_machine(probe)
        actual = estimated.to_machine(probe)
        assert math.hypot(actual.x_mm - expected.x_mm, actual.y_mm - expected.y_mm) < 0.5

    def test_estimate_rejects_length_mismatch(self) -> None:
        """대응점 개수가 다르면 거부한다."""
        with pytest.raises(ValueError, match="개수"):
            MachineCalibration.estimate_from_pairs(
                [Point(0.0, 0.0), Point(1.0, 1.0)], [Point(0.0, 0.0)]
            )

    def test_estimate_rejects_single_pair(self) -> None:
        """대응점이 1쌍이면 거부한다."""
        with pytest.raises(ValueError, match="2쌍"):
            MachineCalibration.estimate_from_pairs([Point(0.0, 0.0)], [Point(0.0, 0.0)])

    def test_estimate_rejects_degenerate_axis(self) -> None:
        """x 좌표가 모두 같으면 축척을 정할 수 없어 거부한다."""
        doc_points = [Point(10.0, 20.0), Point(10.0, 200.0)]
        machine_points = [Point(10.0, 20.0), Point(10.0, 200.0)]
        with pytest.raises(ValueError, match="축척"):
            MachineCalibration.estimate_from_pairs(doc_points, machine_points)

    def test_estimate_rejects_non_finite(self) -> None:
        """NaN 대응점은 거부한다."""
        doc_points = [Point(10.0, 20.0), Point(100.0, 200.0)]
        machine_points = [Point(float("nan"), 20.0), Point(100.0, 200.0)]
        with pytest.raises(ValueError, match="유한"):
            MachineCalibration.estimate_from_pairs(doc_points, machine_points)
