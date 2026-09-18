"""펜 액추에이터 제어 — :class:`docagent.interfaces.MotionController` 프로토콜.

하드웨어는 **에이전트의 액추에이터일 뿐**이다. 목표 좌표는 언제나 에이전트가
:class:`~docagent.contracts.DocumentStructure` 에서 직접 읽어 mm 로 넘겨 주며,
이 모듈은 그것을 기기 좌표로 옮겨 보내고 도착을 확인하는 일만 한다.

시리얼 프로토콜(로드맵 Phase 8~9)
---------------------------------
======================= ==================== =================================
송신                    수신                 의미
======================= ==================== =================================
``MOVE {x:.1f} {y:.1f}`` ``ARRIVED``          지정 좌표로 이동 완료
``HOME``                 ``HOME_REACHED``     원점 복귀 완료
======================= ==================== =================================

모든 명령은 ``\\n`` 으로 끝난다. 응답이 위 문자열이 아니면 **성공으로 보지
않는다**(``ERR ...`` 도, 빈 응답도 실패다). 타임아웃·예상 밖 응답은
:attr:`~docagent.config.DocAgentConfig.serial_retries` 만큼 재시도하고,
그래도 실패하면 :class:`~docagent.errors.ToolExecutionError` 를 던진다.

좌표 변환
---------
:class:`~docagent.vision.geometry.MachineCalibration` 을 주면 문서 좌표를
기기 좌표로 바꾼 뒤 전송한다. :meth:`position` 은 반대로 되돌려 **항상 문서
좌표(A4 mm)** 를 돌려준다. 모듈 경계를 넘는 좌표는 문서 좌표뿐이다.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from docagent.contracts import A4_PAGE_SIZE_MM, Point
from docagent.errors import AdapterUnavailable, ToolExecutionError

__all__ = [
    "MOVE_COMMAND",
    "HOME_COMMAND",
    "ARRIVED_RESPONSE",
    "HOME_RESPONSE",
    "DEFAULT_BAUDRATE",
    "DEFAULT_TIMEOUT_S",
    "DEFAULT_RETRIES",
    "MockMotionController",
    "SerialMotionController",
]

_LOG = logging.getLogger(__name__)

#: 이동 명령 형식. 소수 첫째 자리까지 보낸다(펜 기구 분해능이 0.1mm 수준).
MOVE_COMMAND: str = "MOVE {x:.1f} {y:.1f}\n"
#: 원점 복귀 명령.
HOME_COMMAND: str = "HOME\n"
#: 이동 완료 응답.
ARRIVED_RESPONSE: str = "ARRIVED"
#: 원점 복귀 완료 응답.
HOME_RESPONSE: str = "HOME_REACHED"

#: 기본 통신 속도.
DEFAULT_BAUDRATE: int = 115200
#: 기본 응답 대기 시간(초).
DEFAULT_TIMEOUT_S: float = 5.0
#: 기본 재시도 횟수(최초 시도 제외).
DEFAULT_RETRIES: int = 2


def _check_range(
    x_mm: float, y_mm: float, page_size_mm: tuple[float, float]
) -> tuple[bool, str]:
    """목표 좌표가 가동 범위 안인지 검사한다.

    :param x_mm: 목표 x(mm).
    :param y_mm: 목표 y(mm).
    :param page_size_mm: 가동 범위 ``(가로_mm, 세로_mm)``.
    :returns: ``(허용 여부, 한국어 사유)``. 허용이면 사유는 빈 문자열.
    """
    width, height = page_size_mm
    if not (0.0 <= x_mm <= width):
        return (
            False,
            f"x 좌표 {x_mm:.1f}mm 가 가동 범위 0~{width:.0f}mm 를 벗어났습니다.",
        )
    if not (0.0 <= y_mm <= height):
        return (
            False,
            f"y 좌표 {y_mm:.1f}mm 가 가동 범위 0~{height:.0f}mm 를 벗어났습니다.",
        )
    return (True, "")


class MockMotionController:
    """하드웨어 없이 동작하는 결정론적 펜 제어기.

    이동 이력을 남기고, 필요하면 **고정 오차**를 주입해 "도착했지만 목표에서
    벗어난" 상황을 재현한다(오차는 난수가 아니라 고정값이므로 결정론적이다).

    :param page_size_mm: 가동 범위 ``(가로_mm, 세로_mm)``. 기본 A4.
    :param error_mm: 주입할 고정 오차 ``(dx_mm, dy_mm)``. 이동 후 위치가
        목표에서 이만큼 어긋난다.
    :param fail_targets: 이동을 실패(``False`` 반환)시킬 목표 좌표 목록.
        ``(x_mm, y_mm)`` 튜플을 0.05mm 이내로 비교한다. 재시도 로직 시험용.
    :param calibration: :class:`~docagent.vision.geometry.MachineCalibration`.
        주면 기기 좌표 변환까지 흉내 낸다(:attr:`machine_moves` 에 기록).
    :raises ValueError: 가동 범위가 0 이하인 경우.
    """

    def __init__(
        self,
        page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
        *,
        error_mm: tuple[float, float] = (0.0, 0.0),
        fail_targets: Sequence[tuple[float, float]] = (),
        calibration: Any | None = None,
    ) -> None:
        page = (float(page_size_mm[0]), float(page_size_mm[1]))
        if page[0] <= 0 or page[1] <= 0:
            raise ValueError(
                f"page_size_mm 은 0 보다 커야 합니다: {page_size_mm!r}"
            )
        self.page_size_mm = page
        self.error_mm = (float(error_mm[0]), float(error_mm[1]))
        self._fail_targets = [(float(x), float(y)) for x, y in fail_targets]
        self.calibration = calibration
        self._position = Point(0.0, 0.0)
        #: 이동에 성공한 목표점(문서 좌표) 이력.
        self.moves: list[Point] = []
        #: 전송된 기기 좌표 이력(``calibration`` 이 있을 때만 채워진다).
        self.machine_moves: list[Point] = []
        #: 가동 범위 이탈 등으로 거부한 좌표 이력.
        self.rejected: list[tuple[Point, str]] = []
        #: ``home()`` 호출 횟수.
        self.home_calls: int = 0

    # ------------------------------------------------------------------

    def _is_fail_target(self, x_mm: float, y_mm: float) -> bool:
        """실패시키기로 지정된 좌표인지 확인한다.

        :param x_mm: 목표 x(mm).
        :param y_mm: 목표 y(mm).
        :returns: 지정 좌표면 True.
        """
        return any(
            abs(x_mm - fx) <= 0.05 and abs(y_mm - fy) <= 0.05
            for fx, fy in self._fail_targets
        )

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        """지정한 문서 좌표로 펜을 옮긴다.

        :param x_mm: 목표 x(mm, 좌상단 원점).
        :param y_mm: 목표 y(mm, 아래쪽 +).
        :returns: 이동 성공 여부. 가동 범위 밖이거나 ``fail_targets`` 에
            해당하면 False 이며 위치를 바꾸지 않는다.
        """
        x_mm = float(x_mm)
        y_mm = float(y_mm)
        allowed, reason = _check_range(x_mm, y_mm, self.page_size_mm)
        if not allowed:
            self.rejected.append((Point(x_mm, y_mm), reason))
            _LOG.warning("펜 이동 거부: %s", reason)
            return False
        if self._is_fail_target(x_mm, y_mm):
            self.rejected.append((Point(x_mm, y_mm), "시험용으로 지정된 실패 좌표입니다."))
            return False
        if self.calibration is not None:
            self.machine_moves.append(self.calibration.to_machine(Point(x_mm, y_mm)))
        self._position = Point(x_mm + self.error_mm[0], y_mm + self.error_mm[1])
        self.moves.append(Point(x_mm, y_mm))
        return True

    def home(self) -> bool:
        """원점(0, 0)으로 복귀한다.

        :returns: 항상 True.
        """
        self.home_calls += 1
        self._position = Point(0.0, 0.0)
        return True

    def position(self) -> Point:
        """현재 펜 위치를 문서 좌표로 돌려준다.

        :returns: :class:`~docagent.contracts.Point`.
        """
        return self._position

    def last_move(self) -> Point | None:
        """마지막으로 이동한 목표점. 이동 이력이 없으면 ``None``.

        :returns: :class:`~docagent.contracts.Point` 또는 ``None``.
        """
        return self.moves[-1] if self.moves else None

    def __repr__(self) -> str:
        """이동 횟수와 현재 위치만 노출한다."""
        return (
            f"<MockMotionController 이동 {len(self.moves)}회, "
            f"현재=({self._position.x_mm:.1f}, {self._position.y_mm:.1f})mm>"
        )


class SerialMotionController:
    """시리얼(Arduino) 펜 제어기. ``pyserial`` 을 지연 import 한다.

    ``with`` 문으로 쓰면 종료 시 원점 복귀 후 포트를 닫는다::

        with SerialMotionController("COM3") as pen:
            pen.move_to(22.5, 102.5)

    :param port: 포트 이름(예: ``"COM3"``, ``"/dev/ttyUSB0"``).
    :param baudrate: 통신 속도.
    :param timeout_s: 한 줄 응답 대기 시간(초).
    :param retries: 실패 시 재시도 횟수(최초 시도 제외, 0 이상).
    :param page_size_mm: 가동 범위 ``(가로_mm, 세로_mm)``.
    :param calibration: :class:`~docagent.vision.geometry.MachineCalibration`.
        ``None`` 이면 문서 좌표를 그대로 보낸다.
    :param transport: 이미 열려 있는 시리얼 유사 객체(테스트 주입용).
        ``write(bytes)`` / ``readline() -> bytes`` / ``close()`` 를 제공해야 한다.
        주어지면 ``pyserial`` 을 import 하지 않는다.
    :raises ValueError: 수치 인자가 허용 범위를 벗어난 경우.
    :raises docagent.errors.AdapterUnavailable: ``pyserial`` 이 없는 경우
        (``transport`` 를 주지 않았을 때만).
    :raises docagent.errors.ToolExecutionError: 포트를 열지 못한 경우.
    """

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
        calibration: Any | None = None,
        transport: Any | None = None,
    ) -> None:
        if not port and transport is None:
            raise ValueError("포트 이름이 비어 있습니다.")
        if baudrate <= 0:
            raise ValueError(f"baudrate 는 0 보다 커야 합니다: {baudrate}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s 는 0 보다 커야 합니다: {timeout_s}")
        if retries < 0:
            raise ValueError(f"retries 는 0 이상이어야 합니다: {retries}")
        page = (float(page_size_mm[0]), float(page_size_mm[1]))
        if page[0] <= 0 or page[1] <= 0:
            raise ValueError(f"page_size_mm 은 0 보다 커야 합니다: {page_size_mm!r}")

        self.port = port
        self.baudrate = baudrate
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self.page_size_mm = page
        self.calibration = calibration
        self._position = Point(0.0, 0.0)
        #: 실제로 전송한 명령 문자열 이력(감사·테스트용).
        self.sent: list[str] = []
        self._closed = False
        self._serial = transport if transport is not None else self._open_serial()

    # ------------------------------------------------------------------
    # 연결
    # ------------------------------------------------------------------

    def _open_serial(self) -> Any:
        """``pyserial`` 을 지연 import 해 포트를 연다.

        :returns: 열린 ``serial.Serial`` 객체.
        :raises docagent.errors.AdapterUnavailable: 패키지가 없는 경우.
        :raises docagent.errors.ToolExecutionError: 포트를 열지 못한 경우.
        """
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterUnavailable(
                package="pyserial",
                feature="시리얼(Arduino) 펜 액추에이터 제어",
                extra="serial",
            ) from exc
        try:
            return serial.Serial(
                self.port, baudrate=self.baudrate, timeout=self.timeout_s
            )
        except Exception as exc:  # noqa: BLE001 — 도메인 예외로 감싸 올린다.
            raise ToolExecutionError(
                f"시리얼 포트를 열지 못했습니다({self.port}): {exc}",
                tool_name="SerialMotionController",
            ) from exc

    def close(self) -> None:
        """포트를 닫는다. 두 번 호출해도 안전하다.

        :returns: ``None``.
        :raises docagent.errors.ToolExecutionError: 닫는 중 오류가 난 경우.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._serial.close()
        except Exception as exc:  # noqa: BLE001 — 조용한 실패 금지.
            raise ToolExecutionError(
                f"시리얼 포트를 닫지 못했습니다({self.port}): {exc}",
                tool_name="SerialMotionController",
            ) from exc

    def __enter__(self) -> "SerialMotionController":
        """컨텍스트 매니저 진입. 자기 자신을 돌려준다."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """컨텍스트 종료 시 원점 복귀 후 포트를 닫는다.

        예외로 빠져나가는 경우에는 복귀를 시도하되 실패를 삼키지 않고
        경고 로그만 남긴 뒤 원래 예외를 그대로 전파한다.
        """
        try:
            if exc_type is None:
                self.home()
            else:
                try:
                    self.home()
                except ToolExecutionError as home_exc:
                    _LOG.warning("종료 중 원점 복귀에 실패했습니다: %s", home_exc)
        finally:
            self.close()

    # ------------------------------------------------------------------
    # 통신
    # ------------------------------------------------------------------

    def _exchange(self, command: str, expected: str, *, what: str) -> None:
        """명령 한 줄을 보내고 기대 응답을 확인한다(재시도 포함).

        :param command: 보낼 명령(개행 포함).
        :param expected: 기대 응답 문자열.
        :param what: 오류 메시지에 쓸 동작 이름.
        :returns: ``None``.
        :raises docagent.errors.ToolExecutionError: 모든 시도가 실패한 경우.
        """
        if self._closed:
            raise ToolExecutionError(
                "이미 닫힌 포트로는 명령을 보낼 수 없습니다.",
                tool_name="SerialMotionController",
            )
        problems: list[str] = []
        for attempt in range(self.retries + 1):
            try:
                self._serial.write(command.encode("utf-8"))
                self.sent.append(command)
                raw = self._serial.readline()
            except Exception as exc:  # noqa: BLE001 — 통신 오류를 모아 보고한다.
                problems.append(f"{attempt + 1}회차 통신 오류: {exc}")
                continue
            reply = _decode(raw)
            if reply == expected:
                return
            if not reply:
                problems.append(f"{attempt + 1}회차 응답 없음(타임아웃 {self.timeout_s}초)")
            else:
                problems.append(f"{attempt + 1}회차 예상 밖 응답: {reply!r}")
            _LOG.warning("%s 재시도(%d/%d): %s", what, attempt + 1, self.retries + 1, problems[-1])
        raise ToolExecutionError(
            f"{what}에 실패했습니다(기대 응답 {expected!r}). " + " / ".join(problems),
            tool_name="SerialMotionController",
        )

    # ------------------------------------------------------------------
    # MotionController 프로토콜
    # ------------------------------------------------------------------

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        """문서 좌표로 펜을 옮긴다.

        가동 범위는 **전송 전에** 검사한다. 범위를 벗어난 명령은 장치로
        내보내지 않고 False 를 돌려준다.

        :param x_mm: 목표 x(mm, 문서 좌표).
        :param y_mm: 목표 y(mm, 문서 좌표).
        :returns: 이동 성공 여부(범위 위반이면 False).
        :raises docagent.errors.ToolExecutionError: 통신·응답 확인에 실패한 경우.
        """
        x_mm = float(x_mm)
        y_mm = float(y_mm)
        allowed, reason = _check_range(x_mm, y_mm, self.page_size_mm)
        if not allowed:
            _LOG.warning("펜 이동 거부(전송하지 않음): %s", reason)
            return False
        target = Point(x_mm, y_mm)
        machine = (
            self.calibration.to_machine(target) if self.calibration is not None else target
        )
        self._exchange(
            MOVE_COMMAND.format(x=machine.x_mm, y=machine.y_mm),
            ARRIVED_RESPONSE,
            what="펜 이동",
        )
        self._position = target
        return True

    def home(self) -> bool:
        """원점으로 복귀한다.

        :returns: 복귀 성공 여부(항상 True, 실패는 예외).
        :raises docagent.errors.ToolExecutionError: 통신·응답 확인에 실패한 경우.
        """
        self._exchange(HOME_COMMAND, HOME_RESPONSE, what="원점 복귀")
        self._position = Point(0.0, 0.0)
        return True

    def position(self) -> Point:
        """마지막으로 확인된 펜 위치(문서 좌표)를 돌려준다.

        :returns: :class:`~docagent.contracts.Point`.
        """
        return self._position

    def __repr__(self) -> str:
        """포트와 전송 명령 수만 노출한다."""
        state = "닫힘" if self._closed else "열림"
        return f"<SerialMotionController {self.port} {state}, 전송 {len(self.sent)}건>"


def _decode(raw: Any) -> str:
    """장치 응답 바이트를 문자열로 바꾼다.

    :param raw: ``readline()`` 결과(bytes 또는 str).
    :returns: 앞뒤 공백·개행을 제거한 문자열. ``None`` 이면 빈 문자열.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()
