"""도메인 예외 계층.

원칙: **예외를 삼키지 않는다.** 하위 계층(OpenCV, 파일 I/O, 선택적 어댑터)에서
올라온 예외는 반드시 이 모듈의 도메인 예외로 감싸 ``raise ... from exc`` 형태로
다시 던진다. 조용한 실패(None 반환·빈 리스트 반환으로 오류를 숨기는 것)는 금지한다.

모든 예외 메시지는 **한국어**로 작성한다(사용자 대상 문자열 규약).
이 모듈도 표준 라이브러리만 사용하며 :mod:`docagent.contracts` 에 의존하지 않는다.
"""

from __future__ import annotations

__all__ = [
    "DocAgentError",
    "VisionError",
    "DocumentNotFoundError",
    "LowConfidenceError",
    "PiiEgressBlocked",
    "HandoffRequired",
    "InvalidTransition",
    "ToolExecutionError",
    "AdapterUnavailable",
]


class DocAgentError(Exception):
    """이 프로젝트의 모든 도메인 예외의 최상위 기반 클래스.

    호출자는 ``except DocAgentError`` 하나로 시스템 내부 오류 전체를 잡을 수 있다.
    """


# --------------------------------------------------------------------------
# Vision
# --------------------------------------------------------------------------


class VisionError(DocAgentError):
    """See 단계(이미지 입력·전처리·탐지·검증)에서 발생한 오류의 기반 클래스."""


class DocumentNotFoundError(VisionError):
    """입력 이미지에서 문서를 찾지 못했거나 이미지 자체를 열 수 없는 경우.

    :param message: 한국어 오류 메시지.
    :param source: 문제가 된 이미지 경로 또는 식별자.
    """

    def __init__(self, message: str = "입력 이미지에서 문서를 찾지 못했습니다.", *, source: str | None = None) -> None:
        self.source = source
        if source:
            message = f"{message} (입력: {source})"
        super().__init__(message)


class LowConfidenceError(VisionError):
    """탐지·해석 신뢰도가 임계값에 미달하여 결과를 신뢰할 수 없는 경우.

    :param message: 한국어 오류 메시지.
    :param confidence: 실제 신뢰도(0.0~1.0).
    :param threshold: 요구 임계값(0.0~1.0).
    :param field_id: 문제가 된 항목 id. 문서 전체 수준이면 ``None``.
    """

    def __init__(
        self,
        message: str = "인식 신뢰도가 기준에 미달합니다.",
        *,
        confidence: float | None = None,
        threshold: float | None = None,
        field_id: str | None = None,
    ) -> None:
        self.confidence = confidence
        self.threshold = threshold
        self.field_id = field_id
        detail: list[str] = []
        if field_id is not None:
            detail.append(f"항목={field_id}")
        if confidence is not None:
            detail.append(f"신뢰도={confidence:.3f}")
        if threshold is not None:
            detail.append(f"임계값={threshold:.3f}")
        if detail:
            message = f"{message} ({', '.join(detail)})"
        super().__init__(message)


# --------------------------------------------------------------------------
# PII
# --------------------------------------------------------------------------


class PiiEgressBlocked(DocAgentError):
    """개인정보가 외부(LLM·네트워크·로그)로 나가려는 시도를 차단했을 때 발생한다.

    이 예외는 **막았다는 사실 자체**를 알리는 것이 목적이므로,
    메시지에 원문 값을 절대 포함하지 않는다. 유형과 개수만 남긴다.

    :param message: 한국어 오류 메시지.
    :param pii_types: 탐지된 개인정보 유형 목록(예: ``["rrn", "phone"]``).
    :param count: 탐지된 구간 개수.
    """

    def __init__(
        self,
        message: str = "개인정보가 포함되어 외부 전송을 차단했습니다.",
        *,
        pii_types: list[str] | None = None,
        count: int | None = None,
    ) -> None:
        self.pii_types = list(pii_types or [])
        self.count = count
        detail: list[str] = []
        if self.pii_types:
            detail.append(f"유형={', '.join(self.pii_types)}")
        if count is not None:
            detail.append(f"건수={count}")
        if detail:
            message = f"{message} ({'; '.join(detail)})"
        super().__init__(message)


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


class HandoffRequired(DocAgentError):
    """에이전트가 단독으로 처리할 수 없어 사람 지원으로 넘겨야 하는 상황.

    신뢰도 미달, 법적 판단이 필요한 항목, 반복 실패 등에서 발생한다.

    :param reason: 넘기는 사유(한국어 한 문장). 사용자에게 그대로 낭독 가능해야 한다.
    :param field_id: 문제가 된 항목 id. 특정 항목과 무관하면 ``None``.
    """

    def __init__(self, reason: str, *, field_id: str | None = None) -> None:
        self.reason = reason
        self.field_id = field_id
        message = reason if field_id is None else f"{reason} (항목: {field_id})"
        super().__init__(message)


class InvalidTransition(DocAgentError):
    """See → Understand → Explain → Ask → Act → Verify 상태 전이 규칙 위반.

    :param message: 한국어 오류 메시지.
    :param current: 현재 상태 이름.
    :param requested: 요청된 다음 상태 이름.
    """

    def __init__(
        self,
        message: str = "허용되지 않은 상태 전이입니다.",
        *,
        current: str | None = None,
        requested: str | None = None,
    ) -> None:
        self.current = current
        self.requested = requested
        if current is not None or requested is not None:
            message = f"{message} (현재={current}, 요청={requested})"
        super().__init__(message)


class ToolExecutionError(DocAgentError):
    """도구(Tool) 실행이 실패했을 때 발생한다.

    :param message: 한국어 오류 메시지.
    :param tool_name: 실패한 도구 이름.
    """

    def __init__(
        self,
        message: str = "도구 실행에 실패했습니다.",
        *,
        tool_name: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        if tool_name:
            message = f"[{tool_name}] {message}"
        super().__init__(message)


# --------------------------------------------------------------------------
# 선택적 어댑터
# --------------------------------------------------------------------------


class AdapterUnavailable(DocAgentError):
    """선택적 의존 패키지가 설치되지 않아 어댑터를 쓸 수 없는 경우.

    선택적 패키지(ultralytics, torch, anthropic, pyserial, pytesseract 등)는
    모듈 최상단에서 import 하지 않고 함수·생성자 내부에서 지연 import 한다.
    ``ImportError`` 를 잡아 이 예외로 감싸 올리면, 사용자는 무엇을 어떻게
    설치해야 하는지 한국어 안내를 그대로 받는다.

    :param package: 누락된 패키지 이름(예: ``"ultralytics"``).
    :param feature: 그 패키지가 필요한 기능 설명(예: ``"YOLO 기반 기입란 탐지"``).
    :param extra: ``pip install "docagent[<extra>]"`` 형태로 안내할 optional 그룹 이름.

    사용 예::

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise AdapterUnavailable(
                package="ultralytics", feature="YOLO 기반 기입란 탐지", extra="yolo"
            ) from exc
    """

    def __init__(
        self,
        package: str,
        *,
        feature: str | None = None,
        extra: str | None = None,
    ) -> None:
        self.package = package
        self.feature = feature
        self.extra = extra
        head = f"선택적 패키지 '{package}' 가 설치되어 있지 않습니다."
        if feature:
            head = f"{head} 이 기능({feature})은 해당 패키지를 필요로 합니다."
        if extra:
            install = f'pip install "docagent[{extra}]"'
        else:
            install = f"pip install {package}"
        tail = (
            f" 저장소 venv 인터프리터로 설치하십시오: "
            f".venv\\Scripts\\python.exe -m {install}"
        )
        super().__init__(head + tail)
