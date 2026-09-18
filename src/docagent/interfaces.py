"""모듈 간 경계 인터페이스 — :class:`typing.Protocol` 정의 전용.

이 모듈은 **구현을 담지 않는다.** 구조적 서브타이핑(덕 타이핑)으로만 계약을
표현하므로, 각 구현체는 이 모듈을 상속하거나 import 할 필요조차 없다.
시그니처만 맞으면 계약을 만족한다.

모든 Protocol 은 :func:`typing.runtime_checkable` 로 표시되어 있어
테스트에서 ``isinstance(obj, Detector)`` 로 메서드 존재를 확인할 수 있다.
(runtime_checkable 은 메서드 **존재**만 검사하며 시그니처는 검사하지 않는다.)

선택적 어댑터 규약
------------------
YOLO·LLM·시리얼·OCR 어댑터는 이 Protocol 들을 구현하되, 해당 패키지를
**모듈 최상단에서 import 하지 않는다.** 생성자나 메서드 내부에서 지연 import 하고
``ImportError`` 는 :class:`docagent.errors.AdapterUnavailable` 로 감싸 올린다.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

from docagent.contracts import (
    Detection,
    OcrWord,
    Point,
    RetrievedChunk,
    SanitizedText,
)

__all__ = [
    "ImageArray",
    "Detector",
    "OcrEngine",
    "PiiGate",
    "Retriever",
    "LlmClient",
    "MotionController",
    "SpeechIO",
    "Clock",
]

#: 이미지 배열 타입 별칭.
#:
#: 실제 런타임 타입은 ``numpy.ndarray`` 이며, 형태는 ``(H, W, 3)`` uint8 BGR
#: (OpenCV 기본) 또는 ``(H, W)`` uint8 그레이스케일이다.
#: 계약 계층이 numpy 에 의존하지 않도록 여기서는 :data:`typing.Any` 로 둔다.
ImageArray: TypeAlias = Any


@runtime_checkable
class Detector(Protocol):
    """문서 이미지에서 기입란 후보를 찾는 탐지기(See 단계).

    구현 예: OpenCV 윤곽선 기반 규칙 탐지기(기본), YOLO 어댑터(선택).
    """

    def detect(self, image: ImageArray) -> list[Detection]:
        """이미지에서 기입란 후보를 탐지한다.

        :param image: 정합(deskew·crop)이 끝난 문서 이미지.
            ``(H, W, 3)`` uint8 BGR 또는 ``(H, W)`` uint8 그레이스케일.
        :returns: :class:`~docagent.contracts.Detection` 목록.
            좌표는 반드시 **A4 mm 기준**으로 변환되어 있어야 한다.
            탐지 결과가 없으면 빈 리스트를 반환한다(예외를 던지지 않는다).
        :raises docagent.errors.VisionError: 이미지가 유효하지 않거나 전처리에 실패한 경우.
        """
        ...


@runtime_checkable
class OcrEngine(Protocol):
    """이미지에서 문자열을 읽는 OCR 엔진(Understand 단계).

    구현 예: pytesseract 어댑터(선택), 고정 응답 Mock(테스트).
    """

    def read(self, image: ImageArray) -> list[OcrWord]:
        """이미지에서 단어 단위 인식 결과를 반환한다.

        :param image: 인식 대상 이미지(전체 페이지 또는 잘라낸 영역).
        :returns: :class:`~docagent.contracts.OcrWord` 목록.
            좌표는 **A4 mm 기준**. 인식 결과가 없으면 빈 리스트.
        :raises docagent.errors.VisionError: 인식 자체가 실패한 경우.
        :raises docagent.errors.AdapterUnavailable: 선택적 OCR 패키지가 없는 경우.
        """
        ...


@runtime_checkable
class PiiGate(Protocol):
    """개인정보 유출 차단 게이트.

    외부(LLM·네트워크·영구 로그)로 나가는 **모든** 텍스트는 이 게이트를 통과해야 한다.
    """

    def sanitize(self, text: str) -> SanitizedText:
        """텍스트에서 개인정보를 탐지·마스킹한다.

        :param text: 원문 텍스트.
        :returns: :class:`~docagent.contracts.SanitizedText`.
            ``spans`` 인덱스는 **원문 기준**이며 원문 값을 담지 않는다.
            마스킹으로도 안전을 보장할 수 없으면 ``blocked=True`` 로 표시한다.
        """
        ...

    def assert_clean(self, text: str) -> None:
        """텍스트에 개인정보가 없음을 단언한다. 있으면 즉시 차단한다.

        :param text: 검사 대상 텍스트(보통 이미 마스킹된 문자열).
        :returns: ``None``. 통과 시 아무것도 반환하지 않는다.
        :raises docagent.errors.PiiEgressBlocked: 개인정보가 탐지된 경우.
            예외 메시지에는 유형·건수만 담기며 원문 값은 담기지 않는다.
        """
        ...


@runtime_checkable
class Retriever(Protocol):
    """근거 문서 검색기(Explain 단계의 RAG 구성요소)."""

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """질의에 적합한 근거 조각을 점수 내림차순으로 반환한다.

        :param query: 검색 질의. **개인정보가 제거된 문자열**이어야 한다.
        :param k: 반환할 최대 조각 수(1 이상).
        :returns: :class:`~docagent.contracts.RetrievedChunk` 목록.
            길이는 ``k`` 이하이며, 결과가 없으면 빈 리스트.
        """
        ...


@runtime_checkable
class LlmClient(Protocol):
    """대형 언어모델 클라이언트(선택적 어댑터).

    이 인터페이스로 나가는 ``system`` · ``user`` 문자열은 호출 전에 반드시
    :class:`PiiGate` 를 통과해야 한다. 공개 정보 영역만 전송한다.
    """

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """프롬프트에 대한 모델 응답 텍스트를 반환한다.

        :param system: 시스템 프롬프트(공개 정보만).
        :param user: 사용자 프롬프트(공개 정보만, PII 마스킹 완료).
        :param max_tokens: 생성 최대 토큰 수.
        :returns: 모델 응답 문자열.
        :raises docagent.errors.PiiEgressBlocked: 전송 직전 검사에서 개인정보가 발견된 경우.
        :raises docagent.errors.AdapterUnavailable: 선택적 LLM SDK 가 설치되지 않은 경우.
        """
        ...


@runtime_checkable
class MotionController(Protocol):
    """펜 액추에이터 제어기(Act 단계).

    하드웨어는 에이전트의 액추에이터일 뿐이며, 좌표 판단은 전부 에이전트가 한다.
    이번 범위에서는 Mock 구현까지만 두고 실제 시리얼 연동은 하지 않는다.
    """

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        """지정한 mm 좌표로 펜을 이동한다.

        :param x_mm: 목표 x(mm, 좌상단 원점, 오른쪽 +).
        :param y_mm: 목표 y(mm, 아래쪽 +).
        :returns: 이동 성공 여부.
        :raises docagent.errors.ToolExecutionError: 통신 실패 등 복구 불가한 오류.
        """
        ...

    def home(self) -> bool:
        """원점(0, 0)으로 복귀한다.

        :returns: 복귀 성공 여부.
        :raises docagent.errors.ToolExecutionError: 통신 실패 등 복구 불가한 오류.
        """
        ...

    def position(self) -> Point:
        """현재 펜 위치를 mm 좌표로 반환한다.

        :returns: 현재 위치 :class:`~docagent.contracts.Point`.
        """
        ...


@runtime_checkable
class SpeechIO(Protocol):
    """음성 입출력(TTS/STT). 이번 범위에서는 Mock 구현까지만 둔다."""

    def speak(self, text: str) -> None:
        """한국어 문장을 음성으로 출력한다.

        :param text: 낭독할 한국어 문장.
        :returns: ``None``.
        """
        ...

    def listen(self, timeout_s: float = 10.0) -> str:
        """사용자 발화를 받아 텍스트로 반환한다.

        :param timeout_s: 최대 대기 시간(초).
        :returns: 인식된 발화 문자열. 침묵·인식 실패 시 빈 문자열.
        """
        ...


@runtime_checkable
class Clock(Protocol):
    """시각 공급자.

    테스트 결정성을 위해 시간도 주입식으로 다룬다. 구현체가
    ``datetime.now()`` 를 직접 호출하는 것을 금지하고, 항상 이 인터페이스를 받는다.
    """

    def now_iso(self) -> str:
        """현재 시각을 ISO 8601 문자열로 반환한다.

        :returns: 예 ``"2026-09-09T13:45:00+09:00"``.
        """
        ...
