"""Presidio 선택적 어댑터 — 설치되어 있을 때만 동작하는 보강 탐지기.

.. important::
   이 모듈은 **최상단에서 presidio 를 import 하지 않는다.** 패키지가 없어도
   ``import docagent.pii.presidio_adapter`` 자체는 반드시 성공해야 하며,
   실제 사용 시점(생성자·메서드)에 지연 import 하고 :class:`ImportError` 를
   :class:`~docagent.errors.AdapterUnavailable` 로 감싸 한국어 설치 안내를 돌려준다.

기본 탐지기(:class:`~docagent.pii.detectors.RegexPiiDetector`)는 한국 공공·금융
서식에 맞춘 규칙 기반이며 이번 범위의 표준이다. presidio 는 영문 개체명 등
규칙으로 잡기 어려운 유형을 **추가로** 잡고 싶을 때 쓰는 보강 수단이다.
따라서 :class:`CompositePiiDetector` 로 기본 탐지기와 합쳐 쓰는 것을 권장한다
(교체가 아니라 합집합 — 안전 쪽으로만 움직인다).
"""

from __future__ import annotations

from typing import Any, Sequence

from docagent.contracts import PiiSpan
from docagent.errors import AdapterUnavailable
from docagent.pii.detectors import PiiDetectionError, RegexPiiDetector

__all__ = ["PresidioDetector", "CompositePiiDetector", "PRESIDIO_ENTITY_MAP"]

#: presidio 엔티티 이름 → 이 프로젝트의 ``pii_type`` 식별자.
PRESIDIO_ENTITY_MAP: dict[str, str] = {
    "PERSON": "name",
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone_mobile",
    "CREDIT_CARD": "card",
    "IBAN_CODE": "account",
    "LOCATION": "address",
    "DATE_TIME": "birthdate",
    "KR_RRN": "rrn",
    "US_PASSPORT": "passport",
}


class PresidioDetector:
    """Microsoft Presidio 기반 탐지기(선택적 어댑터).

    :param language: 분석 언어 코드(예: ``"ko"``, ``"en"``).
    :param entities: 탐지할 엔티티 목록. ``None`` 이면 presidio 기본값.
    :param min_confidence: 이 값 미만의 탐지 결과는 버린다(0.0~1.0).
    :raises AdapterUnavailable: ``presidio-analyzer`` 가 설치되지 않은 경우.
    :raises ValueError: ``min_confidence`` 가 0.0~1.0 범위를 벗어난 경우.
    """

    def __init__(
        self,
        *,
        language: str = "ko",
        entities: Sequence[str] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence 는 0.0~1.0 이어야 합니다: {min_confidence}"
            )
        self._language = language
        self._entities = None if entities is None else list(entities)
        self._min_confidence = min_confidence
        self._analyzer: Any = self._build_analyzer()

    @staticmethod
    def _build_analyzer() -> Any:
        """presidio 분석기를 지연 import 로 생성한다.

        :returns: ``presidio_analyzer.AnalyzerEngine`` 인스턴스.
        :raises AdapterUnavailable: 패키지가 없거나 초기화에 실패한 경우.
        """
        try:
            from presidio_analyzer import AnalyzerEngine  # noqa: PLC0415 — 지연 import
        except ImportError as exc:
            raise AdapterUnavailable(
                "presidio-analyzer",
                feature="Presidio 기반 개인정보 보강 탐지",
                extra="pii",
            ) from exc
        try:
            return AnalyzerEngine()
        except Exception as exc:  # noqa: BLE001 — 초기화 실패도 어댑터 불가로 본다.
            raise AdapterUnavailable(
                "presidio-analyzer",
                feature=f"Presidio 분석기 초기화 실패({exc})",
                extra="pii",
            ) from exc

    def detect(self, text: str) -> tuple[PiiSpan, ...]:
        """presidio 분석 결과를 :class:`PiiSpan` 으로 변환해 반환한다.

        :param text: 검사 대상 원문.
        :returns: 시작 위치 오름차순의 :class:`PiiSpan` 튜플(원문 값 미포함).
        :raises PiiDetectionError: 분석 호출이 실패한 경우(조용한 실패 금지).
        """
        if not isinstance(text, str):
            raise PiiDetectionError(
                f"탐지 대상은 문자열이어야 합니다: {type(text).__name__}"
            )
        if not text:
            return ()
        try:
            results = self._analyzer.analyze(
                text=text, language=self._language, entities=self._entities
            )
        except Exception as exc:  # noqa: BLE001 — 도메인 예외로 감싸 올린다.
            raise PiiDetectionError(
                f"Presidio 분석에 실패했습니다: {exc}"
            ) from exc

        spans: list[PiiSpan] = []
        for item in results:
            score = float(getattr(item, "score", 0.0))
            if score < self._min_confidence:
                continue
            start = int(item.start)
            end = int(item.end)
            if end <= start:
                continue
            entity = str(getattr(item, "entity_type", "unknown"))
            spans.append(
                PiiSpan(
                    start=start,
                    end=end,
                    pii_type=PRESIDIO_ENTITY_MAP.get(entity, entity.lower()),
                    raw_len=end - start,
                    confidence=min(max(score, 0.0), 1.0),
                )
            )
        spans.sort(key=lambda s: (s.start, s.end))
        return tuple(spans)

    def __repr__(self) -> str:
        """설정만 노출한다."""
        return (
            f"<PresidioDetector 언어={self._language}, "
            f"최소신뢰도={self._min_confidence}>"
        )


class CompositePiiDetector:
    """여러 탐지기의 결과를 **합집합**으로 모으는 탐지기.

    안전 쪽으로만 움직이도록, 어느 한 탐지기가 잡은 구간은 모두 살린다.
    겹치는 구간은 긴 쪽·신뢰도 높은 쪽을 남긴다.

    :param detectors: ``detect(text) -> Sequence[PiiSpan]`` 를 제공하는 객체들.
    :raises ValueError: 탐지기를 하나도 넘기지 않은 경우.
    """

    def __init__(self, *detectors: Any) -> None:
        if not detectors:
            raise ValueError("CompositePiiDetector 에는 탐지기가 최소 1개 필요합니다.")
        self._detectors: tuple[Any, ...] = tuple(detectors)

    @classmethod
    def with_presidio(cls, **kwargs: Any) -> "CompositePiiDetector":
        """기본 정규식 탐지기 + Presidio 조합을 만든다.

        :param kwargs: :class:`PresidioDetector` 생성자 인자.
        :returns: :class:`CompositePiiDetector`.
        :raises AdapterUnavailable: presidio 가 설치되지 않은 경우.
        """
        return cls(RegexPiiDetector(), PresidioDetector(**kwargs))

    def detect(self, text: str) -> tuple[PiiSpan, ...]:
        """모든 탐지기를 실행하고 겹침을 정리한 결과를 반환한다.

        :param text: 검사 대상 원문.
        :returns: 시작 위치 오름차순, 서로 겹치지 않는 :class:`PiiSpan` 튜플.
        :raises PiiDetectionError: 하위 탐지기 중 하나라도 실패한 경우.
        """
        collected: list[PiiSpan] = []
        for detector in self._detectors:
            collected.extend(detector.detect(text))

        collected.sort(key=lambda s: (-(s.end - s.start), -s.confidence, s.start))
        accepted: list[PiiSpan] = []
        for span in collected:
            if any(
                span.start < other.end and other.start < span.end for other in accepted
            ):
                continue
            accepted.append(span)
        accepted.sort(key=lambda s: (s.start, s.end))
        return tuple(accepted)

    def __repr__(self) -> str:
        """하위 탐지기 개수만 노출한다."""
        return f"<CompositePiiDetector 탐지기 {len(self._detectors)}개>"
