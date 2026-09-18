"""개인정보 탐지기 — 정규화된 텍스트에서 :class:`~docagent.contracts.PiiSpan` 을 만든다.

:mod:`docagent.pii.patterns` 의 규칙 목록을 **정규화 사본마다** 순회해 후보를 모으고,
거짓양성 억제 구간을 적용한 뒤 겹치는 후보를 우선순위로 병합한다.
사본은 :func:`docagent.pii.patterns.normalize_variants` 가 만든다 — 기본 정규화 외에
숫자 사이 구분자를 제거한 사본과 OCR 동형이의 문자를 접은 사본이 포함되어,
단일 정규식으로는 무너지던 회피 변형까지 같은 규칙으로 잡는다.

반환되는 :class:`~docagent.contracts.PiiSpan` 은 **원문(정규화 이전) 인덱스**를
가지며 원문 값을 담지 않는다. 이는 계약이자 안전 KPI의 전제다.

조용한 실패 금지
----------------
정규식 실행 중 예기치 못한 오류가 나면 빈 결과를 돌려주지 않고
:class:`PiiDetectionError` 로 감싸 올린다. 게이트(:mod:`docagent.pii.gate`)는
strict 모드에서 이 예외를 **차단**으로 해석한다(fail-closed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from docagent.contracts import PiiSpan
from docagent.errors import DocAgentError
from docagent.pii.patterns import (
    DENY_EXEMPT_TYPES,
    DENY_PATTERNS,
    PATTERN_SPECS,
    PatternSpec,
    normalize_variants,
)

__all__ = [
    "PiiDetectionError",
    "RegexPiiDetector",
    "detect_pii",
    "contains_pii",
    "summarize_types",
]


class PiiDetectionError(DocAgentError):
    """개인정보 탐지 과정 자체가 실패한 경우.

    탐지 실패는 "개인정보가 없다"는 뜻이 **아니다.** 호출자는 이 예외를
    반드시 차단으로 처리해야 한다(fail-closed).

    :param message: 한국어 오류 메시지.
    :param pii_type: 실패한 규칙의 유형 식별자. 알 수 없으면 ``None``.
    """

    def __init__(
        self, message: str = "개인정보 탐지에 실패했습니다.", *, pii_type: str | None = None
    ) -> None:
        self.pii_type = pii_type
        if pii_type:
            message = f"{message} (규칙: {pii_type})"
        super().__init__(message)


@dataclass(frozen=True)
class _Candidate:
    """병합 전 탐지 후보(내부 전용).

    :param start: 원문 시작 인덱스(포함).
    :param end: 원문 끝 인덱스(제외).
    :param pii_type: 유형 식별자.
    :param priority: 병합 우선순위(클수록 우선).
    :param confidence: 탐지 신뢰도.
    """

    start: int
    end: int
    pii_type: str
    priority: int
    confidence: float

    @property
    def length(self) -> int:
        """원문 기준 구간 길이."""
        return self.end - self.start


def _deny_regions(normalized: str) -> tuple[tuple[int, int], ...]:
    """거짓양성 억제 구간을 정규화 좌표로 계산한다.

    :param normalized: 정규화된 텍스트.
    :returns: ``(start, end)`` 구간 튜플. 정규화 좌표 기준.
    """
    regions: list[tuple[int, int]] = []
    for pattern in DENY_PATTERNS:
        for match in pattern.finditer(normalized):
            if match.end() > match.start():
                regions.append((match.start(), match.end()))
    return tuple(regions)


def _overlaps_any(start: int, end: int, regions: Iterable[tuple[int, int]]) -> bool:
    """``[start, end)`` 가 ``regions`` 중 하나와 겹치면 True."""
    return any(start < region_end and region_start < end for region_start, region_end in regions)


def _collect_candidates(
    text: str,
    specs: Sequence[PatternSpec],
) -> list[_Candidate]:
    """규칙을 모두 적용해 후보 목록을 만든다(병합 전).

    :param text: 원문 텍스트.
    :param specs: 적용할 규칙 목록.
    :returns: :class:`_Candidate` 목록(원문 좌표).
    :raises PiiDetectionError: 규칙 실행 중 오류가 발생한 경우.
    """
    candidates: list[_Candidate] = []
    seen: set[tuple[int, int, str]] = set()

    # 정규화 사본을 전부 훑는다. 사본은 회피 변형(숫자 사이 공백·줄바꿈, OCR
    # 동형이의 문자)을 흡수하며 모두 같은 원문 오프셋을 가리키므로, 어느 사본에서
    # 잡히든 보고되는 좌표는 원문 기준이다.
    for normalized in normalize_variants(text):
        deny = _deny_regions(normalized.text)
        for spec in specs:
            try:
                for match in spec.pattern.finditer(normalized.text):
                    if spec.validator is not None and not spec.validator(match):
                        continue
                    norm_start, norm_end = spec.value_span(match)
                    if norm_end <= norm_start:
                        continue
                    suppressible = (
                        not spec.context_anchored
                        and spec.pii_type not in DENY_EXEMPT_TYPES
                    )
                    if suppressible and _overlaps_any(norm_start, norm_end, deny):
                        continue
                    src_start, src_end = normalized.to_source_span(norm_start, norm_end)
                    if src_end <= src_start:
                        continue
                    key = (src_start, src_end, spec.pii_type)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        _Candidate(
                            start=src_start,
                            end=src_end,
                            pii_type=spec.pii_type,
                            priority=spec.priority,
                            confidence=spec.confidence,
                        )
                    )
            except PiiDetectionError:
                raise
            except (re.error, ValueError, TypeError, IndexError, KeyError) as exc:
                raise PiiDetectionError(
                    f"탐지 규칙 실행 중 오류가 발생했습니다: {exc}",
                    pii_type=spec.pii_type,
                ) from exc
    return candidates


def _merge(candidates: Sequence[_Candidate]) -> tuple[PiiSpan, ...]:
    """겹치는 후보를 우선순위·길이·신뢰도 순으로 병합한다.

    :param candidates: 병합할 후보 목록.
    :returns: 시작 위치 오름차순으로 정렬된 :class:`PiiSpan` 튜플.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (-c.priority, -c.length, -c.confidence, c.start),
    )
    accepted: list[_Candidate] = []
    for candidate in ordered:
        if any(
            candidate.start < other.end and other.start < candidate.end
            for other in accepted
        ):
            continue
        accepted.append(candidate)

    accepted.sort(key=lambda c: (c.start, c.end))
    return tuple(
        PiiSpan(
            start=item.start,
            end=item.end,
            pii_type=item.pii_type,
            raw_len=item.length,
            confidence=item.confidence,
        )
        for item in accepted
    )


def detect_pii(
    text: str,
    *,
    specs: Sequence[PatternSpec] = PATTERN_SPECS,
) -> tuple[PiiSpan, ...]:
    """텍스트에서 개인정보 구간을 탐지한다.

    :param text: 검사 대상 원문. 빈 문자열이면 빈 튜플을 반환한다.
    :param specs: 적용할 규칙 목록. 기본값은
        :data:`docagent.pii.patterns.PATTERN_SPECS`.
    :returns: 시작 위치 오름차순의 :class:`~docagent.contracts.PiiSpan` 튜플.
        구간끼리 절대 겹치지 않는다. **원문 값은 담기지 않는다.**
    :raises PiiDetectionError: 입력 타입이 잘못되었거나 규칙 실행이 실패한 경우.
    """
    if not isinstance(text, str):
        raise PiiDetectionError(
            f"탐지 대상은 문자열이어야 합니다: {type(text).__name__}"
        )
    if not text:
        return ()
    return _merge(_collect_candidates(text, specs))


def contains_pii(text: str) -> bool:
    """개인정보가 하나라도 탐지되면 True.

    :param text: 검사 대상 원문.
    :returns: 탐지 여부.
    :raises PiiDetectionError: 탐지 자체가 실패한 경우.
    """
    return len(detect_pii(text)) > 0


def summarize_types(spans: Sequence[PiiSpan]) -> tuple[str, ...]:
    """탐지 구간 목록에서 중복 없는 유형 목록을 정렬해 반환한다.

    :param spans: 탐지 구간 목록.
    :returns: 사전순 정렬된 유형 식별자 튜플.
    """
    return tuple(sorted({span.pii_type for span in spans}))


class RegexPiiDetector:
    """정규식 규칙 기반 개인정보 탐지기(기본 구현).

    주입 가능한 객체로 두어 게이트·정책 계층이 탐지기를 교체할 수 있게 한다.
    (예: 테스트용 항상 실패 탐지기, :mod:`docagent.pii.presidio_adapter`)

    :param specs: 적용할 규칙 목록. ``None`` 이면 기본 규칙 전체.
    """

    def __init__(self, specs: Sequence[PatternSpec] | None = None) -> None:
        self._specs: tuple[PatternSpec, ...] = tuple(
            PATTERN_SPECS if specs is None else specs
        )

    @property
    def specs(self) -> tuple[PatternSpec, ...]:
        """적용 중인 규칙 목록."""
        return self._specs

    def detect(self, text: str) -> tuple[PiiSpan, ...]:
        """텍스트에서 개인정보 구간을 탐지한다.

        :param text: 검사 대상 원문.
        :returns: :class:`~docagent.contracts.PiiSpan` 튜플(원문 좌표, 값 미포함).
        :raises PiiDetectionError: 탐지가 실패한 경우.
        """
        return detect_pii(text, specs=self._specs)

    def __repr__(self) -> str:
        """규칙 개수만 노출한다(원문·패턴 상세 미노출)."""
        return f"<RegexPiiDetector 규칙 {len(self._specs)}개>"
