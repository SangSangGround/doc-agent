"""개인정보 마스킹 — 탐지된 구간을 안전한 대체 문자열로 바꾼다.

세 가지 전략을 제공한다.

============== ============================== ==========================================
전략           결과 예시                       용도
============== ============================== ==========================================
``FULL``       ``<REDACTED>``                  외부 전송 기본값. 원문 흔적이 남지 않는다.
``PARTIAL``    ``900101-1******``              **로컬 화면·음성 확인 전용.** 외부 전송 금지.
``TYPE_TOKEN`` ``[PII:RRN_1]``                 LLM 이 항목을 지칭해야 할 때. 복원 가능.
============== ============================== ==========================================

.. danger::
   ``PARTIAL`` 은 원문 일부를 그대로 남긴다. 이용자 본인에게 "9001로 시작하는
   주민등록번호가 맞습니까?" 라고 되묻는 **로컬 확인 용도**로만 쓰고,
   LLM·네트워크·영구 로그로는 절대 내보내지 않는다.
   :class:`docagent.pii.gate.LlmEgressGate` 는 이 전략을 거부한다.

``TYPE_TOKEN`` 복원 맵(:class:`RestoreMap`)은 **프로세스 메모리에만** 존재한다.
직렬화(pickle·json)와 로깅을 막기 위해 ``__repr__`` / ``__str__`` 은 값을 감추고
``__reduce__`` / ``__getstate__`` 는 :class:`TypeError` 를 던진다.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Iterator, Mapping, Sequence

from docagent.contracts import PiiSpan, SanitizedText
from docagent.pii.patterns import TOKEN_NAMES

__all__ = [
    "MaskStrategy",
    "FULL_MASK_TOKEN",
    "MASK_CHAR",
    "PARTIAL_KEEP",
    "DEFAULT_PARTIAL_KEEP",
    "RestoreMap",
    "mask",
    "mask_with_restore",
    "token_for",
]


class MaskStrategy(Enum):
    """마스킹 전략."""

    #: 전체를 :data:`FULL_MASK_TOKEN` 으로 치환한다(기본값, 가장 안전).
    FULL = "full"
    #: 앞부분 일부만 남기고 나머지를 ``*`` 로 가린다. **로컬 확인 전용.**
    PARTIAL = "partial"
    #: ``[PII:RRN_1]`` 형태의 유형 토큰으로 치환한다(복원 가능).
    TYPE_TOKEN = "type_token"


#: ``FULL`` 전략의 대체 문자열.
FULL_MASK_TOKEN: Final[str] = "<REDACTED>"
#: ``PARTIAL`` 전략에서 가림에 쓰는 문자.
MASK_CHAR: Final[str] = "*"

#: 유형별 ``PARTIAL`` 노출 허용 길이(앞에서부터 남길 문자 수).
PARTIAL_KEEP: Final[Mapping[str, int]] = {
    "rrn": 8,
    "frn": 8,
    "phone_mobile": 4,
    "phone_landline": 3,
    "card": 4,
    "account": 4,
    "passport": 1,
    "driver_license": 2,
    "business_number": 3,
    "health_insurance": 2,
    "address": 6,
    "birthdate": 4,
    "email": 2,
    "name": 1,
}
#: :data:`PARTIAL_KEEP` 에 없는 유형의 기본 노출 길이.
DEFAULT_PARTIAL_KEEP: Final[int] = 2


def token_for(pii_type: str, index: int) -> str:
    """``TYPE_TOKEN`` 전략의 토큰 문자열을 만든다.

    :param pii_type: 유형 식별자(예: ``"rrn"``).
    :param index: 같은 유형 안에서의 1부터 시작하는 일련번호.
    :returns: 예 ``"[PII:RRN_1]"``.
    :raises ValueError: ``index`` 가 1 미만인 경우.
    """
    if index < 1:
        raise ValueError(f"토큰 일련번호는 1 이상이어야 합니다: {index}")
    name = TOKEN_NAMES.get(pii_type, pii_type.upper())
    return f"[PII:{name}_{index}]"


class RestoreMap:
    """``TYPE_TOKEN`` 토큰 → 원문 값 대응표. **프로세스 메모리 전용.**

    이 객체는 개인정보 원문을 들고 있는 유일한 지점이므로 다음을 강제한다.

    * ``__repr__`` / ``__str__`` 은 항목 수만 노출하고 값을 보여주지 않는다.
    * ``__reduce__`` / ``__getstate__`` 가 :class:`TypeError` 를 던져
      pickle·copy·json 직렬화를 원천 차단한다.
    * 값을 한꺼번에 꺼내는 메서드를 제공하지 않는다. :meth:`restore` 로
      토큰이 들어 있는 문자열을 되돌리는 것만 가능하다.

    :param mapping: ``{토큰: 원문 값}`` 매핑.
    """

    __slots__ = ("_values",)

    def __init__(self, mapping: Mapping[str, str] | None = None) -> None:
        self._values: dict[str, str] = dict(mapping or {})

    def restore(self, text: str) -> str:
        """토큰이 들어 있는 문자열을 원문 값으로 되돌린다.

        :param text: 토큰이 포함된 문자열.
        :returns: 토큰이 원문 값으로 치환된 문자열.
            **이 결과는 다시 외부로 나갈 수 없다.**
        """
        restored = text
        # 긴 토큰부터 치환해 접두 충돌(예: _1 과 _10)을 피한다.
        for token in sorted(self._values, key=len, reverse=True):
            restored = restored.replace(token, self._values[token])
        return restored

    def tokens(self) -> tuple[str, ...]:
        """등록된 토큰 목록(값 미포함)을 정렬해 반환한다."""
        return tuple(sorted(self._values))

    def __contains__(self, token: object) -> bool:
        """토큰 존재 여부."""
        return token in self._values

    def __len__(self) -> int:
        """등록된 항목 수."""
        return len(self._values)

    def __iter__(self) -> Iterator[str]:
        """토큰만 순회한다(값은 노출하지 않는다)."""
        return iter(sorted(self._values))

    def __repr__(self) -> str:
        """값을 노출하지 않는 표현."""
        return f"<RestoreMap 항목 {len(self._values)}개 — 값은 표시하지 않습니다>"

    __str__ = __repr__

    def __reduce__(self) -> Any:
        """직렬화를 원천 차단한다."""
        raise TypeError(
            "RestoreMap 은 직렬화할 수 없습니다. 복원 맵은 프로세스 메모리에만 존재해야 합니다."
        )

    def __getstate__(self) -> Any:
        """직렬화를 원천 차단한다."""
        raise TypeError(
            "RestoreMap 은 직렬화할 수 없습니다. 복원 맵은 프로세스 메모리에만 존재해야 합니다."
        )


def _validate_spans(spans: Sequence[PiiSpan], text_len: int) -> tuple[PiiSpan, ...]:
    """구간이 텍스트 범위 안에 있고 서로 겹치지 않는지 확인한 뒤 정렬해 돌려준다.

    :param spans: 검사할 구간 목록.
    :param text_len: 원문 길이.
    :returns: 시작 위치 오름차순으로 정렬된 구간 튜플.
    :raises ValueError: 범위를 벗어나거나 구간이 겹치는 경우.
    """
    ordered = sorted(spans, key=lambda s: (s.start, s.end))
    previous_end = 0
    for span in ordered:
        if span.end > text_len:
            raise ValueError(
                f"마스킹 구간이 텍스트 범위를 벗어났습니다: end={span.end}, 길이={text_len}"
            )
        if span.start < previous_end:
            raise ValueError(
                "마스킹 구간이 서로 겹칩니다. detect_pii 는 겹치지 않는 구간만 반환해야 합니다: "
                f"start={span.start}, 이전 end={previous_end}"
            )
        previous_end = span.end
    return tuple(ordered)


def _replacement(
    raw: str,
    pii_type: str,
    strategy: MaskStrategy,
    counters: dict[str, int],
) -> str:
    """구간 하나에 대한 대체 문자열을 만든다.

    :param raw: 원문 조각(이 함수 밖으로 나가지 않는다).
    :param pii_type: 유형 식별자.
    :param strategy: 마스킹 전략.
    :param counters: 유형별 토큰 일련번호 누적 dict(호출 간 공유).
    :returns: 대체 문자열.
    :raises ValueError: 정의되지 않은 전략인 경우.
    """
    if strategy is MaskStrategy.FULL:
        return FULL_MASK_TOKEN
    if strategy is MaskStrategy.PARTIAL:
        keep = min(PARTIAL_KEEP.get(pii_type, DEFAULT_PARTIAL_KEEP), len(raw))
        return raw[:keep] + MASK_CHAR * (len(raw) - keep)
    if strategy is MaskStrategy.TYPE_TOKEN:
        counters[pii_type] = counters.get(pii_type, 0) + 1
        return token_for(pii_type, counters[pii_type])
    raise ValueError(f"정의되지 않은 마스킹 전략입니다: {strategy!r}")


def mask_with_restore(
    text: str,
    spans: Sequence[PiiSpan],
    strategy: MaskStrategy = MaskStrategy.TYPE_TOKEN,
) -> tuple[SanitizedText, RestoreMap]:
    """마스킹 결과와 복원 맵을 함께 반환한다.

    복원 맵은 ``TYPE_TOKEN`` 전략에서만 내용을 갖는다. ``FULL`` 은 모든 구간이
    같은 문자열로 바뀌므로 복원이 원리적으로 불가능하고, ``PARTIAL`` 은
    복원 대상이 아니다. 두 경우 모두 **빈** :class:`RestoreMap` 을 돌려준다.

    :param text: 원문 텍스트.
    :param spans: :func:`docagent.pii.detectors.detect_pii` 가 만든 구간 목록
        (원문 좌표, 서로 겹치지 않아야 한다).
    :param strategy: 마스킹 전략.
    :returns: ``(SanitizedText, RestoreMap)``.
        :attr:`SanitizedText.spans` 는 **원문 좌표**를 그대로 유지한다.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우.
    :raises ValueError: 구간이 범위를 벗어나거나 겹치는 경우, 전략이 정의되지 않은 경우.
    """
    if not isinstance(text, str):
        raise TypeError(f"마스킹 대상은 문자열이어야 합니다: {type(text).__name__}")

    ordered = _validate_spans(spans, len(text))
    counters: dict[str, int] = {}
    restore: dict[str, str] = {}
    pieces: list[str] = []
    cursor = 0

    for span in ordered:
        pieces.append(text[cursor : span.start])
        raw = text[span.start : span.end]
        replacement = _replacement(raw, span.pii_type, strategy, counters)
        if strategy is MaskStrategy.TYPE_TOKEN:
            restore[replacement] = raw
        pieces.append(replacement)
        cursor = span.end
    pieces.append(text[cursor:])

    sanitized = SanitizedText(text="".join(pieces), spans=ordered, blocked=False)
    return (sanitized, RestoreMap(restore))


def mask(
    text: str,
    spans: Sequence[PiiSpan],
    strategy: MaskStrategy = MaskStrategy.FULL,
) -> SanitizedText:
    """탐지된 구간을 전략에 따라 마스킹한다.

    복원 맵이 필요하면 :func:`mask_with_restore` 를 쓴다. 이 함수는 복원 맵을
    **만들지 않고 버리므로**, 반환값에는 원문이 남지 않는다.

    :param text: 원문 텍스트.
    :param spans: 원문 좌표 기준 구간 목록(서로 겹치지 않아야 한다).
    :param strategy: 마스킹 전략. 기본값은 가장 안전한 :attr:`MaskStrategy.FULL`.
    :returns: :class:`~docagent.contracts.SanitizedText`.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우.
    :raises ValueError: 구간이 범위를 벗어나거나 겹치는 경우.
    """
    sanitized, _restore = mask_with_restore(text, spans, strategy)
    return sanitized
