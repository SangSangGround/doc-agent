"""LLM 유출 차단 게이트 — 외부로 나가는 **모든** 문자열이 통과하는 단일 관문.

이 모듈은 프로젝트의 안전 KPI("개인정보 LLM 전송 0건")를 단독으로 책임진다.
설계 원칙은 **fail-closed** 다.

* 탐지기가 예외를 던지면 "개인정보가 없다"로 해석하지 않고 **차단**한다.
* 마스킹 후 재검사에서 개인정보가 하나라도 남으면 **차단**한다.
* dict/list 를 통째로 넘길 때도 **모든 문자열과 숫자**를 재귀적으로 검사한다.
  숫자는 마스킹할 방법이 없으므로 개인정보가 발견되면 즉시 차단한다.
* :attr:`~docagent.pii.masker.MaskStrategy.PARTIAL` 은 원문 일부를 남기므로
  게이트 전략으로 **허용하지 않는다**.

사용법::

    gate = LlmEgressGate()
    client = GatedLlmClient(SomeLlmAdapter(), gate=gate)
    answer = client.complete(system="문서를 쉬운 말로 설명하라", user=payload_text)

다른 모듈은 :class:`GatedLlmClient` 를 통해서만 LLM 을 사용한다.
원본 :class:`~docagent.interfaces.LlmClient` 를 직접 호출하는 코드는 계약 위반이다.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final, Mapping, Pattern

from docagent.contracts import PiiSpan, SanitizedText
from docagent.errors import DocAgentError, PiiEgressBlocked, ToolExecutionError
from docagent.pii.audit import AuditLog, count_unknown_digit_runs
from docagent.pii.detectors import RegexPiiDetector
from docagent.pii.masker import MaskStrategy, mask
from docagent.pii.patterns import DENY_PATTERNS

__all__ = [
    "LlmEgressGate",
    "GatedLlmClient",
    "build_default_gate",
    "strip_benign_numbers",
]


#: 결합 검사에서 "숫자 조각"으로 볼 문자열 형태(숫자와 구분자만으로 구성).
_NUMBER_FRAGMENT_RE: Final[Pattern[str]] = re.compile(r"[\d\s.\-]+")


def strip_benign_numbers(text: str) -> str:
    """정당한 숫자(접수번호·금액·일반 날짜·조문·수량)를 지운 사본을 만든다.

    2차 검사(:func:`~docagent.pii.audit.count_unknown_digit_runs`)를 돌리기 전에
    쓴다. 거짓양성 억제 구간(:data:`~docagent.pii.patterns.DENY_PATTERNS`)에
    해당하는 숫자까지 "유래 불명"으로 세면 정상 문서가 통째로 막히기 때문이다.

    :param text: 검사 대상 문자열(보통 마스킹이 끝난 문자열).
    :returns: 정당한 숫자 구간을 공백으로 치환한 사본.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우.
    """
    if not isinstance(text, str):
        raise TypeError(f"문자열만 받습니다: {type(text).__name__}")
    cleaned = text
    for pattern in DENY_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


class LlmEgressGate:
    """개인정보 유출 차단 게이트.

    :param detector: 탐지기. ``detect(text) -> Sequence[PiiSpan]`` 를 제공해야 한다.
        ``None`` 이면 :class:`~docagent.pii.detectors.RegexPiiDetector`.
    :param strategy: 마스킹 전략. ``FULL``(기본) 또는 ``TYPE_TOKEN`` 만 허용한다.
    :param strict: True 이면 탐지기 예외를 :class:`~docagent.errors.PiiEgressBlocked`
        로 바꿔 **차단**한다. False 이면 원래 예외를 그대로 올린다
        (어느 쪽이든 통과시키지는 않는다).
    :param audit: 감사 로그. ``None`` 이면 메모리 전용 로그를 새로 만든다.
    :raises ValueError: ``PARTIAL`` 전략을 넘긴 경우.
    """

    def __init__(
        self,
        detector: Any | None = None,
        *,
        strategy: MaskStrategy = MaskStrategy.FULL,
        strict: bool = True,
        audit: AuditLog | None = None,
    ) -> None:
        if strategy is MaskStrategy.PARTIAL:
            raise ValueError(
                "PARTIAL 전략은 원문 일부를 남기므로 외부 전송 게이트에 쓸 수 없습니다. "
                "FULL 또는 TYPE_TOKEN 을 사용하십시오."
            )
        self._detector = detector if detector is not None else RegexPiiDetector()
        self._strategy = strategy
        self._strict = strict
        self._audit = audit if audit is not None else AuditLog()

    @property
    def audit(self) -> AuditLog:
        """이 게이트가 사용하는 감사 로그."""
        return self._audit

    @property
    def strategy(self) -> MaskStrategy:
        """적용 중인 마스킹 전략."""
        return self._strategy

    @property
    def strict(self) -> bool:
        """탐지기 예외를 차단으로 해석하는지 여부."""
        return self._strict

    # ------------------------------------------------------------------
    # 내부
    # ------------------------------------------------------------------

    def _detect(self, text: str) -> tuple[PiiSpan, ...]:
        """탐지기를 호출한다. 실패는 절대 "안전"으로 해석하지 않는다.

        :param text: 검사 대상 문자열.
        :returns: 탐지 구간 튜플.
        :raises PiiEgressBlocked: strict 모드에서 탐지기가 실패한 경우.
        :raises Exception: strict 가 False 이면 원래 예외를 그대로 올린다.
        """
        try:
            spans = tuple(self._detector.detect(text))
        except Exception as exc:  # noqa: BLE001 — fail-closed 를 위해 광범위하게 잡는다.
            if not self._strict:
                raise
            raise PiiEgressBlocked(
                "개인정보 탐지기가 실패하여 안전을 보장할 수 없으므로 외부 전송을 차단했습니다."
            ) from exc
        for span in spans:
            if not isinstance(span, PiiSpan):
                raise PiiEgressBlocked(
                    "탐지기가 PiiSpan 이 아닌 값을 반환하여 외부 전송을 차단했습니다."
                )
        return spans

    def _sanitize_with_residual(
        self, text: str
    ) -> tuple[SanitizedText, tuple[PiiSpan, ...]]:
        """마스킹 결과와 **마스킹 후 잔존 탐지 결과**를 함께 반환한다.

        :param text: 원문 문자열.
        :returns: ``(SanitizedText, 잔존 구간 튜플)``.
        """
        spans = self._detect(text)
        masked = mask(text, spans, self._strategy)
        residual = self._detect(masked.text)
        return (
            SanitizedText(text=masked.text, spans=spans, blocked=bool(residual)),
            residual,
        )

    # ------------------------------------------------------------------
    # PiiGate 프로토콜
    # ------------------------------------------------------------------

    def sanitize(self, text: str) -> SanitizedText:
        """텍스트에서 개인정보를 탐지·마스킹한다.

        마스킹 후 재검사에서 개인정보가 남아 있으면 ``blocked=True`` 로 표시한다.
        (호출자는 이때 전송을 포기하고 :class:`~docagent.errors.PiiEgressBlocked`
        를 던져야 한다. :meth:`prepare` 를 쓰면 자동으로 처리된다.)

        :param text: 원문 문자열.
        :returns: :class:`~docagent.contracts.SanitizedText`.
            ``spans`` 인덱스는 **원문 기준**이며 원문 값을 담지 않는다.
        :raises TypeError: ``text`` 가 문자열이 아닌 경우.
        :raises PiiEgressBlocked: strict 모드에서 탐지기가 실패한 경우.
        """
        if not isinstance(text, str):
            raise TypeError(f"게이트는 문자열만 검사합니다: {type(text).__name__}")
        sanitized, _residual = self._sanitize_with_residual(text)
        return sanitized

    def assert_clean(self, text: str) -> None:
        """텍스트에 개인정보가 없음을 단언한다. 있으면 즉시 차단한다.

        :param text: 검사 대상(보통 이미 마스킹된 문자열).
        :returns: ``None``.
        :raises TypeError: ``text`` 가 문자열이 아닌 경우.
        :raises PiiEgressBlocked: 개인정보가 탐지된 경우.
            예외 메시지에는 유형·건수만 담기며 원문 값은 담기지 않는다.
        """
        if not isinstance(text, str):
            raise TypeError(f"게이트는 문자열만 검사합니다: {type(text).__name__}")
        spans = self._detect(text)
        if spans:
            raise PiiEgressBlocked(
                pii_types=sorted({span.pii_type for span in spans}),
                count=len(spans),
            )

    # ------------------------------------------------------------------
    # 전송 준비
    # ------------------------------------------------------------------

    def prepare(
        self, text: str, *, caller: str = "unknown", event: str = "egress"
    ) -> str:
        """외부로 내보낼 안전한 문자열을 만들고 감사 로그를 남긴다.

        마스킹이 끝난 문자열에는 :func:`~docagent.pii.audit.count_unknown_digit_runs`
        로 **탐지기와 독립된 2차 검사**를 돌린다. 이 검사는 기록용이 아니라
        **차단 조건**이다 — 탐지 규칙이 아무것도 찾지 못했더라도 유래 불명 숫자열이
        남아 있으면 전송을 막는다(allowlist 방어선, fail-closed).
        :func:`strip_benign_numbers` 로 접수번호·금액·날짜·조문·수량 같은 정당한
        숫자를 먼저 지우므로 정상 문서가 과잉 차단되지는 않는다.

        :param text: 원문 문자열.
        :param caller: 감사 로그에 남길 호출자 식별자.
        :param event: 감사 로그 이벤트 종류.
        :returns: 마스킹이 끝나고 재검사를 통과한 문자열.
        :raises PiiEgressBlocked: 마스킹 후에도 개인정보가 남은 경우,
            유래 불명 숫자열이 남은 경우, 또는 탐지기가 실패한 경우.
        """
        try:
            sanitized, residual = self._sanitize_with_residual(text)
        except PiiEgressBlocked:
            # 탐지기 실패로 차단된 경우에도 감사 흔적을 남긴다(무기록 차단 금지).
            self._audit.record(
                caller=caller,
                text=text,
                spans=(),
                blocked=True,
                residual_count=0,
                event=event,
            )
            raise
        if residual or sanitized.blocked:
            self._audit.record(
                caller=caller,
                text=text,
                spans=sanitized.spans,
                blocked=True,
                residual_count=len(residual),
                event=event,
            )
            raise PiiEgressBlocked(
                "마스킹 후에도 개인정보가 남아 외부 전송을 차단했습니다.",
                pii_types=sorted({span.pii_type for span in residual}),
                count=len(residual),
            )

        # 2차 방어선 — 탐지 규칙과 무관하게, 유래를 설명할 수 없는 긴 숫자열이
        # 남아 있으면 막는다. 기록만 하고 내보내면 "규칙이 모르는 형태의 개인정보"가
        # blocked=False 로 그대로 나가므로, 여기서 fail-closed 로 끊는다.
        screened = strip_benign_numbers(sanitized.text)
        unknown = count_unknown_digit_runs(screened)
        if unknown:
            self._audit.record(
                caller=caller,
                text=text,
                spans=sanitized.spans,
                blocked=True,
                residual_count=0,
                event=event,
            )
            raise PiiEgressBlocked(
                "유래를 확인할 수 없는 숫자열이 남아 외부 전송을 차단했습니다. "
                f"(2차 검사 {unknown}건) 개인정보일 수 있으므로 원문을 그대로 "
                "내보내지 않습니다.",
                count=unknown,
            )

        self._audit.record(
            caller=caller,
            text=text,
            spans=sanitized.spans,
            blocked=False,
            residual_count=0,
            event=event,
            egress_text=screened,
        )
        return sanitized.text

    # ------------------------------------------------------------------
    # 구조체 검사
    # ------------------------------------------------------------------

    def guard(self, payload: Any, *, caller: str = "guard") -> Any:
        """dict/list 를 재귀 순회하며 모든 문자열·숫자를 검사한 사본을 만든다.

        * ``str`` — 마스킹한다. 마스킹 후에도 개인정보가 남으면 차단한다.
        * ``int`` / ``float`` — 마스킹할 방법이 없으므로, 문자열로 바꾼 값에서
          개인정보가 탐지되면 **즉시 차단**한다(예: ``{"rrn": 9001011234567}``).
        * ``bool`` / ``None`` — 그대로 통과.
        * dict 의 **키**도 문자열이므로 함께 검사한다.

        :param payload: 검사 대상. dict / list / tuple / str / 숫자 / bool / None.
        :param caller: 감사 로그에 남길 호출자 식별자.
        :returns: 마스킹이 적용된 **새 객체**(원본은 변경하지 않는다).
        :raises PiiEgressBlocked: 마스킹 불가능한 위치에서 개인정보가 발견된 경우.
        :raises TypeError: 지원하지 않는 타입이 들어 있는 경우(조용한 통과 금지).
        """
        detected: list[PiiSpan] = []
        guarded = self._guard_value(payload, detected, path="$")
        self._audit.record(
            caller=caller,
            text=self._stringify(payload),
            spans=tuple(detected),
            blocked=False,
            residual_count=0,
            event="guard",
        )
        return guarded

    def _guard_value(self, value: Any, detected: list[PiiSpan], *, path: str) -> Any:
        """:meth:`guard` 의 재귀 본체.

        :param value: 현재 노드.
        :param detected: 탐지 구간 누적 리스트(감사용).
        :param path: 오류 메시지에 쓸 위치 표기.
        :returns: 마스킹된 새 값.
        """
        if value is None or isinstance(value, bool):
            return value

        if isinstance(value, str):
            sanitized, residual = self._sanitize_with_residual(value)
            detected.extend(sanitized.spans)
            if residual:
                raise PiiEgressBlocked(
                    f"마스킹 후에도 개인정보가 남아 외부 전송을 차단했습니다. (위치: {path})",
                    pii_types=sorted({span.pii_type for span in residual}),
                    count=len(residual),
                )
            return sanitized.text

        if isinstance(value, (int, float)):
            spans = self._detect(str(value))
            if spans:
                raise PiiEgressBlocked(
                    "숫자 값에 개인정보가 포함되어 외부 전송을 차단했습니다. "
                    f"(위치: {path}) 숫자는 마스킹할 수 없으므로 구조를 바꾸십시오.",
                    pii_types=sorted({span.pii_type for span in spans}),
                    count=len(spans),
                )
            return value

        if isinstance(value, Mapping):
            result: dict[Any, Any] = {}
            for index, (key, item) in enumerate(value.items()):
                # 위치 표기에 **원문 키를 넣지 않는다.** 예외 메시지는 도구 오류
                # 기록을 거쳐 세션 JSON 에 영구 저장될 수 있으므로, 키 이름 자체가
                # 개인정보인 경우(예: "신청인 주민등록번호 900101-...")를 대비해
                # 인덱스만 남긴다.
                safe_key = (
                    self._guard_value(key, detected, path=f"{path}.<key#{index}>")
                    if isinstance(key, str)
                    else key
                )
                result[safe_key] = self._guard_value(
                    item, detected, path=f"{path}.<key#{index}>"
                )
            # 결합 검사는 **마스킹이 끝난 결과**를 대상으로 한다. 개별 검사에서
            # 이미 가려진 값을 다시 이어 붙여 오탐을 만들지 않기 위해서다.
            self._assert_combination_clean(result, path=path)
            return result

        if isinstance(value, (list, tuple)):
            items = [
                self._guard_value(item, detected, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
            # 리스트 항목 사이에 흩어 담은 조각도 결합 검사를 받아야 한다.
            # (dict 분기에만 검사가 있으면 `{"조각": ["900101", "1234567"]}` 처럼
            #  한 단계만 감싸도 검사를 피할 수 있었다.)
            self._assert_combination_clean(items, path=path)
            return tuple(items) if isinstance(value, tuple) else items

        raise TypeError(
            f"게이트가 처리할 수 없는 타입입니다: {type(value).__name__} (위치: {path}). "
            "dict/list/tuple/str/int/float/bool/None 만 전달하십시오."
        )

    @classmethod
    def _descendant_number_fragments(cls, node: Any) -> list[str]:
        """노드 아래의 **숫자 조각** 스칼라를 깊이 우선 순서로 모은다.

        결합 공격의 실체는 "번호를 여러 칸에 나눠 담는 것"이므로, 서브트리
        검사는 숫자 조각만 대상으로 한다. 산문(약관 본문·제목)까지 이어 붙이면
        서로 무관한 문장 사이에서 전화번호 형태가 우연히 만들어져 정상 문서가
        통째로 막힌다.

        숫자 조각의 정의: 숫자와 구분자(공백·하이픈·점)로만 이루어졌고
        숫자가 2개 이상인 문자열, 또는 정수·실수 값.

        :param node: 마스킹이 끝난 노드(dict / list / tuple / 스칼라).
        :returns: 숫자 조각 문자열 목록(등장 순서).
        """
        if node is None or isinstance(node, bool):
            return []
        if isinstance(node, (int, float)):
            text = str(node)
            return [text] if sum(ch.isdigit() for ch in text) >= 2 else []
        if isinstance(node, str):
            if not _NUMBER_FRAGMENT_RE.fullmatch(node):
                return []
            return [node] if sum(ch.isdigit() for ch in node) >= 2 else []
        if isinstance(node, Mapping):
            collected: list[str] = []
            for item in node.values():
                collected.extend(cls._descendant_number_fragments(item))
            return collected
        if isinstance(node, (list, tuple)):
            collected = []
            for item in node:
                collected.extend(cls._descendant_number_fragments(item))
            return collected
        return []

    def _assert_split_digits_clean(self, node: Any, *, path: str) -> None:
        """서브트리 전체에 흩어진 **숫자 조각**을 이어 붙여 검사한다.

        형제 스칼라만 보는 :meth:`_assert_combination_clean` 은 조각을 리스트
        항목이나 하위 dict 로 한 단계만 감싸도 우회된다. 이 검사는 자손을 전부
        훑으므로 ``{"조각": ["900101", "1234567"]}`` 같은 형태도 잡는다.

        :param node: 검사할 노드(마스킹이 끝난 결과).
        :param path: 오류 메시지에 쓸 위치 표기(원문 키를 담지 않는다).
        :returns: ``None``.
        :raises PiiEgressBlocked: 이어 붙였을 때 개인정보가 성립하는 경우.
        """
        fragments = self._descendant_number_fragments(node)
        if len(fragments) < 2:
            return
        digits = "".join(ch for ch in "".join(fragments) if ch.isdigit())
        if not digits:
            return
        spans = self._detect(digits)
        if spans:
            raise PiiEgressBlocked(
                "여러 항목에 나눠 담은 숫자를 합치면 개인정보가 되어 외부 전송을 "
                f"차단했습니다. (위치: {path}) 조각으로 쪼개도 결합 시 식별이 "
                "가능하면 같은 위험입니다.",
                pii_types=sorted({span.pii_type for span in spans}),
                count=len(spans),
            )

    def _assert_combination_clean(self, node: Any, *, path: str) -> None:
        """노드의 스칼라 값들을 이어 붙여 **결합 시 성립하는 개인정보**를 잡는다.

        신청서 데이터는 본래 필드로 쪼개져 있어서, 주민등록번호 앞자리/뒷자리처럼
        조각으로 나누면 각 조각은 무해해 보이고 개별 검사를 모두 통과한다.
        그래서 같은 레벨의 스칼라를 (a) 순서대로 이어 붙인 사본과 (b) 숫자만 이어
        붙인 사본에 대해 한 번 더 탐지한다. 한 단계 아래로 흩어 담은 숫자 조각은
        :meth:`_assert_split_digits_clean` 이 이어서 검사한다.

        :param node: 검사할 노드(마스킹이 끝난 결과). dict / list / tuple.
        :param path: 오류 메시지에 쓸 위치 표기(원문 키를 담지 않는다).
        :returns: ``None``.
        :raises PiiEgressBlocked: 결합했을 때 개인정보가 성립하는 경우.
        """
        self._assert_split_digits_clean(node, path=path)
        values = node.values() if isinstance(node, Mapping) else node
        parts = [
            str(item)
            for item in values
            if isinstance(item, (str, int, float)) and not isinstance(item, bool)
        ]
        if len(parts) < 2:
            return
        joined = "".join(parts)
        digits = "".join(ch for ch in joined if ch.isdigit())
        for candidate in (joined, digits):
            if not candidate:
                continue
            spans = self._detect(candidate)
            if spans:
                raise PiiEgressBlocked(
                    "여러 항목을 합치면 개인정보가 되어 외부 전송을 차단했습니다. "
                    f"(위치: {path}) 항목을 쪼개도 결합 시 식별이 가능하면 같은 위험입니다.",
                    pii_types=sorted({span.pii_type for span in spans}),
                    count=len(spans),
                )

    @staticmethod
    def _stringify(payload: Any) -> str:
        """감사 로그의 길이·해시 계산용 문자열을 만든다(저장되지 않는다).

        :param payload: 대상 객체.
        :returns: JSON 문자열. 직렬화 불가 값은 ``str()`` 로 대체한다.
        """
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(payload)

    def __repr__(self) -> str:
        """설정만 노출한다(원문·탐지 결과 미노출)."""
        return (
            f"<LlmEgressGate 전략={self._strategy.value}, strict={self._strict}, "
            f"탐지기={self._detector!r}>"
        )


class GatedLlmClient:
    """임의의 :class:`~docagent.interfaces.LlmClient` 를 감싸는 안전 데코레이터.

    ``complete()`` 호출 **직전에** 게이트를 강제 적용하고 감사 로그를 남긴다.
    하위 클라이언트는 마스킹된 문자열만 받으므로, 어댑터 구현이 무엇이든
    원문 개인정보가 네트워크로 나갈 수 없다.

    :param inner: 감쌀 실제 LLM 클라이언트.
        ``complete(system, user, max_tokens) -> str`` 를 제공해야 한다.
    :param gate: 사용할 게이트. ``None`` 이면 기본 :class:`LlmEgressGate`.
    :param name: 감사 로그에 남길 이름. ``None`` 이면 하위 클래스 이름.
    :raises TypeError: ``inner`` 에 ``complete`` 가 없는 경우.
    """

    def __init__(
        self,
        inner: Any,
        *,
        gate: LlmEgressGate | None = None,
        name: str | None = None,
    ) -> None:
        if not callable(getattr(inner, "complete", None)):
            raise TypeError(
                "GatedLlmClient 는 complete(system, user, max_tokens) 를 가진 "
                f"클라이언트만 감쌀 수 있습니다: {type(inner).__name__}"
            )
        self._inner = inner
        self._gate = gate if gate is not None else LlmEgressGate()
        self._name = name if name is not None else type(inner).__name__

    @property
    def gate(self) -> LlmEgressGate:
        """적용 중인 게이트."""
        return self._gate

    @property
    def inner(self) -> Any:
        """감싸고 있는 실제 클라이언트."""
        return self._inner

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """게이트를 통과시킨 프롬프트로 하위 클라이언트를 호출한다.

        :param system: 시스템 프롬프트 원문.
        :param user: 사용자 프롬프트 원문.
        :param max_tokens: 생성 최대 토큰 수(1 이상).
        :returns: 모델 응답 문자열.
        :raises ValueError: ``max_tokens`` 가 1 미만인 경우.
        :raises PiiEgressBlocked: 프롬프트에서 개인정보가 발견되어 차단된 경우.
        :raises ToolExecutionError: 하위 클라이언트 호출이 실패한 경우.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens 는 1 이상이어야 합니다: {max_tokens}")

        caller = f"GatedLlmClient({self._name}).complete"
        safe_system = self._gate.prepare(system, caller=caller)
        safe_user = self._gate.prepare(user, caller=caller)

        try:
            answer = self._inner.complete(safe_system, safe_user, max_tokens)
        except DocAgentError:
            raise
        except Exception as exc:  # noqa: BLE001 — 어댑터 예외를 도메인 예외로 감싼다.
            raise ToolExecutionError(
                f"LLM 호출에 실패했습니다: {exc}", tool_name=self._name
            ) from exc

        if not isinstance(answer, str):
            raise ToolExecutionError(
                f"LLM 응답이 문자열이 아닙니다: {type(answer).__name__}",
                tool_name=self._name,
            )

        self._gate.audit.record(
            caller=caller,
            text=f"{safe_system}\n{safe_user}",
            spans=(),
            blocked=False,
            residual_count=0,
            event="llm_call",
        )
        return answer

    def __repr__(self) -> str:
        """감싼 대상과 게이트 설정만 노출한다."""
        return f"<GatedLlmClient 대상={self._name}, 게이트={self._gate!r}>"


def build_default_gate(
    *, audit_path: Any | None = None, strict: bool = True
) -> LlmEgressGate:
    """기본 게이트를 인자 없이 만들어 주는 팩토리.

    :mod:`docagent.agent.llm` 의 ``_resolve_default_gate`` 가 이 이름을 찾는다.
    게이트를 주입하지 않은 호출자가 **조용히 게이트 없이** LLM 을 쓰는 일을
    막기 위한 연결점이며, 여기서 만들어지는 것도 fail-closed 게이트다.

    :param audit_path: 감사 로그 JSONL 경로. ``None`` 이면 메모리에만 남긴다.
    :param strict: 탐지기 실패를 차단으로 해석할지 여부(기본 True).
    :returns: :class:`LlmEgressGate`.
    """
    audit = AuditLog(audit_path) if audit_path is not None else AuditLog()
    return LlmEgressGate(strategy=MaskStrategy.FULL, strict=strict, audit=audit)
