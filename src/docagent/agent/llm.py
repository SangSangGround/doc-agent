"""LLM 어댑터 — 오프라인 기본 구현과 선택적 Claude 어댑터.

이 모듈이 지키는 두 가지 원칙.

1. **API 키 없이도 100% 동작한다.** 기본 구현 :class:`OfflineTemplateLlm` 은
   근거 청크를 규칙 기반으로 재구성할 뿐 외부 호출을 하지 않으며 결정론적이다.
   CI·데모의 기본값이다.
2. **게이트 없는 LLM 은 쓰지 않는다.** :func:`build_llm` 은 :class:`~docagent.interfaces.PiiGate`
   를 반드시 요구하고 :class:`docagent.pii.gate.GatedLlmClient` 로 감싸서 돌려준다.
   오프라인 구현이라도 같은 경로를 지나게 해서, 나중에 원격 어댑터로 바꿔도 게이트가
   빠지지 않게 한다. 래퍼는 저장소에 **하나뿐**이며(마스킹 + 마스킹 후 재검사 +
   감사 기록), 이 모듈은 별도 래퍼를 두지 않는다. 감사 기록이 없는 두 번째 래퍼가
   있으면 안전 KPI(``pii_llm_transmissions``)가 근거 없이 0 으로 보고된다.

프롬프트 규약
-------------
:class:`~docagent.interfaces.LlmClient` 의 시그니처는 ``complete(system, user, max_tokens)``
하나뿐이라 근거 청크를 따로 넘길 자리가 없다. 그래서 근거를 **user 프롬프트 안에**
고정된 형식으로 실어 보낸다(:func:`build_user_prompt`). 오프라인 구현은 이 형식을
:func:`parse_user_prompt` 로 되읽어 근거 문장만 재구성하고, 원격 구현은 그대로 모델에
넘긴다. 형식이 하나이므로 두 구현이 완전히 교체 가능하다.
"""

from __future__ import annotations

import os
import re
from typing import Any, Sequence

from docagent.agent.rag import word_tokens
from docagent.contracts import RetrievedChunk
from docagent.errors import AdapterUnavailable, PiiEgressBlocked
from docagent.interfaces import PiiGate

__all__ = [
    "SYSTEM_PROMPT",
    "PLAIN_LANGUAGE_TERMS",
    "QUESTION_MARKER",
    "EVIDENCE_MARKER",
    "build_user_prompt",
    "parse_user_prompt",
    "OfflineTemplateLlm",
    "ClaudeLlm",
    "build_llm",
]


#: 모든 LLM 호출에 붙는 시스템 프롬프트.
SYSTEM_PROMPT: str = (
    "당신은 시각장애인 이용자에게 공공·금융 서류를 소리로 설명하는 도우미입니다.\n"
    "규칙:\n"
    "1. 제공된 근거 문서 밖의 내용을 추가하지 마십시오. 근거에 없으면 모른다고 말하십시오.\n"
    "2. 법적 판단, 자격 판단, 금액 판단을 하지 마십시오. 그런 질문은 담당 직원 연결이 필요합니다.\n"
    "3. 두세 문장으로 짧게, 초등학생도 알아들을 수 있는 쉬운 한국어로 말하십시오.\n"
    "4. 숫자·기간·기관 이름은 근거에 적힌 그대로만 말하십시오.\n"
    "5. 이용자를 대신해 결정하지 말고, 결정은 이용자 본인이 하도록 안내하십시오."
)

#: user 프롬프트에서 질문 구간을 여는 표지.
QUESTION_MARKER: str = "[질문]"
#: user 프롬프트에서 근거 구간을 여는 표지.
EVIDENCE_MARKER: str = "[근거]"

#: 어려운 표현 → 쉬운 표현 치환 사전.
#:
#: 긴 표현부터 치환해야 부분 치환으로 문장이 깨지지 않으므로,
#: 사용 시 키 길이 내림차순으로 정렬해 적용한다.
PLAIN_LANGUAGE_TERMS: dict[str, str] = {
    # 활용형이 붙는 표현은 활용형을 먼저 등록해 문장이 깨지지 않게 한다.
    "파기해야 합니다": "지워야 합니다",
    "파기합니다": "지웁니다",
    "보유 및 이용 기간": "가지고 있는 기간",
    "보유·이용기간": "가지고 있는 기간",
    "고유식별정보": "주민등록번호 같은 정보",
    "제3자에게 제공": "다른 기관에 넘기는 것",
    "제3자 제공": "다른 기관에 넘기는 것",
    "개인정보처리자": "개인정보를 다루는 기관",
    "정보주체": "본인",
    "제3자": "다른 기관",
}

#: 오프라인 요약이 고르는 최대 문장 수.
_MAX_SUMMARY_SENTENCES: int = 3

#: 문장 분리 정규식.
_SENTENCE_RE = re.compile(r"(?<=다\.)\s+|(?<=[.!?])\s+|\n+")

#: ``[근거: ...]`` 꼬리표 제거 정규식(설명 문장에는 넣지 않는다).
_BASIS_TAIL_RE = re.compile(r"\s*\[근거:[^\]]*\]")

#: 문장 선택에서 제외할 최소 길이(문자 수) — "둘째, 항목입니다." 같은 파편을 거른다.
_MIN_SENTENCE_LEN: int = 12


# --------------------------------------------------------------------------
# 프롬프트 조립·해석
# --------------------------------------------------------------------------


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """질문과 근거 청크를 하나의 user 프롬프트 문자열로 만든다.

    :param question: 사용자 질문(또는 설명 요청 문구). **공개 정보만** 담아야 한다.
    :param chunks: 근거 청크 목록.
    :returns: ``[질문] ... [근거] 1. (출처) 본문 ...`` 형식의 문자열.

    사용 예::

        prompt = build_user_prompt("동의가 뭐예요?", chunks)
        text = llm.complete(SYSTEM_PROMPT, prompt)
    """
    lines = [QUESTION_MARKER, question.strip(), "", EVIDENCE_MARKER]
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"{index}. (출처: {chunk.source}) {chunk.text}")
    return "\n".join(lines)


def parse_user_prompt(prompt: str) -> tuple[str, list[str]]:
    """:func:`build_user_prompt` 가 만든 프롬프트를 되읽는다.

    :param prompt: user 프롬프트 문자열.
    :returns: ``(질문, 근거 본문 목록)``. 표지가 없으면 전체를 질문으로 보고
        근거는 빈 목록으로 돌려준다.
    """
    if EVIDENCE_MARKER not in prompt:
        head = prompt.replace(QUESTION_MARKER, "").strip()
        return head, []
    head, _, tail = prompt.partition(EVIDENCE_MARKER)
    question = head.replace(QUESTION_MARKER, "").strip()
    evidence: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # "3. (출처: ...) 본문" 에서 번호와 출처 표기를 떼어 낸다.
        body = re.sub(r"^\d+\.\s*", "", stripped)
        body = re.sub(r"^\(출처:[^)]*\)\s*", "", body)
        if body:
            evidence.append(body)
    return question, evidence


def _split_sentences(text: str) -> list[str]:
    """텍스트를 문장 목록으로 나눈다.

    :param text: 대상 텍스트.
    :returns: 빈 문장을 제외한 문장 목록.
    """
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _words(text: str) -> set[str]:
    """길이 2 이상의 내용어 집합을 만든다(어휘 겹침 계산용).

    조사를 제거한 어절을 쓴다. ``동의가``/``동의는`` 이 같은 토큰으로 모여야
    질문과 근거 문장이 제대로 맞물린다.

    :param text: 대상 텍스트.
    :returns: 어절 집합.
    """
    return {word for word in word_tokens(text) if len(word) >= 2}


def _to_plain_language(text: str) -> str:
    """어려운 표현을 쉬운 표현으로 치환한다.

    :param text: 원문 문장.
    :returns: 치환된 문장.
    """
    result = text
    for hard in sorted(PLAIN_LANGUAGE_TERMS, key=len, reverse=True):
        result = result.replace(hard, PLAIN_LANGUAGE_TERMS[hard])
    return result


# --------------------------------------------------------------------------
# 오프라인 기본 구현
# --------------------------------------------------------------------------


class OfflineTemplateLlm:
    """근거 청크를 규칙 기반으로 재구성하는 오프라인 LLM(기본 구현).

    :class:`docagent.interfaces.LlmClient` 프로토콜을 만족한다.
    네트워크·API 키가 전혀 필요 없고 같은 입력에 항상 같은 출력을 낸다.

    동작 순서

    1. user 프롬프트에서 질문과 근거를 되읽는다(:func:`parse_user_prompt`).
    2. 근거를 문장으로 쪼갠다.
    3. 질문 어휘와 겹치는 정도로 문장 점수를 매기고 상위 2~3문장을 고른다.
       (질문이 없으면 근거 앞쪽 문장을 순서대로 쓴다.)
    4. :data:`PLAIN_LANGUAGE_TERMS` 로 어려운 표현을 바꾼다.

    **근거 밖 문장은 만들지 않는다.** 출력은 근거 문장의 부분집합을 치환한 것뿐이다.

    :param max_sentences: 요약에 담을 최대 문장 수(1 이상).
    :raises ValueError: ``max_sentences`` 가 1 미만인 경우.
    """

    def __init__(self, *, max_sentences: int = _MAX_SUMMARY_SENTENCES) -> None:
        if max_sentences < 1:
            raise ValueError(f"max_sentences 는 1 이상이어야 합니다: {max_sentences}")
        self._max_sentences = max_sentences

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """근거를 짧고 쉬운 한국어 두세 문장으로 재구성한다.

        :param system: 시스템 프롬프트. 이 구현은 참조하지 않지만 인터페이스를 지킨다.
        :param user: :func:`build_user_prompt` 형식의 user 프롬프트.
        :param max_tokens: 인터페이스 호환용. 문자 수 상한으로 느슨하게 반영한다.
        :returns: 두세 문장의 한국어 설명. 근거가 없으면 빈 문자열.
        """
        del system
        question, evidence = parse_user_prompt(user)
        if not evidence:
            return ""

        sentences: list[str] = []
        for body in evidence:
            for sentence in _split_sentences(_BASIS_TAIL_RE.sub("", body)):
                if sentence not in sentences:
                    sentences.append(sentence)
        if not sentences:
            return ""

        # 파편 문장은 후보에서 뺀다. 모두 짧으면 원래 목록을 그대로 쓴다.
        candidates = [
            index
            for index, sentence in enumerate(sentences)
            if len(sentence) >= _MIN_SENTENCE_LEN
        ] or list(range(len(sentences)))

        question_words = _words(question)
        if question_words:
            ranked = sorted(
                candidates,
                key=lambda i: (-len(question_words & _words(sentences[i])), i),
            )
            chosen = sorted(ranked[: self._max_sentences])
        else:
            chosen = candidates[: self._max_sentences]

        picked = [_to_plain_language(sentences[i]) for i in chosen]
        text = " ".join(picked).strip()
        limit = max(1, max_tokens) * 4  # 토큰당 대략 4자로 보수적으로 환산한다.
        if len(text) > limit:
            text = text[:limit].rstrip()
        return text


# --------------------------------------------------------------------------
# 선택적 원격 어댑터
# --------------------------------------------------------------------------

#: :class:`ClaudeLlm` 기본 모델 id.
_DEFAULT_CLAUDE_MODEL: str = "claude-sonnet-5"
#: API 키를 읽는 환경변수 이름. 키 값 자체는 코드·로그에 남기지 않는다.
_API_KEY_ENV: str = "ANTHROPIC_API_KEY"


class ClaudeLlm:
    """Anthropic Claude 어댑터(선택적).

    ``anthropic`` 패키지는 이번 범위에서 **설치하지 않는다.** 생성자에서 지연 import
    하고, 없으면 :class:`~docagent.errors.AdapterUnavailable` 를 던진다.
    API 키는 환경변수 :data:`_API_KEY_ENV` 에서만 읽으며 인스턴스 속성으로도,
    로그로도 남기지 않는다.

    :param model: 모델 id. 기본값 ``"claude-sonnet-5"``.
    :param max_tokens: 기본 생성 상한.
    :raises docagent.errors.AdapterUnavailable: ``anthropic`` 미설치 또는 API 키 미설정.
    :raises ValueError: ``model`` 이 빈 문자열인 경우.
    """

    def __init__(
        self,
        model: str = _DEFAULT_CLAUDE_MODEL,
        *,
        max_tokens: int = 1024,
    ) -> None:
        if not model:
            raise ValueError("model 은 빈 문자열일 수 없습니다.")
        try:  # 지연 import — 모듈 최상단에서 절대 import 하지 않는다.
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AdapterUnavailable(
                "anthropic",
                feature="Claude 기반 쉬운 말 설명 생성",
                extra="llm",
            ) from exc

        api_key = os.environ.get(_API_KEY_ENV)
        if not api_key:
            raise AdapterUnavailable(
                "anthropic",
                feature=(
                    f"Claude 호출(환경변수 {_API_KEY_ENV} 가 설정되어 있지 않습니다)"
                ),
                extra="llm",
            )
        self._model = model
        self._max_tokens = max_tokens
        # 클라이언트만 보관하고 키 문자열은 인스턴스에 남기지 않는다.
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def model(self) -> str:
        """사용 중인 모델 id."""
        return self._model

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Claude 에 프롬프트를 보내고 응답 텍스트를 돌려준다.

        :param system: 시스템 프롬프트(공개 정보만).
        :param user: user 프롬프트(공개 정보만, PII 마스킹 완료).
        :param max_tokens: 생성 최대 토큰 수.
        :returns: 모델 응답 텍스트.
        :raises docagent.errors.AdapterUnavailable: 호출이 실패한 경우
            (원인 예외를 감싸 올리며 키 값은 메시지에 담지 않는다).
        """
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max(1, min(max_tokens, self._max_tokens)),
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # SDK 예외 타입에 의존하지 않는다.
            raise AdapterUnavailable(
                "anthropic",
                feature=f"Claude 호출 실패({type(exc).__name__})",
                extra="llm",
            ) from exc

        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(str(getattr(block, "text", "")))
        return "".join(parts).strip()


# --------------------------------------------------------------------------
# 게이트 래퍼와 팩토리
# --------------------------------------------------------------------------


class _PiiGateEgressAdapter:
    """임의의 :class:`~docagent.interfaces.PiiGate` 에 전송 게이트 규약을 붙인다.

    :class:`docagent.pii.gate.GatedLlmClient` 는 게이트에 ``prepare()`` 와 ``audit``
    를 요구한다(마스킹 + 마스킹 후 재검사 + 감사 기록). ``sanitize()`` 만 가진
    게이트가 주입되어도 **같은 규약**을 지나가도록 이 어댑터가 나머지를 채운다.
    감사 기록이 빠진 우회 경로를 만들지 않기 위한 장치다.

    :param gate: 감쌀 :class:`~docagent.interfaces.PiiGate` 구현.
    :param audit: 감사 로그. ``None`` 이면 메모리 전용 로그를 새로 만든다.
    """

    def __init__(self, gate: PiiGate, audit: Any | None = None) -> None:
        from docagent.pii.audit import AuditLog  # 지연 import

        self._gate = gate
        #: 이 어댑터가 남기는 감사 로그.
        self.audit = audit if audit is not None else AuditLog()

    @property
    def inner_gate(self) -> PiiGate:
        """감싸고 있는 원본 게이트."""
        return self._gate

    def sanitize(self, text: str) -> Any:
        """원본 게이트의 마스킹을 그대로 위임한다.

        :param text: 원문 문자열.
        :returns: :class:`~docagent.contracts.SanitizedText`.
        """
        return self._gate.sanitize(text)

    def assert_clean(self, text: str) -> None:
        """원본 게이트의 단언을 그대로 위임한다(:class:`PiiGate` 프로토콜 준수).

        원본 게이트에 ``assert_clean`` 이 없으면 마스킹 결과로 대신 판정한다.
        어느 경우에도 **조용히 통과시키지 않는다.**

        :param text: 검사 대상 문자열.
        :returns: ``None``.
        :raises docagent.errors.PiiEgressBlocked: 개인정보가 탐지된 경우.
        """
        inner = getattr(self._gate, "assert_clean", None)
        if callable(inner):
            inner(text)
            return
        sanitized = self._gate.sanitize(text)
        spans = tuple(sanitized.spans)
        if spans or sanitized.blocked:
            raise PiiEgressBlocked(
                "개인정보가 포함되어 외부 전송을 차단했습니다.",
                pii_types=sorted({span.pii_type for span in spans}),
                count=len(spans),
            )

    def prepare(
        self, text: str, *, caller: str = "unknown", event: str = "egress"
    ) -> str:
        """마스킹 → 마스킹 후 재검사 → 감사 기록을 거친 안전 문자열을 만든다.

        :param text: 원문 문자열.
        :param caller: 감사 로그에 남길 호출자 식별자.
        :param event: 감사 로그 이벤트 종류.
        :returns: 외부로 내보낼 수 있는 문자열.
        :raises docagent.errors.PiiEgressBlocked: 게이트가 차단했거나 마스킹
            후에도 개인정보가 남은 경우. 메시지에 원문 값은 담기지 않는다.
        """
        sanitized = self._gate.sanitize(text)
        residual = tuple(self._gate.sanitize(sanitized.text).spans)
        blocked = bool(sanitized.blocked or residual)
        self.audit.record(
            caller=caller,
            text=text,
            spans=tuple(sanitized.spans),
            blocked=blocked,
            residual_count=len(residual),
            event=event,
            egress_text=None if blocked else sanitized.text,
        )
        if blocked:
            raise PiiEgressBlocked(
                "마스킹 후에도 개인정보가 남아 외부 전송을 차단했습니다.",
                pii_types=sorted({span.pii_type for span in residual}),
                count=len(residual),
            )
        return sanitized.text

    def __repr__(self) -> str:
        """감싼 게이트만 노출한다."""
        return f"<_PiiGateEgressAdapter 대상={type(self._gate).__name__}>"


def _as_egress_gate(gate: PiiGate) -> Any:
    """게이트를 전송 게이트 규약(``prepare`` + ``audit``)을 갖춘 객체로 만든다.

    :param gate: 주입된 게이트.
    :returns: 그대로 쓸 수 있으면 원본, 아니면 :class:`_PiiGateEgressAdapter`.
    """
    if callable(getattr(gate, "prepare", None)) and getattr(gate, "audit", None) is not None:
        return gate
    return _PiiGateEgressAdapter(gate)


def _resolve_default_gate() -> PiiGate:
    """PII 게이트 기본 구현을 지연 import 로 찾는다.

    ``docagent.pii`` 는 다른 모듈의 소유이므로 최상단에서 import 하지 않는다.
    찾지 못하면 **조용히 통과시키지 않고** 차단 예외를 던진다.

    :returns: :class:`~docagent.interfaces.PiiGate` 구현 인스턴스.
    :raises docagent.errors.PiiEgressBlocked: 게이트 구현을 찾지 못한 경우.
    """
    try:
        from docagent.pii import gate as gate_module  # type: ignore[attr-defined]
    except ImportError as exc:
        raise PiiEgressBlocked(
            "PII 게이트 구현(docagent.pii.gate)을 찾지 못해 LLM 호출을 차단했습니다. "
            "gate 인자로 게이트를 직접 주입하십시오."
        ) from exc

    for name in ("build_default_gate", "default_gate", "DefaultPiiGate", "PiiGateImpl"):
        factory = getattr(gate_module, name, None)
        if factory is not None:
            return factory()  # type: ignore[no-any-return]
    raise PiiEgressBlocked(
        "docagent.pii.gate 에서 기본 게이트 생성자를 찾지 못해 LLM 호출을 차단했습니다. "
        "gate 인자로 게이트를 직접 주입하십시오."
    )


def build_llm(
    gate: PiiGate | None = None,
    *,
    kind: str = "offline",
    model: str = _DEFAULT_CLAUDE_MODEL,
    inner: Any | None = None,
) -> Any:
    """PII 게이트로 감싼 LLM 클라이언트를 만든다.

    **게이트 없이 LLM 을 돌려주지 않는다.** ``gate`` 를 주지 않으면
    :func:`_resolve_default_gate` 로 기본 구현을 찾고, 찾지 못하면 차단 예외를 던진다.

    :param gate: 개인정보 게이트. ``None`` 이면 기본 구현을 지연 import 로 찾는다.
    :param kind: ``"offline"``(기본) 또는 ``"claude"``.
    :param model: ``kind="claude"`` 일 때 사용할 모델 id.
    :param inner: 이미 만들어 둔 :class:`~docagent.interfaces.LlmClient`.
        주어지면 ``kind`` 를 무시하고 이것을 감싼다(테스트·주입용).
    :returns: :class:`docagent.pii.gate.GatedLlmClient`.
        저장소에 래퍼는 이것 하나뿐이며 마스킹 + 마스킹 후 재검사 + 감사 기록을
        모두 수행한다. ``sanitize()`` 만 가진 게이트가 들어오면
        :class:`_PiiGateEgressAdapter` 가 나머지 규약을 채운다.
    :raises ValueError: ``kind`` 가 알 수 없는 값인 경우.
    :raises docagent.errors.PiiEgressBlocked: 게이트를 확보하지 못한 경우.
    :raises docagent.errors.AdapterUnavailable: ``kind="claude"`` 인데 어댑터를 못 쓰는 경우.
    """
    from docagent.pii.gate import GatedLlmClient  # 지연 import (단일 래퍼)

    resolved_gate = _as_egress_gate(
        gate if gate is not None else _resolve_default_gate()
    )
    if inner is not None:
        return GatedLlmClient(inner=inner, gate=resolved_gate)
    if kind == "offline":
        return GatedLlmClient(inner=OfflineTemplateLlm(), gate=resolved_gate)
    if kind == "claude":
        return GatedLlmClient(inner=ClaudeLlm(model=model), gate=resolved_gate)
    raise ValueError(f"알 수 없는 LLM 종류입니다: {kind!r} (offline 또는 claude)")
