"""Explain 단계 진입점 — 검색·생성·검증·분기를 하나로 묶는다.

오케스트레이터가 쓰는 계약은 두 가지뿐이다.

* ``Explainer.explain(field, question=None) -> ExplanationResult``
* ``Guard.check(user_text, field=None) -> None`` (:mod:`docagent.agent.guardrails`)

처리 흐름
---------
1. **가드 검사** — 자격·법적·금융 판단, 근거 없는 추측 요구, 개인정보가 섞인 질문을
   막는다. 걸리면 생성 경로로 가지 않고 곧바로 핸드오프 결과를 돌려준다.
2. **질의 구성** — 공개 정보 영역만 쓴다. :attr:`~docagent.contracts.Sensitivity.PUBLIC`
   항목이면 ``title``·``clause_text`` 를 쓰고, ``PRIVATE`` 항목이면 ``type`` 에서 유도한
   일반 설명 문구만 쓴다. 개인정보 영역의 문구는 질의에 들어가지 않는다.
3. **근거 검색** — :class:`~docagent.interfaces.Retriever`.
4. **설명 생성** — :class:`~docagent.interfaces.LlmClient` (기본값은 오프라인 구현).
5. **근거 검증** — :func:`~docagent.agent.guardrails.answer_grounding_check` 로
   근거 밖 숫자·고유명사가 섞였는지 본다. 하나라도 있으면 답변을 폐기한다.
6. **신뢰도 분기** — :func:`~docagent.agent.confidence.score_answer` 의 밴드에 따라
   설명 / 설명+원문 병행 / 핸드오프로 나뉜다.

핸드오프 밴드에서는 **생성 텍스트를 버린다.** 원문 낭독 안내와 직원 연결 문구만 남는다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from docagent.agent.confidence import ConfidenceBand, ConfidenceReport, score_answer
from docagent.agent.guardrails import DISCLAIMER, Guard, answer_grounding_check
from docagent.agent.llm import SYSTEM_PROMPT, build_user_prompt
from docagent.contracts import Field, FieldType
from docagent.errors import HandoffRequired
from docagent.interfaces import Clock, LlmClient, Retriever

__all__ = [
    "DEFAULT_TOP_K",
    "HANDOFF_SPEECH",
    "PARTIAL_SUFFIX",
    "FIELD_TYPE_QUERIES",
    "ExplanationResult",
    "Explainer",
]

_LOGGER = logging.getLogger(__name__)

#: 기본 검색 청크 수.
DEFAULT_TOP_K: int = 4

#: 신뢰도 미달 시 낭독할 안내 문구.
HANDOFF_SPEECH: str = (
    "이 부분은 제가 쉬운 말로 설명드리기 어렵습니다. "
    "원문을 그대로 읽어 드리거나, 담당 직원에게 연결해 드리겠습니다."
)

#: 부분 확신(설명+원문 병행) 밴드에서 설명 뒤에 덧붙이는 문구.
PARTIAL_SUFFIX: str = "확실하지 않은 부분이 있어, 원문도 함께 들어 보시기를 권해 드립니다."

#: 개인정보 영역 항목에 쓰는 유형별 일반 질의.
#:
#: 개인정보 영역 항목의 ``title``·``clause_text`` 는 외부로 나가지 않는다
#: (:meth:`docagent.contracts.DocumentStructure.public_payload` 의 강등 규칙과 같은 원칙).
#: 대신 공개 payload 에도 남는 ``type`` 만 가지고 일반적인 질의를 만든다.
FIELD_TYPE_QUERIES: dict[FieldType, str] = {
    FieldType.SIGNATURE: "서명과 기명날인은 무슨 뜻이고 언제 서명해야 하나요",
    FieldType.CHECKBOX: "동의 표시를 하는 칸은 무슨 뜻인가요",
    FieldType.CHOICE: "동의함과 동의하지 않음 중에서 고르는 것은 무슨 뜻인가요",
    FieldType.TEXT_INPUT: "신청서에 적는 개인정보는 어떻게 수집되고 보관되나요",
    FieldType.DATE: "신청일자를 적는 칸은 무슨 뜻인가요",
    FieldType.UNKNOWN: "신청서의 기입 항목은 무슨 뜻인가요",
}


def _summarize_unsupported(tokens: list[str]) -> str:
    """근거 밖 표현 목록을 **원문 없이** 개수·유형으로만 요약한다.

    유출된 개인정보가 답변에 되울릴 경우 이 문자열이 로그와 세션 JSON 에
    남으므로, 토큰 원문은 절대 담지 않는다.

    :param tokens: :func:`~docagent.agent.guardrails.answer_grounding_check` 결과.
    :returns: 예 ``"근거 밖 숫자 2건, 고유명사 1건"``.
    """
    numbers = sum(1 for token in tokens if any(ch.isdigit() for ch in token))
    others = len(tokens) - numbers
    parts: list[str] = []
    if numbers:
        parts.append(f"숫자 {numbers}건")
    if others:
        parts.append(f"고유명사 {others}건")
    return "근거 밖 " + ", ".join(parts) if parts else "근거 밖 표현 없음"


@dataclass(frozen=True)
class ExplanationResult:
    """Explain 단계 결과 — 오케스트레이터가 그대로 낭독·분기에 쓴다.

    :param text: 사용자에게 낭독할 한국어 문장(고지 문구 포함).
    :param confidence: 종합 신뢰도(0.0~1.0).
    :param sources: 근거 출처 표기 목록(중복 제거, 검색 순서 유지).
        감사 추적을 위해 설명에 쓰인 근거를 항상 남긴다.
    :param needs_handoff: 원문 안내·직원 연결로 넘겨야 하면 True.
    :param disclaimer: 설명에 붙인 고지 문구. 핸드오프면 빈 문자열.
    :param mode: 신뢰도 밴드(:class:`~docagent.agent.confidence.ConfidenceBand`).
    :param field_id: 설명 대상 항목 id.
    :param reasons: 신뢰도 판단 근거(한국어). 디버깅·감사용.
    :raises ValueError: ``confidence`` 가 0.0~1.0 범위를 벗어난 경우.
    """

    text: str
    confidence: float
    sources: tuple[str, ...]
    needs_handoff: bool
    disclaimer: str
    mode: ConfidenceBand
    field_id: str = ""
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ExplanationResult.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "reasons", tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "sources": list(self.sources),
            "needs_handoff": self.needs_handoff,
            "disclaimer": self.disclaimer,
            "mode": self.mode.value,
            "field_id": self.field_id,
            "reasons": list(self.reasons),
        }


class Explainer:
    """근거 검색 → 설명 생성 → 검증 → 신뢰도 분기를 수행하는 진입점.

    :param retriever: 근거 검색기(:class:`docagent.interfaces.Retriever`).
    :param llm: 설명 생성기(:class:`docagent.interfaces.LlmClient`).
        기본값으로는 :class:`~docagent.agent.llm.OfflineTemplateLlm` 을 쓰기를 권한다.
    :param guard: 가드레일(:class:`~docagent.agent.guardrails.Guard`).
    :param clock: 시각 공급자. ``None`` 이면 시각을 기록하지 않는다.
        (시간까지 결정론적으로 다루기 위해 직접 ``datetime.now()`` 를 부르지 않는다.)
    :param top_k: 검색할 근거 청크 수(1 이상).
    :raises ValueError: ``top_k`` 가 1 미만인 경우.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LlmClient,
        guard: Guard,
        clock: Clock | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k 는 1 이상이어야 합니다: {top_k}")
        self._retriever = retriever
        self._llm = llm
        self._guard = guard
        self._clock = clock
        self._top_k = top_k

    # ------------------------------------------------------------------ 질의
    def build_query(self, field: Field, question: str | None = None) -> str:
        """검색 질의를 만든다. **공개 정보 영역만** 사용한다.

        질문이 있으면 ``항목명 + 질문`` 으로 짧게 만든다. 약관 전문(``clause_text``)까지
        붙이면 질문과 무관한 어휘가 대량으로 섞여 정작 물어본 내용의 근거가 밀려나기
        때문이다. 질문이 없을 때(항목 자체를 설명해 달라는 요청)만 약관 전문을 쓴다.

        개인정보 영역(:attr:`~docagent.contracts.Sensitivity.PRIVATE`) 항목은
        ``title`` · ``clause_text`` 를 쓰지 않고 ``type`` 에서 유도한 일반 문구만 쓴다.
        공개 payload 강등 규칙과 같은 원칙이다.

        :param field: 설명 대상 항목.
        :param question: 사용자 질문. ``None`` 이면 항목 자체에 대한 설명 요청으로 본다.
        :returns: 검색 질의 문자열.
        """
        has_question = bool(question and question.strip())
        parts: list[str] = []
        if field.is_public():
            if field.title:
                parts.append(field.title)
            if field.clause_text and not has_question:
                parts.append(field.clause_text)
        else:
            # 개인정보 영역 항목은 유형에서 유도한 일반 문구만 쓴다.
            parts.append(
                FIELD_TYPE_QUERIES.get(field.type, FIELD_TYPE_QUERIES[FieldType.UNKNOWN])
            )
        if has_question:
            parts.append(question.strip())  # type: ignore[union-attr]
        return " ".join(parts).strip()

    def _scoring_question(self, field: Field, question: str | None) -> str:
        """신뢰도 계산에 쓸 질의 문자열을 고른다.

        사용자가 실제로 물은 문장이 있으면 그것을 쓴다. 약관 전문을 그대로 넣으면
        분모(질의 어휘)가 지나치게 커져 겹침률이 왜곡되기 때문이다.

        :param field: 설명 대상 항목.
        :param question: 사용자 질문.
        :returns: 신뢰도 계산용 질의 문자열.
        """
        if question and question.strip():
            return question.strip()
        if field.is_public() and field.title:
            return field.title
        return FIELD_TYPE_QUERIES.get(field.type, FIELD_TYPE_QUERIES[FieldType.UNKNOWN])

    # ------------------------------------------------------------------ 진입점
    def explain(self, field: Field, question: str | None = None) -> ExplanationResult:
        """항목(과 질문)에 대한 쉬운 말 설명을 만든다.

        :param field: 설명 대상 항목.
        :param question: 사용자 질문. ``None`` 이면 항목 자체를 설명한다.
        :returns: :class:`ExplanationResult`.
            핸드오프 상황에서도 **예외를 던지지 않고** ``needs_handoff=True`` 인
            결과를 돌려준다(오케스트레이터가 분기 하나로 처리할 수 있게 하기 위함).
        """
        stamp = self._clock.now_iso() if self._clock is not None else ""

        # 1) 가드 검사 -----------------------------------------------------
        try:
            self._guard.check(question or "", field)
        except HandoffRequired as exc:
            _LOGGER.info("가드레일 핸드오프 [%s] field=%s", stamp, field.id)
            return self._handoff(field, exc.reason, reasons=("가드레일 차단.",))

        # 2) 질의 구성 후 검색 ---------------------------------------------
        query = self.build_query(field, question)

        # 질의에는 OCR 로 읽은 항목명·약관 전문이 함께 실린다. 민감도 판정을
        # 신뢰하지 말고 사용자 발화와 **같은 규율**을 질의 전체에 한 번 더 건다.
        # LLM 호출 뒤의 판정은 이미 늦으므로 반드시 호출 전에 검사한다.
        try:
            self._guard.check_outbound(query, field)
        except HandoffRequired as exc:
            _LOGGER.info("질의 개인정보 검사 핸드오프 [%s] field=%s", stamp, field.id)
            return self._handoff(
                field,
                exc.reason,
                reasons=(
                    "질의에 개인정보로 보이는 내용이 있어 외부 설명 경로를 쓰지 않았습니다.",
                ),
            )

        chunks = list(self._retriever.search(query, self._top_k))
        if not chunks:
            return self._handoff(
                field,
                HANDOFF_SPEECH,
                reasons=("질의에 맞는 근거 문서를 찾지 못했습니다.",),
            )

        # 3) 설명 생성 ------------------------------------------------------
        # 프롬프트에는 검색에 쓴 질의를 그대로 넣는다(약관 문맥까지 모델이 본다).
        # 신뢰도 계산에는 사용자가 실제로 물은 문장만 쓴다(_scoring_question).
        prompt = build_user_prompt(query, chunks)
        answer = self._llm.complete(SYSTEM_PROMPT, prompt).strip()
        if not answer:
            return self._handoff(
                field,
                HANDOFF_SPEECH,
                reasons=("설명 문장을 생성하지 못했습니다.",),
                sources=tuple(dict.fromkeys(chunk.source for chunk in chunks)),
            )

        # 4) 근거 검증(할루시네이션 차단) ------------------------------------
        unsupported = answer_grounding_check(answer, chunks)
        if unsupported:
            # 근거 밖 표현의 **원문 토큰은 남기지 않는다.** 프롬프트에 개인정보가
            # 실린 상황에서 모델이 그 숫자를 되풀이하면 여기서 '근거 밖'으로
            # 분류되는데, 그것을 그대로 기록하면 로그·세션 JSON 에 개인정보가
            # 평문으로 남는다. 개수와 유형만 남긴다.
            summary = _summarize_unsupported(unsupported)
            _LOGGER.warning(
                "근거 밖 표현이 발견되어 설명을 폐기했습니다 [%s] field=%s %s",
                stamp,
                field.id,
                summary,
            )
            return self._handoff(
                field,
                HANDOFF_SPEECH,
                reasons=(
                    "근거 문서에서 확인되지 않은 표현이 있어 설명을 사용하지 "
                    f"않았습니다({summary}).",
                ),
                sources=tuple(dict.fromkeys(chunk.source for chunk in chunks)),
            )

        # 5) 신뢰도 분기 ----------------------------------------------------
        report: ConfidenceReport = score_answer(
            chunks, self._scoring_question(field, question), answer
        )
        sources = tuple(dict.fromkeys(chunk.source for chunk in chunks))

        if report.band is ConfidenceBand.HANDOFF:
            return self._handoff(
                field,
                HANDOFF_SPEECH,
                reasons=report.reasons,
                sources=sources,
                confidence=report.score,
            )

        if report.band is ConfidenceBand.EXPLAIN_WITH_ORIGINAL:
            text = f"{answer} {PARTIAL_SUFFIX} {DISCLAIMER}"
        else:
            text = f"{answer} {DISCLAIMER}"

        return ExplanationResult(
            text=text,
            confidence=report.score,
            sources=sources,
            needs_handoff=False,
            disclaimer=DISCLAIMER,
            mode=report.band,
            field_id=field.id,
            reasons=report.reasons,
        )

    # ------------------------------------------------------------------ 내부
    def _handoff(
        self,
        field: Field,
        speech: str,
        *,
        reasons: tuple[str, ...] = (),
        sources: tuple[str, ...] = (),
        confidence: float = 0.0,
    ) -> ExplanationResult:
        """핸드오프 결과를 만든다(생성 텍스트는 버린다).

        :param field: 대상 항목.
        :param speech: 사용자에게 낭독할 안내 문구.
        :param reasons: 판단 근거.
        :param sources: 확보한 근거 출처(있으면 감사 추적용으로 남긴다).
        :param confidence: 산출된 신뢰도. 산출 전이면 0.0.
        :returns: ``needs_handoff=True`` 인 :class:`ExplanationResult`.
        """
        return ExplanationResult(
            text=speech,
            confidence=confidence,
            sources=sources,
            needs_handoff=True,
            disclaimer="",
            mode=ConfidenceBand.HANDOFF,
            field_id=field.id,
            reasons=reasons,
        )
