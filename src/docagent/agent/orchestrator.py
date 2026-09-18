"""에이전트 루프 — 가드 → 의도 분류 → 도구 선택 → 실행 → 발화 → 상태 갱신.

:class:`DocumentAgent` 는 See/Understand 를 마친 :class:`~docagent.contracts.
DocumentStructure` 를 받아, Explain → Ask → Act → Verify 를 대화로 진행한다.
Vision 의 내부 구현도 LLM 의 존재 여부도 알지 못한다.

세 가지 안내 모드
-----------------
항목을 안내할 때마다 **원문 듣기 / 쉬운 설명 / 다음 항목** 세 가지를 항상 제시한다.
쉬운 설명은 원문의 **대체물이 아니다.** 그래서 설명 발화 뒤에는
:data:`docagent.agent.tools.ORIGINAL_NOTICE` 고지가 예외 없이 따라붙는다.

사람 지원(Human-in-the-loop) 트리거
-----------------------------------
아래 조건 중 하나라도 걸리면 :attr:`~docagent.agent.state.SessionPhase.HUMAN_HANDOFF`
로 전이하고 사유를 :attr:`~docagent.agent.state.SessionState.handoff_reason` 과
이력에 남긴다.

1. 항목 인식 신뢰도 < :data:`~docagent.contracts.VISION_TRUST_THRESHOLD` (0.85)
2. 설명 신뢰도 < :data:`~docagent.contracts.PARTIAL_THRESHOLD` (0.70)
3. 자격·법적 판단을 묻는 질문("제가 이 지원금을 받을 수 있나요?")
4. 서명란 후보가 모호함(주체 미상 서명란이 둘 이상)
5. 같은 항목에서 :data:`~docagent.agent.tools.MAX_FIELD_FAILURES` 회 연속 실패
6. 의도 분류 UNKNOWN 2회 연속

결정론
------
시간은 :class:`~docagent.interfaces.Clock` 주입으로만 얻는다. 난수·현재 시각을
직접 호출하는 코드는 없으므로 같은 입력열은 항상 같은 발화열을 만든다.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from docagent.contracts import (
    VISION_TRUST_THRESHOLD,
    AgentTurn,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    ToolCall,
    ToolResult,
)
from docagent.errors import HandoffRequired, InvalidTransition
from docagent.agent.guardrails import Guard
from docagent.agent.intent import (
    AGREE_KEYWORDS,
    DISAGREE_KEYWORDS,
    Intent,
    IntentResult,
    RuleIntentClassifier,
)
from docagent.agent.state import SessionPhase, SessionState
from docagent.agent.tools import ToolRegistry

__all__ = [
    "DocumentAgent",
    "HANDOFF_INTENT",
    "FixedClock",
    "MODES_SENTENCE",
    "MAX_UNKNOWN_STREAK",
]


#: 항목 안내마다 반드시 붙는 세 가지 모드 안내.
MODES_SENTENCE: str = (
    "원문 듣기, 쉬운 설명, 다음 항목 중에서 원하시는 것을 말씀해 주세요."
)

#: 의도를 이 횟수만큼 연속으로 알아듣지 못하면 사람 지원으로 넘긴다.
MAX_UNKNOWN_STREAK: int = 2

#: 사람 지원 카운터 이름.
_UNKNOWN_COUNTER: str = "unknown_streak"

#: 이미 사람에게 넘긴 세션에서 들어온 발화에 붙이는 의도 라벨.
#: :class:`~docagent.agent.intent.Intent` 의 값이 아니라 상태를 나타내는 표식이다.
HANDOFF_INTENT: str = "human_handoff"


class FixedClock:
    """항상 같은 시각을 돌려주는 기본 시계(결정론 보장용).

    운영 환경에서는 실제 시각을 제공하는 :class:`~docagent.interfaces.Clock`
    구현을 주입해야 한다. 기본값을 고정 시각으로 둔 이유는, 시계를 주입하지 않은
    채 돌린 테스트가 우연히 통과했다가 나중에 시각 때문에 깨지는 일을 막기 위해서다.

    :param value: 반환할 ISO 8601 문자열.
    """

    def __init__(self, value: str = "1970-01-01T00:00:00+09:00") -> None:
        self.value = value

    def now_iso(self) -> str:
        """고정 시각을 반환한다.

        :returns: ISO 8601 문자열.
        """
        return self.value


def _norm(text: str) -> str:
    """공백·문장부호를 제거한 소문자 문자열을 만든다.

    :param text: 원문.
    :returns: 정규화 문자열.
    """
    return re.sub(r"[\s.,!?~·\"'()\[\]{}<>:;/\\-]+", "", text.strip().lower())


def _object_particle(word: str) -> str:
    """한글 목적격 조사(을/를)를 받침에 맞춰 고른다.

    :param word: 앞 단어.
    :returns: ``"을"`` 또는 ``"를"``. 한글이 아니면 ``"를"``.
    """
    if not word:
        return "를"
    last = word.strip()[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3:
        return "를" if (code - 0xAC00) % 28 == 0 else "을"
    return "를"


class DocumentAgent:
    """문서 작성 대화를 이끄는 오케스트레이터.

    :param structure: Vision 이 만든 문서 구조. 좌표의 유일한 출처.
    :param state: 세션 상태. ``None`` 이면 ``structure`` 로부터 새로 만든다.
    :param tools: 도구 레지스트리. ``None`` 이면 주입된 의존성으로 새로 만든다.
    :param explainer: ``explain(field, question)`` 를 갖는 설명기. ``None`` 이면
        원문만 낭독하는 기본 설명기를 쓴다.
    :param guard: ``check(user_text, field)`` 를 갖는 가드레일. ``None`` 이면 통과.
    :param speech: :class:`~docagent.interfaces.SpeechIO`. ``None`` 이면 기록만 한다.
    :param motion: :class:`~docagent.interfaces.MotionController`. ``None`` 이면
        메모리 상 이동만 수행한다.
    :param verifier: ``verify(field_id)`` 를 갖는 검증기. ``None`` 이면 데모용 기본값.
    :param clock: :class:`~docagent.interfaces.Clock`. ``None`` 이면 :class:`FixedClock`.
    :param guard: 가드레일. ``None`` 이면 :class:`~docagent.agent.guardrails.Guard`
        기본 인스턴스(개인정보 게이트를 스스로 확보하는 fail-closed 구현)를 만든다.
        검사를 끄려면 :class:`docagent.testing.guards.PassThroughGuard` 를
        명시적으로 주입해야 한다.
    :param classifier: 의도 분류기. ``None`` 이면 :class:`~docagent.agent.intent.
        RuleIntentClassifier`.
    :raises ValueError: ``state.document_id`` 가 ``structure.document_id`` 와 다른 경우.
    """

    def __init__(
        self,
        structure: DocumentStructure,
        state: SessionState | None = None,
        tools: ToolRegistry | None = None,
        explainer: Any | None = None,
        guard: Any | None = None,
        speech: Any | None = None,
        motion: Any | None = None,
        verifier: Any | None = None,
        clock: Any | None = None,
        classifier: Any | None = None,
    ) -> None:
        self.structure = structure
        self.state = state if state is not None else SessionState.from_structure(structure)
        if self.state.document_id != structure.document_id:
            raise ValueError(
                "세션 상태와 문서 구조의 document_id 가 다릅니다: "
                f"{self.state.document_id!r} != {structure.document_id!r}"
            )
        # 기본 가드는 **fail-closed** 다. 가드를 주입하지 않았다고 해서 검사가
        # 사라지지 않는다(Guard 가 개인정보 게이트를 스스로 확보한다).
        # 검사를 끄려면 docagent.testing.guards.PassThroughGuard 를 명시적으로 준다.
        self.guard = guard if guard is not None else Guard()
        self.clock = clock if clock is not None else FixedClock()
        self.classifier = (
            classifier if classifier is not None else RuleIntentClassifier()
        )
        self.tools = (
            tools
            if tools is not None
            else ToolRegistry(
                structure,
                self.state,
                motion=motion,
                explainer=explainer,
                verifier=verifier,
                speech=speech,
            )
        )
        self.speech = speech if speech is not None else self.tools.speech
        self.motion = motion if motion is not None else self.tools.motion
        #: 이 세션에서 만들어진 대화 턴 기록(감사 재료는 ``state.history`` 가 정본).
        self.turns: list[AgentTurn] = []
        self._last_speech: str = ""

    # ------------------------------------------------------------------
    # 생성 · 복원
    # ------------------------------------------------------------------

    @classmethod
    def restore(
        cls, structure: DocumentStructure, session_json: str, **deps: Any
    ) -> "DocumentAgent":
        """저장된 세션 JSON 으로부터 에이전트를 복원한다.

        :param structure: 같은 문서의 구조.
        :param session_json: :meth:`session_json` 이 만든 JSON 문자열.
        :param deps: ``explainer`` / ``guard`` / ``speech`` / ``motion`` /
            ``verifier`` / ``clock`` / ``classifier`` 주입 인자.
        :returns: 복원된 :class:`DocumentAgent`. ``turns`` 는 빈 목록으로 시작한다
            (대화 원문은 저장하지 않기 때문이다 — :meth:`session_json` 참조).
            진행 이력은 ``state.history`` 로 그대로 복원된다.
        :raises ValueError: JSON 이 깨졌거나 문서 식별자가 다른 경우.
        """
        return cls(structure, state=SessionState.from_json(session_json), **deps)

    def session_json(self, *, indent: int | None = 2) -> str:
        """현재 세션 상태를 JSON 문자열로 내보낸다.

        :attr:`turns` 는 **일부러 담지 않는다.** :class:`~docagent.contracts.AgentTurn`
        의 ``user_text`` 는 이용자의 원문 발화이므로, 세션 JSON 에 실으면 개인정보가
        그대로 디스크에 남는다. 복원 정본은 :class:`~docagent.agent.state.SessionState`
        이고 감사 추적은 그 안의 ``history`` 가 담당한다(턴 번호·단계·항목·행위·설명).

        :param indent: JSON 들여쓰기.
        :returns: 세션 상태 JSON.
        """
        return self.state.to_json(indent=indent)

    # ------------------------------------------------------------------
    # 대화 시작
    # ------------------------------------------------------------------

    def start(self) -> AgentTurn:
        """문서 인식 결과를 안내하고 첫 항목까지 읽어 준다.

        :returns: 인식 안내 + 첫 항목 안내가 담긴 :class:`~docagent.contracts.AgentTurn`.
        :raises InvalidTransition: 이미 시작된 세션에서 다시 호출한 경우.
        """
        if self.state.phase is not SessionPhase.IDLE:
            raise InvalidTransition(
                "이미 시작된 세션입니다.",
                current=self.state.phase.value,
                requested=SessionPhase.DOCUMENT_LOADED.value,
            )
        self.state.transition_to(
            SessionPhase.DOCUMENT_LOADED,
            action="document_loaded",
            note=self.clock.now_iso(),
        )
        title = self.structure.doc_title or "문서"
        total = len(self.structure.fields)
        parts = [
            f"{title}{_object_particle(title)} 인식했습니다. "
            f"총 {total}개의 작성 항목이 있습니다."
        ]
        first = self.state.next_pending()
        if first is None:
            self.state.transition_to(
                SessionPhase.COMPLETED,
                action="all_fields_done",
                note="작성할 항목이 없습니다.",
            )
            parts.append("작성이 필요한 항목이 없습니다.")
            return self._emit("", "start", (), parts, 1.0)
        self.state.set_current(first)
        try:
            parts.append(self._announce_current())
        except HandoffRequired as exc:
            return self._handoff("", "start", exc.reason, exc.field_id, 1.0)
        return self._emit("", "start", (), parts, 1.0)

    # ------------------------------------------------------------------
    # 대화 1턴
    # ------------------------------------------------------------------

    def handle(self, user_text: str) -> AgentTurn:
        """사용자 발화 1건을 처리한다.

        절차: 가드 검사 → 의도 분류 → 도구 선택 → 실행 → 발화 생성 → 상태 갱신.

        :param user_text: 사용자 발화(STT 결과).
        :returns: :class:`~docagent.contracts.AgentTurn`.
            사람 지원으로 넘어간 턴은 ``handoff_reason`` 이 채워진다.
        """
        if self.state.phase is SessionPhase.HUMAN_HANDOFF:
            reason = self.state.handoff_reason or "담당 직원에게 넘긴 세션입니다."
            return self._emit(
                user_text,
                HANDOFF_INTENT,
                (),
                [f"{reason} 담당 직원을 기다려 주세요."],
                1.0,
                handoff_reason=reason,
                transition=False,
            )

        self.state.begin_turn()
        self.state.log("turn_time", note=self.clock.now_iso())
        field = self._current_field()

        try:
            self.guard.check(user_text, field)
        except HandoffRequired as exc:
            return self._handoff(user_text, "guard", exc.reason, exc.field_id, 1.0)

        result = self.classifier.classify(
            user_text, state=self.state, structure=self.structure
        )

        if result.intent is Intent.UNKNOWN:
            streak = self.state.bump(_UNKNOWN_COUNTER)
            if streak >= MAX_UNKNOWN_STREAK:
                reason = (
                    f"말씀을 {streak}번 연속으로 알아듣지 못했습니다. "
                    "담당 직원의 도움이 필요합니다."
                )
                return self._handoff(user_text, Intent.UNKNOWN.value, reason, None, 0.0)
            return self._emit(
                user_text,
                Intent.UNKNOWN.value,
                (),
                ["죄송합니다. 말씀을 정확히 알아듣지 못했습니다.", MODES_SENTENCE],
                0.0,
            )
        self.state.reset_counter(_UNKNOWN_COUNTER)

        try:
            return self._dispatch(user_text, result, field)
        except HandoffRequired as exc:
            return self._handoff(
                user_text, result.intent.value, exc.reason, exc.field_id, result.confidence
            )

    # ------------------------------------------------------------------
    # 의도별 처리
    # ------------------------------------------------------------------

    def _dispatch(
        self, user_text: str, result: IntentResult, field: Field | None
    ) -> AgentTurn:
        """분류된 의도를 도구 호출로 옮긴다.

        :param user_text: 사용자 발화.
        :param result: 의도 분류 결과.
        :param field: 턴 시작 시점의 현재 항목.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        :raises HandoffRequired: 사람 지원이 필요한 경우(호출자가 잡는다).
        """
        intent = result.intent
        calls: list[ToolCall] = []
        parts: list[str] = []

        def run(name: str, **arguments: Any) -> ToolResult:
            """도구를 실행하고 호출 기록을 남긴다."""
            calls.append(ToolCall(name=name, arguments=dict(arguments)))
            return self.tools.call(name, arguments)

        # ---- 정보 제공 -------------------------------------------------
        if intent is Intent.EXPLAIN or (
            intent is Intent.QUESTION and not result.slots.get("legal_judgment")
        ):
            outcome = run("explain", question=user_text)
            return self._from_tool(user_text, result, calls, outcome)

        if intent is Intent.QUESTION and result.slots.get("legal_judgment"):
            reason = (
                "자격 여부나 법적 효력에 대한 판단은 제가 대신 내려 드릴 수 없습니다. "
                "담당 직원에게 확인을 요청하겠습니다."
            )
            return self._handoff(
                user_text,
                intent.value,
                reason,
                field.id if field is not None else None,
                result.confidence,
            )

        if intent is Intent.READ_ORIGINAL:
            outcome = run("read_original")
            return self._from_tool(user_text, result, calls, outcome)

        if intent is Intent.REPEAT:
            run("repeat")
            if self._last_speech:
                parts.append(self._last_speech)
            else:
                parts.append(self._announce_current())
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        if intent is Intent.CALL_STAFF:
            outcome = run(
                "request_human", reason="사용자가 담당 직원의 도움을 요청했습니다."
            )
            return self._from_tool(user_text, result, calls, outcome)

        if intent is Intent.CANCEL:
            parts.append(
                "작성을 잠시 멈췄습니다. 계속하시려면 '계속'이라고, "
                "도움이 필요하시면 '직원 불러줘'라고 말씀해 주세요."
            )
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        # ---- 이동 ------------------------------------------------------
        if intent is Intent.NEXT:
            outcome = run("next_field")
            if not outcome.ok:
                return self._from_tool(user_text, result, calls, outcome)
            if self.state.phase is SessionPhase.COMPLETED:
                parts.append(self._completion_speech())
            else:
                parts.append(self._announce_current())
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        if intent is Intent.PREVIOUS:
            outcome = run("previous_field")
            if not outcome.ok:
                return self._from_tool(user_text, result, calls, outcome)
            parts.append(self._announce_current())
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        if intent is Intent.REVISE:
            target = result.slots.get("target_field_id")
            if not target:
                parts.append(
                    "어느 항목을 다시 하시겠습니까? 항목 이름을 말씀해 주세요."
                )
                return self._emit(
                    user_text, intent.value, calls, parts, result.confidence
                )
            outcome = run("revise_field", field_id=str(target))
            if not outcome.ok:
                return self._from_tool(user_text, result, calls, outcome)
            parts.append(outcome.speech)
            parts.append(self._announce_current())
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        # ---- 선택 · 기입 -----------------------------------------------
        if intent is Intent.SELECT_OPTION:
            label = result.slots.get("option_label")
            if not label:
                parts.append(self._option_prompt(field))
                return self._emit(
                    user_text, intent.value, calls, parts, result.confidence
                )
            return self._choose(user_text, result, calls, run, str(label))

        if intent is Intent.AGREE:
            if self.state.phase in (SessionPhase.AWAIT_WRITE, SessionPhase.VERIFYING):
                return self._confirm_write(user_text, result, calls, run)
            if field is not None and field.options:
                label = self._pick_option(field, AGREE_KEYWORDS, DISAGREE_KEYWORDS)
                if label is None:
                    parts.append(self._option_prompt(field))
                    return self._emit(
                        user_text, intent.value, calls, parts, result.confidence
                    )
                return self._choose(user_text, result, calls, run, label)
            return self._guide_write(user_text, result, calls, run, field)

        if intent is Intent.DISAGREE:
            if field is not None and field.options:
                label = self._pick_option(field, DISAGREE_KEYWORDS, ())
                if label is None:
                    parts.append(self._option_prompt(field))
                    return self._emit(
                        user_text, intent.value, calls, parts, result.confidence
                    )
                return self._choose(user_text, result, calls, run, label)
            if self.state.phase in (SessionPhase.AWAIT_WRITE, SessionPhase.VERIFYING):
                parts.append(
                    "알겠습니다. 펜 위치는 그대로 두겠습니다. "
                    "다시 기입하신 뒤 '확인'이라고 말씀해 주세요."
                )
                return self._emit(
                    user_text, intent.value, calls, parts, result.confidence
                )
            parts.append("알겠습니다. 이 항목은 그대로 두겠습니다.")
            parts.append(MODES_SENTENCE)
            return self._emit(user_text, intent.value, calls, parts, result.confidence)

        if intent is Intent.CONFIRM:
            if self.state.phase in (SessionPhase.AWAIT_WRITE, SessionPhase.VERIFYING):
                return self._confirm_write(user_text, result, calls, run)
            if field is not None and field.options and field.id not in self.state.selected_options:
                parts.append(self._option_prompt(field))
                return self._emit(
                    user_text, intent.value, calls, parts, result.confidence
                )
            return self._guide_write(user_text, result, calls, run, field)

        # 여기에 도달하면 의도 표에 처리기가 빠진 것이다. 조용히 넘어가지 않는다.
        parts.append("죄송합니다. 요청을 처리하지 못했습니다.")
        parts.append(MODES_SENTENCE)
        return self._emit(user_text, intent.value, calls, parts, 0.0)

    # ------------------------------------------------------------------
    # 복합 동작
    # ------------------------------------------------------------------

    def _choose(
        self,
        user_text: str,
        result: IntentResult,
        calls: list[ToolCall],
        run: Any,
        label: str,
    ) -> AgentTurn:
        """선택지를 확정하고 곧바로 펜을 그 네모 칸으로 옮긴다.

        :param user_text: 사용자 발화.
        :param result: 의도 분류 결과.
        :param calls: 이 턴의 도구 호출 누적 목록.
        :param run: 도구 실행 함수.
        :param label: 확정할 선택지 라벨.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        selected = run("select_option", option_label=label)
        if not selected.ok:
            return self._from_tool(user_text, result, calls, selected)
        moved = run("move_to_field")
        if not moved.ok:
            return self._from_tool(user_text, result, calls, moved)
        return self._emit(
            user_text,
            result.intent.value,
            calls,
            [selected.speech, moved.speech],
            result.confidence,
        )

    def _guide_write(
        self,
        user_text: str,
        result: IntentResult,
        calls: list[ToolCall],
        run: Any,
        field: Field | None,
    ) -> AgentTurn:
        """선택지가 없는 항목에서 펜을 기입 위치로 옮긴다.

        :param user_text: 사용자 발화.
        :param result: 의도 분류 결과.
        :param calls: 도구 호출 누적 목록.
        :param run: 도구 실행 함수.
        :param field: 현재 항목.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        if field is None:
            return self._emit(
                user_text,
                result.intent.value,
                calls,
                ["지금 안내 중인 항목이 없습니다.", MODES_SENTENCE],
                result.confidence,
            )
        moved = run("move_to_field")
        if not moved.ok:
            return self._from_tool(user_text, result, calls, moved)
        return self._emit(
            user_text, result.intent.value, calls, [moved.speech], result.confidence
        )

    def _confirm_write(
        self,
        user_text: str,
        result: IntentResult,
        calls: list[ToolCall],
        run: Any,
    ) -> AgentTurn:
        """기입 완료 확인 → 검증 → 다음 항목 안내(또는 완료 발화).

        :param user_text: 사용자 발화.
        :param result: 의도 분류 결과.
        :param calls: 도구 호출 누적 목록.
        :param run: 도구 실행 함수.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        verified = run("verify_field")
        if not verified.ok:
            return self._from_tool(user_text, result, calls, verified)
        parts = [verified.speech]
        moved = run("next_field")
        if not moved.ok:
            return self._from_tool(user_text, result, calls, moved)
        if self.state.phase is SessionPhase.COMPLETED:
            parts.append(self._completion_speech())
        else:
            parts.append(self._announce_current())
        return self._emit(
            user_text, result.intent.value, calls, parts, result.confidence
        )

    # ------------------------------------------------------------------
    # 발화 생성
    # ------------------------------------------------------------------

    def _current_field(self) -> Field | None:
        """현재 안내 중인 항목을 반환한다.

        :returns: :class:`~docagent.contracts.Field` 또는 ``None``.
        """
        if self.state.current_field_id is None:
            return None
        return self.structure.field_by_id(self.state.current_field_id)

    def _check_field(self, field: Field) -> None:
        """항목을 안내해도 되는지 사람 지원 트리거로 점검한다.

        :param field: 대상 항목.
        :returns: ``None``.
        :raises HandoffRequired: 인식 신뢰도 미달 또는 서명란 주체 모호.
        """
        if field.confidence < VISION_TRUST_THRESHOLD:
            raise HandoffRequired(
                f"'{field.title or field.id}' 항목을 정확히 읽지 못했습니다. "
                "잘못 안내드리지 않도록 담당 직원의 확인을 요청합니다.",
                field_id=field.id,
            )
        # 서명란의 기입 주체가 모호하면 **서명란 개수와 무관하게** 사람에게 넘긴다.
        # 개수 조건을 달면, 서명란이 하나뿐인 서식에서는 신뢰도 페널티 값에 우연히
        # 기대게 된다(페널티를 조정하거나 다른 Detector 가 confidence>=0.85 인
        # role=UNKNOWN 서명란을 만들면 모호한 칸이 그대로 안내된다).
        if field.type is FieldType.SIGNATURE and field.role is FieldRole.UNKNOWN:
            signatures = [
                item
                for item in self.structure.fields
                if item.type is FieldType.SIGNATURE
            ]
            if len(signatures) > 1:
                raise HandoffRequired(
                    "서명란이 여러 개인데 어느 것이 신청인 서명란인지 확정하지 "
                    "못했습니다. 담당 직원의 확인을 요청합니다.",
                    field_id=field.id,
                )
            raise HandoffRequired(
                "이 서명란에 누가 서명해야 하는지 확정하지 못했습니다. "
                "잘못된 칸에 서명하지 않도록 담당 직원의 확인을 요청합니다.",
                field_id=field.id,
            )

    def _announce_current(self) -> str:
        """현재 항목을 안내하는 발화를 만들고 단계를 맞춘다.

        :returns: 안내 발화. 항상 세 가지 모드 안내로 끝난다.
        :raises HandoffRequired: :meth:`_check_field` 트리거에 걸린 경우.
        """
        field = self._current_field()
        if field is None:
            return "지금 안내할 항목이 없습니다."
        self._check_field(field)

        position = (
            self.state.field_order.index(field.id) + 1
            if field.id in self.state.field_order
            else field.order + 1
        )
        total = len(self.state.field_order) or len(self.structure.fields)
        title = field.title or field.id
        parts = [f"{position}번째 항목, {title}입니다."]
        if field.required:
            parts.append("반드시 작성해야 하는 필수 항목입니다.")
        parts.append(f"전체 {total}개 항목 중 {position}번째입니다.")
        if field.options:
            labels = ", ".join(option.label for option in field.options)
            parts.append(f"선택지는 {labels} 입니다.")
        parts.append(MODES_SENTENCE)

        # 항상 ANNOUNCE_FIELD 를 경유한다. 어느 단계에서 안내를 시작하든
        # 전이 경로가 하나뿐이어야 상태머신이 단순하게 유지된다.
        note = f"'{title}' 항목을 안내했습니다."
        if self.state.phase is SessionPhase.ANNOUNCE_FIELD:
            self.state.log("announce_field", note=note)
        else:
            self.state.transition_to(
                SessionPhase.ANNOUNCE_FIELD, action="announce_field", note=note
            )
        if field.options:
            self.state.transition_to(
                SessionPhase.AWAIT_CHOICE,
                action="await_choice",
                note="선택을 기다립니다.",
            )
        return " ".join(parts)

    def _option_prompt(self, field: Field | None) -> str:
        """어떤 선택지를 고를지 되묻는 발화를 만든다.

        :param field: 현재 항목.
        :returns: 되묻는 발화.
        """
        if field is None or not field.options:
            return "무엇을 하시겠습니까? " + MODES_SENTENCE
        labels = ", ".join(option.label for option in field.options)
        return f"선택지는 {labels} 입니다. 어느 것으로 하시겠습니까?"

    def _completion_speech(self) -> str:
        """모든 항목을 마쳤을 때의 발화를 만든다.

        :returns: 완료 발화. 미작성 필수 항목이 있으면 경고를 덧붙인다.
        """
        done, total, remaining = self.state.progress()
        parts = [
            f"모든 항목의 안내를 마쳤습니다. 전체 {total}개 항목 중 "
            f"{done}개를 작성했습니다."
        ]
        if remaining:
            titles: list[str] = []
            for fid in remaining:
                item = self.structure.field_by_id(fid)
                titles.append(item.title if item is not None and item.title else fid)
            parts.append(
                "아직 작성되지 않은 필수 항목이 있습니다: " + ", ".join(titles) + "."
            )
            parts.append(
                "다시 작성하시려면 항목 이름을 말씀하시고 '다시 할래'라고 해 주세요."
            )
        else:
            parts.append(
                "필수 항목은 모두 작성되었습니다. 서류를 담당 직원에게 제출하시면 됩니다."
            )
        return " ".join(parts)

    @staticmethod
    def _pick_option(
        field: Field,
        include: Sequence[str],
        exclude: Sequence[str],
    ) -> str | None:
        """키워드로 선택지 라벨 하나를 고른다.

        :param field: 선택형 항목.
        :param include: 포함되어야 할 키워드.
        :param exclude: 포함되면 안 되는 키워드.
        :returns: 선택지 라벨 또는 판단 불가 시 ``None``.
        """
        for option in field.options:
            label = _norm(option.label)
            if any(_norm(word) in label for word in exclude):
                continue
            if any(_norm(word) in label for word in include):
                return option.label
        return None

    # ------------------------------------------------------------------
    # 턴 마무리
    # ------------------------------------------------------------------

    def _from_tool(
        self,
        user_text: str,
        result: IntentResult,
        calls: list[ToolCall],
        outcome: ToolResult,
    ) -> AgentTurn:
        """도구 결과 하나를 그대로 턴으로 바꾼다.

        :param user_text: 사용자 발화.
        :param result: 의도 분류 결과.
        :param calls: 도구 호출 목록.
        :param outcome: 도구 실행 결과.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        if outcome.handoff:
            reason = outcome.error or "담당 직원의 도움이 필요합니다."
            return self._handoff(
                user_text,
                result.intent.value,
                reason,
                self.state.current_field_id,
                result.confidence,
            )
        parts = [outcome.speech] if outcome.speech else []
        if not outcome.ok and not parts:
            parts.append(outcome.error or "요청을 처리하지 못했습니다.")
            parts.append(MODES_SENTENCE)
        return self._emit(
            user_text, result.intent.value, calls, parts, result.confidence
        )

    def _handoff(
        self,
        user_text: str,
        intent_name: str,
        reason: str,
        field_id: str | None,
        confidence: float,
    ) -> AgentTurn:
        """사람 지원 단계로 전이하고 그 턴을 만든다.

        :param user_text: 사용자 발화.
        :param intent_name: 분류된 의도 이름.
        :param reason: 넘기는 사유(한국어).
        :param field_id: 관련 항목 id.
        :param confidence: 의도 신뢰도.
        :returns: ``handoff_reason`` 이 채워진 :class:`~docagent.contracts.AgentTurn`.
        """
        if self.state.phase is not SessionPhase.HUMAN_HANDOFF:
            self.state.enter_handoff(reason, field_id=field_id)
        else:
            self.state.handoff_reason = reason
        return self._emit(
            user_text,
            intent_name,
            (),
            [reason, "담당 직원을 호출했습니다. 잠시만 기다려 주세요."],
            confidence,
            handoff_reason=reason,
            transition=False,
        )

    def _emit(
        self,
        user_text: str,
        intent_name: str,
        calls: Sequence[ToolCall],
        parts: Sequence[str],
        confidence: float,
        *,
        handoff_reason: str | None = None,
        transition: bool = True,
    ) -> AgentTurn:
        """발화를 합쳐 낭독하고 :class:`~docagent.contracts.AgentTurn` 을 만든다.

        :param user_text: 사용자 발화.
        :param intent_name: 분류된 의도 이름.
        :param calls: 이 턴의 도구 호출 목록.
        :param parts: 발화 조각(빈 문자열은 버린다).
        :param confidence: 의도 신뢰도.
        :param handoff_reason: 사람 지원 사유.
        :param transition: ``_last_speech`` 를 갱신할지 여부.
        :returns: :class:`~docagent.contracts.AgentTurn`.
        """
        speech = " ".join(part.strip() for part in parts if part and part.strip())
        turn = AgentTurn(
            user_text=user_text,
            intent=intent_name,
            tool_calls=tuple(calls),
            speech=speech,
            confidence=max(0.0, min(1.0, float(confidence))),
            handoff_reason=handoff_reason,
        )
        if speech:
            self.speech.speak(speech)
            if transition:
                self._last_speech = speech
        self.turns.append(turn)
        return turn
