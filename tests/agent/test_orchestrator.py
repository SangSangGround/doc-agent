"""에이전트 루프 테스트 — 로드맵 MVP 시연 장면 Step 1~7 전체 재현.

시나리오
--------
1. 문서 인식 안내
2. 첫 항목 안내(원문 듣기 / 쉬운 설명 / 다음 항목 3모드 제시)
3. "쉽게 설명해줘"
4. "동의할게"
5. 체크박스로 펜 이동(Mock MotionController 호출 검증)
6. 작성 확인 → 서명란 안내 → 이동 → 서명 확인
7. 완료 발화

여기에 번복·세션 복원·사람 지원 시나리오를 더한다.
"""

from __future__ import annotations

import pytest

from docagent.contracts import (
    PARTIAL_THRESHOLD,
    VISION_TRUST_THRESHOLD,
    BoxMm,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    Option,
    Point,
    Sensitivity,
    VerificationResult,
)
from docagent.errors import HandoffRequired, InvalidTransition
from docagent.agent.intent import Intent
from docagent.agent.orchestrator import (
    MAX_UNKNOWN_STREAK,
    MODES_SENTENCE,
    DocumentAgent,
)
from docagent.agent.state import SessionPhase, SessionState
from docagent.agent.tools import MAX_FIELD_FAILURES, ORIGINAL_NOTICE

AGREE_BOX = BoxMm(20.0, 100.0, 5.0, 5.0)
DISAGREE_BOX = BoxMm(60.0, 100.0, 5.0, 5.0)
SIGNATURE_BOX = BoxMm(120.0, 200.0, 50.0, 12.0)


# --------------------------------------------------------------------------
# Mock 구현 (이 파일 전용 — src/docagent/io 는 통합 담당 에이전트 소유)
# --------------------------------------------------------------------------


class MockMotion:
    """이동 이력을 남기는 시험용 펜 제어기."""

    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self._position = Point(0.0, 0.0)

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        self.moves.append((x_mm, y_mm))
        self._position = Point(x_mm, y_mm)
        return True

    def home(self) -> bool:
        self._position = Point(0.0, 0.0)
        return True

    def position(self) -> Point:
        return self._position


class MockSpeech:
    """낭독 문장을 모아 두는 시험용 음성 출력."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def listen(self, timeout_s: float = 10.0) -> str:
        return ""


class PlainExplainer:
    """항상 신뢰할 만한 쉬운 설명을 내놓는 시험용 설명기."""

    class Result:
        def __init__(self, text: str) -> None:
            self.text = text
            self.confidence = 0.95
            self.sources = ("개인정보 보호법 제15조",)
            self.needs_handoff = False
            self.disclaimer = "요약한 설명입니다."

    def explain(self, field: Field, question: str = "") -> "PlainExplainer.Result":
        return self.Result(
            f"'{field.title}' 항목은 신청서 처리에 필요한 정보를 모으는 데 "
            "동의하는지 묻는 것입니다."
        )


class FailingVerifier:
    """항상 미기입으로 판정하는 시험용 검증기."""

    def verify(self, field_id: str) -> VerificationResult:
        return VerificationResult(
            field_id=field_id,
            written=False,
            ink_ratio_before=0.01,
            ink_ratio_after=0.01,
            confidence=0.9,
            reason="잉크 증가량이 기준에 못 미칩니다.",
        )


class BlockingGuard:
    """특정 낱말이 보이면 곧바로 사람 지원을 요구하는 시험용 가드."""

    def __init__(self, trigger: str) -> None:
        self.trigger = trigger
        self.seen: list[str] = []

    def check(self, user_text: str, field: Field | None = None) -> None:
        self.seen.append(user_text)
        if self.trigger in user_text:
            raise HandoffRequired(
                "가드레일이 이 요청을 막았습니다. 담당 직원의 확인이 필요합니다.",
                field_id=None if field is None else field.id,
            )


class StepClock:
    """호출할 때마다 1초씩 나아가는 결정론적 시계."""

    def __init__(self) -> None:
        self.calls = 0

    def now_iso(self) -> str:
        self.calls += 1
        return f"2026-01-01T09:00:{self.calls:02d}+09:00"


# --------------------------------------------------------------------------
# 문서 구조
# --------------------------------------------------------------------------


def demo_structure(
    *, consent_confidence: float = 1.0, signature_role: FieldRole = FieldRole.APPLICANT
) -> DocumentStructure:
    """시연용 2항목 신청서(동의 선택 + 신청인 서명)."""
    return DocumentStructure(
        document_id="doc_demo",
        doc_title="○○지원금 지급 신청서",
        fields=(
            Field(
                id="consent_01",
                type=FieldType.CHOICE,
                title="개인정보 수집·이용 동의",
                role=FieldRole.APPLICANT,
                options=(
                    Option("동의함", AGREE_BOX),
                    Option("동의하지 않음", DISAGREE_BOX),
                ),
                required=True,
                sensitivity=Sensitivity.PUBLIC,
                box_mm=BoxMm(20.0, 90.0, 170.0, 20.0),
                clause_text=(
                    "수집하는 개인정보 항목은 성명, 연락처이며 보유 기간은 3년입니다."
                ),
                order=0,
                confidence=consent_confidence,
            ),
            Field(
                id="signature_01",
                type=FieldType.SIGNATURE,
                title="신청인 서명",
                role=signature_role,
                required=True,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=SIGNATURE_BOX,
                order=1,
                confidence=1.0,
            ),
        ),
    )


def build_agent(
    structure: DocumentStructure | None = None, **deps: object
) -> tuple[DocumentAgent, MockMotion, MockSpeech]:
    """Mock 을 물린 에이전트 한 벌을 만든다."""
    structure = structure if structure is not None else demo_structure()
    motion = MockMotion()
    speech = MockSpeech()
    deps.setdefault("explainer", PlainExplainer())
    agent = DocumentAgent(structure, motion=motion, speech=speech, **deps)  # type: ignore[arg-type]
    return agent, motion, speech


# --------------------------------------------------------------------------
# Step 1~7 시나리오
# --------------------------------------------------------------------------


class TestDemoScenario:
    """로드맵 '최종 MVP 에서 실제로 보여줄 장면' 전체 재현."""

    def test_step1_document_recognized(self) -> None:
        agent, _, speech = build_agent()
        turn = agent.start()
        assert "○○지원금 지급 신청서를 인식했습니다." in turn.speech
        assert "총 2개의 작성 항목이 있습니다." in turn.speech
        assert speech.spoken == [turn.speech]

    def test_step2_first_field_offers_three_modes(self) -> None:
        agent, _, _ = build_agent()
        turn = agent.start()
        assert "개인정보 수집·이용 동의" in turn.speech
        assert "동의함" in turn.speech and "동의하지 않음" in turn.speech
        assert MODES_SENTENCE in turn.speech
        for mode in ("원문 듣기", "쉬운 설명", "다음 항목"):
            assert mode in turn.speech
        assert agent.state.phase is SessionPhase.AWAIT_CHOICE

    def test_step3_plain_explanation_never_replaces_the_original(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("쉽게 설명해줘")
        assert turn.intent == Intent.EXPLAIN.value
        assert "동의하는지 묻는 것입니다" in turn.speech
        assert ORIGINAL_NOTICE in turn.speech
        assert "개인정보 보호법 제15조" in turn.speech

    def test_step4_and_5_agree_moves_pen_to_the_checkbox(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        agent.handle("쉽게 설명해줘")
        turn = agent.handle("동의할게")
        assert turn.intent == Intent.AGREE.value
        assert [call.name for call in turn.tool_calls] == [
            "select_option",
            "move_to_field",
        ]
        assert agent.state.selected_options == {"consent_01": "동의함"}
        assert motion.moves == [(AGREE_BOX.center().x_mm, AGREE_BOX.center().y_mm)]
        assert agent.state.phase is SessionPhase.AWAIT_WRITE

    def test_step6_write_confirmation_advances_to_signature(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        agent.handle("동의할게")
        turn = agent.handle("다 썼어요")
        assert "기입을 확인했습니다" in turn.speech
        assert "신청인 서명" in turn.speech
        assert MODES_SENTENCE in turn.speech
        assert agent.state.completed_fields == ["consent_01"]
        assert agent.state.current_field_id == "signature_01"

    def test_step6_signature_guidance_and_move(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        agent.handle("동의할게")
        agent.handle("다 썼어요")
        turn = agent.handle("네")
        assert "서명해 주세요" in turn.speech
        assert motion.moves[-1] == (
            SIGNATURE_BOX.center().x_mm,
            SIGNATURE_BOX.center().y_mm,
        )

    def test_step7_completion_speech(self) -> None:
        agent, _, speech = build_agent()
        agent.start()
        agent.handle("동의할게")
        agent.handle("다 썼어요")
        agent.handle("네")
        turn = agent.handle("서명했어요")
        assert agent.state.phase is SessionPhase.COMPLETED
        assert "모든 항목의 안내를 마쳤습니다." in turn.speech
        assert "전체 2개 항목 중 2개를 작성했습니다." in turn.speech
        assert "필수 항목은 모두 작성되었습니다." in turn.speech
        assert agent.state.progress() == (2, 2, [])

    def test_full_scenario_moves_exactly_twice(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        for text in ("쉽게 설명해줘", "동의할게", "다 썼어요", "네", "서명했어요"):
            agent.handle(text)
        assert motion.moves == [
            (AGREE_BOX.center().x_mm, AGREE_BOX.center().y_mm),
            (SIGNATURE_BOX.center().x_mm, SIGNATURE_BOX.center().y_mm),
        ]


class TestThreeModes:
    """세 가지 모드가 실제로 동작하는지."""

    def test_read_original_is_verbatim(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("원문 읽어줘")
        assert turn.intent == Intent.READ_ORIGINAL.value
        assert (
            "수집하는 개인정보 항목은 성명, 연락처이며 보유 기간은 3년입니다."
            in turn.speech
        )
        assert ORIGINAL_NOTICE not in turn.speech

    def test_next_skips_without_completing(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("다음 항목")
        assert agent.state.current_field_id == "signature_01"
        assert agent.state.skipped_fields == ["consent_01"]
        assert "신청인 서명" in turn.speech

    def test_repeat_replays_the_last_speech(self) -> None:
        agent, _, _ = build_agent()
        first = agent.start()
        turn = agent.handle("다시 들려줘")
        assert turn.speech == first.speech

    def test_previous_returns_to_earlier_field(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        agent.handle("다음 항목")
        turn = agent.handle("이전으로")
        assert agent.state.current_field_id == "consent_01"
        assert "개인정보 수집·이용 동의" in turn.speech


class TestRevision:
    """번복 시나리오 — '아까 동의한 거 취소할래'."""

    def test_revert_clears_selection_and_allows_reselection(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        agent.handle("동의할게")
        agent.handle("다 썼어요")
        assert agent.state.completed_fields == ["consent_01"]

        turn = agent.handle("아까 동의한 거 취소할래")
        assert turn.intent == Intent.REVISE.value
        assert [call.name for call in turn.tool_calls] == ["revise_field"]
        assert agent.state.current_field_id == "consent_01"
        assert agent.state.selected_options == {}
        assert agent.state.completed_fields == []
        assert agent.state.phase is SessionPhase.AWAIT_CHOICE
        assert "되돌아갑니다" in turn.speech
        assert MODES_SENTENCE in turn.speech

        again = agent.handle("동의하지 않을래요")
        assert agent.state.selected_options == {"consent_01": "동의하지 않음"}
        assert motion.moves[-1] == (
            DISAGREE_BOX.center().x_mm,
            DISAGREE_BOX.center().y_mm,
        )
        assert again.intent == Intent.DISAGREE.value

    def test_revert_is_recorded_in_history(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        agent.handle("동의할게")
        agent.handle("다 썼어요")
        agent.handle("아까 동의한 거 취소할래")
        actions = [entry.action for entry in agent.state.history]
        assert "revert_to_field" in actions

    def test_revert_without_target_asks_back(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("아까 그거 취소할래")
        # 완료된 항목이 없으면 현재 항목으로 되돌린다(추측하지 않는다).
        assert turn.intent == Intent.REVISE.value
        assert agent.state.current_field_id == "consent_01"


class TestSessionRestore:
    """세션 복원율 100% — 어느 시점에 저장해도 이어서 같은 결과."""

    def test_restored_agent_continues_identically(self) -> None:
        agent_a, _, _ = build_agent()
        agent_a.start()
        agent_a.handle("쉽게 설명해줘")
        agent_a.handle("동의할게")
        dumped = agent_a.session_json()

        agent_b = DocumentAgent.restore(
            demo_structure(),
            dumped,
            explainer=PlainExplainer(),
            motion=MockMotion(),
            speech=MockSpeech(),
        )
        assert agent_b.state == agent_a.state

        for text in ("다 썼어요", "네", "서명했어요"):
            assert agent_a.handle(text).speech == agent_b.handle(text).speech
        assert agent_a.state.to_dict() == agent_b.state.to_dict()
        assert agent_b.state.phase is SessionPhase.COMPLETED

    def test_restore_rejects_mismatched_document(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        other = DocumentStructure(document_id="다른문서")
        with pytest.raises(ValueError, match="document_id"):
            DocumentAgent.restore(other, agent.session_json())

    def test_session_json_is_readable_korean(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        agent.handle("동의할게")
        assert "동의함" in agent.session_json()


class TestHumanHandoff:
    """사람 지원 트리거 6종."""

    def test_legal_question_hands_off(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("제가 이 지원금을 받을 수 있나요?")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert turn.handoff_reason is not None
        assert "판단" in turn.handoff_reason
        assert agent.state.handoff_reason == turn.handoff_reason
        assert turn.needs_handoff() is True

    def test_handoff_reason_is_audited(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        agent.handle("제가 이 지원금을 받을 자격이 되나요?")
        entries = [e for e in agent.state.history if e.action == "handoff"]
        assert entries and entries[-1].note == agent.state.handoff_reason

    def test_low_vision_confidence_hands_off_at_announcement(self) -> None:
        structure = demo_structure(consent_confidence=VISION_TRUST_THRESHOLD - 0.05)
        agent, motion, _ = build_agent(structure)
        turn = agent.start()
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "정확히 읽지 못했습니다" in (turn.handoff_reason or "")
        assert motion.moves == []

    def test_ambiguous_signature_hands_off(self) -> None:
        structure = DocumentStructure(
            document_id="doc_sig",
            doc_title="시험용 신청서",
            fields=(
                Field(
                    id="sig_a",
                    type=FieldType.SIGNATURE,
                    title="서명란 1",
                    role=FieldRole.UNKNOWN,
                    box_mm=SIGNATURE_BOX,
                    order=0,
                    confidence=1.0,
                ),
                Field(
                    id="sig_b",
                    type=FieldType.SIGNATURE,
                    title="서명란 2",
                    role=FieldRole.UNKNOWN,
                    box_mm=BoxMm(60.0, 200.0, 50.0, 12.0),
                    order=1,
                    confidence=1.0,
                ),
            ),
        )
        agent, _, _ = build_agent(structure)
        turn = agent.start()
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "서명란이 여러 개" in (turn.handoff_reason or "")

    def test_low_explanation_confidence_hands_off(self) -> None:
        class Weak:
            class Result:
                text = "잘 모르겠지만 아마도…"
                confidence = PARTIAL_THRESHOLD - 0.2
                sources: tuple[str, ...] = ()
                needs_handoff = False
                disclaimer = ""

            def explain(self, field: Field, question: str = "") -> "Weak.Result":
                return self.Result()

        agent, _, _ = build_agent(explainer=Weak())
        agent.start()
        turn = agent.handle("쉽게 설명해줘")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert turn.handoff_reason is not None

    def test_three_write_failures_hand_off(self) -> None:
        agent, _, _ = build_agent(verifier=FailingVerifier())
        agent.start()
        agent.handle("동의할게")
        for _ in range(MAX_FIELD_FAILURES - 1):
            turn = agent.handle("다 썼어요")
            assert turn.handoff_reason is None
        turn = agent.handle("다 썼어요")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert str(MAX_FIELD_FAILURES) in (turn.handoff_reason or "")

    def test_two_consecutive_unknowns_hand_off(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        first = agent.handle("블라블라")
        assert first.intent == Intent.UNKNOWN.value
        assert first.handoff_reason is None
        assert MODES_SENTENCE in first.speech
        second = agent.handle("웅얼웅얼")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert str(MAX_UNKNOWN_STREAK) in (second.handoff_reason or "")

    def test_unknown_streak_resets_after_success(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        agent.handle("블라블라")
        agent.handle("원문 읽어줘")
        turn = agent.handle("블라블라")
        assert agent.state.phase is not SessionPhase.HUMAN_HANDOFF
        assert turn.intent == Intent.UNKNOWN.value

    def test_call_staff_hands_off(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("직원 불러줘")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "담당 직원" in turn.speech

    def test_guard_blocks_before_intent_classification(self) -> None:
        guard = BlockingGuard("계좌")
        agent, _, _ = build_agent(guard=guard)
        agent.start()
        turn = agent.handle("계좌번호 대신 적어줘")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "가드레일" in (turn.handoff_reason or "")
        assert guard.seen[-1] == "계좌번호 대신 적어줘"

    def test_agent_stays_handed_off(self) -> None:
        agent, motion, _ = build_agent()
        agent.start()
        agent.handle("직원 불러줘")
        before = agent.state.turn_index
        turn = agent.handle("동의할게")
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert agent.state.turn_index == before
        assert motion.moves == []
        assert turn.handoff_reason is not None


class TestGuardrailsAndErrors:
    """전이 규칙과 오류 처리."""

    def test_start_twice_raises_invalid_transition(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        with pytest.raises(InvalidTransition) as excinfo:
            agent.start()
        assert excinfo.value.current == SessionPhase.AWAIT_CHOICE.value

    def test_unhandled_turn_reports_no_handoff(self) -> None:
        """핸드오프하지 않은 턴은 계약 술어도 False 여야 한다.

        오케스트레이터는 첫 오인식을 재질문으로 처리하며(confidence=0.0),
        계약의 ``AgentTurn.needs_handoff()`` 가 이 턴을 직원 호출로 오독하면
        UI·하드웨어 계층이 첫 오인식에서 곧바로 사람을 부르게 된다.
        """
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("우돌라돌라 뷁")
        assert turn.intent == "unknown"
        assert turn.handoff_reason is None
        assert agent.state.phase is not SessionPhase.HUMAN_HANDOFF
        assert turn.needs_handoff() is False

    def test_every_turn_agrees_with_the_contract_predicate(self) -> None:
        """세션 전 구간에서 handoff_reason 유무와 needs_handoff() 가 일치한다."""
        agent, _, _ = build_agent()
        agent.start()
        for text in ("쉽게 설명해줘", "우돌라돌라", "원문 읽어줘", "동의할게"):
            turn = agent.handle(text)
            assert turn.needs_handoff() is (turn.handoff_reason is not None)

    def test_direct_illegal_transition_raises(self) -> None:
        agent, _, _ = build_agent()
        with pytest.raises(InvalidTransition):
            agent.state.transition_to(SessionPhase.VERIFYING)

    def test_state_document_mismatch_rejected(self) -> None:
        state = SessionState(document_id="다른문서")
        with pytest.raises(ValueError, match="document_id"):
            DocumentAgent(demo_structure(), state=state)

    def test_defaults_work_without_any_injection(self) -> None:
        """의존성을 하나도 주지 않아도 단독으로 끝까지 진행된다."""
        agent = DocumentAgent(demo_structure())
        agent.start()
        for text in ("원문 읽어줘", "동의할게", "다 썼어요", "네", "서명했어요"):
            agent.handle(text)
        assert agent.state.phase is SessionPhase.COMPLETED

    def test_explain_without_explainer_reads_the_clause(self) -> None:
        agent = DocumentAgent(demo_structure())
        agent.start()
        turn = agent.handle("쉽게 설명해줘")
        assert "보유 기간은 3년입니다." in turn.speech
        assert ORIGINAL_NOTICE in turn.speech

    def test_cancel_does_not_change_state(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        before = agent.state.phase
        turn = agent.handle("그만할래")
        assert agent.state.phase is before
        assert "직원 불러줘" in turn.speech


class TestDeterminism:
    """같은 입력열은 항상 같은 발화열을 만든다."""

    SCRIPT = ("쉽게 설명해줘", "원문 읽어줘", "동의할게", "다 썼어요", "네", "서명했어요")

    def _run(self) -> list[str]:
        agent, _, speech = build_agent(clock=StepClock())
        agent.start()
        for text in self.SCRIPT:
            agent.handle(text)
        return speech.spoken

    def test_two_runs_produce_identical_speech(self) -> None:
        assert self._run() == self._run()

    def test_clock_is_the_only_time_source(self) -> None:
        clock = StepClock()
        agent, _, _ = build_agent(clock=clock)
        agent.start()
        agent.handle("원문 읽어줘")
        notes = [e.note for e in agent.state.history if e.action == "turn_time"]
        assert notes == ["2026-01-01T09:00:02+09:00"]

    def test_turn_records_tool_calls(self) -> None:
        agent, _, _ = build_agent()
        agent.start()
        turn = agent.handle("동의할게")
        assert turn.tool_calls[0].arguments == {"option_label": "동의함"}
        assert turn.tool_calls[1].arguments == {}
        assert "x_mm" not in str(turn.tool_calls)


class TestSharedFixtureIntegration:
    """공용 픽스처(``sample_structure``, 8개 항목)로 전체 흐름을 한 번 돈다."""

    @staticmethod
    def _agent(sample_structure: DocumentStructure) -> tuple[DocumentAgent, MockMotion]:
        motion = MockMotion()
        return (
            DocumentAgent(
                sample_structure,
                motion=motion,
                speech=MockSpeech(),
                explainer=PlainExplainer(),
            ),
            motion,
        )

    def test_start_announces_every_field(
        self, sample_structure: DocumentStructure
    ) -> None:
        agent, _ = self._agent(sample_structure)
        turn = agent.start()
        assert f"총 {len(sample_structure.fields)}개의 작성 항목이 있습니다." in turn.speech
        assert MODES_SENTENCE in turn.speech

    def test_walks_the_whole_form_to_completion(
        self, sample_structure: DocumentStructure
    ) -> None:
        agent, motion = self._agent(sample_structure)
        agent.start()
        for _ in range(len(sample_structure.fields)):
            agent.handle("네")
            agent.handle("다 썼어요")
        assert agent.state.phase is SessionPhase.COMPLETED
        assert len(agent.state.completed_fields) == len(sample_structure.fields)
        assert agent.state.progress()[2] == []
        assert len(motion.moves) == len(sample_structure.fields)

    def test_private_field_values_are_never_spoken(
        self, sample_structure: DocumentStructure
    ) -> None:
        """개인정보 항목에서도 예시 값이 발화에 섞이지 않는다."""
        agent, _ = self._agent(sample_structure)
        speech = agent.speech
        agent.start()
        for _ in range(len(sample_structure.fields)):
            agent.handle("쉽게 설명해줘")
            agent.handle("원문 읽어줘")
            agent.handle("네")
            agent.handle("다 썼어요")
        blob = " ".join(speech.spoken)
        for leak in ("홍길동", "900101", "010-1234-5678", "세종대로"):
            assert leak not in blob

    def test_representative_signature_is_optional(
        self, sample_structure: DocumentStructure
    ) -> None:
        agent, _ = self._agent(sample_structure)
        agent.start()
        required = set(agent.state.required_field_ids)
        assert "signature_applicant" in required
        assert "signature_representative" not in required


class TestSingleAmbiguousSignature:
    """서명란이 하나뿐이어도 기입 주체가 모호하면 사람에게 넘긴다.

    개수 조건을 달면 신뢰도 페널티 값에 우연히 기대게 된다(페널티를 조정하거나
    다른 Detector 가 confidence>=0.85 인 role=UNKNOWN 서명란을 만들면 모호한
    칸이 그대로 안내된다).
    """

    def _single_signature_structure(self) -> DocumentStructure:
        """주체를 확정하지 못한 서명란 하나짜리 문서."""
        return DocumentStructure(
            document_id="doc_single_sig",
            doc_title="시험용 신청서",
            fields=(
                Field(
                    id="sig_only",
                    type=FieldType.SIGNATURE,
                    title="서명",
                    role=FieldRole.UNKNOWN,
                    box_mm=SIGNATURE_BOX,
                    order=0,
                    confidence=1.0,
                ),
            ),
        )

    def test_hands_off_even_with_high_confidence(self) -> None:
        """신뢰도가 1.0 이어도 주체 미상 서명란은 안내하지 않는다."""
        agent, motion, _ = build_agent(self._single_signature_structure())
        turn = agent.start()
        assert agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "서명" in (turn.handoff_reason or "")
        assert motion.moves == []

    def test_known_role_signature_is_announced(self) -> None:
        """역할이 확정된 서명란은 종전대로 정상 안내된다(과잉 차단 금지)."""
        structure = DocumentStructure(
            document_id="doc_known_sig",
            doc_title="시험용 신청서",
            fields=(
                Field(
                    id="sig_applicant",
                    type=FieldType.SIGNATURE,
                    title="신청인 서명",
                    role=FieldRole.APPLICANT,
                    box_mm=SIGNATURE_BOX,
                    order=0,
                    confidence=1.0,
                ),
            ),
        )
        agent, _, _ = build_agent(structure)
        agent.start()
        assert agent.state.phase is not SessionPhase.HUMAN_HANDOFF
