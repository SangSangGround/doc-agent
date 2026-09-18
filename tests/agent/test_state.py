"""세션 상태머신 테스트 — 전이 규칙 · 되돌림 · 세션 복원율 100%."""

from __future__ import annotations

import pytest

from docagent.contracts import (
    BoxMm,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    Option,
    Sensitivity,
)
from docagent.errors import InvalidTransition
from docagent.agent.state import (
    MAX_CHECKPOINTS,
    TRANSITIONS,
    HistoryEntry,
    SessionPhase,
    SessionState,
)


def _structure() -> DocumentStructure:
    """3개 항목(선택형 1 + 서명 1 + 개인정보 1)짜리 시험용 문서 구조를 만든다."""
    return DocumentStructure(
        document_id="doc_state",
        doc_title="시험용 신청서",
        fields=(
            Field(
                id="name_01",
                type=FieldType.TEXT_INPUT,
                title="성명",
                role=FieldRole.APPLICANT,
                required=True,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=BoxMm(20.0, 40.0, 60.0, 8.0),
                order=0,
                confidence=1.0,
            ),
            Field(
                id="consent_01",
                type=FieldType.CHOICE,
                title="개인정보 수집·이용 동의",
                role=FieldRole.APPLICANT,
                options=(
                    Option("동의함", BoxMm(20.0, 100.0, 5.0, 5.0)),
                    Option("동의하지 않음", BoxMm(60.0, 100.0, 5.0, 5.0)),
                ),
                required=True,
                sensitivity=Sensitivity.PUBLIC,
                box_mm=BoxMm(20.0, 90.0, 170.0, 20.0),
                clause_text="수집 항목은 성명과 연락처입니다.",
                order=1,
                confidence=1.0,
            ),
            Field(
                id="signature_01",
                type=FieldType.SIGNATURE,
                title="신청인 서명",
                role=FieldRole.APPLICANT,
                required=False,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=BoxMm(120.0, 200.0, 50.0, 12.0),
                order=2,
                confidence=1.0,
            ),
        ),
    )


class TestTransitionTable:
    """전이 테이블 자체의 정합성."""

    def test_every_phase_has_an_entry(self) -> None:
        assert set(TRANSITIONS) == set(SessionPhase)

    def test_every_phase_can_reach_handoff(self) -> None:
        for phase, allowed in TRANSITIONS.items():
            assert SessionPhase.HUMAN_HANDOFF in allowed, phase

    def test_handoff_is_absorbing(self) -> None:
        assert TRANSITIONS[SessionPhase.HUMAN_HANDOFF] == frozenset(
            {SessionPhase.HUMAN_HANDOFF}
        )

    def test_allowed_transitions_is_readonly(self) -> None:
        assert isinstance(
            SessionState.allowed_transitions(SessionPhase.IDLE), frozenset
        )


class TestTransitions:
    """전이 실행과 위반 처리."""

    def test_valid_transition_records_history(self) -> None:
        state = SessionState.from_structure(_structure())
        state.transition_to(SessionPhase.DOCUMENT_LOADED, note="문서를 읽었습니다.")
        assert state.phase is SessionPhase.DOCUMENT_LOADED
        assert state.history[-1].phase is SessionPhase.DOCUMENT_LOADED
        assert state.history[-1].note == "문서를 읽었습니다."

    def test_invalid_transition_raises(self) -> None:
        state = SessionState.from_structure(_structure())
        with pytest.raises(InvalidTransition) as excinfo:
            state.transition_to(SessionPhase.VERIFYING)
        assert excinfo.value.current == SessionPhase.IDLE.value
        assert excinfo.value.requested == SessionPhase.VERIFYING.value

    def test_invalid_transition_leaves_state_untouched(self) -> None:
        state = SessionState.from_structure(_structure())
        before = len(state.history)
        with pytest.raises(InvalidTransition):
            state.transition_to(SessionPhase.COMPLETED)
        assert state.phase is SessionPhase.IDLE
        assert len(state.history) == before

    def test_handoff_is_reachable_from_any_phase(self) -> None:
        for phase in SessionPhase:
            state = SessionState.from_structure(_structure())
            state.phase = phase
            state.enter_handoff("담당 직원의 확인이 필요합니다.")
            assert state.phase is SessionPhase.HUMAN_HANDOFF
            assert state.handoff_reason == "담당 직원의 확인이 필요합니다."

    def test_agent_cannot_leave_handoff(self) -> None:
        state = SessionState.from_structure(_structure())
        state.enter_handoff("사유")
        with pytest.raises(InvalidTransition):
            state.transition_to(SessionPhase.ANNOUNCE_FIELD)


class TestFieldProgress:
    """항목 진행 · 건너뜀 · 완료."""

    def test_from_structure_fills_queues(self) -> None:
        state = SessionState.from_structure(_structure())
        assert state.field_order == ["name_01", "consent_01", "signature_01"]
        assert state.pending_fields == state.field_order
        assert state.required_field_ids == ["name_01", "consent_01"]
        assert state.phase is SessionPhase.IDLE

    def test_next_pending_walks_document_order(self) -> None:
        state = SessionState.from_structure(_structure())
        assert state.next_pending() == "name_01"
        state.set_current("name_01")
        assert state.next_pending() == "consent_01"

    def test_next_pending_wraps_back_to_skipped_gap(self) -> None:
        """뒤쪽에 남은 항목이 없으면 앞쪽에 건너뛴 항목으로 되돌아간다."""
        state = SessionState.from_structure(_structure())
        state.complete_field("consent_01")
        state.complete_field("signature_01")
        state.set_current("signature_01")
        assert state.next_pending() == "name_01"

    def test_next_pending_never_returns_current(self) -> None:
        state = SessionState.from_structure(_structure())
        state.complete_field("consent_01")
        state.complete_field("signature_01")
        state.set_current("name_01")
        assert state.next_pending() is None

    def test_complete_removes_from_pending(self) -> None:
        state = SessionState.from_structure(_structure())
        state.complete_field("name_01")
        assert "name_01" not in state.pending_fields
        assert state.completed_fields == ["name_01"]

    def test_skip_is_not_completion(self) -> None:
        state = SessionState.from_structure(_structure())
        state.skip_field("name_01")
        assert state.skipped_fields == ["name_01"]
        assert state.completed_fields == []
        done, total, remaining = state.progress()
        assert (done, total) == (0, 3)
        assert remaining == ["name_01", "consent_01"]

    def test_progress_reports_remaining_required(self) -> None:
        state = SessionState.from_structure(_structure())
        state.complete_field("name_01")
        done, total, remaining = state.progress()
        assert (done, total, remaining) == (1, 3, ["consent_01"])

    def test_is_complete_when_pending_empty(self) -> None:
        state = SessionState.from_structure(_structure())
        for fid in list(state.pending_fields):
            state.complete_field(fid)
        assert state.is_complete() is True

    def test_previous_field(self) -> None:
        state = SessionState.from_structure(_structure())
        state.set_current("consent_01")
        assert state.previous_field() == "name_01"
        state.set_current("name_01")
        assert state.previous_field() is None

    def test_set_current_rejects_unknown_field(self) -> None:
        state = SessionState.from_structure(_structure())
        with pytest.raises(ValueError, match="존재하지 않는"):
            state.set_current("없는항목")


class TestRevert:
    """번복(revert_to_field) — '아까 동의한 거 다시 바꿀래'."""

    def _advanced(self) -> SessionState:
        state = SessionState.from_structure(_structure())
        state.transition_to(SessionPhase.DOCUMENT_LOADED)
        state.transition_to(SessionPhase.ANNOUNCE_FIELD, field_id="name_01")
        state.complete_field("name_01")
        state.set_current("consent_01")
        state.select_option("consent_01", "동의함")
        state.mark_verified("consent_01")
        state.complete_field("consent_01")
        state.transition_to(SessionPhase.FIELD_DONE)
        state.set_current("signature_01")
        return state

    def test_revert_clears_selection_and_completion(self) -> None:
        state = self._advanced()
        state.revert_to_field("consent_01")
        assert state.current_field_id == "consent_01"
        assert state.phase is SessionPhase.ANNOUNCE_FIELD
        assert "consent_01" not in state.completed_fields
        assert "consent_01" not in state.verified_fields
        assert "consent_01" not in state.selected_options
        assert state.pending_fields == ["consent_01", "signature_01"]

    def test_revert_keeps_earlier_fields_done(self) -> None:
        state = self._advanced()
        state.revert_to_field("consent_01")
        assert state.completed_fields == ["name_01"]

    def test_revert_is_audited(self) -> None:
        state = self._advanced()
        state.revert_to_field("consent_01")
        actions = [entry.action for entry in state.history]
        assert "revert_to_field" in actions

    def test_revert_clears_failure_counter(self) -> None:
        state = self._advanced()
        state.counters["fail:consent_01"] = 2
        state.revert_to_field("consent_01")
        assert "fail:consent_01" not in state.counters

    def test_revert_rejects_unknown_field(self) -> None:
        state = self._advanced()
        with pytest.raises(ValueError, match="존재하지 않는"):
            state.revert_to_field("없는항목")

    def test_revert_from_completed_phase(self) -> None:
        state = self._advanced()
        state.complete_field("signature_01")
        state.transition_to(SessionPhase.COMPLETED)
        state.revert_to_field("consent_01")
        assert state.phase is SessionPhase.ANNOUNCE_FIELD
        assert state.pending_fields == ["consent_01", "signature_01"]

    def test_revert_is_blocked_after_handoff(self) -> None:
        state = self._advanced()
        state.enter_handoff("직원 호출")
        with pytest.raises(InvalidTransition):
            state.revert_to_field("consent_01")


class TestUndo:
    """직전 턴 취소."""

    def test_undo_restores_previous_turn(self) -> None:
        state = SessionState.from_structure(_structure())
        state.transition_to(SessionPhase.DOCUMENT_LOADED)
        state.begin_turn()
        state.set_current("consent_01")
        state.select_option("consent_01", "동의함")
        assert state.undo_last() is True
        assert state.selected_options == {}
        assert state.current_field_id is None
        assert state.turn_index == 0

    def test_undo_keeps_audit_trail(self) -> None:
        state = SessionState.from_structure(_structure())
        state.begin_turn()
        state.select_option("consent_01", "동의함")
        before = len(state.history)
        state.undo_last()
        assert len(state.history) > before
        assert state.history[-1].action == "undo_last"
        assert any(entry.action == "select_option" for entry in state.history)

    def test_undo_without_checkpoint_returns_false(self) -> None:
        state = SessionState.from_structure(_structure())
        assert state.undo_last() is False
        assert state.history[-1].action == "undo_last_noop"

    def test_checkpoints_are_capped(self) -> None:
        state = SessionState.from_structure(_structure())
        for _ in range(MAX_CHECKPOINTS + 5):
            state.begin_turn()
        assert len(state.checkpoints) == MAX_CHECKPOINTS


class TestSerialization:
    """세션 복원율 100% 를 증명한다."""

    def _rich_state(self) -> SessionState:
        state = SessionState.from_structure(_structure())
        state.transition_to(SessionPhase.DOCUMENT_LOADED)
        state.begin_turn()
        state.transition_to(SessionPhase.ANNOUNCE_FIELD, field_id="consent_01")
        state.select_option("consent_01", "동의하지 않음")
        state.bump("fail:consent_01")
        state.begin_turn()
        state.skip_field("name_01")
        state.mark_verified("consent_01")
        state.complete_field("consent_01")
        return state

    def test_json_roundtrip_is_lossless(self) -> None:
        state = self._rich_state()
        assert SessionState.from_json(state.to_json()) == state

    def test_dict_roundtrip_is_lossless(self) -> None:
        state = self._rich_state()
        assert SessionState.from_dict(state.to_dict()) == state

    def test_json_keeps_korean_readable(self) -> None:
        state = self._rich_state()
        assert "동의하지 않음" in state.to_json()

    def test_history_entry_roundtrip(self) -> None:
        entry = HistoryEntry(3, SessionPhase.GUIDING, "consent_01", "select", "메모")
        assert HistoryEntry.from_dict(entry.to_dict()) == entry

    def test_missing_document_id_raises(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            SessionState.from_dict({"phase": "idle"})

    def test_unknown_phase_value_raises(self) -> None:
        with pytest.raises(ValueError, match="정의되지 않은"):
            SessionState.from_dict({"document_id": "d", "phase": "없는단계"})

    def test_broken_json_raises(self) -> None:
        with pytest.raises(ValueError, match="파싱"):
            SessionState.from_json("{이건 JSON 이 아니다")

    def test_empty_document_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="빈 문자열"):
            SessionState(document_id="")

    def test_restored_state_continues_identically(self) -> None:
        state = self._rich_state()
        restored = SessionState.from_json(state.to_json())
        state.set_current("signature_01")
        restored.set_current("signature_01")
        assert state.to_dict() == restored.to_dict()
