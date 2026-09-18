"""Tool Calling 파이프라인 테스트.

핵심 검증: **LLM 은 좌표를 만들 수 없다.** ``move_to_field`` 스키마에 좌표 인자가
없고, 펜의 목표점은 항상 문서 구조에서 로컬 코드가 직접 읽어야 한다.
"""

from __future__ import annotations

import json

import pytest

from docagent.contracts import (
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
from docagent.errors import ToolExecutionError
from docagent.agent.state import SessionPhase, SessionState
from docagent.agent.tools import (
    MAX_FIELD_FAILURES,
    ORIGINAL_NOTICE,
    Tool,
    ToolRegistry,
)


# --------------------------------------------------------------------------
# Mock 구현 (이 파일 안에서만 쓴다 — src/docagent/io 는 통합 담당 소유)
# --------------------------------------------------------------------------


class MockMotion:
    """호출 이력을 남기는 시험용 펜 제어기."""

    def __init__(self, *, succeed: bool = True, drift_mm: float = 0.0) -> None:
        self.succeed = succeed
        self.drift_mm = drift_mm
        self.moves: list[tuple[float, float]] = []
        self._position = Point(0.0, 0.0)

    def move_to(self, x_mm: float, y_mm: float) -> bool:
        self.moves.append((x_mm, y_mm))
        if not self.succeed:
            return False
        self._position = Point(x_mm + self.drift_mm, y_mm)
        return True

    def home(self) -> bool:
        self._position = Point(0.0, 0.0)
        return True

    def position(self) -> Point:
        return self._position


class MockSpeech:
    """낭독 내용을 모아 두는 시험용 음성 출력."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    def listen(self, timeout_s: float = 10.0) -> str:
        return ""


class FailingVerifier:
    """항상 미기입으로 판정하는 검증기."""

    def verify(self, field_id: str) -> VerificationResult:
        return VerificationResult(
            field_id=field_id,
            written=False,
            ink_ratio_before=0.01,
            ink_ratio_after=0.01,
            confidence=0.95,
            reason="잉크 증가량이 기준에 못 미칩니다.",
        )


class LowConfidenceExplainer:
    """신뢰도가 낮은 설명을 내놓는 설명기."""

    class Result:
        text = "아마도 이런 뜻일 겁니다."
        confidence = 0.42
        sources: tuple[str, ...] = ()
        needs_handoff = False
        disclaimer = ""

    def explain(self, field: Field, question: str = "") -> "LowConfidenceExplainer.Result":
        return self.Result()


# --------------------------------------------------------------------------
# 픽스처
# --------------------------------------------------------------------------


def _structure(*, consent_confidence: float = 1.0) -> DocumentStructure:
    return DocumentStructure(
        document_id="doc_tools",
        doc_title="시험용 신청서",
        fields=(
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
                clause_text="수집 항목은 성명과 연락처이며 보유 기간은 3년입니다.",
                order=0,
                confidence=consent_confidence,
            ),
            Field(
                id="signature_01",
                type=FieldType.SIGNATURE,
                title="신청인 서명",
                role=FieldRole.APPLICANT,
                required=True,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=BoxMm(120.0, 200.0, 50.0, 12.0),
                order=1,
                confidence=1.0,
            ),
            Field(
                id="floating_01",
                type=FieldType.TEXT_INPUT,
                title="위치 미상 항목",
                role=FieldRole.APPLICANT,
                required=False,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=None,
                order=2,
                confidence=1.0,
            ),
        ),
    )


def _ready_state(structure: DocumentStructure, field_id: str) -> SessionState:
    """항목 안내 단계까지 진행된 세션 상태를 만든다."""
    state = SessionState.from_structure(structure)
    state.transition_to(SessionPhase.DOCUMENT_LOADED)
    state.transition_to(SessionPhase.ANNOUNCE_FIELD, field_id=field_id)
    return state


@pytest.fixture()
def registry() -> ToolRegistry:
    structure = _structure()
    state = SessionState.from_structure(structure)
    state.transition_to(SessionPhase.DOCUMENT_LOADED)
    state.transition_to(SessionPhase.ANNOUNCE_FIELD, field_id="consent_01")
    return ToolRegistry(structure, state, motion=MockMotion(), speech=MockSpeech())


# --------------------------------------------------------------------------
# 테스트
# --------------------------------------------------------------------------


class TestRegistryBasics:
    def test_default_tools_are_registered(self, registry: ToolRegistry) -> None:
        assert set(registry.names()) == {
            "explain",
            "read_original",
            "next_field",
            "previous_field",
            "go_to_field",
            "select_option",
            "move_to_field",
            "verify_field",
            "repeat",
            "revise_field",
            "request_human",
            "summarize_progress",
        }

    def test_duplicate_registration_rejected(self, registry: ToolRegistry) -> None:
        tool = registry.get("repeat")
        with pytest.raises(ToolExecutionError, match="이미 등록된"):
            registry.register(tool)

    def test_unknown_tool_is_reported_not_raised(self, registry: ToolRegistry) -> None:
        result = registry.call("없는도구")
        assert result.ok is False
        assert "등록되지 않은" in (result.error or "")

    def test_tool_schema_must_be_object(self) -> None:
        with pytest.raises(ValueError, match="object"):
            Tool("t", "설명", {"type": "string"}, lambda args: None)  # type: ignore[arg-type]


class TestDescribeSafety:
    """LLM 에 넘기는 스키마에 좌표·개인정보가 새지 않는지."""

    def test_describe_returns_function_schemas(self, registry: ToolRegistry) -> None:
        described = registry.describe()
        assert len(described) == len(registry.names())
        for item in described:
            assert set(item) == {"name", "description", "parameters"}
            assert item["parameters"]["type"] == "object"

    def test_no_coordinate_arguments_anywhere(self, registry: ToolRegistry) -> None:
        blob = json.dumps(
            [item["parameters"] for item in registry.describe()], ensure_ascii=False
        ).lower()
        for token in ("x_mm", "y_mm", "w_mm", "h_mm", "box", "coord", "point", "px"):
            assert token not in blob

    def test_move_to_field_takes_only_field_id(self, registry: ToolRegistry) -> None:
        schema = registry.get("move_to_field").json_schema
        assert set(schema["properties"]) == {"field_id"}
        assert schema["additionalProperties"] is False

    def test_no_document_values_leak_into_schemas(
        self, registry: ToolRegistry
    ) -> None:
        blob = json.dumps(registry.describe(), ensure_ascii=False)
        assert "동의함" not in blob
        assert "개인정보 수집·이용 동의" not in blob
        assert "수집 항목은" not in blob

    def test_describe_rejects_coordinate_schema(self, registry: ToolRegistry) -> None:
        registry.register(
            Tool(
                name="bad_tool",
                description="좌표를 직접 받는 잘못된 도구",
                json_schema={
                    "type": "object",
                    "properties": {"x_mm": {"type": "number"}},
                    "required": ["x_mm"],
                },
                handler=lambda args: None,  # type: ignore[arg-type,return-value]
            )
        )
        with pytest.raises(ToolExecutionError, match="좌표"):
            registry.describe()


class TestExplainAndRead:
    def test_explain_always_appends_original_notice(
        self, registry: ToolRegistry
    ) -> None:
        result = registry.call("explain")
        assert result.ok is True
        assert ORIGINAL_NOTICE in result.speech

    def test_default_explainer_reads_clause_verbatim(
        self, registry: ToolRegistry
    ) -> None:
        result = registry.call("explain")
        assert "수집 항목은 성명과 연락처이며 보유 기간은 3년입니다." in result.speech

    def test_low_confidence_explanation_hands_off(self) -> None:
        structure = _structure()
        state = _ready_state(structure, "consent_01")
        registry = ToolRegistry(
            structure, state, motion=MockMotion(), explainer=LowConfidenceExplainer()
        )
        result = registry.call("explain")
        assert result.handoff is True
        assert result.ok is False
        assert "직원" in (result.error or "")

    def test_read_original_includes_options(self, registry: ToolRegistry) -> None:
        result = registry.call("read_original")
        assert "동의함" in result.speech
        assert "동의하지 않음" in result.speech

    def test_private_choice_still_reads_its_clause(self) -> None:
        """PRIVATE 로 분류된 선택 항목도 조항 본문을 그대로 낭독한다.

        제목에 개인정보 낱말이 들어갔다는 이유로 인쇄된 조항 본문을 버리면,
        사용자는 '원문 듣기' 로도 그 문구를 들을 수 없다.
        """
        clause = "수집 항목은 주민등록번호이며 보유 기간은 5년입니다."
        structure = DocumentStructure(
            document_id="doc_private_choice",
            doc_title="시험용 신청서",
            fields=(
                Field(
                    id="consent_01",
                    type=FieldType.CHOICE,
                    title="주민등록번호 수집·이용 동의",
                    role=FieldRole.APPLICANT,
                    options=(
                        Option("동의함", BoxMm(20.0, 100.0, 5.0, 5.0)),
                        Option("동의하지 않음", BoxMm(60.0, 100.0, 5.0, 5.0)),
                    ),
                    required=True,
                    sensitivity=Sensitivity.PRIVATE,
                    box_mm=BoxMm(20.0, 90.0, 170.0, 20.0),
                    clause_text=clause,
                    order=0,
                    confidence=1.0,
                ),
            ),
        )
        state = _ready_state(structure, "consent_01")
        registry = ToolRegistry(
            structure, state, motion=MockMotion(), speech=MockSpeech()
        )
        result = registry.call("read_original")
        assert result.ok is True
        assert clause in result.speech

    def test_unknown_field_id_is_reported(self, registry: ToolRegistry) -> None:
        result = registry.call("read_original", {"field_id": "없는항목"})
        assert result.ok is False
        assert "존재하지 않는" in (result.error or "")


class TestSelectOption:
    def test_valid_label_is_recorded(self, registry: ToolRegistry) -> None:
        result = registry.call("select_option", {"option_label": "동의함"})
        assert result.ok is True
        assert registry.state.selected_options["consent_01"] == "동의함"
        assert registry.state.phase is SessionPhase.GUIDING

    def test_invented_label_is_rejected(self, registry: ToolRegistry) -> None:
        result = registry.call("select_option", {"option_label": "아마도 동의"})
        assert result.ok is False
        assert "선택지가 아닙니다" in (result.error or "")
        assert registry.state.selected_options == {}

    def test_non_choice_field_is_rejected(self, registry: ToolRegistry) -> None:
        result = registry.call(
            "select_option", {"field_id": "signature_01", "option_label": "동의함"}
        )
        assert result.ok is False
        assert "선택지가 없습니다" in (result.error or "")


class TestMoveToField:
    def test_coordinates_come_from_the_document(self, registry: ToolRegistry) -> None:
        registry.call("select_option", {"option_label": "동의함"})
        result = registry.call("move_to_field")
        assert result.ok is True
        # 동의함 네모 칸 BoxMm(20, 100, 5, 5) 의 중심.
        assert registry.motion.moves == [(22.5, 102.5)]
        assert registry.state.phase is SessionPhase.AWAIT_WRITE

    def test_disagree_moves_to_the_other_box(self, registry: ToolRegistry) -> None:
        registry.call("select_option", {"option_label": "동의하지 않음"})
        registry.call("move_to_field")
        assert registry.motion.moves == [(62.5, 102.5)]

    def test_move_requires_a_selection_first(self, registry: ToolRegistry) -> None:
        result = registry.call("move_to_field")
        assert result.ok is False
        assert "선택지를 먼저" in (result.error or "")
        assert registry.motion.moves == []

    def test_low_vision_confidence_hands_off(self) -> None:
        structure = _structure(consent_confidence=VISION_TRUST_THRESHOLD - 0.1)
        state = _ready_state(structure, "consent_01")
        motion = MockMotion()
        registry = ToolRegistry(structure, state, motion=motion)
        registry.state.select_option("consent_01", "동의함")
        result = registry.call("move_to_field")
        assert result.handoff is True
        assert motion.moves == []

    def test_missing_box_hands_off(self, registry: ToolRegistry) -> None:
        result = registry.call("move_to_field", {"field_id": "floating_01"})
        assert result.handoff is True
        assert "위치를 확정하지 못했습니다" in (result.error or "")

    def test_arrival_drift_hands_off(self) -> None:
        structure = _structure()
        state = _ready_state(structure, "signature_01")
        registry = ToolRegistry(structure, state, motion=MockMotion(drift_mm=5.0))
        result = registry.call("move_to_field")
        assert result.handoff is True
        assert "도달하지 못했습니다" in (result.error or "")

    def test_motion_failure_is_reported(self) -> None:
        structure = _structure()
        state = _ready_state(structure, "signature_01")
        registry = ToolRegistry(structure, state, motion=MockMotion(succeed=False))
        result = registry.call("move_to_field")
        assert result.ok is False
        assert result.handoff is False
        assert "펜 이동에 실패" in (result.error or "")

    def test_out_of_page_target_is_refused(self) -> None:
        structure = DocumentStructure(
            document_id="doc_far",
            fields=(
                Field(
                    id="far_01",
                    type=FieldType.SIGNATURE,
                    title="용지 밖 항목",
                    role=FieldRole.APPLICANT,
                    box_mm=BoxMm(400.0, 400.0, 10.0, 10.0),
                    order=0,
                    confidence=1.0,
                ),
            ),
        )
        state = _ready_state(structure, "far_01")
        motion = MockMotion()
        registry = ToolRegistry(structure, state, motion=motion)
        result = registry.call("move_to_field")
        assert result.ok is False
        assert "가동 범위" in (result.error or "")
        assert motion.moves == []


class TestVerify:
    def test_default_verifier_completes_the_field(
        self, registry: ToolRegistry
    ) -> None:
        registry.call("select_option", {"option_label": "동의함"})
        registry.call("move_to_field")
        result = registry.call("verify_field")
        assert result.ok is True
        assert registry.state.phase is SessionPhase.FIELD_DONE
        assert registry.state.completed_fields == ["consent_01"]
        assert registry.state.verified_fields == ["consent_01"]

    def test_default_verifier_states_it_did_not_verify(
        self, registry: ToolRegistry
    ) -> None:
        registry.call("select_option", {"option_label": "동의함"})
        registry.call("move_to_field")
        registry.call("verify_field")
        notes = [entry.note for entry in registry.state.history]
        assert any("연결되지 않아" in note for note in notes)

    def test_failure_keeps_pen_and_asks_again(self) -> None:
        structure = _structure()
        state = _ready_state(structure, "signature_01")
        registry = ToolRegistry(
            structure, state, motion=MockMotion(), verifier=FailingVerifier()
        )
        registry.call("move_to_field")
        result = registry.call("verify_field")
        assert result.ok is False
        assert registry.state.phase is SessionPhase.AWAIT_WRITE
        assert "다시 표시" in result.speech

    def test_three_consecutive_failures_hand_off(self) -> None:
        structure = _structure()
        state = _ready_state(structure, "signature_01")
        registry = ToolRegistry(
            structure, state, motion=MockMotion(), verifier=FailingVerifier()
        )
        registry.call("move_to_field")
        outcomes = [registry.call("verify_field") for _ in range(MAX_FIELD_FAILURES)]
        assert outcomes[-1].handoff is True
        assert str(MAX_FIELD_FAILURES) in (outcomes[-1].error or "")


class TestNavigation:
    def test_next_field_skips_without_completing(
        self, registry: ToolRegistry
    ) -> None:
        registry.call("next_field")
        assert registry.state.current_field_id == "signature_01"
        assert registry.state.skipped_fields == ["consent_01"]
        assert registry.state.completed_fields == []

    def test_previous_field_at_start_is_reported(
        self, registry: ToolRegistry
    ) -> None:
        result = registry.call("previous_field")
        assert result.ok is False
        assert "첫 번째" in result.speech

    def test_go_to_field_moves_pointer_only(self, registry: ToolRegistry) -> None:
        registry.call("go_to_field", {"field_id": "signature_01"})
        assert registry.state.current_field_id == "signature_01"
        assert registry.state.pending_fields == [
            "consent_01",
            "signature_01",
            "floating_01",
        ]

    def test_revise_field_reverts_progress(self, registry: ToolRegistry) -> None:
        registry.call("select_option", {"option_label": "동의함"})
        registry.call("move_to_field")
        registry.call("verify_field")
        registry.call("next_field")
        result = registry.call("revise_field", {"field_id": "consent_01"})
        assert result.ok is True
        assert registry.state.current_field_id == "consent_01"
        assert registry.state.selected_options == {}
        assert registry.state.completed_fields == []

    def test_repeat_does_not_change_state(self, registry: ToolRegistry) -> None:
        before = registry.state.phase
        registry.call("repeat")
        assert registry.state.phase is before


class TestMisc:
    def test_request_human_enters_handoff(self, registry: ToolRegistry) -> None:
        result = registry.call("request_human", {"reason": "서식이 이해되지 않습니다."})
        assert result.handoff is True
        assert registry.state.phase is SessionPhase.HUMAN_HANDOFF
        assert registry.state.handoff_reason == "서식이 이해되지 않습니다."

    def test_summarize_progress_lists_remaining_required(
        self, registry: ToolRegistry
    ) -> None:
        result = registry.call("summarize_progress")
        assert "전체 3개 항목 중 0개" in result.speech
        assert "개인정보 수집·이용 동의" in result.speech
        assert "신청인 서명" in result.speech

    def test_handlers_never_raise(self, registry: ToolRegistry) -> None:
        """모든 도구를 빈 인자로 호출해도 예외가 밖으로 나오지 않는다."""
        for name in registry.names():
            result = registry.call(name)
            assert result.error is None or isinstance(result.error, str)

    def test_tool_errors_are_audited(self, registry: ToolRegistry) -> None:
        registry.call("select_option", {"option_label": "없는 선택지"})
        actions = [entry.action for entry in registry.state.history]
        assert "tool_error" in actions


class OptionAwareVerifier:
    """선택지별 판정을 제공하는 검증기(실제 ImageVerifier 와 같은 계약).

    :param marked: 실제로 표시가 확인된 선택지 라벨.
    """

    def __init__(self, marked: str) -> None:
        self.marked = marked

    def verify(self, field_id: str) -> VerificationResult:
        """항목 전체로는 '기입됨' 으로 판정한다(잉크가 늘었으므로)."""
        return VerificationResult(
            field_id=field_id,
            written=True,
            ink_ratio_before=0.01,
            ink_ratio_after=0.12,
            confidence=0.95,
            reason=f"선택지 '{self.marked}' 에 체크 표시를 확인했습니다.",
        )

    def verify_options(self, field_id: str) -> tuple[VerificationResult, ...]:
        """선택지별 판정을 돌려준다."""
        labels = ("동의함", "동의하지 않음")
        return tuple(
            VerificationResult(
                field_id=f"{field_id}:{label}",
                written=label == self.marked,
                ink_ratio_before=0.01,
                ink_ratio_after=0.12 if label == self.marked else 0.01,
                confidence=0.95,
                reason="선택지별 판정",
            )
            for label in labels
        )


class TestSelectedOptionIsCrossChecked:
    """'말로 고른 선택지' 와 '실제로 표시된 선택지' 를 대조한다."""

    def _registry(self, marked: str) -> ToolRegistry:
        structure = _structure()
        state = _ready_state(structure, "consent_01")
        return ToolRegistry(
            structure,
            state,
            verifier=OptionAwareVerifier(marked),
            motion=MockMotion(),
            speech=MockSpeech(),
        )

    def test_wrong_box_is_not_accepted(self) -> None:
        """다른 칸에 표시하면 기입 확인으로 통과시키지 않는다."""
        registry = self._registry(marked="동의하지 않음")
        registry.call("select_option", {"field_id": "consent_01", "option_label": "동의함"})
        result = registry.call("verify_field", {"field_id": "consent_01"})
        assert result.ok is False
        assert "다른 칸" in result.speech
        assert "consent_01" not in registry.state.verified_fields
        assert "consent_01" not in registry.state.completed_fields

    def test_matching_box_is_accepted(self) -> None:
        """말한 선택지와 같은 칸에 표시하면 정상 완료된다."""
        registry = self._registry(marked="동의함")
        registry.call("select_option", {"field_id": "consent_01", "option_label": "동의함"})
        result = registry.call("verify_field", {"field_id": "consent_01"})
        assert result.ok is True
        assert "consent_01" in registry.state.verified_fields
        assert "consent_01" in registry.state.completed_fields

    def test_repeated_mismatch_hands_off(self) -> None:
        """같은 항목에서 계속 어긋나면 사람에게 넘긴다."""
        registry = self._registry(marked="동의하지 않음")
        registry.call("select_option", {"field_id": "consent_01", "option_label": "동의함"})
        for _ in range(MAX_FIELD_FAILURES - 1):
            registry.call("verify_field", {"field_id": "consent_01"})
        result = registry.call("verify_field", {"field_id": "consent_01"})
        assert result.handoff is True
        assert "consent_01" not in registry.state.completed_fields


class TestSingleDisclaimer:
    """고지 문구는 한 곳에서 정의되고 한 발화에 한 번만 낭독된다."""

    def test_notice_constants_are_the_same_object(self) -> None:
        """tools 의 고지는 guardrails 의 고지를 그대로 참조한다."""
        from docagent.agent.guardrails import DISCLAIMER

        assert ORIGINAL_NOTICE is DISCLAIMER

    def test_notice_is_spoken_once_when_explainer_already_added_it(self) -> None:
        """설명기가 본문에 이미 고지를 넣었으면 도구가 또 붙이지 않는다."""
        from docagent.agent.guardrails import DISCLAIMER

        class DisclaimingExplainer:
            """본문 끝에 고지를 넣고 같은 문장을 disclaimer 로도 돌려주는 설명기."""

            class Result:
                text = f"쉬운 설명 본문입니다. {DISCLAIMER}"
                confidence = 0.95
                sources: tuple[str, ...] = ("근거문서",)
                needs_handoff = False
                disclaimer = DISCLAIMER

            def explain(self, field: Field, question: str = "") -> "DisclaimingExplainer.Result":
                return self.Result()

        structure = _structure()
        state = _ready_state(structure, "consent_01")
        registry = ToolRegistry(
            structure,
            state,
            explainer=DisclaimingExplainer(),
            motion=MockMotion(),
            speech=MockSpeech(),
        )
        speech = registry.call("explain", {"field_id": "consent_01"}).speech
        assert speech.count(DISCLAIMER) == 1
        assert speech.count("원문 읽어줘") == 1
