"""한국어 규칙 기반 의도 분류기 테스트.

가장 중요한 회귀 방지 항목은 **부정이 긍정으로 새지 않는 것**과
**번복이 동의·취소로 새지 않는 것**이다. 둘 다 잘못 분류되면 사용자가 원치 않는
칸에 표시가 들어간다.
"""

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
from docagent.errors import AdapterUnavailable
from docagent.agent.intent import (
    Intent,
    IntentResult,
    LlmIntentClassifier,
    RuleIntentClassifier,
)
from docagent.agent.state import SessionState


@pytest.fixture()
def structure() -> DocumentStructure:
    """선택형 1개 + 서명 1개짜리 시험용 문서."""
    return DocumentStructure(
        document_id="doc_intent",
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
                clause_text="수집 항목은 성명과 연락처입니다.",
                order=0,
                confidence=1.0,
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
        ),
    )


@pytest.fixture()
def classifier() -> RuleIntentClassifier:
    return RuleIntentClassifier()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 동의
        ("동의할게", Intent.AGREE),
        ("동의합니다", Intent.AGREE),
        ("네", Intent.AGREE),
        ("그렇게 해줘", Intent.AGREE),
        ("좋아요", Intent.AGREE),
        ("알겠습니다", Intent.AGREE),
        # 비동의 — 절대 AGREE 로 새면 안 된다
        ("동의 안 해", Intent.DISAGREE),
        ("동의안해", Intent.DISAGREE),
        ("동의하지 않을래요", Intent.DISAGREE),
        ("아니요", Intent.DISAGREE),
        ("거부할래", Intent.DISAGREE),
        ("싫어요", Intent.DISAGREE),
        # 설명
        ("쉽게 설명해줘", Intent.EXPLAIN),
        ("무슨 뜻이야", Intent.EXPLAIN),
        ("이게 뭐예요", Intent.EXPLAIN),
        ("어려워요 풀어서 말해줘", Intent.EXPLAIN),
        # 원문
        ("원문 읽어줘", Intent.READ_ORIGINAL),
        ("전문 들려줘", Intent.READ_ORIGINAL),
        ("그대로 읽어주세요", Intent.READ_ORIGINAL),
        # 이동
        ("다음", Intent.NEXT),
        ("다음 항목", Intent.NEXT),
        ("넘어가 주세요", Intent.NEXT),
        ("이전으로", Intent.PREVIOUS),
        ("뒤로 가줘", Intent.PREVIOUS),
        # 재생
        ("다시 들려줘", Intent.REPEAT),
        ("한번 더 말해줘", Intent.REPEAT),
        ("못 들었어요", Intent.REPEAT),
        # 번복
        ("아까 동의한 거 취소할래", Intent.REVISE),
        ("방금 선택한 거 바꿀래요", Intent.REVISE),
        ("잘못 눌렀어요", Intent.REVISE),
        ("아까 동의 안 한다고 한 거 다시 바꿀래", Intent.REVISE),
        # 직원 호출
        ("직원 불러줘", Intent.CALL_STAFF),
        ("사람 연결해 주세요", Intent.CALL_STAFF),
        ("도와주세요", Intent.CALL_STAFF),
        # 확인 · 취소
        ("다 썼어요", Intent.CONFIRM),
        ("서명했어요", Intent.CONFIRM),
        ("확인", Intent.CONFIRM),
        ("그만할래", Intent.CANCEL),
        # 질문
        ("제가 이 지원금을 받을 수 있나요?", Intent.QUESTION),
        ("이거 안 쓰면 어떻게 되나요", Intent.QUESTION),
    ],
)
def test_korean_phrase_mapping(
    classifier: RuleIntentClassifier, text: str, expected: Intent
) -> None:
    """대표 한국어 표현이 의도된 분류로 떨어진다."""
    assert classifier.classify(text).intent is expected


class TestNegationSafety:
    """부정 표현이 긍정으로 새지 않는지 — 사용자 피해가 가장 큰 오분류."""

    @pytest.mark.parametrize(
        "text",
        [
            "동의 안 해",
            "동의하지 않겠습니다",
            "동의 못 하겠어요",
            "비동의",
            "동의는 안 할래요",
        ],
    )
    def test_negations_never_become_agree(
        self, classifier: RuleIntentClassifier, text: str
    ) -> None:
        assert classifier.classify(text).intent is not Intent.AGREE

    @pytest.mark.parametrize(
        "text",
        ["아까 동의한 거 취소할래", "방금 동의한 거 무를래요", "아까 그거 잘못 골랐어요"],
    )
    def test_revision_never_becomes_agree_or_cancel(
        self, classifier: RuleIntentClassifier, text: str
    ) -> None:
        assert classifier.classify(text).intent is Intent.REVISE


class TestUnknown:
    """추측 금지 — 모르면 UNKNOWN 으로 떨어뜨린다."""

    @pytest.mark.parametrize("text", ["", "   ", "블라블라", "@@@"])
    def test_unrecognized_input_is_unknown(
        self, classifier: RuleIntentClassifier, text: str
    ) -> None:
        result = classifier.classify(text)
        assert result.intent is Intent.UNKNOWN
        assert result.confidence == 0.0

    def test_ambiguous_selection_is_unknown(
        self, classifier: RuleIntentClassifier
    ) -> None:
        """무엇을 고를지 모르는 '선택해줘'는 추측하지 않는다."""
        result = classifier.classify("선택해줘")
        assert result.intent is Intent.UNKNOWN

    def test_high_min_confidence_downgrades_everything(self) -> None:
        strict = RuleIntentClassifier(min_confidence=0.99)
        assert strict.classify("동의할게").intent is Intent.UNKNOWN

    def test_invalid_min_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="0.0~1.0"):
            RuleIntentClassifier(min_confidence=1.5)


class TestSlots:
    """슬롯 추출 — 번복 대상 · 선택지."""

    def test_option_label_is_extracted(
        self, classifier: RuleIntentClassifier, structure: DocumentStructure
    ) -> None:
        state = SessionState.from_structure(structure)
        state.set_current("consent_01")
        result = classifier.classify(
            "동의하지 않음으로 해주세요", state=state, structure=structure
        )
        assert result.intent is Intent.SELECT_OPTION
        assert result.slots["option_label"] == "동의하지 않음"

    def test_ordinal_maps_to_option_label(
        self, classifier: RuleIntentClassifier, structure: DocumentStructure
    ) -> None:
        state = SessionState.from_structure(structure)
        state.set_current("consent_01")
        result = classifier.classify("첫 번째로 해줘", state=state, structure=structure)
        assert result.intent is Intent.SELECT_OPTION
        assert result.slots["option_index"] == 0
        assert result.slots["option_label"] == "동의함"

    def test_revise_target_is_latest_completed_choice(
        self, classifier: RuleIntentClassifier, structure: DocumentStructure
    ) -> None:
        state = SessionState.from_structure(structure)
        state.complete_field("consent_01")
        state.set_current("signature_01")
        result = classifier.classify(
            "아까 그 동의 항목 취소할래", state=state, structure=structure
        )
        assert result.intent is Intent.REVISE
        assert result.slots["target_field_id"] == "consent_01"

    def test_revise_target_falls_back_to_latest_completed(
        self, classifier: RuleIntentClassifier, structure: DocumentStructure
    ) -> None:
        state = SessionState.from_structure(structure)
        state.complete_field("consent_01")
        state.complete_field("signature_01")
        result = classifier.classify(
            "아까 한 거 취소할래", state=state, structure=structure
        )
        assert result.slots["target_field_id"] == "signature_01"

    def test_revise_without_state_has_no_target(
        self, classifier: RuleIntentClassifier
    ) -> None:
        result = classifier.classify("아까 그거 취소할래")
        assert result.intent is Intent.REVISE
        assert "target_field_id" not in result.slots

    def test_legal_question_is_flagged(self, classifier: RuleIntentClassifier) -> None:
        result = classifier.classify("제가 이 지원금을 받을 수 있나요?")
        assert result.intent is Intent.QUESTION
        assert result.slots.get("legal_judgment") is True

    def test_plain_question_is_not_flagged(
        self, classifier: RuleIntentClassifier
    ) -> None:
        result = classifier.classify("이건 언제까지 내야 하나요")
        assert result.intent is Intent.QUESTION
        assert result.slots.get("legal_judgment") is None


class TestDeterminism:
    """같은 입력은 항상 같은 결과."""

    def test_repeated_calls_agree(
        self, classifier: RuleIntentClassifier, structure: DocumentStructure
    ) -> None:
        state = SessionState.from_structure(structure)
        state.set_current("consent_01")
        first = classifier.classify("동의할게", state=state, structure=structure)
        for _ in range(20):
            again = classifier.classify("동의할게", state=state, structure=structure)
            assert again.to_dict() == first.to_dict()

    def test_whitespace_variants_agree(
        self, classifier: RuleIntentClassifier
    ) -> None:
        assert (
            classifier.classify("동의 안 해").intent
            is classifier.classify("동의안해!!").intent
        )


class TestIntentResult:
    """결과 dataclass 계약."""

    def test_roundtrip(self) -> None:
        result = IntentResult(Intent.REVISE, 0.92, {"target_field_id": "consent_01"})
        assert IntentResult.from_dict(result.to_dict()) == result

    def test_confidence_range_enforced(self) -> None:
        with pytest.raises(ValueError, match="0.0~1.0"):
            IntentResult(Intent.AGREE, 1.4)

    def test_unknown_intent_value_rejected(self) -> None:
        with pytest.raises(ValueError, match="정의되지 않은"):
            IntentResult.from_dict({"intent": "없는의도"})

    def test_missing_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="intent"):
            IntentResult.from_dict({"confidence": 0.5})


class TestLlmAdapter:
    """LLM 분류기는 선택적 어댑터일 뿐 기본 경로가 아니다."""

    def test_missing_package_raises_korean_guidance(self) -> None:
        with pytest.raises(AdapterUnavailable) as excinfo:
            LlmIntentClassifier().classify("동의할게")
        message = str(excinfo.value)
        assert "anthropic" in message
        assert "설치" in message
        assert ".venv" in message

    def test_injected_client_is_used(self) -> None:
        class StubLlm:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                return "read_original"

        result = LlmIntentClassifier(StubLlm()).classify("아무 말")
        assert result.intent is Intent.READ_ORIGINAL
        assert result.slots["source"] == "llm"

    def test_raw_utterance_never_reaches_the_inner_client(self) -> None:
        """발화 원문의 개인정보가 게이트를 거치지 않고 LLM 으로 나가지 않는다."""

        class RecordingLlm:
            def __init__(self) -> None:
                self.seen: list[str] = []

            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                self.seen.append(user)
                return "read_original"

        inner = RecordingLlm()
        classifier = LlmIntentClassifier(inner)
        classifier.classify("제 주민번호 900101-1234567 읽어줘")
        assert inner.seen, "내부 클라이언트가 호출되지 않았습니다."
        assert all("900101" not in text for text in inner.seen)

    def test_gate_audit_records_the_call(self) -> None:
        """게이트 감사 로그에 분류기 호출이 남는다(무기록 전송 금지)."""
        from docagent.pii.gate import build_default_gate

        class StubLlm:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                return "read_original"

        gate = build_default_gate()
        LlmIntentClassifier(StubLlm(), gate=gate).classify("원문 읽어줘")
        assert gate.audit.counters()["llm_calls"] == 1

    def test_unparseable_response_falls_back_to_rules(self) -> None:
        class NoisyLlm:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                return "저는 잘 모르겠습니다"

        result = LlmIntentClassifier(NoisyLlm()).classify("원문 읽어줘")
        assert result.intent is Intent.READ_ORIGINAL
        assert result.slots.get("source") != "llm"
