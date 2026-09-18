"""가드레일·할루시네이션 차단 테스트."""

from __future__ import annotations

import logging
import re

import pytest

from docagent.agent.guardrails import (
    DISCLAIMER,
    Guard,
    GuardCategory,
    NullPiiGate,
    answer_grounding_check,
)
from docagent.contracts import (
    Field,
    FieldType,
    PiiSpan,
    RetrievedChunk,
    SanitizedText,
    Sensitivity,
)
from docagent.errors import HandoffRequired
from docagent.interfaces import PiiGate

#: 주민등록번호 형태(가상 예시) 탐지용 정규식.
_RRN_RE = re.compile(r"\d{6}-\d{7}")
#: 휴대전화 번호 형태 탐지용 정규식.
_PHONE_RE = re.compile(r"01\d-\d{3,4}-\d{4}")


class FakePiiGate:
    """테스트용 PII 게이트.

    실제 구현(:mod:`docagent.pii`)은 다른 모듈의 소유이므로, 여기서는
    :class:`docagent.interfaces.PiiGate` 프로토콜만 만족하는 최소 구현을 쓴다.
    주민등록번호·휴대전화 번호 형태만 탐지한다.
    """

    def sanitize(self, text: str) -> SanitizedText:
        """정규식으로 개인정보 구간을 찾아 마스킹한다.

        :param text: 원문.
        :returns: :class:`SanitizedText`.
        """
        spans: list[PiiSpan] = []
        masked = text
        for pii_type, pattern in (("rrn", _RRN_RE), ("phone", _PHONE_RE)):
            for match in pattern.finditer(text):
                spans.append(
                    PiiSpan(
                        start=match.start(),
                        end=match.end(),
                        pii_type=pii_type,
                        raw_len=len(match.group()),
                    )
                )
                masked = masked.replace(match.group(), "*" * len(match.group()))
        spans.sort(key=lambda span: span.start)
        return SanitizedText(text=masked, spans=tuple(spans), blocked=False)

    def assert_clean(self, text: str) -> None:
        """개인정보가 있으면 차단 예외를 던진다.

        :param text: 검사 대상.
        :returns: ``None``.
        :raises docagent.errors.PiiEgressBlocked: 개인정보가 탐지된 경우.
        """
        from docagent.errors import PiiEgressBlocked

        result = self.sanitize(text)
        if result.has_pii:
            raise PiiEgressBlocked(
                pii_types=sorted({span.pii_type for span in result.spans}),
                count=len(result.spans),
            )


CONSENT_FIELD = Field(
    id="consent_01",
    type=FieldType.CHOICE,
    title="개인정보 수집·이용 동의",
    sensitivity=Sensitivity.PUBLIC,
    required=True,
)


def chunk(text: str) -> RetrievedChunk:
    """테스트용 근거 청크를 만든다.

    :param text: 본문.
    :returns: :class:`RetrievedChunk`.
    """
    return RetrievedChunk(chunk_id="c1", text=text, source="테스트.md", score=0.3)


class TestGuardProtocol:
    """게이트 주입 규약."""

    def test_fake_gate_satisfies_protocol(self) -> None:
        """테스트용 게이트가 PiiGate 프로토콜을 만족한다."""
        assert isinstance(FakePiiGate(), PiiGate)

    def test_has_pii_gate_flag(self) -> None:
        """실제 검사가 이루어지는지 여부를 노출한다."""
        assert Guard(NullPiiGate()).has_pii_gate is False
        assert Guard(FakePiiGate()).has_pii_gate is True

    def test_default_gate_is_real_not_pass_through(self) -> None:
        """게이트를 주지 않아도 기본값은 '검사 생략' 이 아니라 실제 게이트다."""
        assert Guard().has_pii_gate is True


class TestBlockedCategories:
    """차단 범주별 트리거."""

    @pytest.mark.parametrize(
        "text",
        [
            "이 지원금을 제 아들도 받을 수 있나요?",
            "제가 자격이 되나요?",
            "저도 신청 가능한가요?",
            "돈이 얼마나 나와요?",
        ],
    )
    def test_eligibility_blocked(self, text: str) -> None:
        """자격·금액 판단 요구를 막는다."""
        verdict = Guard().inspect(text)
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.ELIGIBILITY

    @pytest.mark.parametrize(
        "text",
        [
            "이건 법적으로 유효한가요?",
            "나중에 소송을 걸 수 있나요?",
            "제가 불이익이 있나요?",
            "책임은 누가 지나요?",
        ],
    )
    def test_legal_blocked(self, text: str) -> None:
        """법적 판단 요구를 막는다."""
        verdict = Guard().inspect(text)
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.LEGAL

    @pytest.mark.parametrize(
        "text", ["돈을 어디에 투자하면 좋아요?", "어느 쪽이 이자가 유리한가요?"]
    )
    def test_financial_blocked(self, text: str) -> None:
        """세무·금융 조언 요구를 막는다."""
        verdict = Guard().inspect(text)
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.FINANCIAL

    @pytest.mark.parametrize("text", ["네 생각은 어때?", "그냥 추천해줘", "알아서 해줘"])
    def test_speculation_blocked(self, text: str) -> None:
        """근거 없는 추측·추천 요구를 막는다."""
        verdict = Guard().inspect(text)
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.SPECULATION

    @pytest.mark.parametrize(
        "text",
        [
            "개인정보 수집 동의가 뭐예요?",
            "왜 주민번호를 물어봐?",
            "동의하지 않으면 어떻게 되나요?",
            "이 칸에는 무엇을 적나요?",
            "",
        ],
    )
    def test_normal_questions_pass(self, text: str) -> None:
        """설명 요청은 통과시킨다(과잉 차단 금지)."""
        assert Guard().inspect(text).blocked is False

    def test_whitespace_insensitive(self) -> None:
        """띄어쓰기가 달라도 같은 표현으로 본다."""
        assert Guard().inspect("법적으로유효한가요").blocked is True
        assert Guard().inspect("법 적 으 로 유효 한가요").blocked is True

    def test_check_raises_handoff_with_korean_reason(self) -> None:
        """check 는 낭독 가능한 한국어 사유와 함께 HandoffRequired 를 던진다."""
        with pytest.raises(HandoffRequired) as info:
            Guard().check("이건 법적으로 유효한가요?", CONSENT_FIELD)
        assert "법적" in info.value.reason
        assert info.value.field_id == "consent_01"

    def test_check_passes_silently(self) -> None:
        """통과 시 아무것도 반환하지 않는다."""
        assert Guard().check("개인정보 수집 동의가 뭐예요?") is None


class TestPiiGuard:
    """개인정보가 섞인 질문 처리."""

    def test_pii_question_blocked(self) -> None:
        """질문에 주민등록번호가 있으면 LLM 경로를 막는다."""
        guard = Guard(FakePiiGate())
        verdict = guard.inspect("제 번호 900101-1234567 맞게 적었나요?")
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.PII
        assert verdict.local_only is True
        assert "rrn" in verdict.trigger

    def test_pii_reason_has_no_raw_value(self) -> None:
        """차단 사유에 원문 개인정보가 들어가지 않는다."""
        guard = Guard(FakePiiGate())
        verdict = guard.inspect("제 번호는 900101-1234567 입니다")
        assert "900101" not in verdict.reason
        assert "원문" in verdict.reason

    def test_pii_free_question_passes_with_gate(self) -> None:
        """개인정보가 없으면 게이트가 있어도 통과한다."""
        assert Guard(FakePiiGate()).inspect("동의가 뭐예요?").blocked is False

    def test_default_guard_blocks_pii_without_injection(self) -> None:
        """게이트를 주입하지 않아도 개인정보가 섞인 질문은 막힌다(fail-open 금지)."""
        verdict = Guard().inspect("제 주민번호는 900101-1234567 입니다")
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.PII

    def test_explicit_null_gate_is_the_only_way_to_skip(self) -> None:
        """검사를 끄려면 NullPiiGate 를 명시적으로 주입해야 한다."""
        assert Guard(NullPiiGate()).inspect("제 번호 900101-1234567").blocked is False

    def test_unexplained_long_digits_are_blocked(self) -> None:
        """탐지 규칙이 못 잡은 긴 숫자열도 allowlist 방어선에서 막힌다."""
        verdict = Guard().inspect("9 8 7 6 5 4 3 2 1 0 이 번호 맞나요?")
        assert verdict.blocked is True
        assert verdict.category is GuardCategory.PII
        assert verdict.trigger == "unexplained_digits"

    def test_short_numbers_are_not_over_blocked(self) -> None:
        """연도·조문 같은 짧은 숫자는 막지 않는다(과잉 차단 금지)."""
        assert Guard().inspect("2024년 제17조가 무슨 뜻이에요?").blocked is False


class TestDisclaimer:
    """고지 문구."""

    def test_disclaimer_text(self) -> None:
        """고지 문구는 요약임을 밝히고 원문 듣기를 안내한다."""
        assert "쉽게 설명한 내용입니다" in DISCLAIMER
        assert "원문" in DISCLAIMER


class TestAnswerGroundingCheck:
    """근거 밖 사실 탐지."""

    def test_clean_answer_passes(self) -> None:
        """근거 문장을 그대로 쓴 답변은 통과한다."""
        evidence = chunk("보유 기간은 지급 완료일부터 5년입니다. 개인정보 보호법이 정한 바입니다.")
        assert answer_grounding_check("보유 기간은 5년입니다.", [evidence]) == []

    def test_fabricated_number_detected(self) -> None:
        """근거에 없는 숫자·금액·기간을 잡아낸다."""
        evidence = chunk("보유 기간은 지급 완료일부터 5년입니다.")
        unsupported = answer_grounding_check(
            "지원금 300만원을 3개월 안에 받을 수 있습니다.", [evidence]
        )
        assert "300만원" in unsupported
        assert "3개월" in unsupported

    def test_fabricated_institution_detected(self) -> None:
        """근거에 없는 기관·법령 이름을 잡아낸다."""
        evidence = chunk("개인정보 보호법은 별도 동의를 요구합니다.")
        unsupported = answer_grounding_check(
            "국민행복공단에 문의하면 신용정보법에 따라 처리됩니다.", [evidence]
        )
        assert "국민행복공단" in unsupported
        assert "신용정보법" in unsupported

    def test_supported_institution_passes(self) -> None:
        """근거에 있는 법령 이름은 통과한다."""
        evidence = chunk("개인정보 보호법은 별도 동의를 요구합니다.")
        assert answer_grounding_check("개인정보 보호법이 정한 내용입니다.", [evidence]) == []

    def test_empty_answer(self) -> None:
        """빈 답변은 검사 대상이 없다."""
        assert answer_grounding_check("", [chunk("본문")]) == []

    def test_result_has_no_duplicates(self) -> None:
        """같은 표현이 여러 번 나와도 한 번만 보고한다."""
        evidence = chunk("근거 본문입니다.")
        unsupported = answer_grounding_check("7년 그리고 다시 7년입니다.", [evidence])
        assert unsupported.count("7년") == 1


class TestPiiGateProtocolCompliance:
    """PiiGate 자리에 들어가는 구현은 프로토콜을 실제로 만족해야 한다."""

    def test_null_gate_satisfies_the_protocol(self) -> None:
        """NullPiiGate 는 ``assert_clean`` 까지 갖춰 isinstance 검사를 통과한다."""
        from docagent.agent.guardrails import NullPiiGate
        from docagent.interfaces import PiiGate

        gate = NullPiiGate()
        assert isinstance(gate, PiiGate)
        gate.assert_clean("주민등록번호 900101-1234567")  # 검사를 끈 게이트다.

    def test_egress_adapter_satisfies_the_protocol(self) -> None:
        """llm 의 전송 어댑터도 ``assert_clean`` 을 위임한다."""
        from docagent.agent.llm import _PiiGateEgressAdapter
        from docagent.errors import PiiEgressBlocked
        from docagent.interfaces import PiiGate
        from docagent.pii.gate import LlmEgressGate

        adapter = _PiiGateEgressAdapter(LlmEgressGate())
        assert isinstance(adapter, PiiGate)
        adapter.assert_clean("접수번호 제2026-0001호입니다.")
        with pytest.raises(PiiEgressBlocked):
            adapter.assert_clean("주민등록번호 900101-1234567")

    def test_adapter_without_assert_clean_still_blocks(self) -> None:
        """원본 게이트에 assert_clean 이 없어도 조용히 통과시키지 않는다."""
        from docagent.agent.llm import _PiiGateEgressAdapter
        from docagent.contracts import PiiSpan, SanitizedText
        from docagent.errors import PiiEgressBlocked

        class SanitizeOnlyGate:
            """``sanitize`` 만 가진 게이트."""

            def sanitize(self, text: str) -> SanitizedText:
                if "900101" in text:
                    return SanitizedText(
                        text="<REDACTED>",
                        spans=(PiiSpan(start=0, end=6, pii_type="rrn", raw_len=6),),
                    )
                return SanitizedText(text=text, spans=(), blocked=False)

        adapter = _PiiGateEgressAdapter(SanitizeOnlyGate())
        adapter.assert_clean("안전한 문장입니다.")
        with pytest.raises(PiiEgressBlocked):
            adapter.assert_clean("900101 로 시작합니다.")


class TestOutboundInspection:
    """문서에서 읽어 온 문구도 외부 전송 전에 검사한다."""

    def test_document_text_with_pii_is_blocked(self) -> None:
        """약관 전문에 개인정보가 섞이면 외부 경로를 막는다."""
        guard = Guard()
        verdict = guard.inspect_outbound(
            "개인정보 수집·이용 동의 신청인 김철수, 건강보험증번호는 12345678901"
        )
        assert verdict.blocked is True
        assert verdict.local_only is True

    def test_benign_document_numbers_pass(self) -> None:
        """접수번호·조문 같은 정당한 숫자는 통과한다."""
        guard = Guard()
        verdict = guard.inspect_outbound(
            "접수번호 제2026-0001호로 접수되었으며 근거는 제12조 제3항입니다."
        )
        assert verdict.blocked is False

    def test_block_rules_do_not_apply_to_document_text(self) -> None:
        """문서 원문에 '법적 효력' 같은 표현이 있어도 그 이유로 막지 않는다."""
        guard = Guard()
        verdict = guard.inspect_outbound("이 동의서는 법적효력을 가집니다.")
        assert verdict.blocked is False
