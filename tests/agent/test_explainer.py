"""Explain 단계 통합 테스트 — 로드맵 Phase 2 테스트 케이스 4종 포함.

이 파일의 모든 테스트는 **API 키 없이** 통과해야 한다. 기본 LLM 은
:class:`~docagent.agent.llm.OfflineTemplateLlm` 이며 네트워크를 쓰지 않는다.
"""

from __future__ import annotations

import re

import pytest

from docagent.agent.confidence import ConfidenceBand
from docagent.agent.explainer import (
    FIELD_TYPE_QUERIES,
    HANDOFF_SPEECH,
    PARTIAL_SUFFIX,
    ExplanationResult,
    Explainer,
)
from docagent.agent.guardrails import DISCLAIMER, Guard
from docagent.agent.llm import (
    SYSTEM_PROMPT,
    ClaudeLlm,
    OfflineTemplateLlm,
    build_llm,
    build_user_prompt,
    parse_user_prompt,
)
from docagent.agent.rag import LocalTfidfRetriever
from docagent.contracts import (
    Field,
    FieldType,
    PiiSpan,
    RetrievedChunk,
    SanitizedText,
    Sensitivity,
)
from docagent.errors import AdapterUnavailable, PiiEgressBlocked
from docagent.pii.gate import GatedLlmClient
from docagent.interfaces import LlmClient
from docagent.testing.synthetic import (
    APPLICANT_SIGNATURE_FIELD_ID,
    CONSENT_FIELD_ID,
    DEFAULT_SEED,
    FormSpec,
    build_truth,
)

_RRN_RE = re.compile(r"\d{6}-\d{7}")


class RrnPiiGate:
    """주민등록번호 형태만 탐지하는 테스트용 게이트(PiiGate 프로토콜 구현)."""

    def sanitize(self, text: str) -> SanitizedText:
        """주민등록번호 형태를 찾아 마스킹한다.

        :param text: 원문.
        :returns: :class:`SanitizedText`.
        """
        spans = tuple(
            PiiSpan(
                start=match.start(),
                end=match.end(),
                pii_type="rrn",
                raw_len=len(match.group()),
            )
            for match in _RRN_RE.finditer(text)
        )
        return SanitizedText(text=_RRN_RE.sub("*************", text), spans=spans)

    def assert_clean(self, text: str) -> None:
        """개인정보가 있으면 차단한다.

        :param text: 검사 대상.
        :returns: ``None``.
        :raises docagent.errors.PiiEgressBlocked: 개인정보가 탐지된 경우.
        """
        if _RRN_RE.search(text):
            raise PiiEgressBlocked(pii_types=["rrn"], count=1)


class BlockingGate(RrnPiiGate):
    """무조건 전송을 차단하는 게이트(차단 경로 검증용)."""

    def sanitize(self, text: str) -> SanitizedText:
        """항상 ``blocked=True`` 로 표시한다.

        :param text: 원문.
        :returns: 차단 표시된 :class:`SanitizedText`.
        """
        return SanitizedText(text="", spans=(), blocked=True)


class FixedLlm:
    """항상 같은 문장을 돌려주는 스텁 LLM(할루시네이션 검증용).

    :param text: 돌려줄 문장.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """고정 문장을 돌려준다.

        :param system: 무시.
        :param user: 무시.
        :param max_tokens: 무시.
        :returns: 생성자에 준 문장.
        """
        del system, user, max_tokens
        return self._text


class FixedClock:
    """고정 시각을 돌려주는 테스트용 시계."""

    def now_iso(self) -> str:
        """고정 ISO 8601 문자열을 돌려준다.

        :returns: ``"2026-09-10T09:00:00+09:00"``.
        """
        return "2026-09-10T09:00:00+09:00"


# --------------------------------------------------------------------------
# 픽스처
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def retriever() -> LocalTfidfRetriever:
    """저장소 기본 코퍼스 검색기."""
    return LocalTfidfRetriever.from_corpus()


@pytest.fixture(scope="module")
def structure():
    """정답 문서 구조(이미지 없이 생성)."""
    return build_truth(
        FormSpec(
            document_id="explainer_fixture",
            dpi=200,
            include_representative=True,
            seed=DEFAULT_SEED,
        )
    )


@pytest.fixture(scope="module")
def consent_field(structure) -> Field:
    """공개 정보 영역의 필수 동의 항목."""
    field = structure.field_by_id(CONSENT_FIELD_ID)
    assert field is not None and field.sensitivity is Sensitivity.PUBLIC
    return field


@pytest.fixture(scope="module")
def rrn_field(structure) -> Field:
    """개인정보 영역의 주민등록번호 항목."""
    field = structure.field_by_id("applicant_rrn")
    assert field is not None and field.sensitivity is Sensitivity.PRIVATE
    return field


@pytest.fixture
def explainer(retriever: LocalTfidfRetriever) -> Explainer:
    """오프라인 LLM + 게이트가 붙은 가드레일로 구성한 설명기."""
    return Explainer(
        retriever,
        OfflineTemplateLlm(),
        Guard(RrnPiiGate()),
        FixedClock(),
    )


# --------------------------------------------------------------------------
# 로드맵 Phase 2 테스트 케이스
# --------------------------------------------------------------------------


class TestRoadmapCases:
    """로드맵이 지정한 질문 4종의 밴드를 그대로 단언한다."""

    def test_q1_consent_meaning_explains(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """Q1 '개인정보 수집 동의가 뭐예요?' → 설명 제공(EXPLAIN)."""
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert result.mode is ConfidenceBand.EXPLAIN
        assert result.needs_handoff is False
        assert result.sources

    def test_q2_eligibility_hands_off(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """Q2 '제 아들도 받을 수 있나요?' → 자격 판단이므로 직원 연결(HANDOFF)."""
        result = explainer.explain(consent_field, "이 지원금을 제 아들도 받을 수 있나요?")
        assert result.mode is ConfidenceBand.HANDOFF
        assert result.needs_handoff is True
        assert "직원" in result.text

    def test_q3_rrn_reason_explains(self, explainer: Explainer, rrn_field: Field) -> None:
        """Q3 '왜 주민번호를 물어봐?' → 설명 또는 설명+원문 병행."""
        result = explainer.explain(rrn_field, "왜 주민번호를 물어봐?")
        assert result.mode in (
            ConfidenceBand.EXPLAIN,
            ConfidenceBand.EXPLAIN_WITH_ORIGINAL,
        )
        assert result.needs_handoff is False
        assert any("unique_identifier" in source for source in result.sources)

    def test_q4_legal_validity_hands_off(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """Q4 '이건 법적으로 유효한가요?' → 법적 판단이므로 직원 연결(HANDOFF)."""
        result = explainer.explain(consent_field, "이건 법적으로 유효한가요?")
        assert result.mode is ConfidenceBand.HANDOFF
        assert result.needs_handoff is True
        assert "법적" in result.text


class TestDisclaimerPolicy:
    """설명 응답에는 항상 고지 문구가 붙는다."""

    @pytest.mark.parametrize(
        "question",
        [
            None,
            "개인정보 수집 동의가 뭐예요?",
            "동의하지 않으면 어떻게 되나요?",
            "제 정보를 다른 곳에도 넘기나요?",
        ],
    )
    def test_explanation_always_has_disclaimer(
        self, explainer: Explainer, consent_field: Field, question: str | None
    ) -> None:
        """핸드오프가 아닌 모든 응답에 고지 문구가 포함된다."""
        result = explainer.explain(consent_field, question)
        # 파라미터는 모두 '핸드오프가 아닌 질의' 로 고정되어 있다.
        # 핸드오프로 퇴행하면 skip 이 아니라 실패로 드러나야 한다.
        assert result.needs_handoff is False
        assert result.disclaimer == DISCLAIMER
        assert result.text.endswith(DISCLAIMER)

    def test_handoff_has_no_disclaimer(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """핸드오프 응답에는 고지 문구를 붙이지 않는다(설명이 아니므로)."""
        result = explainer.explain(consent_field, "이건 법적으로 유효한가요?")
        assert result.disclaimer == ""
        assert DISCLAIMER not in result.text

    def test_partial_band_adds_original_suggestion(
        self, retriever: LocalTfidfRetriever, consent_field: Field
    ) -> None:
        """부분 확신 밴드에서는 원문 병행 안내가 함께 붙는다."""
        explainer = Explainer(
            retriever, OfflineTemplateLlm(), Guard(RrnPiiGate()), top_k=4
        )
        result = explainer.explain(consent_field, "제 정보를 다른 곳에도 넘기나요?")
        # 밴드 자체를 단언으로 고정한다. EXPLAIN 으로 퇴행하면 skip 이 아니라 실패다.
        assert result.mode is ConfidenceBand.EXPLAIN_WITH_ORIGINAL
        assert PARTIAL_SUFFIX in result.text
        assert result.text.endswith(DISCLAIMER)


class TestHallucinationBlocking:
    """근거 밖 사실이 섞이면 답변을 폐기한다."""

    def test_fake_numbers_are_blocked(
        self, retriever: LocalTfidfRetriever, consent_field: Field
    ) -> None:
        """근거에 없는 금액·기간을 넣은 가짜 답변은 핸드오프로 폐기된다."""
        fake = "이 지원금은 1인당 250만원이며 신청 후 14일 안에 지급됩니다."
        explainer = Explainer(retriever, FixedLlm(fake), Guard(RrnPiiGate()))
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert result.mode is ConfidenceBand.HANDOFF
        assert result.needs_handoff is True
        assert fake not in result.text
        assert result.text == HANDOFF_SPEECH
        # 사유에는 근거 밖 표현이 **몇 건** 있었는지만 남는다. 원문 토큰을 남기면
        # 프롬프트로 새어 들어간 개인정보가 답변에 되울릴 때 그대로 기록된다.
        assert any("근거 밖" in reason for reason in result.reasons)
        assert any("숫자 2건" in reason for reason in result.reasons)
        for reason in result.reasons:
            assert "250만원" not in reason
            assert "250" not in reason

    def test_fake_institution_is_blocked(
        self, retriever: LocalTfidfRetriever, consent_field: Field
    ) -> None:
        """근거에 없는 기관 이름도 폐기 대상이다."""
        fake = "자세한 내용은 국민행복공단에 문의하십시오."
        explainer = Explainer(retriever, FixedLlm(fake), Guard(RrnPiiGate()))
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert result.needs_handoff is True
        assert fake not in result.text

    def test_empty_generation_is_handoff(
        self, retriever: LocalTfidfRetriever, consent_field: Field
    ) -> None:
        """설명을 만들지 못하면 조용히 빈 문장을 내보내지 않는다."""
        explainer = Explainer(retriever, FixedLlm("  "), Guard(RrnPiiGate()))
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert result.needs_handoff is True
        assert result.text == HANDOFF_SPEECH


class TestPrivacyBoundary:
    """공개/개인정보 영역 분리."""

    def test_private_field_query_excludes_title_and_clause(
        self, explainer: Explainer, rrn_field: Field
    ) -> None:
        """개인정보 영역 항목의 제목·약관 문구는 질의에 들어가지 않는다."""
        query = explainer.build_query(rrn_field)
        assert query == FIELD_TYPE_QUERIES[FieldType.TEXT_INPUT]
        assert rrn_field.title not in query

    def test_public_field_query_uses_clause_without_question(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """공개 영역 항목은 질문이 없을 때 약관 전문을 질의에 쓴다."""
        query = explainer.build_query(consent_field)
        assert consent_field.title in query
        assert consent_field.clause_text in query

    def test_public_field_query_drops_clause_with_question(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """질문이 있으면 약관 전문 대신 질문을 쓴다(검색 희석 방지)."""
        query = explainer.build_query(consent_field, "동의가 뭐예요?")
        assert "동의가 뭐예요?" in query
        assert consent_field.clause_text not in query

    def test_pii_in_question_blocks_llm_path(
        self, retriever: LocalTfidfRetriever, rrn_field: Field
    ) -> None:
        """질문에 주민등록번호가 섞이면 생성 경로를 타지 않고 핸드오프한다."""
        explainer = Explainer(retriever, OfflineTemplateLlm(), Guard(RrnPiiGate()))
        result = explainer.explain(rrn_field, "900101-1234567 이렇게 적으면 되나요?")
        assert result.needs_handoff is True
        assert "900101" not in result.text


class TestExplainerBehaviour:
    """설명기 일반 동작."""

    def test_result_is_deterministic(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """같은 입력은 항상 같은 결과."""
        first = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        second = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert first == second

    def test_sources_are_unique_and_traceable(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """출처는 중복 없이 남고 파일명을 포함해 감사 가능하다."""
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert len(result.sources) == len(set(result.sources))
        assert all(".md" in source for source in result.sources)

    def test_offline_answer_stays_within_evidence(
        self, explainer: Explainer, consent_field: Field
    ) -> None:
        """오프라인 생성물은 근거 검증을 스스로 통과한다(근거 밖 생성 없음)."""
        result = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert result.needs_handoff is False

    def test_no_matching_evidence_is_handoff(
        self, retriever: LocalTfidfRetriever
    ) -> None:
        """코퍼스와 겹치지 않는 항목은 근거 없음으로 핸드오프한다."""
        explainer = Explainer(retriever, OfflineTemplateLlm(), Guard())
        field = Field(id="unknown_01", type=FieldType.UNKNOWN, title="zzzq")
        result = explainer.explain(field, "zzzq wwwx")
        assert result.needs_handoff is True

    def test_rejects_bad_top_k(self, retriever: LocalTfidfRetriever) -> None:
        """top_k 가 1 미만이면 ValueError."""
        with pytest.raises(ValueError, match="top_k"):
            Explainer(retriever, OfflineTemplateLlm(), Guard(), top_k=0)

    def test_result_to_dict(self, explainer: Explainer, consent_field: Field) -> None:
        """결과를 JSON 직렬화 가능한 dict 로 바꿀 수 있다."""
        payload = explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?").to_dict()
        assert payload["mode"] == "explain"
        assert payload["disclaimer"] == DISCLAIMER
        assert isinstance(payload["sources"], list)

    def test_result_validates_confidence(self) -> None:
        """신뢰도 범위 위반은 생성 시 거부한다."""
        with pytest.raises(ValueError, match="0.0~1.0"):
            ExplanationResult(
                text="x",
                confidence=1.5,
                sources=(),
                needs_handoff=False,
                disclaimer=DISCLAIMER,
                mode=ConfidenceBand.EXPLAIN,
            )

    def test_signature_field_is_explained(
        self, explainer: Explainer, structure
    ) -> None:
        """서명란도 근거 문서로 설명된다."""
        field = structure.field_by_id(APPLICANT_SIGNATURE_FIELD_ID)
        result = explainer.explain(field)
        assert result.needs_handoff is False
        assert any("signature" in source for source in result.sources)


# --------------------------------------------------------------------------
# LLM 어댑터
# --------------------------------------------------------------------------


class TestOfflineTemplateLlm:
    """오프라인 기본 구현."""

    def test_satisfies_llm_protocol(self) -> None:
        """LlmClient 프로토콜을 만족한다."""
        assert isinstance(OfflineTemplateLlm(), LlmClient)

    def test_prompt_roundtrip(self) -> None:
        """프롬프트를 만들고 되읽으면 질문과 근거가 복원된다."""
        chunks = [
            RetrievedChunk(chunk_id="c1", text="첫 근거입니다.", source="a.md", score=0.3),
            RetrievedChunk(chunk_id="c2", text="둘째 근거입니다.", source="b.md", score=0.2),
        ]
        question, evidence = parse_user_prompt(build_user_prompt("질문입니다?", chunks))
        assert question == "질문입니다?"
        assert evidence == ["첫 근거입니다.", "둘째 근거입니다."]

    def test_generates_only_from_evidence(self) -> None:
        """근거에 없는 문장을 만들지 않는다."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                text=(
                    "정보주체는 본인을 말합니다. "
                    "제3자 제공은 다른 곳에 정보를 넘기는 것입니다. "
                    "고유식별정보는 엄격하게 다룹니다."
                ),
                source="a.md",
                score=0.3,
            )
        ]
        answer = OfflineTemplateLlm().complete(
            SYSTEM_PROMPT, build_user_prompt("제3자 제공이 뭔가요?", chunks)
        )
        assert answer
        assert "정보주체" not in answer  # 쉬운 말로 치환되었다.
        assert "본인" in answer

    def test_plain_language_substitution(self) -> None:
        """어려운 표현이 쉬운 표현으로 바뀐다."""
        chunks = [
            RetrievedChunk(
                chunk_id="c1",
                text="고유식별정보는 사람을 한 명으로 구별하는 정보입니다.",
                source="a.md",
                score=0.3,
            )
        ]
        answer = OfflineTemplateLlm().complete(
            SYSTEM_PROMPT, build_user_prompt("고유식별정보가 뭐예요?", chunks)
        )
        assert "주민등록번호 같은 정보" in answer

    def test_no_evidence_returns_empty(self) -> None:
        """근거가 없으면 빈 문자열을 돌려준다(억지 생성 금지)."""
        assert OfflineTemplateLlm().complete(SYSTEM_PROMPT, "[질문]\n뭐예요?") == ""

    def test_is_deterministic(self) -> None:
        """같은 프롬프트는 항상 같은 출력."""
        chunks = [
            RetrievedChunk(chunk_id="c1", text="근거 문장이 여기 있습니다.", source="a.md", score=0.3)
        ]
        prompt = build_user_prompt("질문", chunks)
        llm = OfflineTemplateLlm()
        assert llm.complete(SYSTEM_PROMPT, prompt) == llm.complete(SYSTEM_PROMPT, prompt)

    def test_rejects_bad_max_sentences(self) -> None:
        """문장 수 상한이 1 미만이면 ValueError."""
        with pytest.raises(ValueError, match="max_sentences"):
            OfflineTemplateLlm(max_sentences=0)

    def test_system_prompt_states_prohibitions(self) -> None:
        """시스템 프롬프트에 근거 밖 생성 금지와 판단 금지가 명시되어 있다."""
        assert "근거 문서 밖의 내용을 추가하지" in SYSTEM_PROMPT
        assert "법적 판단" in SYSTEM_PROMPT
        assert "자격 판단" in SYSTEM_PROMPT


class TestClaudeAdapter:
    """선택적 Claude 어댑터."""

    def test_unavailable_without_package(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """anthropic 미설치 환경에서는 한국어 설치 안내와 함께 예외를 던진다."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(AdapterUnavailable) as info:
            ClaudeLlm()
        message = str(info.value)
        assert "anthropic" in message
        assert ".venv" in message

    def test_rejects_empty_model(self) -> None:
        """모델 id 가 비면 ValueError."""
        with pytest.raises(ValueError, match="model"):
            ClaudeLlm("")


class TestGatedLlm:
    """게이트 래퍼와 팩토리."""

    def test_build_llm_wraps_with_gate(self) -> None:
        """팩토리는 항상 게이트로 감싼 클라이언트를 돌려준다."""
        client = build_llm(RrnPiiGate())
        assert isinstance(client, GatedLlmClient)
        assert isinstance(client.inner, OfflineTemplateLlm)

    def test_build_llm_wraps_injected_inner(self) -> None:
        """주입한 클라이언트도 게이트로 감싼다."""
        inner = FixedLlm("고정 응답입니다.")
        client = build_llm(RrnPiiGate(), inner=inner)
        assert client.inner is inner

    def test_build_llm_client_is_audited(self) -> None:
        """build_llm 이 만든 클라이언트로 호출하면 감사 로그에 기록이 남는다.

        감사 기록이 없는 두 번째 래퍼가 있으면 안전 KPI
        (``pii_llm_transmissions``)가 아무 근거 없이 0 으로 보고된다.
        """
        client = build_llm(RrnPiiGate(), inner=FixedLlm("응답입니다."))
        before = client.gate.audit.counters()["llm_calls"]
        client.complete("시스템", "동의가 뭐예요?")
        assert client.gate.audit.counters()["llm_calls"] == before + 1

    def test_default_gate_client_shares_the_gate_audit(self) -> None:
        """기본 게이트로 만든 클라이언트도 같은 감사 로그를 쓴다."""
        from docagent.pii.gate import build_default_gate

        gate = build_default_gate()
        client = build_llm(gate, inner=FixedLlm("응답입니다."))
        client.complete("시스템", "동의가 뭐예요?")
        assert gate.audit.counters()["llm_calls"] == 1

    def test_build_llm_rejects_unknown_kind(self) -> None:
        """알 수 없는 종류는 ValueError."""
        with pytest.raises(ValueError, match="offline 또는 claude"):
            build_llm(RrnPiiGate(), kind="mystery")

    def test_build_llm_without_gate_resolves_the_default_gate(self) -> None:
        """게이트를 주지 않으면 기본 구현을 찾아 감싼다(통합 후 연결된 경로).

        ``docagent.pii.gate.build_default_gate`` 가 생기면서
        :func:`docagent.agent.llm._resolve_default_gate` 가 실제로 게이트를
        확보한다. 감싸지 않은 클라이언트가 돌아오는 일은 여전히 없다.
        """
        from docagent.pii.gate import LlmEgressGate

        client = build_llm(None)
        assert isinstance(client, GatedLlmClient)
        assert isinstance(client.gate, LlmEgressGate)

    def test_build_llm_is_blocked_when_no_default_gate_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """기본 게이트를 **확보하지 못하면** 조용히 통과시키지 않고 차단한다."""
        from docagent.pii import gate as gate_module

        for name in (
            "build_default_gate",
            "default_gate",
            "DefaultPiiGate",
            "PiiGateImpl",
        ):
            monkeypatch.delattr(gate_module, name, raising=False)
        with pytest.raises(PiiEgressBlocked, match="게이트"):
            build_llm(None)

    def test_gate_masks_prompt(self) -> None:
        """게이트가 마스킹한 문자열만 내부 구현으로 내려간다."""
        seen: dict[str, str] = {}

        class Recorder:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                seen["user"] = user
                return "ok"

        client = build_llm(RrnPiiGate(), inner=Recorder())
        client.complete("시스템", "주민등록번호 900101-1234567 입니다")
        assert "900101" not in seen["user"]

    def test_blocked_gate_raises(self) -> None:
        """게이트가 차단하면 PiiEgressBlocked 를 던지고 호출하지 않는다."""
        client = build_llm(BlockingGate(), inner=FixedLlm("응답"))
        with pytest.raises(PiiEgressBlocked):
            client.complete("시스템", "사용자")


class SpyLlm:
    """호출될 때마다 프롬프트를 기록하는 스파이 LLM."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """프롬프트를 기록하고 무해한 문장을 돌려준다.

        :param system: 시스템 프롬프트.
        :param user: 사용자 프롬프트.
        :param max_tokens: 무시.
        :returns: 고정 문장.
        """
        del max_tokens
        self.calls.append((system, user))
        return "이 항목은 서류에 적힌 대로 작성하시면 됩니다."


class TestDocumentTextIsScreenedBeforeTheLlm:
    """문서에서 읽어 온 문구도 LLM **호출 전에** 개인정보 검사를 받는다.

    ``sensitivity=PUBLIC`` 판정만 믿고 ``title``·``clause_text`` 를 프롬프트에
    실으면, OCR 로 딸려 들어온 기입값이 아무 검사 없이 어댑터에 도달한다.
    """

    def _field_with_pii(self) -> Field:
        """개인정보가 섞여 들어온 '공개' 항목을 만든다."""
        return Field(
            id="consent_pii",
            type=FieldType.CHOICE,
            title="개인정보 수집·이용 동의",
            clause_text=(
                "신청인 김철수, 건강보험증번호는 12345678901, "
                "계좌번호는 352 0123 4567 89 입니다."
            ),
            required=True,
            order=1,
            sensitivity=Sensitivity.PUBLIC,
        )

    def test_llm_is_never_called(self, retriever: LocalTfidfRetriever) -> None:
        """개인정보가 섞인 질의는 LLM 을 호출하지 않고 핸드오프한다."""
        spy = SpyLlm()
        explainer = Explainer(retriever, spy, Guard())
        result = explainer.explain(self._field_with_pii())
        assert spy.calls == []
        assert result.needs_handoff is True
        assert result.mode is ConfidenceBand.HANDOFF

    def test_clean_public_field_still_reaches_the_llm(
        self, retriever: LocalTfidfRetriever, consent_field: Field
    ) -> None:
        """개인정보가 없는 정상 항목은 그대로 설명 경로를 탄다(과잉 차단 금지)."""
        spy = SpyLlm()
        explainer = Explainer(retriever, spy, Guard())
        explainer.explain(consent_field, "개인정보 수집 동의가 뭐예요?")
        assert spy.calls, "정상 항목까지 막혀 설명 경로가 끊겼습니다."

    def test_document_number_in_clause_is_not_over_blocked(
        self, retriever: LocalTfidfRetriever
    ) -> None:
        """접수번호·조문 같은 정당한 숫자만 있는 문구는 막히지 않는다."""
        spy = SpyLlm()
        field = Field(
            id="notice_01",
            type=FieldType.CHECKBOX,
            title="안내 사항 확인",
            clause_text=(
                "접수번호 제2026-0001호로 접수되었으며 근거는 제12조 제3항입니다."
            ),
            required=True,
            order=1,
            sensitivity=Sensitivity.PUBLIC,
        )
        Explainer(retriever, spy, Guard()).explain(field)
        assert spy.calls, "정당한 문서 번호까지 개인정보로 오인해 막았습니다."
