"""게이트 테스트 — 적대적 관점에서 "개인정보 LLM 전송 0건" 을 검증한다.

핵심 단언은 두 가지다.

1. 마스킹되지 않은 텍스트는 :class:`PiiEgressBlocked` 로 막힌다.
2. 하위 LLM 클라이언트가 **실제로 받은 인자**에 원문 개인정보가 없다
   (스파이 객체로 인자를 캡처해 확인한다).
"""

from __future__ import annotations

from typing import Any

import pytest

from docagent.contracts import PiiSpan, SanitizedText
from docagent.errors import PiiEgressBlocked, ToolExecutionError
from docagent.interfaces import PiiGate
from docagent.pii.audit import AuditLog, FixedClock
from docagent.pii.detectors import PiiDetectionError, detect_pii
from docagent.pii.gate import GatedLlmClient, LlmEgressGate
from docagent.pii.masker import MaskStrategy

SAMPLE_DOC = (
    "성명 김철수\n"
    "주민등록번호 900101-1234567\n"
    "주소 서울특별시 중구 세종대로 110\n"
    "전화 010-1234-5678\n"
)

FORBIDDEN_FRAGMENTS = (
    "김철수",
    "900101",
    "1234567",
    "세종대로 110",
    "010-1234-5678",
)

BENIGN_DOC = "접수번호 제2026-0001호, 지급 금액 1,234,567원, 기한 2026년 9월 30일."


class SpyLlmClient:
    """하위 LLM 클라이언트 스파이 — 실제로 받은 인자를 그대로 보관한다."""

    def __init__(self, answer: str = "설명 결과입니다.") -> None:
        self.calls: list[tuple[str, str, int]] = []
        self._answer = answer

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """호출 인자를 기록하고 고정 응답을 돌려준다."""
        self.calls.append((system, user, max_tokens))
        return self._answer


class ExplodingDetector:
    """항상 예외를 던지는 탐지기(fail-closed 검증용)."""

    def detect(self, text: str) -> tuple[PiiSpan, ...]:
        """언제나 실패한다."""
        raise PiiDetectionError("탐지기 내부 오류(테스트용)")


class LyingDetector:
    """개인정보를 못 본 척하는 탐지기(게이트 우회 시도 검증용)."""

    def detect(self, text: str) -> tuple[PiiSpan, ...]:
        """항상 빈 결과를 돌려준다."""
        return ()


class BadTypeDetector:
    """계약을 어기고 PiiSpan 이 아닌 값을 반환하는 탐지기."""

    def detect(self, text: str) -> Any:
        """잘못된 타입을 돌려준다."""
        return [{"start": 0, "end": 3}]


def make_gate(**kwargs: Any) -> LlmEgressGate:
    """결정론적 시계를 붙인 게이트를 만드는 테스트 헬퍼."""
    kwargs.setdefault("audit", AuditLog(clock=FixedClock(step_seconds=1)))
    return LlmEgressGate(**kwargs)


class TestProtocolCompliance:
    """계약 준수."""

    def test_gate_satisfies_pii_gate_protocol(self) -> None:
        """:class:`LlmEgressGate` 가 PiiGate 프로토콜을 만족한다."""
        assert isinstance(make_gate(), PiiGate)

    def test_partial_strategy_is_rejected(self) -> None:
        """PARTIAL 은 원문을 남기므로 게이트 전략으로 쓸 수 없다."""
        with pytest.raises(ValueError, match="PARTIAL"):
            LlmEgressGate(strategy=MaskStrategy.PARTIAL)

    def test_repr_does_not_leak(self) -> None:
        """repr 이 설정만 노출한다."""
        assert "김철수" not in repr(make_gate())


class TestSanitize:
    """:meth:`LlmEgressGate.sanitize`."""

    def test_sanitize_removes_all_pii(self) -> None:
        """마스킹 결과에 원문 조각이 남지 않는다."""
        result = make_gate().sanitize(SAMPLE_DOC)
        assert isinstance(result, SanitizedText)
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in result.text

    def test_sanitize_reports_spans_with_source_indices(self) -> None:
        """구간 인덱스는 원문 기준이며 원문 값을 담지 않는다."""
        result = make_gate().sanitize(SAMPLE_DOC)
        assert result.spans == detect_pii(SAMPLE_DOC)
        assert not result.blocked

    def test_sanitize_of_clean_text_is_identity(self) -> None:
        """개인정보가 없으면 원문이 그대로 유지된다."""
        result = make_gate().sanitize(BENIGN_DOC)
        assert result.text == BENIGN_DOC
        assert not result.has_pii

    def test_sanitize_rejects_non_string(self) -> None:
        """문자열이 아닌 입력은 TypeError."""
        with pytest.raises(TypeError):
            make_gate().sanitize(123)  # type: ignore[arg-type]


class TestAssertClean:
    """:meth:`LlmEgressGate.assert_clean` — 요구사항 3."""

    def test_unmasked_text_is_blocked(self) -> None:
        """마스킹하지 않은 원문은 PiiEgressBlocked 로 막힌다."""
        with pytest.raises(PiiEgressBlocked):
            make_gate().assert_clean(SAMPLE_DOC)

    def test_exception_carries_types_and_count_only(self) -> None:
        """예외에 유형·건수만 담기고 원문은 담기지 않는다."""
        with pytest.raises(PiiEgressBlocked) as excinfo:
            make_gate().assert_clean(SAMPLE_DOC)
        error = excinfo.value
        assert set(error.pii_types) == {"name", "rrn", "address", "phone_mobile"}
        assert error.count == 4
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in str(error)

    def test_masked_text_passes(self) -> None:
        """마스킹된 문자열은 통과한다."""
        gate = make_gate()
        gate.assert_clean(gate.sanitize(SAMPLE_DOC).text)

    def test_benign_text_passes(self) -> None:
        """접수번호·금액·날짜만 있는 문장은 막히지 않는다(요구사항 6)."""
        make_gate().assert_clean(BENIGN_DOC)

    @pytest.mark.parametrize(
        "payload",
        [
            "900101-1234567",
            "9001011234567",
            "900101 1234567",
            "900101–1234567",
            "900101－1234567",
            "900101.1234567",
            "900101-\n1234567",
            "900101​-1234567",
            "９００１０１-１２３４５６７",
            "라벨 없이 9001011234567 만 있는 문장",
            "hong@example.com",
            "4111-1111-1111-1111",
        ],
    )
    def test_evasion_variants_are_blocked(self, payload: str) -> None:
        """회피 변형 12종이 모두 차단된다."""
        with pytest.raises(PiiEgressBlocked):
            make_gate().assert_clean(payload)


class TestFailClosed:
    """탐지기가 고장 나도 통과시키지 않는다."""

    def test_detector_failure_blocks_in_strict_mode(self) -> None:
        """strict 모드에서 탐지기 예외는 차단으로 바뀐다."""
        gate = make_gate(detector=ExplodingDetector())
        with pytest.raises(PiiEgressBlocked, match="탐지기"):
            gate.assert_clean("아무 문자열")

    def test_detector_failure_propagates_when_not_strict(self) -> None:
        """strict=False 여도 통과시키지 않고 원래 예외를 올린다."""
        gate = make_gate(detector=ExplodingDetector(), strict=False)
        with pytest.raises(PiiDetectionError):
            gate.assert_clean("아무 문자열")

    def test_bad_detector_return_type_is_blocked(self) -> None:
        """탐지기가 계약을 어긴 값을 돌려주면 차단한다."""
        gate = make_gate(detector=BadTypeDetector())
        with pytest.raises(PiiEgressBlocked, match="PiiSpan"):
            gate.assert_clean("아무 문자열")

    def test_lying_detector_is_recorded_but_not_crashed(self) -> None:
        """탐지기를 교체해도 게이트 자체는 계약대로 동작한다."""
        gate = make_gate(detector=LyingDetector())
        assert gate.sanitize(SAMPLE_DOC).text == SAMPLE_DOC


class TestGuard:
    """:meth:`LlmEgressGate.guard` — dict/list 재귀 검사."""

    def test_nested_strings_are_masked(self) -> None:
        """중첩 구조 안의 문자열이 모두 마스킹된다."""
        payload = {
            "doc": {"title": "지원금 신청서", "note": "주민등록번호 900101-1234567"},
            "items": ["연락처 010-1234-5678", {"name": "성명 김철수"}],
            "count": 3,
            "ok": True,
            "empty": None,
        }
        guarded = make_gate().guard(payload)
        flat = repr(guarded)
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in flat
        assert guarded["count"] == 3
        assert guarded["ok"] is True
        assert guarded["empty"] is None

    def test_original_payload_is_not_mutated(self) -> None:
        """원본 dict 는 변경되지 않는다."""
        payload = {"note": "주민등록번호 900101-1234567"}
        make_gate().guard(payload)
        assert payload["note"] == "주민등록번호 900101-1234567"

    def test_dict_keys_are_checked(self) -> None:
        """dict 의 키도 검사 대상이다."""
        guarded = make_gate().guard({"900101-1234567": "값"})
        assert "900101" not in repr(guarded)

    def test_numeric_pii_is_blocked(self) -> None:
        """숫자에 담긴 주민등록번호는 마스킹 불가이므로 차단한다."""
        with pytest.raises(PiiEgressBlocked, match="숫자"):
            make_gate().guard({"rrn": 9001011234567})

    def test_benign_number_passes(self) -> None:
        """무해한 숫자는 그대로 통과한다."""
        assert make_gate().guard({"amount": 1234567}) == {"amount": 1234567}

    def test_unsupported_type_raises(self) -> None:
        """지원하지 않는 타입은 조용히 통과시키지 않는다."""
        with pytest.raises(TypeError, match="처리할 수 없는"):
            make_gate().guard({"blob": object()})

    def test_tuple_stays_tuple(self) -> None:
        """튜플은 튜플로 유지된다."""
        assert isinstance(make_gate().guard(("가", "나")), tuple)


class TestGatedLlmClient:
    """:class:`GatedLlmClient` — 요구사항 4."""

    def test_inner_client_never_sees_raw_pii(self) -> None:
        """하위 클라이언트가 실제로 받은 인자에 원문 개인정보가 없다."""
        spy = SpyLlmClient()
        client = GatedLlmClient(spy, gate=make_gate())
        client.complete(system="문서를 쉬운 말로 설명하라", user=SAMPLE_DOC)

        assert len(spy.calls) == 1
        system, user, max_tokens = spy.calls[0]
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in system
            assert fragment not in user
        assert max_tokens == 1024
        assert "<REDACTED>" in user

    def test_pii_in_system_prompt_is_also_masked(self) -> None:
        """시스템 프롬프트에 섞인 개인정보도 마스킹된다."""
        spy = SpyLlmClient()
        client = GatedLlmClient(spy, gate=make_gate())
        client.complete(system=f"참고: {SAMPLE_DOC}", user="이 항목을 설명하라")
        system, _user, _max_tokens = spy.calls[0]
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in system

    def test_answer_is_returned(self) -> None:
        """하위 응답이 그대로 반환된다."""
        client = GatedLlmClient(SpyLlmClient("동의 여부를 묻는 항목입니다."), gate=make_gate())
        assert client.complete("시스템", "질문") == "동의 여부를 묻는 항목입니다."

    def test_detector_failure_prevents_the_call(self) -> None:
        """탐지기가 실패하면 하위 클라이언트를 아예 호출하지 않는다."""
        spy = SpyLlmClient()
        client = GatedLlmClient(spy, gate=make_gate(detector=ExplodingDetector()))
        with pytest.raises(PiiEgressBlocked):
            client.complete("시스템", SAMPLE_DOC)
        assert spy.calls == []

    def test_inner_failure_is_wrapped(self) -> None:
        """하위 어댑터 예외는 도메인 예외로 감싸 올린다."""

        class Broken:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                raise RuntimeError("네트워크 오류")

        client = GatedLlmClient(Broken(), gate=make_gate())
        with pytest.raises(ToolExecutionError, match="LLM 호출"):
            client.complete("시스템", "질문")

    def test_non_string_answer_is_rejected(self) -> None:
        """문자열이 아닌 응답은 조용히 넘기지 않는다."""

        class Weird:
            def complete(self, system: str, user: str, max_tokens: int = 1024) -> Any:
                return 42

        client = GatedLlmClient(Weird(), gate=make_gate())
        with pytest.raises(ToolExecutionError, match="문자열이 아닙니다"):
            client.complete("시스템", "질문")

    def test_wrapping_non_client_raises(self) -> None:
        """complete 가 없는 객체는 감쌀 수 없다."""
        with pytest.raises(TypeError, match="complete"):
            GatedLlmClient(object())

    def test_invalid_max_tokens_raises(self) -> None:
        """max_tokens 는 1 이상이어야 한다."""
        client = GatedLlmClient(SpyLlmClient(), gate=make_gate())
        with pytest.raises(ValueError, match="max_tokens"):
            client.complete("시스템", "질문", max_tokens=0)

    def test_audit_counters_report_zero_leak(self) -> None:
        """감사 카운터의 KPI 지표(pii_leaked)가 0 이다."""
        gate = make_gate()
        client = GatedLlmClient(SpyLlmClient(), gate=gate)
        client.complete(system="설명하라", user=SAMPLE_DOC)
        counters = gate.audit.counters()
        assert counters["llm_calls"] == 1
        assert counters["pii_detected"] == 4
        assert counters["blocked"] == 0
        assert counters["pii_leaked"] == 0

    def test_blocked_call_is_counted_and_not_forwarded(self) -> None:
        """차단된 호출은 llm_calls 로 세지 않는다."""
        gate = make_gate(detector=ExplodingDetector())
        spy = SpyLlmClient()
        client = GatedLlmClient(spy, gate=gate)
        with pytest.raises(PiiEgressBlocked):
            client.complete("시스템", SAMPLE_DOC)
        assert gate.audit.counters()["llm_calls"] == 0
        assert spy.calls == []


class TestErrorMessagesCarryNoRawValues:
    """차단 메시지에 원문(키 이름 포함)이 섞이지 않는지 확인한다."""

    def test_dict_key_is_not_echoed_in_the_message(self) -> None:
        """위치 표기에 dict 의 원문 키를 넣지 않는다.

        이 메시지는 ToolRegistry 를 거쳐 세션 이력 → 세션 복원 JSON 으로
        영구 저장될 수 있으므로, 키 이름 자체가 개인정보인 경우를 대비한다.
        """
        payload = {"신청인 주민등록번호 900101-1234567": 9001011234567}
        with pytest.raises(PiiEgressBlocked) as info:
            make_gate().guard(payload)
        message = str(info.value)
        assert "900101" not in message
        assert "1234567" not in message
        assert "key#" in message


class TestSplitFieldCombination:
    """조각으로 나눈 개인정보(신청서의 본래 구조)를 결합해 검사한다."""

    def test_split_rrn_is_blocked(self) -> None:
        """앞자리/뒷자리로 쪼갠 주민등록번호는 결합 검사에서 막힌다."""
        with pytest.raises(PiiEgressBlocked, match="합치면"):
            make_gate().guard({"rrn_front": "900101", "rrn_back": "1234567"})

    def test_unrelated_fields_are_not_over_blocked(self) -> None:
        """무해한 항목을 이어 붙였다고 막지는 않는다(과잉 차단 금지)."""
        guarded = make_gate().guard(
            {"title": "지원금 신청서", "page": 3, "year": "2026"}
        )
        assert guarded["page"] == 3

    def test_split_rrn_inside_a_list_is_blocked(self) -> None:
        """리스트 항목으로 흩어 담은 조각도 결합 검사를 받는다."""
        with pytest.raises(PiiEgressBlocked, match="합치면"):
            make_gate().guard({"조각": ["900101", "1234567"]})

    def test_split_rrn_across_nested_dicts_is_blocked(self) -> None:
        """하위 dict 로 흩어 담은 조각도 결합 검사를 받는다."""
        with pytest.raises(PiiEgressBlocked, match="합치면"):
            make_gate().guard({"a": {"x": "900101"}, "b": {"y": "1234567"}})

    def test_split_rrn_in_list_of_dicts_is_blocked(self) -> None:
        """공개 payload 와 같은 '항목 dict 의 리스트' 형태도 검사한다."""
        payload = {
            "fields": [
                {"id": "f1", "value": "900101"},
                {"id": "f2", "value": "1234567"},
            ]
        }
        with pytest.raises(PiiEgressBlocked, match="합치면"):
            make_gate().guard(payload)


class TestSecondOpinion:
    """탐지기 자기보고와 독립된 2차 검사."""

    def test_masked_prompt_records_zero_unknown_runs(self) -> None:
        """정상 마스킹 결과에는 유래 불명 숫자열이 남지 않는다."""
        gate = make_gate()
        gate.prepare(SAMPLE_DOC, caller="test")
        assert gate.audit.counters()["pii_residual_unknown"] == 0

    def test_undetected_long_digits_are_blocked(self) -> None:
        """탐지 규칙이 못 본 긴 숫자열은 2차 검사가 **차단**한다(fail-closed).

        기록만 하고 내보내면 '규칙이 모르는 형태의 개인정보'가 blocked=False 로
        그대로 나간다. 2차 검사는 감사 지표이자 차단 조건이어야 한다.
        """

        class BlindDetector:
            """아무것도 탐지하지 못하는 탐지기(규칙 누락 재현)."""

            def detect(self, text: str) -> tuple[PiiSpan, ...]:
                return ()

        gate = make_gate(detector=BlindDetector())
        with pytest.raises(PiiEgressBlocked, match="숫자열"):
            gate.prepare("확인 번호 12345678901 입니다", caller="test")
        counters = gate.audit.counters()
        assert counters["pii_leaked"] == 0
        assert counters["blocked"] == 1
        # 나가지 않았으므로 유출 지표는 0 이어야 한다.
        assert counters["pii_residual_unknown"] == 0

    def test_short_numbers_are_not_counted(self) -> None:
        """연도·조문 같은 짧은 숫자는 위반으로 세지 않는다."""
        gate = make_gate()
        gate.prepare("2026년 제17조를 참고하십시오.", caller="test")
        assert gate.audit.counters()["pii_residual_unknown"] == 0

    def test_benign_document_numbers_are_not_over_blocked(self) -> None:
        """접수번호·금액·날짜 같은 정당한 숫자는 2차 검사가 막지 않는다."""
        gate = make_gate()
        text = (
            "접수번호 제2026-0001호로 접수되었으며 지급 금액은 1,234,567원입니다. "
            "처리 기한은 2026년 9월 30일이고 근거는 제12조 제3항입니다."
        )
        assert gate.prepare(text, caller="test") == text
        assert gate.audit.counters()["blocked"] == 0

    def test_masked_pii_still_passes_second_opinion(self) -> None:
        """정상적으로 마스킹된 문장은 2차 검사도 통과한다."""
        gate = make_gate()
        out = gate.prepare("주민등록번호 900101-1234567 입니다", caller="test")
        assert "900101" not in out
        assert gate.audit.counters()["blocked"] == 0
