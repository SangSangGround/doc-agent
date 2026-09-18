"""감사 로그 테스트 — 기록에 원문이 남지 않는가, KPI 카운터가 맞는가.

요구사항 5("감사 로그 파일 내용에 원문 PII 가 없음")를 실제 파일 내용으로 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docagent.contracts import PiiSpan
from docagent.errors import PiiEgressBlocked
from docagent.interfaces import Clock
from docagent.pii.audit import (
    HASH_PREFIX_LEN,
    AuditLog,
    AuditRecord,
    FixedClock,
    SystemClock,
    text_fingerprint,
)
from docagent.pii.detectors import detect_pii
from docagent.pii.gate import GatedLlmClient, LlmEgressGate

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
    "세종대로",
    "010-1234-5678",
)


class EchoClient:
    """받은 프롬프트를 그대로 되돌려주는 테스트용 LLM."""

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """마스킹된 프롬프트를 그대로 반환한다."""
        return f"{system}|{user}"


class TestClocks:
    """시각 공급자."""

    def test_fixed_clock_is_deterministic(self) -> None:
        """step 이 0 이면 항상 같은 값을 돌려준다."""
        clock = FixedClock("2026-01-01T09:00:00+09:00")
        assert clock.now_iso() == clock.now_iso() == "2026-01-01T09:00:00+09:00"

    def test_fixed_clock_steps_forward(self) -> None:
        """step 을 주면 호출마다 시간이 진행한다."""
        clock = FixedClock("2026-01-01T09:00:00+09:00", step_seconds=30)
        assert clock.now_iso() == "2026-01-01T09:00:00+09:00"
        assert clock.now_iso() == "2026-01-01T09:00:30+09:00"

    def test_invalid_start_raises(self) -> None:
        """해석할 수 없는 시각 문자열은 ValueError."""
        with pytest.raises(ValueError, match="ISO 8601"):
            FixedClock("어제")

    def test_clocks_satisfy_protocol(self) -> None:
        """두 구현 모두 Clock 프로토콜을 만족한다."""
        assert isinstance(SystemClock(), Clock)
        assert isinstance(FixedClock(), Clock)


class TestFingerprint:
    """텍스트 지문."""

    def test_length_is_fixed(self) -> None:
        """지문 길이가 상수와 일치한다."""
        assert len(text_fingerprint("아무 문자열")) == HASH_PREFIX_LEN

    def test_same_text_same_fingerprint(self) -> None:
        """같은 텍스트는 같은 지문."""
        assert text_fingerprint(SAMPLE_DOC) == text_fingerprint(SAMPLE_DOC)

    def test_different_text_different_fingerprint(self) -> None:
        """다른 텍스트는 다른 지문."""
        assert text_fingerprint("가") != text_fingerprint("나")

    def test_fingerprint_does_not_contain_source(self) -> None:
        """지문에서 원문 조각을 찾을 수 없다."""
        fingerprint = text_fingerprint(SAMPLE_DOC)
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in fingerprint


class TestAuditRecord:
    """레코드 계약."""

    def test_record_keys_are_fixed(self) -> None:
        """dict 키 집합이 고정되어 있다(원문 필드 추가 금지)."""
        record = AuditRecord(
            timestamp="2026-01-01T09:00:00+09:00",
            caller="test",
            event="llm_call",
            text_len=10,
            pii_types=("rrn",),
            pii_count=1,
            blocked=False,
            residual_count=0,
            text_sha256_12="0123456789ab",
        )
        assert set(record.to_dict()) == {
            "timestamp",
            "caller",
            "event",
            "text_len",
            "pii_types",
            "pii_count",
            "blocked",
            "residual_count",
            "text_sha256_12",
            "residual_unknown",
        }

    def test_round_trip(self) -> None:
        """to_dict → from_dict 왕복이 무손실이다."""
        record = AuditRecord(
            timestamp="2026-01-01T09:00:00+09:00",
            caller="test",
            event="egress",
            text_len=10,
            pii_types=("name", "rrn"),
            pii_count=2,
            blocked=True,
            residual_count=0,
            text_sha256_12="0123456789ab",
        )
        assert AuditRecord.from_dict(record.to_dict()) == record

    def test_missing_key_raises(self) -> None:
        """필수 키가 없으면 ValueError."""
        with pytest.raises(ValueError, match="필수 키"):
            AuditRecord.from_dict({"caller": "x"})

    def test_negative_values_raise(self) -> None:
        """음수 길이·건수는 거부된다."""
        with pytest.raises(ValueError, match="0 이상"):
            AuditRecord(
                timestamp="t",
                caller="c",
                event="e",
                text_len=-1,
                pii_types=(),
                pii_count=0,
                blocked=False,
                residual_count=0,
                text_sha256_12="x",
            )


class TestAuditLog:
    """메모리·파일 기록."""

    def test_record_stores_metadata_only(self) -> None:
        """레코드에 원문이 아니라 메타데이터만 남는다."""
        log = AuditLog(clock=FixedClock())
        spans = detect_pii(SAMPLE_DOC)
        record = log.record(caller="테스트", text=SAMPLE_DOC, spans=spans)
        assert record.text_len == len(SAMPLE_DOC)
        assert record.pii_count == 4
        assert record.pii_types == ("address", "name", "phone_mobile", "rrn")
        assert record.text_sha256_12 == text_fingerprint(SAMPLE_DOC)

    def test_counters(self) -> None:
        """KPI 카운터가 이벤트 종류별로 집계된다."""
        log = AuditLog(clock=FixedClock(step_seconds=1))
        spans = detect_pii(SAMPLE_DOC)
        log.record(caller="a", text=SAMPLE_DOC, spans=spans, event="egress")
        log.record(caller="a", text="안전한 문자열", spans=(), event="llm_call")
        log.record(
            caller="a", text=SAMPLE_DOC, spans=spans, blocked=True, event="egress"
        )
        counters = log.counters()
        assert counters == {
            "llm_calls": 1,
            "pii_detected": 8,
            "blocked": 1,
            "pii_leaked": 0,
            "pii_residual_unknown": 0,
        }

    def test_leak_counter_reflects_residual(self) -> None:
        """잔존 개인정보가 기록되면 KPI 지표가 즉시 드러난다."""
        log = AuditLog(clock=FixedClock())
        log.record(
            caller="a",
            text=SAMPLE_DOC,
            spans=(PiiSpan(start=0, end=3, pii_type="rrn", raw_len=3),),
            blocked=True,
            residual_count=1,
        )
        assert log.counters()["pii_leaked"] == 1

    def test_jsonl_file_contains_no_raw_pii(self, tmp_path: Path) -> None:
        """감사 로그 파일 내용에 원문 개인정보가 없다(요구사항 5)."""
        path = tmp_path / "audit" / "egress.jsonl"
        gate = LlmEgressGate(audit=AuditLog(path, clock=FixedClock(step_seconds=1)))
        client = GatedLlmClient(EchoClient(), gate=gate)
        client.complete(system="문서를 설명하라", user=SAMPLE_DOC)

        raw = path.read_text(encoding="utf-8")
        assert raw.strip(), "감사 로그가 비어 있습니다."
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in raw, f"감사 로그 원문 유출: {fragment}"

    def test_jsonl_records_are_readable(self, tmp_path: Path) -> None:
        """파일에서 레코드를 복원할 수 있고 메모리 기록과 일치한다."""
        path = tmp_path / "egress.jsonl"
        log = AuditLog(path, clock=FixedClock(step_seconds=1))
        log.record(caller="a", text=SAMPLE_DOC, spans=detect_pii(SAMPLE_DOC))
        log.record(caller="b", text="안전", spans=())
        assert log.read_file_records() == log.records()

    def test_jsonl_is_appended_not_overwritten(self, tmp_path: Path) -> None:
        """기록은 append 이므로 이전 줄이 사라지지 않는다."""
        path = tmp_path / "egress.jsonl"
        first = AuditLog(path, clock=FixedClock())
        first.record(caller="a", text="첫 줄", spans=())
        second = AuditLog(path, clock=FixedClock())
        second.record(caller="b", text="둘째 줄", spans=())
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_korean_is_not_escaped(self, tmp_path: Path) -> None:
        """한국어 호출자 이름이 이스케이프 없이 그대로 읽힌다."""
        path = tmp_path / "egress.jsonl"
        log = AuditLog(path, clock=FixedClock())
        log.record(caller="설명단계", text="안전", spans=())
        assert "설명단계" in path.read_text(encoding="utf-8")

    def test_corrupted_line_raises(self, tmp_path: Path) -> None:
        """손상된 줄을 조용히 건너뛰지 않는다."""
        path = tmp_path / "egress.jsonl"
        log = AuditLog(path, clock=FixedClock())
        log.record(caller="a", text="안전", spans=())
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{망가진 줄\n")
        with pytest.raises(ValueError, match="JSON"):
            log.read_file_records()

    def test_memory_only_log_has_no_path(self) -> None:
        """경로를 주지 않으면 파일을 만들지 않는다."""
        log = AuditLog()
        log.record(caller="a", text="안전", spans=())
        assert log.path is None
        assert log.read_file_records() == ()

    def test_repr_does_not_leak(self) -> None:
        """repr 이 레코드 수와 대상만 노출한다."""
        log = AuditLog()
        log.record(caller="a", text=SAMPLE_DOC, spans=detect_pii(SAMPLE_DOC))
        assert "900101" not in repr(log)
        assert "레코드 1건" in repr(log)


class TestBlockedCallIsAudited:
    """차단 사건도 반드시 기록된다."""

    def test_blocked_egress_leaves_a_record(self, tmp_path: Path) -> None:
        """마스킹 실패로 차단된 전송이 감사 로그에 남는다."""

        class LyingDetector:
            """마스킹 이후에도 개인정보를 계속 보고하는 탐지기."""

            def detect(self, text: str) -> tuple[PiiSpan, ...]:
                return (PiiSpan(start=0, end=1, pii_type="rrn", raw_len=1),)

        path = tmp_path / "egress.jsonl"
        gate = LlmEgressGate(
            detector=LyingDetector(),
            audit=AuditLog(path, clock=FixedClock(step_seconds=1)),
        )
        with pytest.raises(PiiEgressBlocked):
            gate.prepare("아무 문자열", caller="테스트")

        records = gate.audit.read_file_records()
        assert len(records) == 1
        assert records[0].blocked is True
        assert records[0].residual_count == 1
        assert gate.audit.counters()["pii_leaked"] == 1

    def test_detector_failure_leaves_a_record(self) -> None:
        """탐지기 실패로 차단된 경우에도 무기록으로 넘어가지 않는다."""

        class ExplodingDetector:
            """항상 실패하는 탐지기."""

            def detect(self, text: str) -> tuple[PiiSpan, ...]:
                raise RuntimeError("탐지기 고장(테스트용)")

        gate = LlmEgressGate(
            detector=ExplodingDetector(), audit=AuditLog(clock=FixedClock())
        )
        with pytest.raises(PiiEgressBlocked):
            gate.prepare(SAMPLE_DOC, caller="테스트")
        assert gate.audit.counters()["blocked"] == 1

    def test_audit_json_is_valid_jsonl(self, tmp_path: Path) -> None:
        """모든 줄이 개별 JSON 객체다."""
        path = tmp_path / "egress.jsonl"
        gate = LlmEgressGate(audit=AuditLog(path, clock=FixedClock(step_seconds=1)))
        GatedLlmClient(EchoClient(), gate=gate).complete("시스템", SAMPLE_DOC)
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            assert isinstance(json.loads(line), dict)
