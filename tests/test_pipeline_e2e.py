"""통합 E2E 테스트 — 합성 문서 한 장으로 파이프라인 전 구간을 관통한다.

이 파일은 "모듈이 각자 잘 돈다"가 아니라 **"연결이 실제로 성립한다"** 를
검증한다. 그래서 Mock 구조를 새로 만들지 않고, 데모가 쓰는 것과 같은 경로
(:func:`docagent.demo.build_demo_session` → :func:`docagent.demo.run_demo`)로
정합 → 탐지 → OCR → 구조화 → PII → RAG → 대화 → 이동 → 검증을 그대로 태운다.

핵심 단언
---------
1. 필수 항목이 전부 완료된다(사람 지원 없이).
2. 체크·서명 검증이 실제 이미지 비교로 ``written=True`` 가 된다.
3. 펜에 전달된 좌표가 정답(:attr:`SyntheticForm.truth`) 대비 3mm 이내다.
4. ``audit.counters()["pii_leaked"] == 0``.
5. 모든 설명 응답에 "이 설명은 원문을 대신하지 않습니다" 고지가 붙는다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from docagent.agent.state import SessionPhase
from docagent.agent.tools import ORIGINAL_NOTICE
from docagent.config import DocAgentConfig
from docagent.contracts import (
    VISION_TRUST_THRESHOLD,
    BoxMm,
    FieldType,
    Point,
    Sensitivity,
)
from docagent.demo import (
    DEMO_SCRIPT,
    build_demo_form,
    build_demo_session,
    draw_handwriting,
    run_demo,
    synthetic_ocr_words,
)
from docagent.errors import AdapterUnavailable, ToolExecutionError, VisionError
from docagent.io.motion import (
    ARRIVED_RESPONSE,
    HOME_RESPONSE,
    MockMotionController,
    SerialMotionController,
)
from docagent.io.speech import ConsoleSpeechIO, GoogleSpeechIO, ScriptedSpeechIO
from docagent.pii.gate import GatedLlmClient
from docagent.pipeline import ImageVerifier, align_to_page, build_session
from docagent.testing.synthetic import (
    APPLICANT_SIGNATURE_FIELD_ID,
    CONSENT_FIELD_ID,
    DATE_FIELD_ID,
    SyntheticForm,
)
from docagent.vision.ocr import StubOcr

#: 펜 안내 좌표 허용 오차(mm). 체크칸이 6mm 안팎이므로 3mm 를 넘으면 칸 밖이다.
POSITION_TOLERANCE_MM = 3.0


# --------------------------------------------------------------------------
# 픽스처
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo_form() -> SyntheticForm:
    """데모와 같은 합성 신청서(렌더링 1회 캐시)."""
    return build_demo_form()


@pytest.fixture(scope="module")
def demo_run() -> Any:
    """Step 1~7 을 끝까지 돌린 데모 결과(모듈 scope 캐시)."""
    return run_demo(quiet=True)


# --------------------------------------------------------------------------
# 조립
# --------------------------------------------------------------------------


class TestPipelineAssembly:
    """:func:`docagent.pipeline.build_session` 이 6개 모듈을 계약대로 잇는가."""

    def test_structure_is_built_from_real_vision_chain(
        self, demo_form: SyntheticForm
    ) -> None:
        """정합 → 탐지 → OCR → 구조화가 실제로 이어져 항목이 나온다."""
        session = build_demo_session(demo_form)
        assert session.structure.doc_title
        assert session.structure.fields
        types = {item.type for item in session.structure.fields}
        assert FieldType.CHOICE in types
        assert FieldType.SIGNATURE in types

    def test_every_field_confidence_is_trustworthy(
        self, demo_form: SyntheticForm
    ) -> None:
        """모든 항목이 인식 신뢰 임계값을 넘어야 사람 지원 없이 진행된다."""
        session = build_demo_session(demo_form)
        for item in session.structure.fields:
            assert item.confidence >= VISION_TRUST_THRESHOLD, item.id

    def test_stage_reports_cover_every_stage(self, demo_form: SyntheticForm) -> None:
        """단계별 신뢰도 보고가 네 단계 모두를 담는다."""
        session = build_demo_session(demo_form)
        assert [report.stage for report in session.stages] == [
            "normalize",
            "detect",
            "structure",
            "pii",
        ]
        assert 0.0 <= session.document_confidence <= 1.0

    def test_llm_is_always_wrapped_by_the_session_gate(
        self, demo_form: SyntheticForm
    ) -> None:
        """게이트를 우회하는 LLM 경로가 존재하지 않는다."""
        session = build_demo_session(demo_form)
        assert isinstance(session.llm, GatedLlmClient)
        assert session.llm.gate is session.gate

    def test_raw_llm_injection_is_forcibly_gated(
        self, demo_form: SyntheticForm
    ) -> None:
        """날 것의 LLM 을 주입해도 게이트로 강제로 감싼다."""

        class RawLlm:
            """게이트 없이 문자열을 그대로 돌려주는 시험용 클라이언트."""

            def __init__(self) -> None:
                self.seen: list[str] = []

            def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
                self.seen.append(user)
                return "확인했습니다."

        raw = RawLlm()
        session = build_session(
            demo_form.image,
            ocr=StubOcr(synthetic_ocr_words(demo_form)),
            dpi=demo_form.dpi,
            llm=raw,
            config=DocAgentConfig(dpi=demo_form.dpi),
            document_id="gate_check",
        )
        assert isinstance(session.llm, GatedLlmClient)
        assert session.llm.inner is raw
        session.llm.complete("시스템", "주민등록번호는 900101-1234567 입니다.", 128)
        assert raw.seen, "하위 클라이언트가 호출되지 않았습니다."
        assert "900101-1234567" not in raw.seen[0]

    def test_public_payload_has_no_coordinates_or_private_text(
        self, demo_form: SyntheticForm
    ) -> None:
        """공개 payload 에 좌표와 개인정보 영역 문구가 들어가지 않는다."""
        session = build_demo_session(demo_form)
        payload = session.public_payload
        blob = str(payload)
        assert "box_mm" not in blob
        assert "x_mm" not in blob
        for item in session.structure.fields:
            if item.sensitivity is Sensitivity.PRIVATE and item.title:
                assert item.title not in blob, item.id
        assert payload["redacted_field_ids"], "개인정보 강등 항목이 하나도 없습니다."

    def test_missing_ocr_is_reported_not_hidden(
        self, demo_form: SyntheticForm
    ) -> None:
        """OCR 이 없으면 조용히 넘어가지 않고 경고로 알린다."""
        session = build_session(
            demo_form.image,
            dpi=demo_form.dpi,
            config=DocAgentConfig(dpi=demo_form.dpi),
            document_id="no_ocr",
        )
        assert any("OCR" in message for message in session.structure.warnings)

    def test_blank_page_raises_document_not_found(self) -> None:
        """탐지·OCR 이 모두 비면 빈 구조를 돌려주지 않고 실패한다."""
        blank = np.full((1169, 827, 3), 255, dtype=np.uint8)
        with pytest.raises(VisionError):
            build_session(blank, dpi=100, document_id="blank")


# --------------------------------------------------------------------------
# Step 1~7 관통
# --------------------------------------------------------------------------


class TestEndToEndScenario:
    """로드맵 Step 1~7 이 실제 파이프라인에서 끝까지 진행되는가."""

    def test_all_required_fields_completed(self, demo_run: Any) -> None:
        """필수 항목이 전부 완료되고 세션이 완료 단계로 끝난다."""
        state = demo_run.session.agent.state
        done, total, remaining = state.progress()
        assert remaining == [], f"미작성 필수 항목: {remaining}"
        assert done == total
        assert state.phase is SessionPhase.COMPLETED

    def test_no_human_handoff_happened(self, demo_run: Any) -> None:
        """정상 서식에서는 사람 지원 없이 끝난다."""
        assert demo_run.session.agent.state.handoff_reason is None
        assert all(turn.handoff_reason is None for turn in demo_run.turns)

    def test_checkbox_and_signature_are_verified_by_image(
        self, demo_run: Any
    ) -> None:
        """체크·서명 검증이 실제 잉크 비율 비교로 True 가 된다."""
        results = {item.field_id: item for item in demo_run.session.verifier.results}
        assert results, "검증이 한 번도 수행되지 않았습니다."
        checked = [
            item
            for item in results.values()
            if item.written and item.ink_delta > 0.0
        ]
        assert len(checked) >= 3, "체크·날짜·서명 세 건이 확인되어야 합니다."
        for item in results.values():
            assert item.written is True, f"{item.field_id} 기입이 확인되지 않았습니다."
            assert item.confidence >= 0.7

    def test_pen_targets_match_truth_within_tolerance(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """펜에 전달된 좌표가 정답 대비 3mm 이내다."""
        moves = demo_run.session.motion.moves
        assert len(moves) >= 3, "이동이 세 번(체크·날짜·서명) 일어나야 합니다."

        truth_points: list[Point] = []
        for item in demo_form.truth.fields:
            for option in item.options:
                truth_points.append(option.box_mm.center())
            if item.box_mm is not None and not item.options:
                truth_points.append(item.box_mm.center())

        for move in moves:
            nearest = min(
                math.hypot(move.x_mm - p.x_mm, move.y_mm - p.y_mm)
                for p in truth_points
            )
            assert nearest <= POSITION_TOLERANCE_MM, (
                f"펜 좌표 ({move.x_mm:.2f}, {move.y_mm:.2f})mm 가 "
                f"정답에서 {nearest:.2f}mm 떨어져 있습니다."
            )

    def test_agree_option_box_is_the_move_target(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """'동의할게' 는 반드시 '동의함' 네모 칸으로 펜을 옮긴다."""
        agree = demo_form.option_box("동의함").center()
        first = demo_run.session.motion.moves[0]
        assert math.hypot(first.x_mm - agree.x_mm, first.y_mm - agree.y_mm) <= (
            POSITION_TOLERANCE_MM
        )

    def test_no_pii_left_the_gate(self, demo_run: Any) -> None:
        """안전 KPI — 마스킹 후에도 남아 나간 개인정보가 0 건이다."""
        counters = demo_run.session.audit.counters()
        assert counters["pii_leaked"] == 0
        assert counters["llm_calls"] >= 1, "설명 경로가 LLM 을 한 번도 타지 않았습니다."

    def test_every_explanation_carries_the_original_notice(
        self, demo_run: Any
    ) -> None:
        """설명 응답에는 예외 없이 '원문을 대신하지 않습니다' 고지가 붙는다."""
        explanations = [
            turn for turn in demo_run.turns if turn.intent in ("explain", "question")
        ]
        assert explanations, "설명 턴이 하나도 없습니다."
        for turn in explanations:
            assert ORIGINAL_NOTICE in turn.speech, turn.speech[:80]

    def test_three_modes_are_offered_on_every_announcement(
        self, demo_run: Any
    ) -> None:
        """항목 안내에는 세 가지 모드가 항상 제시된다."""
        announcements = [
            turn.speech for turn in demo_run.turns if "번째 항목" in turn.speech
        ]
        assert announcements
        for speech in announcements:
            for mode in ("원문 듣기", "쉬운 설명", "다음 항목"):
                assert mode in speech

    def test_no_private_value_is_spoken(self, demo_run: Any) -> None:
        """합성 문서에 인쇄된 예시 개인정보가 발화에 섞이지 않는다."""
        joined = " ".join(turn.speech for turn in demo_run.turns)
        for secret in ("홍길동", "900101-1234567", "010-1234-5678"):
            assert secret not in joined

    def test_run_is_deterministic(self) -> None:
        """같은 입력이면 발화열이 완전히 같다."""
        first = run_demo(quiet=True)
        second = run_demo(quiet=True)
        assert [turn.speech for turn in first.turns] == [
            turn.speech for turn in second.turns
        ]
        assert first.session.summary()["fields_completed"] == (
            second.session.summary()["fields_completed"]
        )

    def test_session_can_be_saved_and_restored(self, demo_run: Any) -> None:
        """세션 상태가 JSON 왕복으로 손실 없이 복원된다."""
        from docagent.agent.state import SessionState

        payload = demo_run.session.session_json()
        restored = SessionState.from_json(payload)
        assert restored == demo_run.session.agent.state

    def test_demo_result_reports_success(self, demo_run: Any) -> None:
        """데모 결과가 성공으로 판정된다(종료 코드 0 의 근거)."""
        assert demo_run.ok is True
        assert demo_run.kpi["metrics"]["pii_llm_transmissions"] == 0


# --------------------------------------------------------------------------
# Verify 어댑터
# --------------------------------------------------------------------------


class TestImageVerifier:
    """이미지 비교 검증기가 "쓰지 않았는데 확인" 을 낼 수 없는가."""

    def test_unwritten_field_is_not_verified(self, demo_form: SyntheticForm) -> None:
        """기입 후 이미지를 갱신하기 전에는 어떤 항목도 확인되지 않는다."""
        session = build_demo_session(demo_form)
        for item in session.structure.fields:
            result = session.verifier.verify(item.id)
            assert result.written is False, item.id

    def test_frame_mismatch_is_rejected(self, demo_form: SyntheticForm) -> None:
        """크기가 다른 이미지는 조용히 받지 않고 예외를 던진다."""
        session = build_demo_session(demo_form)
        smaller = np.full((100, 100, 3), 255, dtype=np.uint8)
        with pytest.raises(VisionError):
            session.verifier.update(smaller)

    def test_align_to_page_keeps_the_frame(self, demo_form: SyntheticForm) -> None:
        """``align_to_page`` 는 최초 정합과 같은 크기의 프레임을 만든다."""
        session = build_demo_session(demo_form)
        aligned = align_to_page(session.normalized, demo_form.image)
        assert aligned.shape[:2] == session.normalized.image.shape[:2]

    def test_missing_field_raises(self, demo_form: SyntheticForm) -> None:
        """문서에 없는 항목을 검증하면 조용히 False 를 내지 않고 실패한다."""
        session = build_demo_session(demo_form)
        with pytest.raises(VisionError):
            session.verifier.verify("존재하지_않는_항목")

    def test_verifier_satisfies_the_tool_contract(
        self, demo_form: SyntheticForm
    ) -> None:
        """도구가 요구하는 ``verify(field_id)`` 계약을 만족한다."""
        session = build_demo_session(demo_form)
        assert isinstance(session.verifier, ImageVerifier)
        assert callable(session.verifier.verify)


# --------------------------------------------------------------------------
# io 어댑터
# --------------------------------------------------------------------------


class TestSpeechAdapters:
    """:mod:`docagent.io.speech` 구현이 프로토콜을 만족하는가."""

    def test_scripted_speech_collects_and_replays(self) -> None:
        """발화는 모으고 대사는 순서대로 돌려준다."""
        io = ScriptedSpeechIO(["첫 번째", "두 번째"])
        io.speak("안내드립니다.")
        assert io.listen() == "첫 번째"
        assert io.listen() == "두 번째"
        assert io.listen() == ""
        assert io.exhausted is True
        assert io.transcript() == ("안내드립니다.",)

    def test_scripted_speech_rejects_non_string(self) -> None:
        """문자열이 아닌 스크립트는 조용히 무시하지 않고 거부한다."""
        with pytest.raises(ValueError):
            ScriptedSpeechIO([1])  # type: ignore[list-item]

    def test_console_speech_writes_tagged_lines(self) -> None:
        """콘솔 구현은 태그를 붙여 출력하고 스크립트를 되돌려준다."""
        import io as _io

        buffer = _io.StringIO()
        speech = ConsoleSpeechIO(["동의할게"], stream=buffer)
        speech.speak("안녕하세요.")
        assert speech.listen() == "동의할게"
        text = buffer.getvalue()
        assert "[에이전트] 안녕하세요." in text
        assert "[사용자] 동의할게" in text

    def test_console_speech_reads_stdin_when_no_script(self) -> None:
        """스크립트가 없으면 주입된 입력 함수를 쓴다."""
        import io as _io

        answers = iter(["네", ""])
        speech = ConsoleSpeechIO(
            None, stream=_io.StringIO(), input_fn=lambda: next(answers)
        )
        assert speech.listen() == "네"
        assert speech.listen() == ""

    def test_google_speech_is_unavailable_without_package(self) -> None:
        """선택적 패키지가 없으면 한국어 설치 안내와 함께 실패한다."""
        with pytest.raises(AdapterUnavailable) as info:
            GoogleSpeechIO()
        message = str(info.value)
        assert "google-cloud-speech" in message
        assert ".venv" in message

    def test_google_speech_module_import_does_not_fail(self) -> None:
        """모듈 import 와 상수 접근만으로는 절대 실패하지 않는다."""
        from docagent.io import speech as speech_module

        assert speech_module.AGENT_TAG == "[에이전트]"
        assert speech_module.GoogleSpeechIO.PACKAGES


class TestMotionAdapters:
    """:mod:`docagent.io.motion` 구현이 가동 범위·프로토콜을 지키는가."""

    def test_mock_records_moves_and_rejects_out_of_range(self) -> None:
        """가동 범위를 벗어난 좌표는 이동하지 않고 False 를 돌려준다."""
        pen = MockMotionController()
        assert pen.move_to(20.0, 30.0) is True
        assert pen.position() == Point(20.0, 30.0)
        assert pen.move_to(-1.0, 30.0) is False
        assert pen.move_to(20.0, 400.0) is False
        assert len(pen.moves) == 1
        assert len(pen.rejected) == 2
        assert pen.home() is True
        assert pen.position() == Point(0.0, 0.0)

    def test_mock_error_injection_is_deterministic(self) -> None:
        """주입한 오차는 난수가 아니라 고정값이다."""
        pen = MockMotionController(error_mm=(0.4, -0.3))
        pen.move_to(50.0, 60.0)
        assert pen.position() == Point(50.4, 59.7)
        again = MockMotionController(error_mm=(0.4, -0.3))
        again.move_to(50.0, 60.0)
        assert again.position() == pen.position()

    def test_mock_fail_targets_simulate_hardware_failure(self) -> None:
        """지정한 좌표에서만 이동이 실패한다(재시도 경로 시험용)."""
        pen = MockMotionController(fail_targets=[(10.0, 10.0)])
        assert pen.move_to(10.0, 10.0) is False
        assert pen.move_to(10.0, 20.0) is True

    def test_serial_speaks_the_roadmap_protocol(self) -> None:
        """``MOVE x y`` → ``ARRIVED`` / ``HOME`` → ``HOME_REACHED`` 규약을 지킨다."""

        class FakePort:
            """정상 응답만 돌려주는 시험용 포트."""

            def __init__(self) -> None:
                self.written: list[bytes] = []
                self.closed = False

            def write(self, payload: bytes) -> None:
                self.written.append(payload)

            def readline(self) -> bytes:
                last = self.written[-1].decode("utf-8")
                return (
                    f"{HOME_RESPONSE}\n" if last.startswith("HOME") else f"{ARRIVED_RESPONSE}\n"
                ).encode("utf-8")

            def close(self) -> None:
                self.closed = True

        port = FakePort()
        with SerialMotionController("COM_TEST", transport=port) as pen:
            assert pen.move_to(22.5, 102.5) is True
            assert pen.position() == Point(22.5, 102.5)
        assert port.written[0] == b"MOVE 22.5 102.5\n"
        assert port.written[-1] == b"HOME\n"
        assert port.closed is True

    def test_serial_retries_then_fails_loudly(self) -> None:
        """예상 밖 응답은 재시도 후 조용히 성공 처리하지 않고 예외로 끝난다."""

        class BadPort:
            """항상 엉뚱한 응답을 돌려주는 포트."""

            def __init__(self) -> None:
                self.calls = 0

            def write(self, payload: bytes) -> None:
                self.calls += 1

            def readline(self) -> bytes:
                return b"ERR 3\n"

            def close(self) -> None:
                return None

        port = BadPort()
        pen = SerialMotionController("COM_TEST", retries=2, transport=port)
        with pytest.raises(ToolExecutionError) as info:
            pen.move_to(10.0, 10.0)
        assert port.calls == 3
        assert "펜 이동" in str(info.value)

    def test_serial_does_not_send_out_of_range_commands(self) -> None:
        """가동 범위를 벗어난 명령은 장치로 내보내지 않는다."""

        class CountingPort:
            """전송 횟수만 세는 포트."""

            def __init__(self) -> None:
                self.calls = 0

            def write(self, payload: bytes) -> None:
                self.calls += 1

            def readline(self) -> bytes:
                return f"{ARRIVED_RESPONSE}\n".encode("utf-8")

            def close(self) -> None:
                return None

        port = CountingPort()
        pen = SerialMotionController("COM_TEST", transport=port)
        assert pen.move_to(500.0, 10.0) is False
        assert port.calls == 0

    def test_serial_uses_calibration_for_machine_coordinates(self) -> None:
        """보정이 있으면 기기 좌표로 바꿔 전송하고, 위치는 문서 좌표로 돌려준다."""
        from docagent.vision.geometry import MachineCalibration

        class EchoPort:
            """마지막 명령을 기억하는 포트."""

            def __init__(self) -> None:
                self.last = b""

            def write(self, payload: bytes) -> None:
                self.last = payload

            def readline(self) -> bytes:
                return f"{ARRIVED_RESPONSE}\n".encode("utf-8")

            def close(self) -> None:
                return None

        port = EchoPort()
        pen = SerialMotionController(
            "COM_TEST",
            transport=port,
            calibration=MachineCalibration(origin_offset_mm=Point(5.0, 7.0)),
        )
        assert pen.move_to(20.0, 30.0) is True
        assert port.last == b"MOVE 25.0 37.0\n"
        assert pen.position() == Point(20.0, 30.0)

    def test_serial_requires_pyserial_without_transport(self) -> None:
        """``transport`` 없이 만들면 pyserial 부재를 한국어로 알린다."""
        with pytest.raises(AdapterUnavailable) as info:
            SerialMotionController("COM_TEST")
        assert "pyserial" in str(info.value)


# --------------------------------------------------------------------------
# 보조 유틸
# --------------------------------------------------------------------------


class TestWritingSimulation:
    """기입 시뮬레이션이 결정론적이고 실제로 잉크를 늘리는가."""

    def test_handwriting_is_deterministic(self, demo_form: SyntheticForm) -> None:
        """같은 인자로 두 번 그리면 픽셀까지 같다."""
        box = BoxMm(50.0, 60.0, 40.0, 10.0)
        first = draw_handwriting(demo_form.image, box, demo_form.px_per_mm)
        second = draw_handwriting(demo_form.image, box, demo_form.px_per_mm)
        assert np.array_equal(first, second)

    def test_handwriting_does_not_mutate_the_source(
        self, demo_form: SyntheticForm
    ) -> None:
        """원본 이미지를 바꾸지 않는다(픽스처 오염 금지)."""
        before = demo_form.image.copy()
        draw_handwriting(demo_form.image, BoxMm(50.0, 60.0, 40.0, 10.0), demo_form.px_per_mm)
        assert np.array_equal(demo_form.image, before)

    def test_handwriting_rejects_bad_arguments(
        self, demo_form: SyntheticForm
    ) -> None:
        """잘못된 인자는 조용히 넘기지 않는다."""
        box = BoxMm(50.0, 60.0, 40.0, 10.0)
        with pytest.raises(ValueError):
            draw_handwriting(demo_form.image, box, 0.0)
        with pytest.raises(ValueError):
            draw_handwriting(demo_form.image, box, 5.0, strokes=0)


class TestSyntheticOcrStandIn:
    """오프라인 OCR 대역이 문서의 핵심 문자열을 담는가."""

    def test_words_contain_labels_and_clause(self, demo_form: SyntheticForm) -> None:
        """항목 라벨·선택지·약관 어휘가 모두 들어 있다."""
        texts = " ".join(word.text for word in synthetic_ocr_words(demo_form))
        for needle in ("신청일자", "신청인", "동의함", "개인정보"):
            assert needle in texts

    def test_words_are_inside_the_page(self, demo_form: SyntheticForm) -> None:
        """모든 단어 좌표가 A4 페이지 안에 있다."""
        for word in synthetic_ocr_words(demo_form):
            assert 0.0 <= word.box_mm.x_mm <= 210.0
            assert 0.0 <= word.box_mm.y_mm <= 297.0


class TestKnownFieldIdentities:
    """구조화 결과가 정답의 핵심 항목을 실제로 잡아내는가."""

    def test_consent_choice_is_public_with_clause(
        self, demo_form: SyntheticForm
    ) -> None:
        """동의 항목은 공개 영역이고 약관 원문을 갖는다."""
        session = build_demo_session(demo_form)
        consent = [
            item for item in session.structure.fields if item.type is FieldType.CHOICE
        ]
        assert len(consent) == 1
        assert consent[0].sensitivity is Sensitivity.PUBLIC
        assert "개인정보" in consent[0].clause_text
        assert {option.label for option in consent[0].options} == {
            "동의함",
            "동의하지 않음",
        }

    def test_date_line_is_not_mistaken_for_a_signature(
        self, demo_form: SyntheticForm
    ) -> None:
        """신청일자 기입선이 서명란으로 분류되지 않는다(연결 시 발견된 오분류)."""
        session = build_demo_session(demo_form)
        dates = [
            item for item in session.structure.fields if item.type is FieldType.DATE
        ]
        signatures = [
            item
            for item in session.structure.fields
            if item.type is FieldType.SIGNATURE
        ]
        assert len(dates) == 1
        assert "신청일자" in dates[0].title
        assert len(signatures) == 1
        assert "신청인" in signatures[0].title

    def test_truth_ids_exist_in_the_reference_structure(
        self, demo_form: SyntheticForm
    ) -> None:
        """정답 구조에는 비교 기준이 되는 항목들이 그대로 있다."""
        for field_id in (
            CONSENT_FIELD_ID,
            DATE_FIELD_ID,
            APPLICANT_SIGNATURE_FIELD_ID,
        ):
            assert demo_form.truth.field_by_id(field_id) is not None


class TestDemoScript:
    """데모 스크립트가 Step 1~7 을 빠짐없이 태우는가."""

    def test_turn_count_matches_the_script(self, demo_run: Any) -> None:
        """시작 안내 1턴 + 스크립트 줄 수만큼의 턴이 생긴다."""
        assert len(demo_run.turns) == len(DEMO_SCRIPT) + 1

    def test_script_covers_every_step(self, demo_run: Any) -> None:
        """설명 · 원문 · 선택 · 이동 · 검증 · 완료가 모두 등장한다."""
        intents = [turn.intent for turn in demo_run.turns]
        tools = [call.name for turn in demo_run.turns for call in turn.tool_calls]
        assert "explain" in intents
        assert "read_original" in intents
        assert "select_option" in tools
        assert tools.count("move_to_field") == 3
        assert tools.count("verify_field") == 3
        assert "모든 항목의 안내를 마쳤습니다" in demo_run.turns[-1].speech

    def test_log_uses_korean_tags(self) -> None:
        """콘솔 로그가 약속한 다섯 가지 태그를 쓴다."""
        result = run_demo(quiet=True)
        joined = "\n".join(result.log)
        for tag in ("[사용자]", "[에이전트]", "[하드웨어]", "[검증]", "[시스템]"):
            assert tag in joined

    def test_json_output_has_no_private_values(self) -> None:
        """``--json`` 산출물에 개인정보가 섞이지 않는다."""
        import json

        payload = json.dumps(run_demo(quiet=True).to_dict(), ensure_ascii=False)
        for secret in ("홍길동", "900101-1234567", "010-1234-5678"):
            assert secret not in payload

    def test_main_returns_zero(self) -> None:
        """``python -m docagent.demo --json`` 은 종료 코드 0 으로 끝난다."""
        import contextlib
        import io as _io

        from docagent.demo import main

        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(["--json"])
        assert code == 0
        assert buffer.getvalue().strip().startswith("{")

    def test_kpi_only_prints_table(self) -> None:
        """``--kpi-only`` 는 대화 로그 없이 표만 출력한다."""
        import contextlib
        import io as _io

        from docagent.demo import main

        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert main(["--kpi-only"]) == 0
        text = buffer.getvalue()
        assert "KPI 요약" in text
        assert "[에이전트]" not in text


class TestConfig:
    """:class:`docagent.config.DocAgentConfig` 의 검증·직렬화."""

    def test_defaults_are_offline_and_deterministic(self) -> None:
        """기본값은 네트워크·키 없이 도는 조합이다."""
        cfg = DocAgentConfig()
        assert cfg.llm_kind == "offline"
        assert cfg.detector_prefer == "auto"
        assert cfg.detector_weights is None
        assert cfg.audit_path is None
        assert cfg.strict_pii is True

    def test_corpus_dir_defaults_to_repo_data(self) -> None:
        """기본 코퍼스 디렉터리가 실제로 존재한다."""
        cfg = DocAgentConfig()
        corpus = cfg.resolved_corpus_dir()
        assert corpus.is_dir()
        assert list(corpus.glob("*.md"))

    def test_round_trip(self) -> None:
        """``from_dict(to_dict(x)) == x`` 가 성립한다."""
        from pathlib import Path

        cfg = DocAgentConfig(
            dpi=200,
            corpus_dir=Path("data/corpus"),
            serial_port="COM9",
            explain_top_k=2,
        )
        assert DocAgentConfig.from_dict(cfg.to_dict()) == cfg

    def test_invalid_values_are_rejected(self) -> None:
        """허용 범위를 벗어난 설정은 조용히 보정하지 않고 거부한다."""
        for kwargs in (
            {"dpi": 10},
            {"llm_kind": "mystery"},
            {"detector_prefer": "magic"},
            {"motion_tolerance_mm": 0.0},
            {"serial_retries": -1},
            {"explain_top_k": 0},
            {"page_size_mm": (0.0, 297.0)},
        ):
            with pytest.raises(ValueError):
                DocAgentConfig(**kwargs)  # type: ignore[arg-type]

    def test_from_env_overrides(self) -> None:
        """환경변수는 주입한 매핑에서만 읽는다(전역 환경 오염 없음)."""
        cfg = DocAgentConfig.from_env(
            {"DOCAGENT_DPI": "150", "DOCAGENT_LLM_KIND": "claude", "DOCAGENT_STRICT_PII": "0"}
        )
        assert cfg.dpi == 150
        assert cfg.llm_kind == "claude"
        assert cfg.strict_pii is False

    def test_from_env_rejects_bad_numbers(self) -> None:
        """숫자가 아닌 값은 무시하지 않고 실패한다."""
        with pytest.raises(ValueError):
            DocAgentConfig.from_env({"DOCAGENT_DPI": "매우 큼"})

    def test_thresholds_come_from_contracts(self) -> None:
        """임계값은 계약에서 읽어 오고 여기서 다시 정의하지 않는다."""
        from docagent.config import thresholds

        assert thresholds()["vision_trust"] == VISION_TRUST_THRESHOLD


class TestModuleHygiene:
    """통합 모듈이 프로젝트 규약(선택적 패키지 지연 import, UTF-8)을 지키는가."""

    #: 모듈 최상단에서 import 하면 안 되는 선택적 패키지.
    FORBIDDEN = (
        "ultralytics",
        "torch",
        "anthropic",
        "openai",
        "langchain",
        "faiss",
        "presidio_analyzer",
        "pydantic",
        "serial",
        "pytesseract",
        "google",
    )

    @staticmethod
    def _source_files() -> list[Path]:
        """src/docagent 아래 모든 .py 파일 목록.

        하드코딩한 모듈 목록 대신 패키지 전체를 순회하므로,
        새로 추가되는 모듈도 자동으로 이 회귀 가드의 보호를 받는다.
        """
        import docagent

        root = Path(docagent.__file__).resolve().parent
        files = sorted(root.rglob("*.py"))
        assert len(files) >= 20, f"소스 탐색 실패: {len(files)}개만 찾았습니다."
        return files

    def test_no_optional_package_at_module_level(self) -> None:
        """선택적 패키지가 어떤 모듈의 최상단 import 에도 없어야 한다."""
        import ast

        for path in self._source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            top: list[str] = []
            for node in tree.body:
                if isinstance(node, ast.Import):
                    top += [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top.append(node.module.split(".")[0])
            for banned in self.FORBIDDEN:
                assert banned not in top, (
                    f"{path.name} 최상단에 {banned} import 가 있습니다."
                )

    def test_sources_are_utf8(self) -> None:
        """모든 소스가 UTF-8 로 읽힌다."""
        for path in self._source_files():
            path.read_text(encoding="utf-8")


class TestSafetyScene:
    """자격 판단 질문은 반드시 사람에게 넘어가는가."""

    def test_eligibility_question_is_handed_off(self, demo_form: SyntheticForm) -> None:
        """'제가 받을 수 있나요' 는 가드레일에 걸려 직원 연결로 끝난다."""
        from docagent.demo import SAFETY_QUESTION, run_safety_scene

        session = run_safety_scene(demo_form)
        turn = session.turns[-1]
        assert turn.user_text == SAFETY_QUESTION
        assert turn.handoff_reason is not None
        assert session.agent.state.phase is SessionPhase.HUMAN_HANDOFF
        assert "자격" in turn.handoff_reason

    def test_handoff_does_not_move_the_pen(self, demo_form: SyntheticForm) -> None:
        """차단된 질문에서는 펜이 움직이지 않는다."""
        from docagent.demo import run_safety_scene

        session = run_safety_scene(demo_form)
        assert session.motion.moves == []

    def test_demo_exposes_the_safety_session_separately(self, demo_run: Any) -> None:
        """안전 시나리오는 본 세션 대화에 섞이지 않는다."""
        assert demo_run.safety is not None
        assert demo_run.safety is not demo_run.session
        assert all(turn.handoff_reason is None for turn in demo_run.turns)
        assert demo_run.safety.agent.state.handoff_reason is not None

    def test_handoff_accuracy_is_measured(self, demo_run: Any) -> None:
        """안전 시나리오 덕분에 직원 연결 정확도가 '측정 불가' 가 아니다."""
        accuracy = demo_run.kpi["metrics"]["handoff_accuracy"]
        assert accuracy is not None
        assert accuracy == 1.0

    def test_probe_is_excluded_from_completion_denominator(
        self, demo_run: Any
    ) -> None:
        """핸드오프 검증 세션은 완료율 분모에서 빠진다."""
        assert demo_run.kpi["counts"]["probe_sessions"] == 1
        assert demo_run.kpi["counts"]["sessions"] == 1
        assert demo_run.kpi["metrics"]["independent_completion_rate"] == 1.0


class TestExplanationComposition:
    """설명 발화가 같은 문장을 두 번 낭독하지 않는가(통합 시 발견한 중복)."""

    def test_disclaimer_is_not_repeated(self, demo_run: Any) -> None:
        """설명기가 본문에 이미 넣은 고지를 도구가 또 붙이지 않는다."""
        from docagent.agent.guardrails import DISCLAIMER

        explanations = [
            turn.speech for turn in demo_run.turns if turn.intent == "explain"
        ]
        assert explanations
        for speech in explanations:
            assert speech.count(DISCLAIMER) == 1, speech

    def test_disclaimer_is_appended_when_missing(self) -> None:
        """본문에 고지가 없는 설명기에는 도구가 정상적으로 고지를 붙인다."""
        from docagent.agent.state import SessionState
        from docagent.agent.tools import ToolRegistry
        from docagent.contracts import DocumentStructure, Field

        class Explainer:
            """고지를 본문에 넣지 않는 설명기."""

            class Result:
                text = "쉬운 말 설명입니다."
                confidence = 0.95
                sources: tuple[str, ...] = ()
                needs_handoff = False
                disclaimer = "따로 붙는 고지입니다."

            def explain(self, field: Any, question: str = "") -> "Explainer.Result":
                return self.Result()

        structure = DocumentStructure(
            document_id="dup_check",
            fields=(
                Field(
                    id="f1",
                    type=FieldType.CHOICE,
                    title="항목",
                    sensitivity=Sensitivity.PUBLIC,
                    clause_text="약관 문구",
                    confidence=1.0,
                ),
            ),
        )
        state = SessionState.from_structure(structure)
        state.set_current("f1")
        tools = ToolRegistry(structure, state, explainer=Explainer())
        speech = tools.call("explain", {"field_id": "f1"}).speech
        assert speech.count("따로 붙는 고지입니다.") == 1
        assert ORIGINAL_NOTICE in speech
