"""KPI 측정(:mod:`docagent.kpi`) 테스트.

지표를 "계산되는지"만 보는 것이 아니라, **틀린 결과를 넣었을 때 지표가 실제로
나빠지는지**까지 확인한다. 그래야 KPI 가 장식이 아니라 회귀 감지기가 된다.
"""

from __future__ import annotations

from typing import Any

import pytest

from docagent.contracts import BoxMm, DocumentStructure, Field, FieldType, Option
from docagent.demo import build_demo_form, run_demo
from docagent.kpi import (
    METRIC_LABELS,
    POSITION_TOLERANCE_MM,
    TARGETS,
    KpiCase,
    evaluate,
    meets_target,
    render_table,
    target_for,
)
from docagent.testing.synthetic import SyntheticForm


@pytest.fixture(scope="module")
def demo_form() -> SyntheticForm:
    """데모와 같은 합성 신청서."""
    return build_demo_form()


@pytest.fixture(scope="module")
def demo_run() -> Any:
    """Step 1~7 을 끝까지 돌린 데모 결과."""
    return run_demo(quiet=True)


def _shift(structure: DocumentStructure, dx_mm: float) -> DocumentStructure:
    """모든 좌표를 x 방향으로 옮긴 '틀린 정답'을 만든다.

    :param structure: 원본 구조.
    :param dx_mm: 이동량(mm).
    :returns: 좌표가 옮겨진 새 구조.
    """

    def move(box: BoxMm) -> BoxMm:
        return BoxMm(box.x_mm + dx_mm, box.y_mm, box.w_mm, box.h_mm)

    return DocumentStructure(
        document_id=structure.document_id,
        doc_title=structure.doc_title,
        fields=tuple(
            Field(
                id=item.id,
                type=item.type,
                title=item.title,
                role=item.role,
                options=tuple(
                    Option(label=o.label, box_mm=move(o.box_mm), checked=o.checked)
                    for o in item.options
                ),
                required=item.required,
                sensitivity=item.sensitivity,
                box_mm=None if item.box_mm is None else move(item.box_mm),
                clause_text=item.clause_text,
                order=item.order,
                confidence=item.confidence,
            )
            for item in structure.fields
        ),
        page_size_mm=structure.page_size_mm,
        source_image=structure.source_image,
        warnings=structure.warnings,
    )


class TestEvaluate:
    """:func:`docagent.kpi.evaluate` 의 계산."""

    def test_demo_session_meets_every_measured_target(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """데모 세션은 측정된 모든 지표에서 목표를 만족한다."""
        report = evaluate([KpiCase(demo_run.session, demo_form.truth, name="demo")])
        for name in METRIC_LABELS:
            verdict = meets_target(name, report["metrics"][name])
            assert verdict is not False, f"{name} 지표가 목표에 미달했습니다."

    def test_report_shape(self, demo_run: Any, demo_form: SyntheticForm) -> None:
        """리포트가 약속한 키를 모두 갖는다."""
        report = evaluate([KpiCase(demo_run.session, demo_form.truth)])
        assert set(report) == {
            "cases",
            "vision_measured",
            "metrics",
            "targets",
            "counts",
        }
        assert set(report["metrics"]) == set(METRIC_LABELS)
        assert report["vision_measured"] is True
        assert report["cases"] == ["demo_form"]

    def test_safety_metric_is_zero(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """개인정보 LLM 전송 건수는 반드시 0 이다."""
        report = evaluate([KpiCase(demo_run.session, demo_form.truth)])
        assert report["metrics"]["pii_llm_transmissions"] == 0
        assert meets_target("pii_llm_transmissions", 0.0) is True
        assert meets_target("pii_llm_transmissions", 1.0) is False

    def test_vision_metrics_are_none_without_truth(self, demo_run: Any) -> None:
        """정답이 없으면 Vision 지표를 추정하지 않고 측정 불가로 둔다."""
        report = evaluate([demo_run.session])
        assert report["vision_measured"] is False
        assert report["metrics"]["checkbox_recall"] is None
        assert report["metrics"]["signature_recall"] is None
        assert report["metrics"]["misguided_position_rate"] is None
        # Agent·UX 지표는 정답 없이도 측정된다.
        assert report["metrics"]["independent_completion_rate"] == 1.0

    def test_wrong_truth_makes_position_metric_fail(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """정답 좌표를 크게 옮기면 잘못된 위치 안내율이 실제로 올라간다."""
        shifted = _shift(demo_form.truth, POSITION_TOLERANCE_MM + 25.0)
        report = evaluate([KpiCase(demo_run.session, shifted)])
        assert report["metrics"]["misguided_position_rate"] == 1.0
        assert meets_target("misguided_position_rate", 1.0) is False
        assert report["metrics"]["signature_misguide_rate"] == 1.0

    def test_recall_drops_when_truth_has_extra_fields(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """정답에 없던 체크칸을 더하면 재현율이 떨어진다."""
        extra = Field(
            id="extra_checkbox",
            type=FieldType.CHOICE,
            title="추가 동의",
            options=(
                Option(label="예", box_mm=BoxMm(150.0, 250.0, 6.0, 6.0)),
                Option(label="아니오", box_mm=BoxMm(170.0, 250.0, 6.0, 6.0)),
            ),
            order=99,
            confidence=1.0,
        )
        truth = DocumentStructure(
            document_id=demo_form.truth.document_id,
            doc_title=demo_form.truth.doc_title,
            fields=(*demo_form.truth.fields, extra),
            page_size_mm=demo_form.truth.page_size_mm,
        )
        report = evaluate([KpiCase(demo_run.session, truth)])
        assert report["metrics"]["checkbox_recall"] < 1.0

    def test_multiple_sessions_are_aggregated(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """세션이 여러 개면 분자·분모를 합산한다."""
        case = KpiCase(demo_run.session, demo_form.truth)
        one = evaluate([case])
        two = evaluate([case, case])
        assert two["counts"]["sessions"] == 2
        assert two["counts"]["turns"] == one["counts"]["turns"] * 2
        assert two["metrics"]["turns_mean"] == one["metrics"]["turns_mean"]

    def test_tuple_input_is_accepted(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """``(세션, 정답구조)`` 튜플도 그대로 받는다."""
        report = evaluate([(demo_run.session, demo_form.truth)])
        assert report["vision_measured"] is True

    def test_empty_dataset_raises(self) -> None:
        """평가 대상이 없으면 0 으로 채우지 않고 실패한다."""
        with pytest.raises(ValueError):
            evaluate([])

    def test_unknown_input_type_raises(self) -> None:
        """알 수 없는 입력은 조용히 건너뛰지 않는다."""
        with pytest.raises(TypeError):
            evaluate(["세션이 아님"])

    def test_tuple_with_bad_truth_raises(self, demo_run: Any) -> None:
        """튜플 두 번째 값이 구조가 아니면 거부한다."""
        with pytest.raises(TypeError):
            evaluate([(demo_run.session, "정답이 아님")])


class TestTargets:
    """목표치 조회·판정."""

    def test_every_metric_has_a_target(self) -> None:
        """표에 실리는 지표는 전부 목표치를 갖는다."""
        assert set(TARGETS) == set(METRIC_LABELS)

    def test_target_for_returns_operator_and_value(self) -> None:
        """목표치는 ``(연산자, 값)`` 형태다."""
        assert target_for("checkbox_recall") == (">=", 0.95)
        assert target_for("없는지표") is None

    def test_meets_target_boundaries(self) -> None:
        """경계값에서 판정이 흔들리지 않는다."""
        assert meets_target("checkbox_recall", 0.95) is True
        assert meets_target("checkbox_recall", 0.9499) is False
        assert meets_target("misguided_position_rate", 0.02) is True
        assert meets_target("misguided_position_rate", 0.03) is False
        assert meets_target("checkbox_recall", None) is None
        assert meets_target("없는지표", 1.0) is None


class TestRenderTable:
    """한국어 표 렌더링."""

    def test_table_lists_every_metric(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """표에 모든 지표의 한국어 이름이 나온다."""
        table = render_table(evaluate([KpiCase(demo_run.session, demo_form.truth)]))
        assert "KPI 요약" in table
        for _, label in METRIC_LABELS.values():
            assert label in table

    def test_table_marks_achievement(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """목표 달성 여부를 한국어로 표시한다."""
        table = render_table(evaluate([KpiCase(demo_run.session, demo_form.truth)]))
        assert "달성" in table
        assert "측정된 모든 지표가 목표를 만족했습니다." in table

    def test_table_reports_failures(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """미달 지표는 표 아래에 따로 모아 보고한다."""
        shifted = _shift(demo_form.truth, POSITION_TOLERANCE_MM + 25.0)
        table = render_table(evaluate([KpiCase(demo_run.session, shifted)]))
        assert "미달 지표:" in table
        assert "잘못된 위치 안내율" in table

    def test_table_notes_missing_truth(self, demo_run: Any) -> None:
        """정답이 없으면 그 사실을 표에 명시한다."""
        table = render_table(evaluate([demo_run.session]))
        assert "Vision 지표는 측정하지 않았습니다" in table
        assert "측정 불가" in table

    def test_table_rows_are_aligned(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """구분선과 본문 줄이 모두 존재하고 빈 줄이 없다."""
        table = render_table(evaluate([KpiCase(demo_run.session, demo_form.truth)]))
        lines = table.splitlines()
        assert len(lines) >= len(METRIC_LABELS) + 5
        assert all(line.strip() for line in lines)


class TestHandoffProbe:
    """직원 연결 검증 세션이 완료율을 왜곡하지 않는가."""

    def test_probe_is_excluded_from_completion_and_turns(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """검증 세션은 완료율·턴 수 분모에서 빠지고 연결 정확도에는 들어간다."""
        main = KpiCase(demo_run.session, demo_form.truth, name="main")
        probe = KpiCase(
            demo_run.safety, demo_form.truth, name="probe", handoff_probe=True
        )
        with_probe = evaluate([main, probe])
        assert with_probe["counts"]["sessions"] == 1
        assert with_probe["counts"]["probe_sessions"] == 1
        assert with_probe["metrics"]["independent_completion_rate"] == 1.0
        assert with_probe["metrics"]["handoff_accuracy"] == 1.0

    def test_probe_without_flag_lowers_completion(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """플래그를 빼면 실제로 완료율이 내려간다(플래그가 실효적임을 증명)."""
        report = evaluate(
            [
                KpiCase(demo_run.session, demo_form.truth),
                KpiCase(demo_run.safety, demo_form.truth),
            ]
        )
        assert report["metrics"]["independent_completion_rate"] == 0.5

    def test_table_notes_excluded_probes(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """표가 제외 사실을 명시한다(숨기지 않는다)."""
        table = render_table(
            evaluate(
                [
                    KpiCase(demo_run.session, demo_form.truth),
                    KpiCase(demo_run.safety, demo_form.truth, handoff_probe=True),
                ]
            )
        )
        assert "완료율·턴 수 분모에서 제외했습니다" in table

    def test_handoff_reason_must_be_verifiable(
        self, demo_form: SyntheticForm
    ) -> None:
        """확인할 수 없는 사유로 넘긴 연결은 정확도에서 오답으로 센다.

        모듈 scope 픽스처를 건드리지 않도록 **새 세션**을 만들어 사유만 바꿔치기한다.
        """
        from docagent.contracts import AgentTurn
        from docagent.demo import run_safety_scene

        session = run_safety_scene(demo_form)
        session.turns[:] = [
            AgentTurn(
                user_text=turn.user_text,
                intent=turn.intent,
                tool_calls=turn.tool_calls,
                speech=turn.speech,
                confidence=turn.confidence,
                handoff_reason=(
                    "그냥 넘겼습니다." if turn.handoff_reason else None
                ),
            )
            for turn in session.turns
        ]
        report = evaluate([KpiCase(session, demo_form.truth)])
        assert report["counts"]["handoff_total"] == 1
        assert report["metrics"]["handoff_accuracy"] == 0.0


class TestSafetyMetricIsFalsifiable:
    """안전 KPI 가 자기 자신을 반증할 수 있는지 확인한다."""

    def test_unknown_digits_in_prompt_break_the_safety_metric(
        self, demo_form: SyntheticForm
    ) -> None:
        """탐지기가 못 본 개인정보가 나가면 안전 KPI 가 위반으로 드러난다.

        ``pii_leaked`` 는 '탐지된 PII 가 마스킹 후에도 남은 수' 라서, 규칙이 애초에
        보지 못한 유출은 구조적으로 0 이 된다. 탐지기와 독립된 2차 검사
        (``pii_residual_unknown``)가 그 구멍을 메우는지 본다.
        """
        from docagent.demo import run_safety_scene

        session = run_safety_scene(demo_form)
        before = evaluate([KpiCase(session, demo_form.truth)])
        assert before["metrics"]["pii_llm_transmissions"] == 0.0

        # 탐지 규칙이 하나도 걸리지 않는 '유래 불명 숫자열'이 그대로 나간 상황.
        session.audit.record(
            caller="test",
            text="유출 재현",
            spans=(),
            blocked=False,
            residual_count=0,
            event="llm_call",
            egress_text="확인 번호 12345678901 입니다",
        )
        after = evaluate([KpiCase(session, demo_form.truth)])
        assert after["counts"]["pii_leaked"] == 0.0
        assert after["metrics"]["pii_llm_transmissions"] == 1.0
        assert meets_target("pii_llm_transmissions", 1.0) is False


class TestQuestionResolutionDenominator:
    """정당한 직원 연결이 질문 해결률을 깎지 않는지 확인한다."""

    def test_legal_judgment_handoff_is_excluded(
        self, demo_form: SyntheticForm
    ) -> None:
        """법적·자격 판단으로 넘긴 질문 턴은 분모에 들어가지 않는다.

        오케스트레이터는 ``Intent.QUESTION`` + ``legal_judgment`` 턴에서 explain
        도구를 부르지 않고 곧바로 사람에게 넘긴다. 그 턴의 intent 는 ``"question"``
        이라 분모에는 들어가지만 ``state.history`` 에 ``explain`` 이 남지 않아
        분자에는 절대 들어갈 수 없다. 분모에 남겨 두면 "안전하게 넘길수록 지표가
        깎이는" 모순이 되고, 이는 '법적 판단은 반드시 사람에게' 라는 안전 요구와
        정면으로 충돌한다.
        """
        from docagent.contracts import AgentTurn
        from docagent.demo import run_safety_scene

        session = run_safety_scene(demo_form)
        session.turns[:] = [
            AgentTurn(
                user_text="이건 법적으로 유효한가요?",
                intent="question",
                speech="담당 직원에게 확인을 요청하겠습니다.",
                confidence=0.9,
                handoff_reason=(
                    "자격 여부나 법적 효력에 대한 판단은 제가 대신 내려 드릴 수 없습니다."
                ),
            )
        ]
        report = evaluate([KpiCase(session, demo_form.truth, handoff_probe=True)])
        assert report["counts"]["question_requests"] == 0.0
        assert report["metrics"]["question_resolution_rate"] is None


class TestSessionRestore:
    """세션 복원율 KPI 와 대칭 저장·복원 진입점."""

    def test_restore_rate_is_measured_and_met(
        self, demo_run: Any, demo_form: SyntheticForm
    ) -> None:
        """세션 복원율이 '측정 불가' 가 아니라 실제로 측정된다."""
        report = evaluate([KpiCase(demo_run.session, demo_form.truth)])
        rate = report["metrics"]["session_restore_rate"]
        assert rate == 1.0
        assert meets_target("session_restore_rate", rate) is True

    def test_snapshot_restores_structure_and_state(self, demo_run: Any) -> None:
        """스냅숏 한 봉투로 문서 구조와 세션 상태가 함께 복원된다."""
        session = demo_run.session
        snapshot = session.snapshot_json()
        agent = type(session).restore_agent(snapshot)
        assert agent.structure.to_dict() == session.structure.to_dict()
        assert agent.state.to_dict() == session.agent.state.to_dict()

    def test_broken_snapshot_is_not_silently_ignored(self, demo_run: Any) -> None:
        """깨진 스냅숏은 조용히 넘어가지 않고 한국어 오류로 올라온다."""
        session_cls = type(demo_run.session)
        with pytest.raises(ValueError, match="JSON"):
            session_cls.restore_agent("{깨진 JSON")
        with pytest.raises(ValueError, match="필수 키"):
            session_cls.restore_agent('{"version": 1}')


class TestTurnsAreNotPersisted:
    """대화 턴(원문 발화)은 세션 JSON 에 저장하지 않는다는 계약을 고정한다.

    :class:`~docagent.contracts.AgentTurn` 의 ``user_text`` 는 이용자의 발화 원문
    이므로 세션 JSON 에 실으면 개인정보가 디스크에 남는다. 복원 정본은
    ``SessionState`` 이고 감사 추적은 ``SessionState.history`` 가 담당한다.
    """

    def test_session_json_has_no_user_text(self, demo_run: Any) -> None:
        """세션 JSON 어디에도 발화 원문이 들어 있지 않다."""
        agent = demo_run.session.agent
        payload = agent.session_json()
        assert "turns" not in payload
        for turn in agent.turns:
            # 짧은 낱말("확인" 등)은 상태 이력의 한국어 설명과 우연히 겹칠 수
            # 있으므로, 발화로 식별 가능한 길이의 문장만 대조한다.
            if len(turn.user_text.strip()) >= 8:
                assert turn.user_text not in payload

    def test_history_is_the_restored_audit_trail(self, demo_run: Any) -> None:
        """복원 후에도 진행 이력(history)은 그대로 살아난다."""
        session = demo_run.session
        agent = type(session).restore_agent(session.snapshot_json())
        assert [entry.to_dict() for entry in agent.state.history] == [
            entry.to_dict() for entry in session.agent.state.history
        ]
        assert agent.state.history, "복원된 세션에 감사 이력이 비어 있습니다."
