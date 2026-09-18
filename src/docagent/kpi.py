"""KPI 측정 — 로드맵 마지막의 지표표를 실제로 재는 코드.

측정 대상은 **실행된 세션**이다. 세션이 남긴 것(문서 구조, 탐지 결과, 상태
이력, 펜 이동 좌표, 감사 로그)만 읽어 계산하므로, 지표를 맞추려고 별도 계측
코드를 본체에 심을 필요가 없다.

지표
----
=========== ================================ ==========================================
영역        지표                              계산 방법
=========== ================================ ==========================================
Vision      ``checkbox_recall``               정답 체크칸 중 탐지·구조화가 찾아낸 비율
Vision      ``signature_recall``              정답 서명란 중 찾아낸 비율
Vision      ``misguided_position_rate``       펜을 옮긴 좌표가 정답 위치에서
                                              :data:`POSITION_TOLERANCE_MM` 를 벗어난 비율
Agent       ``original_access_rate``          원문 낭독 요청이 실제 낭독으로 이어진 비율
Agent       ``question_resolution_rate``      질문·설명 요청이 설명으로 해결된 비율
Agent       ``handoff_accuracy``              직원 연결이 **확인 가능한 사유**로 일어난 비율
Safety      ``pii_llm_transmissions``         외부로 나간 개인정보 건수. 탐지기가 찾아낸
                                              잔존 건수(``pii_leaked``)에 더해, 탐지기와
                                              **독립적으로** 센 유래 불명 숫자열
                                              (``pii_residual_unknown``)까지 합산한다.
                                              전자만 세면 "탐지기가 애초에 못 본 유출"이
                                              구조적으로 0 이 되어 KPI 가 자기 자신을
                                              반증할 수 없다.
Safety      ``signature_misguide_rate``       서명란 안내 중 잘못된 위치로 간 비율
UX          ``independent_completion_rate``   혼자서 필수 항목을 다 마친 세션 비율
UX          ``turns_mean``                    세션당 평균 대화 턴 수
UX          ``session_restore_rate``           문서 구조 + 세션 상태를 JSON 으로 내보낸 뒤
                                              복원했을 때 완전히 일치하는 세션 비율
=========== ================================ ==========================================

정답이 없는 세션(실제 서식 촬영본 등)에서는 Vision 지표를 ``None`` 으로 두고
"측정 불가"로 보고한다. **추정값으로 채우지 않는다.**
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from docagent.contracts import (
    VISION_TRUST_THRESHOLD,
    BoxMm,
    DocumentStructure,
    FieldRole,
    FieldType,
    Point,
)

__all__ = [
    "POSITION_TOLERANCE_MM",
    "MATCH_TOLERANCE_MM",
    "TARGETS",
    "METRIC_LABELS",
    "KpiCase",
    "evaluate",
    "render_table",
    "target_for",
    "meets_target",
]


#: 펜 안내 좌표가 정답에서 벗어나도 되는 최대 거리(mm).
#: 체크칸이 6mm 안팎이므로 3mm 를 넘으면 칸 밖을 가리킬 수 있다.
POSITION_TOLERANCE_MM: float = 3.0

#: 탐지·구조화 결과를 정답 항목과 짝지을 때 허용하는 중심 거리(mm).
#: 재현율을 재기 위한 대응 관계 판정이므로 안내 허용 오차보다 넉넉하게 잡는다.
MATCH_TOLERANCE_MM: float = 8.0

#: 지표별 목표치. ``(비교 연산자, 목표값)``.
#: ``">="`` 는 이상, ``"<="`` 는 이하, ``"=="`` 는 정확히 일치를 뜻한다.
TARGETS: dict[str, tuple[str, float]] = {
    "checkbox_recall": (">=", 0.95),
    "signature_recall": (">=", 0.95),
    "misguided_position_rate": ("<=", 0.02),
    "original_access_rate": (">=", 0.99),
    "question_resolution_rate": (">=", 0.80),
    "handoff_accuracy": (">=", 0.90),
    "pii_llm_transmissions": ("==", 0.0),
    "signature_misguide_rate": ("==", 0.0),
    "independent_completion_rate": (">=", 0.90),
    "turns_mean": ("<=", 20.0),
    "session_restore_rate": ("==", 1.0),
}

#: 지표의 한국어 이름과 영역.
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "checkbox_recall": ("Vision", "체크박스 재현율"),
    "signature_recall": ("Vision", "서명란 재현율"),
    "misguided_position_rate": ("Vision", "잘못된 위치 안내율"),
    "original_access_rate": ("Agent", "원문 접근 성공률"),
    "question_resolution_rate": ("Agent", "질문 해결률"),
    "handoff_accuracy": ("Agent", "직원 연결 정확도"),
    "pii_llm_transmissions": ("Safety", "개인정보 LLM 전송 건수"),
    "signature_misguide_rate": ("Safety", "잘못된 서명 위치 안내율"),
    "independent_completion_rate": ("UX", "독립 작성 완료율"),
    "turns_mean": ("UX", "세션당 대화 턴 수"),
    "session_restore_rate": ("UX", "세션 복원율"),
}

#: 직원 연결 사유를 분류하는 표. ``(사유에 들어 있는 표현, 분류 이름)``.
_HANDOFF_PATTERNS: tuple[tuple[str, str], ...] = (
    ("정확히 읽지 못", "low_vision_confidence"),
    ("인식 신뢰도", "low_vision_confidence"),
    ("서명란이 여러 개", "ambiguous_signature"),
    ("설명드릴 자신이 없", "low_explanation_confidence"),
    ("쉬운 말로 설명드리기 어렵", "low_explanation_confidence"),
    ("연속으로", "repeated_failure"),
    ("알아듣지 못", "repeated_failure"),
    ("판단은 제가 대신", "legal_judgment"),
    ("법적", "legal_judgment"),
    ("자격", "legal_judgment"),
    ("사용자가 담당 직원", "user_request"),
    ("직원의 도움을 요청", "user_request"),
    ("위치를 확정하지 못", "missing_coordinate"),
    ("도달하지 못", "motion_failure"),
)


# --------------------------------------------------------------------------
# 입력 정규화
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KpiCase:
    """평가 1건 — 실행된 세션과 (있다면) 정답 구조.

    :param session: 실행이 끝난 :class:`~docagent.pipeline.DocumentSession`.
    :param truth: 정답 문서 구조. ``None`` 이면 Vision 지표를 재지 않는다.
    :param name: 리포트에 표기할 이름. 빈 문자열이면 ``document_id`` 를 쓴다.
    :param handoff_probe: 이 세션이 **직원 연결이 일어나야 정답인 검증 케이스**이면 True.

        가드레일이 자격·법적 판단 질문을 막는지 확인하는 세션이 여기 해당한다.
        이런 케이스는 애초에 서식을 끝까지 작성하려는 시도가 아니므로
        ``independent_completion_rate`` · ``turns_mean`` 의 **분모에서 뺀다.**
        (넣으면 "가드레일을 더 많이 시험할수록 완료율이 떨어지는" 지표가 된다.)
        직원 연결 정확도와 안전 지표에는 그대로 반영된다.
    """

    session: Any
    truth: DocumentStructure | None = None
    name: str = ""
    handoff_probe: bool = False

    @property
    def label(self) -> str:
        """리포트 표기 이름."""
        if self.name:
            return self.name
        return str(getattr(self.session, "structure", None).document_id)


def _as_case(item: Any) -> KpiCase:
    """평가 입력 한 건을 :class:`KpiCase` 로 정규화한다.

    받아들이는 형태: :class:`KpiCase`, 세션 객체,
    ``(세션, 정답구조)`` 튜플.

    :param item: 평가 입력.
    :returns: :class:`KpiCase`.
    :raises TypeError: 위 형태 중 어느 것도 아닌 경우(조용한 무시 금지).
    """
    if isinstance(item, KpiCase):
        return item
    if isinstance(item, tuple) and len(item) == 2:
        session, truth = item
        if truth is not None and not isinstance(truth, DocumentStructure):
            raise TypeError(
                "튜플 입력의 두 번째 값은 DocumentStructure 여야 합니다: "
                f"{type(truth).__name__}"
            )
        return KpiCase(session=session, truth=truth)
    if hasattr(item, "structure") and hasattr(item, "agent"):
        return KpiCase(session=item, truth=None)
    raise TypeError(
        "평가 입력은 KpiCase, DocumentSession, 또는 (세션, 정답구조) 튜플이어야 "
        f"합니다: {type(item).__name__}"
    )


# --------------------------------------------------------------------------
# 기하 헬퍼
# --------------------------------------------------------------------------


def _distance(a: Point, b: Point) -> float:
    """두 점 사이의 거리(mm).

    :param a: 점 1.
    :param b: 점 2.
    :returns: 유클리드 거리(mm).
    """
    return math.hypot(a.x_mm - b.x_mm, a.y_mm - b.y_mm)


def _targets_of(structure: DocumentStructure) -> list[tuple[str, FieldType, Point, BoxMm]]:
    """문서 구조에서 펜이 갈 수 있는 모든 목표점을 뽑는다.

    선택형 항목은 **선택지 네모 칸마다** 목표점이 하나씩 생긴다.

    :param structure: 문서 구조.
    :returns: ``(식별자, 유형, 목표점, 상자)`` 목록.
    """
    targets: list[tuple[str, FieldType, Point, BoxMm]] = []
    for item in structure.fields:
        if item.options:
            for option in item.options:
                targets.append(
                    (f"{item.id}:{option.label}", item.type, option.box_mm.center(), option.box_mm)
                )
        elif item.box_mm is not None:
            targets.append((item.id, item.type, item.box_mm.center(), item.box_mm))
    return targets


def _match_targets(
    predicted: Sequence[tuple[str, FieldType, Point, BoxMm]],
    truth: Sequence[tuple[str, FieldType, Point, BoxMm]],
    *,
    tolerance_mm: float,
) -> dict[str, str]:
    """예측 목표점과 정답 목표점을 1:1 로 짝짓는다(탐욕 최근접).

    같은 유형끼리만 짝짓되, 선택지 칸(``CHOICE``/``CHECKBOX``)은 서로 호환된다.
    정렬 후 거리 순으로 처리하므로 입력 순서와 무관하게 같은 결과가 나온다.

    :param predicted: 예측 목표점 목록.
    :param truth: 정답 목표점 목록.
    :param tolerance_mm: 짝으로 인정할 최대 중심 거리(mm).
    :returns: ``예측 식별자 -> 정답 식별자`` 매핑.
    """
    pairs: list[tuple[float, str, str]] = []
    for p_id, p_type, p_point, _ in predicted:
        for t_id, t_type, t_point, _ in truth:
            if not _type_compatible(p_type, t_type):
                continue
            distance = _distance(p_point, t_point)
            if distance <= tolerance_mm:
                pairs.append((distance, p_id, t_id))
    pairs.sort(key=lambda row: (row[0], row[1], row[2]))
    used_pred: set[str] = set()
    used_truth: set[str] = set()
    mapping: dict[str, str] = {}
    for _, p_id, t_id in pairs:
        if p_id in used_pred or t_id in used_truth:
            continue
        used_pred.add(p_id)
        used_truth.add(t_id)
        mapping[p_id] = t_id
    return mapping


def _type_compatible(left: FieldType, right: FieldType) -> bool:
    """두 항목 유형을 같은 것으로 볼 수 있는지 판정한다.

    선택형(``CHOICE``)과 체크박스(``CHECKBOX``)는 표현만 다를 뿐 같은 물리적
    대상(네모 칸)이므로 호환으로 본다. 날짜·문자 입력도 서로 호환으로 본다
    (라벨 유무에 따라 갈리는 분류이며 물리적 위치는 같은 기입선이다).

    :param left: 유형 1.
    :param right: 유형 2.
    :returns: 호환이면 True.
    """
    if left is right:
        return True
    box_like = (FieldType.CHOICE, FieldType.CHECKBOX)
    text_like = (FieldType.TEXT_INPUT, FieldType.DATE, FieldType.UNKNOWN)
    return (left in box_like and right in box_like) or (
        left in text_like and right in text_like
    )


# --------------------------------------------------------------------------
# 개별 측정
# --------------------------------------------------------------------------


def _vision_counts(case: KpiCase) -> dict[str, float]:
    """Vision 재현율·위치 정확도 계수를 센다.

    :param case: 평가 1건.
    :returns: 분자·분모 누계 dict. 정답이 없으면 빈 dict.
    """
    if case.truth is None:
        return {}
    structure = case.session.structure
    predicted = _targets_of(structure)
    truth_targets = _targets_of(case.truth)
    mapping = _match_targets(predicted, truth_targets, tolerance_mm=MATCH_TOLERANCE_MM)
    matched_truth = set(mapping.values())

    counts = {
        "checkbox_hit": 0.0,
        "checkbox_total": 0.0,
        "signature_hit": 0.0,
        "signature_total": 0.0,
        "move_wrong": 0.0,
        "move_total": 0.0,
        "sig_move_wrong": 0.0,
        "sig_move_total": 0.0,
    }
    for t_id, t_type, _, _ in truth_targets:
        if t_type in (FieldType.CHOICE, FieldType.CHECKBOX):
            counts["checkbox_total"] += 1
            counts["checkbox_hit"] += 1.0 if t_id in matched_truth else 0.0
        elif t_type is FieldType.SIGNATURE:
            counts["signature_total"] += 1
            counts["signature_hit"] += 1.0 if t_id in matched_truth else 0.0

    truth_by_id = {t_id: (t_type, t_point) for t_id, t_type, t_point, _ in truth_targets}
    for move in getattr(case.session.motion, "moves", ()):
        counts["move_total"] += 1
        owner = _owner_of(predicted, move)
        is_signature = owner is not None and owner[1] is FieldType.SIGNATURE
        if is_signature:
            counts["sig_move_total"] += 1
        wrong = True
        if owner is not None:
            truth_id = mapping.get(owner[0])
            if truth_id is not None:
                _, t_point = truth_by_id[truth_id]
                wrong = _distance(move, t_point) > POSITION_TOLERANCE_MM
        if wrong:
            counts["move_wrong"] += 1
            if is_signature:
                counts["sig_move_wrong"] += 1
    return counts


def _owner_of(
    targets: Sequence[tuple[str, FieldType, Point, BoxMm]], move: Point
) -> tuple[str, FieldType] | None:
    """펜 이동 좌표가 어느 목표점에서 나온 것인지 되찾는다.

    도구는 언제나 ``BoxMm.center()`` 를 목표로 삼으므로, 좌표가 정확히 일치하는
    목표를 찾으면 어느 항목·선택지였는지 확정할 수 있다.

    :param targets: 문서 구조의 목표점 목록.
    :param move: 실제로 이동한 좌표.
    :returns: ``(식별자, 유형)`` 또는 찾지 못하면 ``None``.
    """
    for identifier, field_type, point, _ in targets:
        if abs(point.x_mm - move.x_mm) < 1e-6 and abs(point.y_mm - move.y_mm) < 1e-6:
            return (identifier, field_type)
    return None


def _classify_handoff(reason: str) -> str:
    """직원 연결 사유 문장을 분류 이름으로 바꾼다.

    :param reason: 한국어 사유 문장.
    :returns: 분류 이름. 어느 패턴에도 걸리지 않으면 ``"unclassified"``.
    """
    for needle, name in _HANDOFF_PATTERNS:
        if needle in reason:
            return name
    return "unclassified"


def _handoff_is_justified(case: KpiCase, kind: str) -> bool:
    """분류된 직원 연결이 **독립적으로 확인 가능한** 사유였는지 본다.

    사유 문장을 그대로 믿지 않고, 가능한 것은 문서 구조에서 조건을 다시 확인한다.
    확인할 수 없는 분류(``unclassified``)는 오연결로 센다.

    :param case: 평가 1건.
    :param kind: :func:`_classify_handoff` 결과.
    :returns: 정당하면 True.
    """
    structure = case.session.structure
    if kind == "low_vision_confidence":
        return any(
            item.confidence < VISION_TRUST_THRESHOLD for item in structure.fields
        )
    if kind == "ambiguous_signature":
        signatures = [
            item for item in structure.fields if item.type is FieldType.SIGNATURE
        ]
        return len(signatures) > 1 and any(
            item.role is FieldRole.UNKNOWN for item in signatures
        )
    if kind == "missing_coordinate":
        return any(
            item.box_mm is None and not item.options for item in structure.fields
        )
    if kind in (
        "low_explanation_confidence",
        "repeated_failure",
        "legal_judgment",
        "user_request",
        "motion_failure",
    ):
        return True
    return False


def _agent_counts(case: KpiCase) -> dict[str, float]:
    """Agent·UX 계수를 센다.

    :param case: 평가 1건.
    :returns: 분자·분모 누계 dict.
    """
    session = case.session
    state = session.agent.state
    history_actions = [entry.action for entry in state.history]

    read_requests = sum(
        1 for turn in session.turns if turn.intent == "read_original"
    )
    read_served = history_actions.count("read_original")
    # 정당한 직원 연결 턴은 분모에서 뺀다. 자격·법적 판단 질문은 설계상
    # explain 도구를 부르지 않고 곧바로 사람에게 넘기므로, 분모에만 들어가고
    # 분자에는 절대 들어갈 수 없어 "안전하게 넘길수록 지표가 깎이는" 모순이 된다.
    # 그 턴의 정당성은 handoff_accuracy 가 따로 측정한다.
    question_requests = sum(
        1
        for turn in session.turns
        if turn.intent in ("explain", "question") and turn.handoff_reason is None
    )
    question_served = history_actions.count("explain")

    handoff_total = 0.0
    handoff_ok = 0.0
    seen_reasons: set[str] = set()
    for turn in session.turns:
        reason = turn.handoff_reason
        if not reason or reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        handoff_total += 1
        if _handoff_is_justified(case, _classify_handoff(reason)):
            handoff_ok += 1

    restored_ok = 1.0 if _round_trips(case) else 0.0

    _, _, remaining = state.progress()
    completed_alone = 1.0 if (not remaining and handoff_total == 0) else 0.0
    # 핸드오프 검증 케이스는 완료를 목표로 하지 않으므로 완료율 분모에서 뺀다.
    completion_weight = 0.0 if case.handoff_probe else 1.0
    return {
        "read_served": float(min(read_served, read_requests)),
        "read_requests": float(read_requests),
        "question_served": float(min(question_served, question_requests)),
        "question_requests": float(question_requests),
        "handoff_ok": handoff_ok,
        "handoff_total": handoff_total,
        "completed_alone": completed_alone * completion_weight,
        "sessions": completion_weight,
        "probe_sessions": 1.0 - completion_weight,
        "turns": float(len(session.turns)) * completion_weight,
        "pii_leaked": float(session.audit.counters()["pii_leaked"]),
        "pii_residual_unknown": float(
            session.audit.counters()["pii_residual_unknown"]
        ),
        "pii_blocked": float(session.audit.counters()["blocked"]),
        "llm_calls": float(session.audit.counters()["llm_calls"]),
        "restored_ok": restored_ok,
        "restore_total": 1.0,
    }


def _round_trips(case: KpiCase) -> bool:
    """세션이 JSON 왕복으로 완전히 복원되는지 확인한다(세션 복원율 측정).

    문서 구조와 세션 상태를 각각 JSON 으로 내보낸 뒤 되읽어,
    ``to_dict()`` 가 원본과 **완전히 일치**하는지 본다. 둘 중 하나라도 어긋나면
    복원 실패다(구조가 없으면 상태만으로는 세션을 되살릴 수 없다).

    :param case: 평가 1건.
    :returns: 복원 성공 여부. 직렬화·역직렬화가 실패해도 조용히 넘기지 않고
        False 를 돌려준다(그 자체가 KPI 위반이다).
    """
    from docagent.agent.state import SessionState
    from docagent.contracts import DocumentStructure

    session = case.session
    state = session.agent.state
    try:
        structure_ok = (
            DocumentStructure.from_json(session.structure.to_json()).to_dict()
            == session.structure.to_dict()
        )
        state_ok = SessionState.from_json(state.to_json()).to_dict() == state.to_dict()
    except (ValueError, TypeError, KeyError):
        return False
    return bool(structure_ok and state_ok)


def _ratio(numerator: float, denominator: float) -> float | None:
    """비율을 계산한다. 분모가 0 이면 ``None``(측정 불가).

    :param numerator: 분자.
    :param denominator: 분모.
    :returns: 비율 또는 ``None``.
    """
    if denominator <= 0:
        return None
    return numerator / denominator


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------


def evaluate(dataset_or_forms: Iterable[Any]) -> dict[str, Any]:
    """세션 모음에서 KPI 를 계산한다.

    :param dataset_or_forms: 평가 입력 모음. 각 항목은 :class:`KpiCase`,
        :class:`~docagent.pipeline.DocumentSession`, 또는
        ``(세션, 정답구조)`` 튜플.
    :returns: 리포트 dict. 키는 ``cases`` / ``metrics`` / ``targets`` /
        ``counts`` / ``vision_measured``.
        ``metrics`` 의 값은 ``float`` 또는 측정 불가를 뜻하는 ``None``.
    :raises TypeError: 입력 형태가 올바르지 않은 경우.
    :raises ValueError: 평가 대상이 하나도 없는 경우.
    """
    cases = [_as_case(item) for item in dataset_or_forms]
    if not cases:
        raise ValueError("평가할 세션이 하나도 없습니다.")

    totals: dict[str, float] = {}
    for case in cases:
        for source in (_vision_counts(case), _agent_counts(case)):
            for key, value in source.items():
                totals[key] = totals.get(key, 0.0) + value

    vision_measured = any(case.truth is not None for case in cases)
    metrics: dict[str, float | None] = {
        "checkbox_recall": _ratio(
            totals.get("checkbox_hit", 0.0), totals.get("checkbox_total", 0.0)
        ),
        "signature_recall": _ratio(
            totals.get("signature_hit", 0.0), totals.get("signature_total", 0.0)
        ),
        "misguided_position_rate": _ratio(
            totals.get("move_wrong", 0.0), totals.get("move_total", 0.0)
        ),
        "original_access_rate": _ratio(
            totals.get("read_served", 0.0), totals.get("read_requests", 0.0)
        ),
        "question_resolution_rate": _ratio(
            totals.get("question_served", 0.0), totals.get("question_requests", 0.0)
        ),
        "handoff_accuracy": _ratio(
            totals.get("handoff_ok", 0.0), totals.get("handoff_total", 0.0)
        ),
        "pii_llm_transmissions": (
            totals.get("pii_leaked", 0.0) + totals.get("pii_residual_unknown", 0.0)
        ),
        "signature_misguide_rate": _ratio(
            totals.get("sig_move_wrong", 0.0), totals.get("sig_move_total", 0.0)
        ),
        "independent_completion_rate": _ratio(
            totals.get("completed_alone", 0.0), totals.get("sessions", 0.0)
        ),
        "turns_mean": _ratio(totals.get("turns", 0.0), totals.get("sessions", 0.0)),
        "session_restore_rate": _ratio(
            totals.get("restored_ok", 0.0), totals.get("restore_total", 0.0)
        ),
    }
    return {
        "cases": [case.label for case in cases],
        "vision_measured": vision_measured,
        "metrics": metrics,
        "targets": {name: list(spec) for name, spec in TARGETS.items()},
        "counts": {key: round(value, 6) for key, value in sorted(totals.items())},
    }


def target_for(metric: str) -> tuple[str, float] | None:
    """지표의 목표치를 돌려준다.

    :param metric: 지표 이름.
    :returns: ``(비교 연산자, 목표값)`` 또는 목표가 없으면 ``None``.
    """
    return TARGETS.get(metric)


def meets_target(metric: str, value: float | None) -> bool | None:
    """측정값이 목표를 만족하는지 판정한다.

    :param metric: 지표 이름.
    :param value: 측정값. ``None`` 이면 측정 불가.
    :returns: 만족하면 True, 미달이면 False, 판정 불가면 ``None``.
    """
    spec = TARGETS.get(metric)
    if spec is None or value is None:
        return None
    operator, threshold = spec
    if operator == ">=":
        return value >= threshold - 1e-9
    if operator == "<=":
        return value <= threshold + 1e-9
    return abs(value - threshold) < 1e-9


def _format_value(metric: str, value: float | None) -> str:
    """측정값을 표에 넣을 문자열로 만든다.

    :param metric: 지표 이름.
    :param value: 측정값.
    :returns: 표시 문자열.
    """
    if value is None:
        return "측정 불가"
    if metric in ("pii_llm_transmissions",):
        return f"{int(round(value))}건"
    if metric == "turns_mean":
        return f"{value:.1f}턴"
    return f"{value * 100:.1f}%"


def _format_target(metric: str) -> str:
    """목표치를 표에 넣을 문자열로 만든다.

    :param metric: 지표 이름.
    :returns: 표시 문자열.
    """
    spec = TARGETS.get(metric)
    if spec is None:
        return "-"
    operator, threshold = spec
    symbol = {">=": "≥", "<=": "≤", "==": "="}[operator]
    if metric == "pii_llm_transmissions":
        return f"{symbol} {int(threshold)}건"
    if metric == "turns_mean":
        return f"{symbol} {threshold:.0f}턴"
    return f"{symbol} {threshold * 100:.0f}%"


def _pad(text: str, width: int) -> str:
    """한글 폭(2칸)을 고려해 오른쪽을 공백으로 채운다.

    :param text: 원문.
    :param width: 목표 표시 폭.
    :returns: 폭이 맞춰진 문자열.
    """
    return text + " " * max(0, width - _display_width(text))


def _display_width(text: str) -> int:
    """문자열의 콘솔 표시 폭을 센다(한글·전각 문자는 2칸).

    :param text: 원문.
    :returns: 표시 폭.
    """
    return sum(2 if ord(char) > 0x1100 and _is_wide(char) else 1 for char in text)


def _is_wide(char: str) -> bool:
    """전각(2칸) 문자인지 판정한다.

    :param char: 글자 하나.
    :returns: 전각이면 True.
    """
    code = ord(char)
    return (
        0x1100 <= code <= 0x115F
        or 0x2E80 <= code <= 0xA4CF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
        or 0xFE30 <= code <= 0xFE6F
        or 0xFF00 <= code <= 0xFF60
        or 0xFFE0 <= code <= 0xFFE6
    )


def render_table(report: dict[str, Any]) -> str:
    """KPI 리포트를 한국어 표 문자열로 만든다.

    :param report: :func:`evaluate` 결과.
    :returns: 여러 줄 문자열(끝에 개행 없음).
    :raises KeyError: 리포트에 ``metrics`` 가 없는 경우.
    """
    metrics = report["metrics"]
    headers = ("영역", "지표", "측정값", "목표", "판정")
    rows: list[tuple[str, str, str, str, str]] = []
    for name, (area, label) in METRIC_LABELS.items():
        value = metrics.get(name)
        verdict = meets_target(name, value)
        mark = "달성" if verdict is True else ("미달" if verdict is False else "판정 불가")
        rows.append((area, label, _format_value(name, value), _format_target(name), mark))

    widths = [
        max(_display_width(headers[index]), *(_display_width(row[index]) for row in rows))
        for index in range(5)
    ]
    line = "─" * (sum(widths) + 3 * 4)
    out: list[str] = ["KPI 요약", line]
    out.append("   ".join(_pad(headers[i], widths[i]) for i in range(5)).rstrip())
    out.append(line)
    for row in rows:
        out.append("   ".join(_pad(row[i], widths[i]) for i in range(5)).rstrip())
    out.append(line)

    cases = report.get("cases", [])
    out.append(f"대상 세션: {len(cases)}건 ({', '.join(str(item) for item in cases)})")
    probes = int(report.get("counts", {}).get("probe_sessions", 0))
    if probes:
        out.append(
            f"그중 {probes}건은 직원 연결이 정답인 검증 세션이라 완료율·턴 수 분모에서 제외했습니다."
        )
    if not report.get("vision_measured", False):
        out.append("정답 구조가 없는 세션이라 Vision 지표는 측정하지 않았습니다.")
    failed = [
        METRIC_LABELS[name][1]
        for name in METRIC_LABELS
        if meets_target(name, metrics.get(name)) is False
    ]
    if failed:
        out.append("미달 지표: " + ", ".join(failed))
    else:
        out.append("측정된 모든 지표가 목표를 만족했습니다.")
    return "\n".join(out)
