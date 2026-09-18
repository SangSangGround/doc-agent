"""세션 상태머신(FSM) — 대화 세션의 단일 진실 원천.

이 모듈은 에이전트 루프가 "지금 어느 단계에 있고, 어떤 항목을 다루고 있으며,
무엇이 끝났는지"를 보관한다. 발화 생성·도구 실행·LLM 호출은 하지 않는다.

설계 원칙
---------
1. **명시적 전이 테이블.** 허용된 전이는 :data:`TRANSITIONS` 에만 존재한다.
   테이블에 없는 전이는 :class:`docagent.errors.InvalidTransition` 으로 즉시 실패한다
   (조용한 실패 금지).
2. **모든 상태 변경은 감사 가능하다.** 전이·선택·검증·되돌림이 전부
   :class:`HistoryEntry` 로 ``history`` 에 남는다.
3. **완전 왕복 직렬화.** ``SessionState.from_json(s.to_json()) == s`` 가 항상 성립한다.
   이것이 "세션 복원율 100%" KPI 의 기술적 근거다.
4. **되돌림 지원.** :meth:`SessionState.revert_to_field` 는 "아까 동의 안 한다고 한 거
   다시 바꿀래" 를, :meth:`SessionState.undo_last` 는 직전 턴 취소를 처리한다.
   되돌림 자체도 이력에 남으므로 감사 추적이 끊기지 않는다.

용어
----
* **완료(completed)** — 기입이 확인되어 더 다룰 필요가 없는 항목.
* **대기(pending)** — 아직 다루지 않은 항목.
* **건너뜀(skipped)** — 사용자가 "다음"으로 넘긴 항목. 대기에서 빠지지만
  완료도 아니다. 필수 항목이 여기에 있으면 완료 발화에서 경고한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Mapping

from docagent.contracts import DocumentStructure
from docagent.errors import InvalidTransition

__all__ = [
    "SessionPhase",
    "HistoryEntry",
    "SessionState",
    "TRANSITIONS",
    "MAX_CHECKPOINTS",
]


#: 되돌림을 위해 보관하는 턴 시작 스냅샷의 최대 개수.
MAX_CHECKPOINTS: int = 20

#: :meth:`SessionState.transition_to` 에서 "현재 항목을 그대로 둔다"를 뜻하는 sentinel.
_KEEP: Any = object()


class SessionPhase(Enum):
    """대화 세션의 단계.

    See → Understand → Explain → Ask → Act → Verify 골격을 대화 관점으로 펼친 것이다.

    * :attr:`IDLE` — 문서가 아직 올라오지 않은 초기 상태.
    * :attr:`DOCUMENT_LOADED` — 문서 구조 해석 완료, 첫 안내 직전.
    * :attr:`ANNOUNCE_FIELD` — 항목을 안내하고 사용자의 응답을 기다리는 상태.
    * :attr:`AWAIT_CHOICE` — 선택지를 읽어 주고 선택을 기다리는 상태.
    * :attr:`GUIDING` — 선택이 확정되어 펜 이동을 안내하는 상태.
    * :attr:`AWAIT_WRITE` — 펜이 목표 위치에 도달했고 사용자의 기입을 기다리는 상태.
    * :attr:`VERIFYING` — 기입 여부를 확인하는 중.
    * :attr:`FIELD_DONE` — 한 항목이 확정된 상태.
    * :attr:`COMPLETED` — 모든 항목 처리 종료.
    * :attr:`HUMAN_HANDOFF` — 사람 지원으로 넘긴 종료 상태.
    """

    IDLE = "idle"
    DOCUMENT_LOADED = "document_loaded"
    ANNOUNCE_FIELD = "announce_field"
    AWAIT_CHOICE = "await_choice"
    GUIDING = "guiding"
    AWAIT_WRITE = "await_write"
    VERIFYING = "verifying"
    FIELD_DONE = "field_done"
    COMPLETED = "completed"
    HUMAN_HANDOFF = "human_handoff"


#: 허용된 상태 전이 테이블. ``TRANSITIONS[현재]`` 에 없는 단계로는 갈 수 없다.
#:
#: :attr:`SessionPhase.HUMAN_HANDOFF` 는 흡수 상태(absorbing state)다. 사람에게
#: 넘긴 세션을 에이전트가 스스로 회수하지 못하게 막는다.
TRANSITIONS: dict[SessionPhase, frozenset[SessionPhase]] = {
    SessionPhase.IDLE: frozenset(
        {SessionPhase.DOCUMENT_LOADED, SessionPhase.HUMAN_HANDOFF}
    ),
    SessionPhase.DOCUMENT_LOADED: frozenset(
        {
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.COMPLETED,
            SessionPhase.IDLE,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.ANNOUNCE_FIELD: frozenset(
        {
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.AWAIT_CHOICE,
            SessionPhase.GUIDING,
            SessionPhase.AWAIT_WRITE,
            SessionPhase.FIELD_DONE,
            SessionPhase.COMPLETED,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.AWAIT_CHOICE: frozenset(
        {
            SessionPhase.AWAIT_CHOICE,
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.GUIDING,
            SessionPhase.COMPLETED,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.GUIDING: frozenset(
        {
            SessionPhase.GUIDING,
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.AWAIT_WRITE,
            SessionPhase.VERIFYING,
            SessionPhase.COMPLETED,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.AWAIT_WRITE: frozenset(
        {
            SessionPhase.AWAIT_WRITE,
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.GUIDING,
            SessionPhase.VERIFYING,
            SessionPhase.COMPLETED,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.VERIFYING: frozenset(
        {
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.AWAIT_WRITE,
            SessionPhase.FIELD_DONE,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.FIELD_DONE: frozenset(
        {
            SessionPhase.FIELD_DONE,
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.AWAIT_CHOICE,
            SessionPhase.COMPLETED,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.COMPLETED: frozenset(
        {
            SessionPhase.COMPLETED,
            SessionPhase.ANNOUNCE_FIELD,
            SessionPhase.IDLE,
            SessionPhase.HUMAN_HANDOFF,
        }
    ),
    SessionPhase.HUMAN_HANDOFF: frozenset({SessionPhase.HUMAN_HANDOFF}),
}


def _as_phase(value: Any) -> SessionPhase:
    """문자열 또는 :class:`SessionPhase` 를 :class:`SessionPhase` 로 변환한다.

    :param value: Enum 인스턴스이거나 ``value`` 문자열.
    :returns: :class:`SessionPhase`.
    :raises ValueError: 정의되지 않은 값인 경우.
    """
    if isinstance(value, SessionPhase):
        return value
    try:
        return SessionPhase(value)
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in SessionPhase)
        raise ValueError(
            f"SessionPhase 에 정의되지 않은 값입니다: {value!r}. 허용 값: {allowed}"
        ) from exc


@dataclass(frozen=True)
class HistoryEntry:
    """상태 변경 1건의 감사 기록.

    :param turn_index: 이 변경이 일어난 대화 턴 번호(0 부터).
    :param phase: 변경 **직후**의 세션 단계.
    :param field_id: 관련 항목 id. 항목과 무관한 변경이면 ``None``.
    :param action: 변경 종류 식별자(예: ``"transition"``, ``"select_option"``,
        ``"revert_to_field"``, ``"undo_last"``).
    :param note: 사람이 읽는 한국어 부연 설명.
    """

    turn_index: int
    phase: SessionPhase
    field_id: str | None = None
    action: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "turn_index": self.turn_index,
            "phase": self.phase.value,
            "field_id": self.field_id,
            "action": self.action,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HistoryEntry":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다.

        :param data: :meth:`to_dict` 가 만든 매핑.
        :returns: :class:`HistoryEntry`.
        :raises ValueError: 필수 키 누락 또는 미정의 단계 값.
        """
        for key in ("turn_index", "phase"):
            if key not in data:
                raise ValueError(f"HistoryEntry.from_dict 에 필수 키가 없습니다: {key}")
        field_id = data.get("field_id")
        return cls(
            turn_index=int(data["turn_index"]),
            phase=_as_phase(data["phase"]),
            field_id=None if field_id is None else str(field_id),
            action=str(data.get("action", "")),
            note=str(data.get("note", "")),
        )


@dataclass
class SessionState:
    """문서 작성 세션 1건의 전체 상태.

    :param document_id: 대상 문서 식별자(세션 복원 키).
    :param phase: 현재 세션 단계.
    :param current_field_id: 지금 안내 중인 항목 id. 없으면 ``None``.
    :param field_order: 문서의 전체 항목 id 를 ``order`` 순으로 나열한 목록.
    :param required_field_ids: 필수 항목 id 목록(``field_order`` 부분집합).
    :param completed_fields: 기입이 확정된 항목 id(확정된 순서).
    :param pending_fields: 아직 다루지 않은 항목 id(문서 순서).
    :param skipped_fields: 사용자가 건너뛴 항목 id.
    :param selected_options: ``항목 id -> 선택한 선택지 라벨``.
    :param verified_fields: Verify 단계를 통과한 항목 id.
    :param turn_index: 현재 대화 턴 번호(0 부터).
    :param handoff_reason: 사람 지원으로 넘긴 사유. 넘기지 않았으면 ``None``.
    :param counters: 재시도·실패 누적 카운터(``이름 -> 횟수``).
    :param history: 모든 상태 변경의 감사 기록.
    :param checkpoints: 턴 시작 시점 스냅샷 스택(:meth:`undo_last` 전용).
        최신 :data:`MAX_CHECKPOINTS` 개만 유지한다.
    :raises ValueError: ``document_id`` 가 빈 문자열인 경우.
    """

    document_id: str
    phase: SessionPhase = SessionPhase.IDLE
    current_field_id: str | None = None
    field_order: list[str] = dc_field(default_factory=list)
    required_field_ids: list[str] = dc_field(default_factory=list)
    completed_fields: list[str] = dc_field(default_factory=list)
    pending_fields: list[str] = dc_field(default_factory=list)
    skipped_fields: list[str] = dc_field(default_factory=list)
    selected_options: dict[str, str] = dc_field(default_factory=dict)
    verified_fields: list[str] = dc_field(default_factory=list)
    turn_index: int = 0
    handoff_reason: str | None = None
    counters: dict[str, int] = dc_field(default_factory=dict)
    history: list[HistoryEntry] = dc_field(default_factory=list)
    checkpoints: list[dict[str, Any]] = dc_field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("SessionState.document_id 는 빈 문자열일 수 없습니다.")

    # ------------------------------------------------------------------
    # 생성
    # ------------------------------------------------------------------

    @classmethod
    def from_structure(cls, structure: DocumentStructure) -> "SessionState":
        """Vision 이 만든 문서 구조로부터 초기 세션 상태를 만든다.

        :param structure: :class:`~docagent.contracts.DocumentStructure`.
        :returns: ``phase`` 가 :attr:`SessionPhase.IDLE` 이고 모든 항목이
            ``pending_fields`` 에 들어 있는 새 세션 상태.
        """
        ordered = sorted(structure.fields, key=lambda f: f.order)
        ids = [item.id for item in ordered]
        return cls(
            document_id=structure.document_id,
            field_order=list(ids),
            required_field_ids=[item.id for item in ordered if item.required],
            pending_fields=list(ids),
        )

    # ------------------------------------------------------------------
    # 이력
    # ------------------------------------------------------------------

    def log(
        self,
        action: str,
        *,
        note: str = "",
        field_id: Any = _KEEP,
    ) -> HistoryEntry:
        """상태 변경 1건을 이력에 남긴다.

        :param action: 변경 종류 식별자.
        :param note: 한국어 부연 설명.
        :param field_id: 관련 항목 id. 생략하면 ``current_field_id`` 를 쓴다.
        :returns: 방금 추가한 :class:`HistoryEntry`.
        """
        entry = HistoryEntry(
            turn_index=self.turn_index,
            phase=self.phase,
            field_id=self.current_field_id if field_id is _KEEP else field_id,
            action=action,
            note=note,
        )
        self.history.append(entry)
        return entry

    # ------------------------------------------------------------------
    # 전이
    # ------------------------------------------------------------------

    @staticmethod
    def allowed_transitions(phase: SessionPhase) -> frozenset[SessionPhase]:
        """``phase`` 에서 갈 수 있는 단계 집합을 반환한다.

        :param phase: 기준 단계.
        :returns: 허용 단계 집합(불변).
        """
        return TRANSITIONS[phase]

    def can_transition_to(self, phase: SessionPhase) -> bool:
        """``phase`` 로의 전이가 허용되면 True.

        :param phase: 목표 단계.
        :returns: 허용 여부.
        """
        return phase in TRANSITIONS[self.phase]

    def transition_to(
        self,
        phase: SessionPhase,
        *,
        action: str = "transition",
        note: str = "",
        field_id: Any = _KEEP,
    ) -> None:
        """세션 단계를 옮기고 이력을 남긴다.

        :param phase: 목표 단계.
        :param action: 이력에 남길 변경 종류.
        :param note: 이력에 남길 한국어 설명.
        :param field_id: 함께 갱신할 현재 항목 id. 생략하면 유지한다.
        :returns: ``None``.
        :raises InvalidTransition: 전이 테이블에 없는 전이인 경우.
        """
        if phase not in TRANSITIONS[self.phase]:
            raise InvalidTransition(
                "허용되지 않은 상태 전이입니다.",
                current=self.phase.value,
                requested=phase.value,
            )
        if field_id is not _KEEP:
            self.current_field_id = None if field_id is None else str(field_id)
        self.phase = phase
        self.log(action, note=note)

    def enter_handoff(self, reason: str, *, field_id: str | None = None) -> None:
        """사람 지원 단계로 넘어가고 사유를 기록한다.

        :param reason: 넘기는 사유(한국어 한 문장). 사용자에게 그대로 낭독 가능해야 한다.
        :param field_id: 문제가 된 항목 id. 없으면 현재 항목을 쓴다.
        :returns: ``None``.
        """
        if field_id is not None:
            self.current_field_id = field_id
        self.handoff_reason = reason
        # HUMAN_HANDOFF 는 모든 단계에서 도달 가능하므로 전이 실패가 발생하지 않는다.
        self.transition_to(
            SessionPhase.HUMAN_HANDOFF, action="handoff", note=reason
        )

    # ------------------------------------------------------------------
    # 턴 · 되돌림
    # ------------------------------------------------------------------

    def begin_turn(self) -> int:
        """새 대화 턴을 시작한다(스냅샷 저장 + 턴 번호 증가).

        :meth:`undo_last` 가 되돌릴 지점이 여기서 만들어진다.

        :returns: 새 턴 번호.
        """
        snapshot = self.to_dict(include_checkpoints=False)
        self.checkpoints.append(snapshot)
        if len(self.checkpoints) > MAX_CHECKPOINTS:
            del self.checkpoints[0 : len(self.checkpoints) - MAX_CHECKPOINTS]
        self.turn_index += 1
        self.log("begin_turn", note=f"{self.turn_index}번째 턴을 시작했습니다.")
        return self.turn_index

    def undo_last(self) -> bool:
        """직전 턴을 통째로 취소하고 그 턴 시작 시점으로 되돌린다.

        되돌린 사실 자체는 이력에 ``undo_last`` 로 남으므로 감사 추적이 끊기지 않는다.

        :returns: 되돌릴 스냅샷이 있어 실제로 되돌렸으면 True, 없으면 False.
        """
        if not self.checkpoints:
            self.log("undo_last_noop", note="되돌릴 이전 턴이 없습니다.")
            return False
        snapshot = self.checkpoints.pop()
        restored = SessionState.from_dict(snapshot)
        preserved_history = list(self.history)
        self._adopt(restored)
        # 이력은 되돌리지 않는다. 감사 가능성을 위해 되돌림 이전 기록을 모두 남긴다.
        self.history = preserved_history
        self.log("undo_last", note="직전 턴을 취소하고 이전 상태로 되돌렸습니다.")
        return True

    def revert_to_field(self, field_id: str) -> None:
        """``field_id`` 항목으로 되돌아가고, 그 이후의 진행 기록을 취소한다.

        "아까 동의 안 한다고 한 거 다시 바꿀래" 를 처리하는 연산이다.
        해당 항목과 그 이후 항목의 완료·검증·선택 기록을 모두 지우고
        ``pending_fields`` 에 문서 순서대로 다시 넣는다.

        :param field_id: 되돌아갈 항목 id.
        :returns: ``None``.
        :raises ValueError: 문서에 없는 항목 id 인 경우.
        :raises InvalidTransition: 현재 단계에서 항목 안내로 돌아갈 수 없는 경우
            (예: :attr:`SessionPhase.IDLE`, :attr:`SessionPhase.HUMAN_HANDOFF`).
        """
        if field_id not in self.field_order:
            raise ValueError(
                f"문서에 존재하지 않는 항목으로는 되돌릴 수 없습니다: {field_id}"
            )
        index = {fid: i for i, fid in enumerate(self.field_order)}
        pivot = index[field_id]
        reverted = [fid for fid in self.field_order[pivot:]]
        reverted_set = set(reverted)

        self.completed_fields = [
            fid for fid in self.completed_fields if fid not in reverted_set
        ]
        self.verified_fields = [
            fid for fid in self.verified_fields if fid not in reverted_set
        ]
        self.skipped_fields = [
            fid for fid in self.skipped_fields if fid not in reverted_set
        ]
        self.selected_options = {
            fid: label
            for fid, label in self.selected_options.items()
            if fid not in reverted_set
        }
        for fid in reverted:
            self.counters.pop(f"fail:{fid}", None)
        # 되돌린 구간을 문서 순서대로 다시 대기열에 넣는다(중복 없이).
        kept_pending = [fid for fid in self.pending_fields if fid not in reverted_set]
        self.pending_fields = sorted(
            kept_pending + reverted, key=lambda fid: index[fid]
        )
        self.current_field_id = field_id
        self.transition_to(
            SessionPhase.ANNOUNCE_FIELD,
            action="revert_to_field",
            note=f"{field_id} 항목 이후의 진행 기록을 취소하고 되돌아갔습니다.",
        )

    # ------------------------------------------------------------------
    # 항목 진행
    # ------------------------------------------------------------------

    def set_current(self, field_id: str | None) -> None:
        """현재 안내 항목 포인터를 옮긴다(단계는 바꾸지 않는다).

        :param field_id: 새 현재 항목 id. ``None`` 이면 포인터를 비운다.
        :returns: ``None``.
        :raises ValueError: 문서에 없는 항목 id 인 경우.
        """
        if field_id is not None and field_id not in self.field_order:
            raise ValueError(f"문서에 존재하지 않는 항목입니다: {field_id}")
        self.current_field_id = field_id
        self.log("set_current", note="현재 안내 항목을 변경했습니다.")

    def select_option(self, field_id: str, label: str) -> None:
        """선택형 항목의 선택 결과를 기록한다.

        :param field_id: 대상 항목 id.
        :param label: 선택한 선택지 라벨.
        :returns: ``None``.
        """
        self.selected_options[field_id] = label
        self.log(
            "select_option",
            note=f"'{label}' 을(를) 선택했습니다.",
            field_id=field_id,
        )

    def mark_verified(self, field_id: str) -> None:
        """항목이 Verify 를 통과했음을 기록한다.

        :param field_id: 대상 항목 id.
        :returns: ``None``.
        """
        if field_id not in self.verified_fields:
            self.verified_fields.append(field_id)
        self.log("verified", note="기입이 확인되었습니다.", field_id=field_id)

    def complete_field(self, field_id: str) -> None:
        """항목을 완료 처리하고 대기열에서 제거한다.

        :param field_id: 대상 항목 id.
        :returns: ``None``.
        """
        if field_id in self.pending_fields:
            self.pending_fields.remove(field_id)
        if field_id in self.skipped_fields:
            self.skipped_fields.remove(field_id)
        if field_id not in self.completed_fields:
            self.completed_fields.append(field_id)
        self.log("complete_field", note="항목을 완료 처리했습니다.", field_id=field_id)

    def skip_field(self, field_id: str) -> None:
        """항목을 건너뛴 것으로 표시한다(완료가 아니다).

        :param field_id: 대상 항목 id.
        :returns: ``None``.
        """
        if field_id in self.completed_fields:
            return
        if field_id in self.pending_fields:
            self.pending_fields.remove(field_id)
        if field_id not in self.skipped_fields:
            self.skipped_fields.append(field_id)
        self.log("skip_field", note="항목을 건너뛰었습니다.", field_id=field_id)

    def next_pending(self, after: str | None = None) -> str | None:
        """``after`` 다음에 처리할 대기 항목 id 를 문서 순서로 찾는다.

        :param after: 기준 항목 id. ``None`` 이면 현재 항목을 기준으로 한다.
        :returns: 다음 대기 항목 id. 남은 대기 항목이 없으면 ``None``.
        """
        pending = set(self.pending_fields)
        if not pending:
            return None
        anchor = self.current_field_id if after is None else after
        start = 0
        if anchor is not None and anchor in self.field_order:
            start = self.field_order.index(anchor) + 1
        for fid in self.field_order[start:]:
            if fid in pending:
                return fid
        # 앞쪽에 남은 항목이 있으면(되돌림·건너뜀 이후) 문서 순서로 되돌아간다.
        for fid in self.field_order:
            if fid in pending and fid != anchor:
                return fid
        return None

    def previous_field(self) -> str | None:
        """문서 순서상 현재 항목의 바로 앞 항목 id 를 반환한다.

        :returns: 앞 항목 id. 첫 항목이거나 현재 항목이 없으면 ``None``.
        """
        if self.current_field_id is None:
            return None
        if self.current_field_id not in self.field_order:
            return None
        index = self.field_order.index(self.current_field_id)
        return None if index == 0 else self.field_order[index - 1]

    def progress(self) -> tuple[int, int, list[str]]:
        """진행 상황을 요약한다.

        :returns: ``(완료 항목 수, 전체 항목 수, 남은 필수 항목 id 목록)``.
            남은 필수 항목은 문서 순서를 유지한다.
        """
        done = set(self.completed_fields)
        remaining = [fid for fid in self.required_field_ids if fid not in done]
        return (len(self.completed_fields), len(self.field_order), remaining)

    def is_complete(self) -> bool:
        """더 다룰 대기 항목이 없으면 True.

        :returns: ``pending_fields`` 가 비어 있는지 여부.
        """
        return not self.pending_fields

    # ------------------------------------------------------------------
    # 카운터
    # ------------------------------------------------------------------

    def bump(self, name: str) -> int:
        """카운터를 1 올리고 올린 뒤 값을 반환한다.

        :param name: 카운터 이름(예: ``"unknown_streak"``, ``"fail:consent_01"``).
        :returns: 증가 후 값.
        """
        value = self.counters.get(name, 0) + 1
        self.counters[name] = value
        return value

    def reset_counter(self, name: str) -> None:
        """카운터를 지운다.

        :param name: 카운터 이름.
        :returns: ``None``.
        """
        self.counters.pop(name, None)

    # ------------------------------------------------------------------
    # 직렬화
    # ------------------------------------------------------------------

    def _adopt(self, other: "SessionState") -> None:
        """``other`` 의 모든 필드를 이 인스턴스에 그대로 옮겨 담는다(내부용).

        :param other: 복사 원본.
        :returns: ``None``.
        """
        self.document_id = other.document_id
        self.phase = other.phase
        self.current_field_id = other.current_field_id
        self.field_order = list(other.field_order)
        self.required_field_ids = list(other.required_field_ids)
        self.completed_fields = list(other.completed_fields)
        self.pending_fields = list(other.pending_fields)
        self.skipped_fields = list(other.skipped_fields)
        self.selected_options = dict(other.selected_options)
        self.verified_fields = list(other.verified_fields)
        self.turn_index = other.turn_index
        self.handoff_reason = other.handoff_reason
        self.counters = dict(other.counters)
        self.history = list(other.history)

    def to_dict(self, *, include_checkpoints: bool = True) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다.

        :param include_checkpoints: 되돌림 스냅샷 스택을 포함할지 여부.
            ``False`` 는 스냅샷 자체를 만들 때 재귀를 끊기 위해 쓴다.
        :returns: dict.
        """
        data: dict[str, Any] = {
            "document_id": self.document_id,
            "phase": self.phase.value,
            "current_field_id": self.current_field_id,
            "field_order": list(self.field_order),
            "required_field_ids": list(self.required_field_ids),
            "completed_fields": list(self.completed_fields),
            "pending_fields": list(self.pending_fields),
            "skipped_fields": list(self.skipped_fields),
            "selected_options": dict(self.selected_options),
            "verified_fields": list(self.verified_fields),
            "turn_index": self.turn_index,
            "handoff_reason": self.handoff_reason,
            "counters": dict(self.counters),
            "history": [entry.to_dict() for entry in self.history],
        }
        if include_checkpoints:
            data["checkpoints"] = [dict(item) for item in self.checkpoints]
        return data

    def to_json(self, *, indent: int | None = 2) -> str:
        """세션 상태 JSON 문자열을 반환한다(한글은 이스케이프하지 않는다).

        :param indent: JSON 들여쓰기. ``None`` 이면 한 줄로 직렬화한다.
        :returns: UTF-8 그대로 읽히는 JSON 문자열.
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionState":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다.

        :param data: :meth:`to_dict` 가 만든 매핑.
        :returns: :class:`SessionState`.
        :raises ValueError: 필수 키 누락 또는 미정의 단계 값.
        """
        if "document_id" not in data:
            raise ValueError("SessionState.from_dict 에 필수 키가 없습니다: document_id")
        current = data.get("current_field_id")
        reason = data.get("handoff_reason")
        return cls(
            document_id=str(data["document_id"]),
            phase=_as_phase(data.get("phase", SessionPhase.IDLE.value)),
            current_field_id=None if current is None else str(current),
            field_order=[str(item) for item in data.get("field_order", ())],
            required_field_ids=[
                str(item) for item in data.get("required_field_ids", ())
            ],
            completed_fields=[str(item) for item in data.get("completed_fields", ())],
            pending_fields=[str(item) for item in data.get("pending_fields", ())],
            skipped_fields=[str(item) for item in data.get("skipped_fields", ())],
            selected_options={
                str(key): str(value)
                for key, value in dict(data.get("selected_options", {})).items()
            },
            verified_fields=[str(item) for item in data.get("verified_fields", ())],
            turn_index=int(data.get("turn_index", 0)),
            handoff_reason=None if reason is None else str(reason),
            counters={
                str(key): int(value)
                for key, value in dict(data.get("counters", {})).items()
            },
            history=[HistoryEntry.from_dict(item) for item in data.get("history", ())],
            checkpoints=[dict(item) for item in data.get("checkpoints", ())],
        )

    @classmethod
    def from_json(cls, text: str) -> "SessionState":
        """세션 상태 JSON 문자열로부터 인스턴스를 복원한다.

        :param text: :meth:`to_json` 이 만든 JSON 문자열.
        :returns: :class:`SessionState`.
        :raises ValueError: JSON 파싱 실패 또는 스키마 불일치.
        """
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"세션 상태 JSON 파싱에 실패했습니다: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("세션 상태 JSON 최상위는 객체여야 합니다.")
        return cls.from_dict(payload)
