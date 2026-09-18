"""전송 감사 로그 — "개인정보 LLM 전송 0건" KPI 의 증거를 남긴다.

기록 원칙
---------
**원문도, 마스킹 전 값도 저장하지 않는다.** 남기는 것은 다음뿐이다.

* 타임스탬프(주입된 :class:`~docagent.interfaces.Clock` 사용 — 결정론 보장)
* 호출자 식별자와 이벤트 종류
* 텍스트 **길이**
* 탐지된 개인정보 **유형 목록과 건수**
* 차단 여부
* 텍스트의 SHA-256 앞 12자 — 동일 텍스트 재전송 여부는 대조할 수 있으나
  해시로부터 내용을 복원할 수는 없다.

파일 저장은 JSONL(한 줄 = 한 레코드) append 이며 항상 ``encoding="utf-8"`` 이다.

KPI 지표
--------
:meth:`AuditLog.counters` 는 두 개의 안전 지표를 함께 낸다.

``pii_leaked``
    탐지기가 **찾아낸** 개인정보가 마스킹 후에도 남은 건수. 게이트가 이 경우
    전송을 차단하므로 정상 동작 시 항상 0 이다.

``pii_residual_unknown``
    탐지기가 **아무것도 찾지 못한 채** 외부로 나간 문자열에 남아 있는
    유래 불명 숫자열(구분자를 지운 뒤 7자리 이상 연속 숫자) 건수.
    ``pii_leaked`` 만으로는 "탐지기가 애초에 못 본 유출"이 구조적으로 0 으로
    집계되어 KPI 가 자기 자신을 반증할 수 없다. 이 2차 지표는 탐지기의
    자기보고와 **독립적으로** 계산되므로, 규칙이 놓친 개인정보가 나가면
    KPI 가 위반으로 드러난다. (연도·조문 같은 4자리 이하 숫자는 세지 않는다.)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from docagent.contracts import PiiSpan

__all__ = [
    "HASH_PREFIX_LEN",
    "SystemClock",
    "FixedClock",
    "AuditRecord",
    "AuditLog",
    "text_fingerprint",
    "count_unknown_digit_runs",
]

#: 감사 로그에 남기는 SHA-256 접두 길이(문자 수). 내용 복원 불가 길이로 고정한다.
HASH_PREFIX_LEN: Final[int] = 12

#: 한국 표준시(KST).
_KST: Final[timezone] = timezone(timedelta(hours=9))

#: 유래 불명 숫자열로 세기 시작하는 최소 자리수.
#: 연도(4자리)·금액·조문 번호를 위반으로 오인하지 않도록 넉넉히 잡되,
#: 전화번호(10~11)·주민등록번호(13)·계좌번호(11~14)·카드번호(16)는 모두 걸린다.
UNKNOWN_DIGIT_RUN_MIN: Final[int] = 7

#: 숫자 사이 구분자(공백·하이픈·점)를 지우는 정규식. 회피 변형을 흡수한다.
_DIGIT_JOIN_RE: Final[re.Pattern[str]] = re.compile(r"(?<=\d)[\s.\-]+(?=\d)")

#: 유래 불명 숫자열 정규식.
_LONG_DIGITS_RE: Final[re.Pattern[str]] = re.compile(
    rf"\d{{{UNKNOWN_DIGIT_RUN_MIN},}}"
)


def count_unknown_digit_runs(text: str) -> int:
    """문자열에 남은 유래 불명 숫자열 개수를 센다(탐지기와 독립된 2차 검사).

    구분자(공백·줄바꿈·하이픈·점)로 끊긴 숫자를 먼저 이어 붙인 뒤
    :data:`UNKNOWN_DIGIT_RUN_MIN` 자리 이상 연속 숫자를 센다.
    개인정보 탐지 규칙을 전혀 쓰지 않으므로, 규칙이 놓친 유출도 여기서는 잡힌다.

    :param text: 검사 대상 문자열(보통 마스킹이 끝나 외부로 나갈 문자열).
    :returns: 유래 불명 숫자열 개수. 0 이면 위반 없음.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우.
    """
    if not isinstance(text, str):
        raise TypeError(f"2차 검사는 문자열만 받습니다: {type(text).__name__}")
    joined = _DIGIT_JOIN_RE.sub("", text)
    return len(_LONG_DIGITS_RE.findall(joined))


def text_fingerprint(text: str) -> str:
    """텍스트의 SHA-256 앞 :data:`HASH_PREFIX_LEN` 자를 반환한다.

    :param text: 대상 텍스트. 내용은 반환값에서 복원할 수 없다.
    :returns: 소문자 16진 문자열 12자.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]


class SystemClock:
    """실제 시스템 시각을 쓰는 :class:`~docagent.interfaces.Clock` 구현.

    운영 환경 기본값이다. 테스트에서는 :class:`FixedClock` 을 주입한다.
    """

    def now_iso(self) -> str:
        """현재 시각(KST)을 ISO 8601 문자열로 반환한다."""
        return datetime.now(tz=_KST).isoformat(timespec="seconds")


class FixedClock:
    """고정 시각 :class:`~docagent.interfaces.Clock` 구현(테스트 결정론용).

    :param start: 시작 시각 ISO 8601 문자열.
    :param step_seconds: 호출마다 더할 초. 0 이면 항상 같은 값을 돌려준다.
    :raises ValueError: ``start`` 를 ISO 8601 로 해석할 수 없는 경우.
    """

    def __init__(
        self, start: str = "2026-01-01T09:00:00+09:00", step_seconds: int = 0
    ) -> None:
        try:
            self._current = datetime.fromisoformat(start)
        except ValueError as exc:
            raise ValueError(
                f"FixedClock 시작 시각을 ISO 8601 로 해석할 수 없습니다: {start!r}"
            ) from exc
        self._step = timedelta(seconds=step_seconds)

    def now_iso(self) -> str:
        """현재 시각을 반환하고 ``step_seconds`` 만큼 진행시킨다."""
        value = self._current.isoformat(timespec="seconds")
        self._current = self._current + self._step
        return value


@dataclass(frozen=True)
class AuditRecord:
    """감사 로그 한 줄. **개인정보 원문을 담지 않는다.**

    :param timestamp: ISO 8601 시각 문자열.
    :param caller: 호출자 식별자(예: ``"GatedLlmClient.complete"``).
    :param event: 이벤트 종류(``"llm_call"`` / ``"guard"`` / ``"sanitize"`` 등).
    :param text_len: 검사 대상 텍스트 길이(문자 수).
    :param pii_types: 탐지된 개인정보 유형 목록(정렬됨).
    :param pii_count: 탐지된 구간 개수.
    :param blocked: 전송을 차단했는지 여부.
    :param residual_count: 마스킹 **후** 재검사에서 남아 있던 개인정보 개수.
        정상 동작 시 항상 0 이며, 이 값의 합이 KPI ``pii_leaked`` 다.
    :param text_sha256_12: 텍스트 SHA-256 앞 12자.
    :param residual_unknown: 탐지기와 **독립적으로** 센 유래 불명 숫자열 개수
        (:func:`count_unknown_digit_runs`). 이 값의 합이 KPI
        ``pii_residual_unknown`` 이며, 탐지기가 놓친 유출을 드러내는 반증 지표다.
    """

    timestamp: str
    caller: str
    event: str
    text_len: int
    pii_types: tuple[str, ...]
    pii_count: int
    blocked: bool
    residual_count: int
    text_sha256_12: str
    residual_unknown: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pii_types", tuple(self.pii_types))
        if self.text_len < 0:
            raise ValueError(f"AuditRecord.text_len 은 0 이상이어야 합니다: {self.text_len}")
        if self.pii_count < 0:
            raise ValueError(
                f"AuditRecord.pii_count 는 0 이상이어야 합니다: {self.pii_count}"
            )
        if self.residual_count < 0:
            raise ValueError(
                f"AuditRecord.residual_count 는 0 이상이어야 합니다: {self.residual_count}"
            )
        if self.residual_unknown < 0:
            raise ValueError(
                "AuditRecord.residual_unknown 은 0 이상이어야 합니다: "
                f"{self.residual_unknown}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다(원문 미포함)."""
        return {
            "timestamp": self.timestamp,
            "caller": self.caller,
            "event": self.event,
            "text_len": self.text_len,
            "pii_types": list(self.pii_types),
            "pii_count": self.pii_count,
            "blocked": self.blocked,
            "residual_count": self.residual_count,
            "text_sha256_12": self.text_sha256_12,
            "residual_unknown": self.residual_unknown,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditRecord":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다.

        :param data: :meth:`to_dict` 가 만든 매핑.
        :returns: :class:`AuditRecord`.
        :raises ValueError: 필수 키가 없는 경우.
        """
        missing = [
            key
            for key in ("timestamp", "caller", "event", "text_len", "text_sha256_12")
            if key not in data
        ]
        if missing:
            raise ValueError(
                f"AuditRecord.from_dict 에 필수 키가 없습니다: {', '.join(missing)}"
            )
        return cls(
            timestamp=str(data["timestamp"]),
            caller=str(data["caller"]),
            event=str(data["event"]),
            text_len=int(data["text_len"]),
            pii_types=tuple(str(item) for item in data.get("pii_types", ())),
            pii_count=int(data.get("pii_count", 0)),
            blocked=bool(data.get("blocked", False)),
            residual_count=int(data.get("residual_count", 0)),
            text_sha256_12=str(data["text_sha256_12"]),
            residual_unknown=int(data.get("residual_unknown", 0)),
        )


class AuditLog:
    """전송 감사 로그(메모리 + 선택적 JSONL 파일).

    :param path: JSONL 파일 경로. ``None`` 이면 메모리에만 보관한다.
    :param clock: 시각 공급자. ``None`` 이면 :class:`SystemClock`.
        테스트에서는 :class:`FixedClock` 을 주입해 결정론을 확보한다.
    """

    def __init__(self, path: Path | str | None = None, *, clock: Any | None = None) -> None:
        self._path: Path | None = None if path is None else Path(path)
        self._clock = clock if clock is not None else SystemClock()
        self._records: list[AuditRecord] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        """JSONL 파일 경로. 메모리 전용이면 ``None``."""
        return self._path

    def record(
        self,
        *,
        caller: str,
        text: str,
        spans: Sequence[PiiSpan] = (),
        blocked: bool = False,
        residual_count: int = 0,
        event: str = "llm_call",
        egress_text: str | None = None,
    ) -> AuditRecord:
        """감사 레코드를 남긴다.

        ``text`` 는 **길이와 해시 계산에만** 쓰이며 어디에도 저장되지 않는다.

        :param caller: 호출자 식별자.
        :param text: 검사 대상 텍스트(저장되지 않음).
        :param spans: 탐지된 개인정보 구간 목록.
        :param blocked: 전송을 차단했는지 여부.
        :param residual_count: 마스킹 후에도 남아 있던 개인정보 개수.
        :param event: 이벤트 종류.
        :param egress_text: 실제로 외부에 나갈 문자열(마스킹 완료본). 주어지면
            :func:`count_unknown_digit_runs` 로 **탐지기와 독립된 2차 검사**를 수행해
            ``residual_unknown`` 에 기록한다. ``None`` 이면 2차 검사를 하지 않는다
            (이 문자열도 저장되지 않는다).
        :returns: 기록된 :class:`AuditRecord`.
        :raises OSError: JSONL 파일 기록에 실패한 경우(조용히 넘기지 않는다).
        """
        types = tuple(sorted({span.pii_type for span in spans}))
        unknown = 0 if egress_text is None else count_unknown_digit_runs(egress_text)
        record = AuditRecord(
            timestamp=self._clock.now_iso(),
            caller=caller,
            event=event,
            text_len=len(text),
            pii_types=types,
            pii_count=len(spans),
            blocked=blocked,
            residual_count=residual_count,
            text_sha256_12=text_fingerprint(text),
            residual_unknown=unknown,
        )
        self._records.append(record)
        self._append(record)
        return record

    def _append(self, record: AuditRecord) -> None:
        """레코드를 JSONL 파일에 한 줄 덧붙인다.

        :param record: 기록할 레코드.
        :raises OSError: 파일 기록 실패.
        """
        if self._path is None:
            return
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def records(self) -> tuple[AuditRecord, ...]:
        """메모리에 쌓인 레코드 전체를 시간순으로 반환한다."""
        return tuple(self._records)

    def counters(self) -> dict[str, int]:
        """KPI 카운터를 반환한다.

        :returns: ``{"llm_calls", "pii_detected", "blocked", "pii_leaked",
            "pii_residual_unknown"}``.
            ``pii_leaked`` 또는 ``pii_residual_unknown`` 이 0 이 아니면 안전 KPI 위반이다.
        """
        return {
            "llm_calls": sum(1 for r in self._records if r.event == "llm_call"),
            "pii_detected": sum(r.pii_count for r in self._records),
            "blocked": sum(1 for r in self._records if r.blocked),
            "pii_leaked": sum(r.residual_count for r in self._records),
            "pii_residual_unknown": sum(
                r.residual_unknown for r in self._records
            ),
        }

    def read_file_records(self) -> tuple[AuditRecord, ...]:
        """JSONL 파일에 기록된 레코드를 읽어 돌려준다(검증용).

        :returns: 파일에서 복원한 레코드 튜플. 파일이 없으면 빈 튜플.
        :raises ValueError: 파일에 손상된 줄이 있는 경우(조용한 실패 금지).
        """
        if self._path is None or not self._path.exists():
            return ()
        restored: list[AuditRecord] = []
        raw = self._path.read_text(encoding="utf-8")
        for line_no, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"감사 로그 {line_no}번째 줄을 JSON 으로 읽을 수 없습니다: {exc}"
                ) from exc
            restored.append(AuditRecord.from_dict(payload))
        return tuple(restored)

    def __repr__(self) -> str:
        """레코드 수와 파일 경로만 노출한다."""
        target = "메모리" if self._path is None else str(self._path)
        return f"<AuditLog 레코드 {len(self._records)}건, 대상={target}>"
