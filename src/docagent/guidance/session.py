"""검토된 분석 결과와 순서에 묶인 고정 문구 안내 세션."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from docagent.guidance.analysis import observation_from_analysis
from docagent.guidance.rules import FALLBACK, RuleCatalog


def digest(data: Any) -> str:
    """JSON 내용 변경 감지값. 검토자 신원 인증을 대신하지 않는다."""
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def context_digest(context: dict[str, Any]) -> str:
    return digest({k: v for k, v in context.items() if k != "approval"})


def validate_context(context: dict[str, Any], report: dict[str, Any] | None = None) -> None:
    """사람이 확인한 문서 정보·분석 해시·항목 순서만 수락한다."""
    from docagent.guidance.rules import _confidence
    if (not isinstance(context, dict) or type(context.get("schema_version")) is not int or context["schema_version"] != 1
            or context.get("identity_verified") is not True
            or not _confidence(context.get("confidence")) or context["confidence"] < .85
            or any(not isinstance(context.get(k), str) or not context[k].strip()
                   for k in ("document_id", "document_version", "analysis_sha256"))
            or type(context.get("page")) is not int or context["page"] < 1):
        raise ValueError("문서 식별 정보를 검토해야 합니다.")
    order = context.get("field_order")
    if (not isinstance(order, list) or not order
            or any(not isinstance(x, str) or not x for x in order) or len(set(order)) != len(order)):
        raise ValueError("중복 없는 안내 순서가 필요합니다.")
    approval = context.get("approval", {})
    if (not isinstance(approval, dict) or approval.get("status") != "approved"
            or any(not isinstance(approval.get(k), str) or not approval[k].strip()
                   for k in ("reviewed_by", "reviewed_at"))
            or approval.get("content_sha256") != context_digest(context)):
        raise ValueError("문서 정보와 안내 순서를 다시 검토해야 합니다.")
    if report is not None:
        if context["analysis_sha256"] != digest(report):
            raise ValueError("검토한 분석 결과가 변경되었습니다.")
        fields = [f["id"] for f in report["fields"] if f["type"] in ("choice", "checkbox", "signature")]
        if len(fields) != len(order) or set(fields) != set(order):
            raise ValueError("검토한 순서와 현재 작성 영역이 다릅니다.")


class GuidanceSession:
    """모든 대상 항목을 검증한 후에만 이동과 재생을 허용한다."""
    def __init__(self, catalog: RuleCatalog, report: dict[str, Any], context: dict[str, Any],
                 *, demo: bool = False, backend: Any = None):
        validate_context(context, report)
        self.results = [catalog.match(observation_from_analysis(report, context, fid), demo=demo)
                        for fid in context["field_order"]]
        self.reason = next((r.status for r in self.results if r.status != "matched"), None)
        identities = [r.field_id for r in self.results]
        if self.reason is None and len(set(identities)) != len(identities):
            self.reason = "ambiguous_field_mapping"
        self.backend = backend
        self.index = 0

    @property
    def ready(self) -> bool:
        return bool(self.results) and self.reason is None

    def current(self) -> dict[str, Any]:
        if not self.ready:
            return {"status": "blocked", "reason": self.reason or "no_fields", "instruction": FALLBACK}
        return {**self.results[self.index].to_dict(), "position": self.index + 1,
                "total": len(self.results)}

    def command(self, command: str) -> dict[str, Any]:
        """n/p/r/s/q. 경계에서는 현재 항목 유지; 거절 상태는 재생하지 않는다."""
        if command not in ("n", "p", "r", "s", "q"):
            raise ValueError("지원하지 않는 안내 제어입니다.")
        if self.backend is not None:
            self.backend.stop()
        if not self.ready or command in ("s", "q"):
            return self.current()
        if command == "n":
            self.index = min(len(self.results) - 1, self.index + 1)
        elif command == "p":
            self.index = max(0, self.index - 1)
        if self.backend is not None:
            self.backend.play(self.results[self.index].instruction)
        return self.current()
