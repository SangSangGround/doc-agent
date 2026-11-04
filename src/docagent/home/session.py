"""로컬 문서 작업 상태. 사용자 선택과 이미지 관측을 서로 독립적으로 기록한다."""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from docagent.home.templates import HomeTemplate


MESSAGES = {
    "matched": "확인하신 선택 내용과 표시된 칸이 일치합니다.",
    "mismatch": "확인하신 선택 내용과 표시된 칸이 다릅니다. 다시 확인해주세요.",
    "unmarked": "이 항목에서 표시를 확인하지 못했습니다.",
    "multiple": "허용된 수보다 많은 칸에 표시가 확인되었습니다. 다시 확인해주세요.",
    "uncertain": "표시 여부를 정확하게 확인하기 어렵습니다. 다시 촬영하거나 확인해주세요.",
    "unconfirmed": "선택 내용을 먼저 확인해주세요.",
}


class HomeSession:
    """검토된 원문을 그대로 읽고, 명시적으로 확인된 선택만 비교한다."""

    def __init__(self, template: HomeTemplate):
        self.template = template
        self._manifest = copy.deepcopy(template.manifest)
        self._groups = {group["id"]: group for group in self._manifest["groups"]}
        self.index = 0
        self.selections: dict[str, list[str] | None] = {key: None for key in self._groups}
        self.pending: dict[str, list[str] | None] = {key: None for key in self._groups}
        self.preferences = {"speech_rate": 1.0}
        self._verification: dict[str, dict[str, Any]] = {}
        self.invalidate_verification()

    def _group(self, group_id: str) -> dict[str, Any]:
        if not isinstance(group_id, str) or group_id not in self._groups:
            raise ValueError("문서에 없는 항목입니다.")
        return self._groups[group_id]

    def _validate_choices(self, group_id: str, option_ids: Any) -> list[str]:
        group = self._group(group_id)
        allowed = [option["id"] for option in group["options"]]
        if (not isinstance(option_ids, list) or any(not isinstance(x, str) for x in option_ids)
                or len(set(option_ids)) != len(option_ids) or not set(option_ids) <= set(allowed)
                or not group["min_selected"] <= len(option_ids) <= group["max_selected"]):
            raise ValueError("선택 내용이나 선택 가능한 개수를 확인하세요.")
        return [option_id for option_id in allowed if option_id in option_ids]

    def invalidate_verification(self) -> dict[str, Any]:
        """촬영본·선택 내용 변경 시 이전 이미지 판정을 폐기한다."""
        self._verification = {
            key: {"status": "uncertain", "message": MESSAGES["uncertain"], "marked_option_ids": []}
            for key in self._groups
        }
        return self.view()

    def read_page(self) -> str:
        return self._manifest["pages"][0]["source_text"]

    def read_current(self) -> str:
        return self._manifest["groups"][self.index]["source_text"]

    def move(self, direction: str) -> dict[str, Any]:
        if direction == "next":
            self.index = min(len(self._groups) - 1, self.index + 1)
        elif direction == "previous":
            self.index = max(0, self.index - 1)
        elif direction != "replay":
            raise ValueError("지원하지 않는 이동 명령입니다.")
        return self.view()

    def propose_selection(self, group_id: str, option_ids: list[str]) -> dict[str, Any]:
        self.pending[group_id] = self._validate_choices(group_id, option_ids)
        return self.invalidate_verification()

    def confirm_selection(self, group_id: str) -> dict[str, Any]:
        self._group(group_id)
        pending = self.pending[group_id]
        if pending is None:
            raise ValueError("확인할 선택 내용이 없습니다.")
        self.selections[group_id] = list(pending)
        self.pending[group_id] = None
        return self.invalidate_verification()

    def cancel_selection(self, group_id: str) -> dict[str, Any]:
        self._group(group_id)
        self.pending[group_id] = None
        return self.invalidate_verification()

    def set_speech_rate(self, rate: float) -> dict[str, Any]:
        if type(rate) not in (int, float) or not math.isfinite(rate) or not .5 <= rate <= 2:
            raise ValueError("읽기 속도는 0.5~2 사이여야 합니다.")
        self.preferences["speech_rate"] = float(rate)
        return self.view()

    def _validate_observations(self, result: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(result, dict) or result.get("status") not in ("ok", "uncertain"):
            raise ValueError("검증 결과의 상태가 잘못되었습니다.")
        observations = result.get("groups")
        if not isinstance(observations, list) or len(observations) != len(self._groups):
            raise ValueError("모든 항목의 검증 결과가 필요합니다.")
        indexed: dict[str, dict[str, Any]] = {}
        for observed in observations:
            if not isinstance(observed, dict):
                raise ValueError("검증 항목 형식이 잘못되었습니다.")
            group_id = observed.get("id")
            group = self._group(group_id)
            if group_id in indexed:
                raise ValueError("검증 항목이 중복되었습니다.")
            allowed = {option["id"] for option in group["options"]}
            marked = observed.get("marked_option_ids")
            state = observed.get("state")
            if (not isinstance(marked, list) or any(not isinstance(x, str) for x in marked)
                    or len(set(marked)) != len(marked) or not set(marked) <= allowed
                    or state not in ("single", "unmarked", "multiple", "uncertain")):
                raise ValueError("관측한 선택지 목록이 잘못되었습니다.")
            expected_state = "unmarked" if not marked else "single" if len(marked) == 1 else "multiple"
            if state != "uncertain" and state != expected_state:
                raise ValueError("관측 상태와 표시된 칸 수가 다릅니다.")
            options = observed.get("options")
            if not isinstance(options, list) or len(options) != len(allowed):
                raise ValueError("모든 선택지의 관측 결과가 필요합니다.")
            seen: set[str] = set()
            for option in options:
                if (not isinstance(option, dict) or not isinstance(option.get("id"), str)
                        or option["id"] not in allowed or option["id"] in seen
                        or option.get("state") not in ("marked", "unmarked", "uncertain")):
                    raise ValueError("선택지 관측 결과가 잘못되었습니다.")
                seen.add(option["id"])
                if state != "uncertain":
                    wanted = "marked" if option["id"] in marked else "unmarked"
                    if option["state"] != wanted:
                        raise ValueError("선택지 관측과 항목 판정이 다릅니다.")
            indexed[group_id] = observed
        return indexed

    def apply_verification(self, result: dict[str, Any]) -> dict[str, Any]:
        """관측 결과가 불완전하거나 새 입력이 실패하면 기존 완료 판정도 폐기한다."""
        self.invalidate_verification()
        observations = self._validate_observations(result)
        for group_id, group in self._groups.items():
            observed = observations[group_id]
            marked = observed["marked_option_ids"]
            selected = self.selections[group_id]
            if result["status"] != "ok" or observed["state"] == "uncertain":
                status = "uncertain"
            elif selected is None or self.pending[group_id] is not None:
                status = "unconfirmed"
            elif len(marked) > group["max_selected"]:
                status = "multiple"
            elif not marked and group["min_selected"] > 0:
                status = "unmarked"
            elif set(selected) != set(marked) or len(marked) < group["min_selected"]:
                status = "mismatch"
            else:
                status = "matched"
            self._verification[group_id] = {
                "status": status, "message": MESSAGES[status], "marked_option_ids": list(marked),
            }
        return self.view()

    def view(self) -> dict[str, Any]:
        groups = []
        for group_id, group in self._groups.items():
            groups.append({**copy.deepcopy(group),
                           "selected_option_ids": copy.deepcopy(self.selections[group_id]),
                           "pending_option_ids": copy.deepcopy(self.pending[group_id]),
                           "confirmed": self.selections[group_id] is not None and self.pending[group_id] is None,
                           "verification": copy.deepcopy(self._verification[group_id])})
        complete = bool(groups) and all(group["confirmed"] and group["verification"]["status"] == "matched"
                                        for group in groups)
        return {"document_id": self._manifest["document_id"],
                "document_version": self._manifest["document_version"],
                "title": self._manifest["title"], "purpose": self._manifest["purpose"],
                "current_index": self.index, "current_group": copy.deepcopy(groups[self.index]),
                "groups": groups, "complete": complete, "preferences": copy.deepcopy(self.preferences)}

    def save(self, path: str | Path) -> None:
        """원문·이미지·검증 증거 없이 선택 ID와 환경설정만 원자적으로 저장한다."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": 1, "template_sha256": self.template.manifest_sha256,
                "current_index": self.index, "selections": self.selections,
                "pending": self.pending, "preferences": self.preferences}
        encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        fd, name = tempfile.mkstemp(prefix=".docagent-session-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    @classmethod
    def load(cls, path: str | Path, template: HomeTemplate) -> HomeSession:
        """문서 버전이 맞는 선택만 복원하고 이미지 검증은 반드시 다시 수행한다."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"schema_version", "template_sha256", "current_index", "selections", "pending", "preferences"}
        if (not isinstance(data, dict) or set(data) != expected
                or type(data.get("schema_version")) is not int or data["schema_version"] != 1
                or data.get("template_sha256") != template.manifest_sha256):
            raise ValueError("저장된 작업이 현재 문서 양식과 다릅니다.")
        session = cls(template)
        if type(data["current_index"]) is not int or not 0 <= data["current_index"] < len(session._groups):
            raise ValueError("저장된 항목 위치가 잘못되었습니다.")
        for name in ("selections", "pending"):
            values = data[name]
            if not isinstance(values, dict) or set(values) != set(session._groups):
                raise ValueError("저장된 선택 항목이 현재 문서와 다릅니다.")
            for group_id, value in values.items():
                if value is not None:
                    values[group_id] = session._validate_choices(group_id, value)
        preferences = data["preferences"]
        if not isinstance(preferences, dict) or set(preferences) != {"speech_rate"}:
            raise ValueError("저장된 읽기 설정이 잘못되었습니다.")
        session.set_speech_rate(preferences["speech_rate"])
        session.index = data["current_index"]
        session.selections = copy.deepcopy(data["selections"])
        session.pending = copy.deepcopy(data["pending"])
        return session
