"""검토된 이미지별 계획으로 불투명 마스킹. 자동 PII 검출기는 아니다."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from docagent.errors import PiiEgressBlocked


def _validate_image(image: np.ndarray) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] not in (3, 4))):
        raise PiiEgressBlocked("마스킹 입력은 비어 있지 않은 uint8 이미지여야 합니다.")


def image_digest(image: np.ndarray) -> str:
    _validate_image(image)
    h = hashlib.sha256(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


def plan_digest(plan: dict[str, Any]) -> str:
    """검토자가 확인한 픽셀·좌표·여백에 대한 변경 감지값."""
    keys = ("schema_version", "coordinate_system", "image_sha256", "image_shape", "boxes", "padding_px")
    body = {key: plan[key] for key in keys}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def draft_plan(image: np.ndarray) -> dict[str, Any]:
    return dict(schema_version=1, coordinate_system="normalized_image_px",
                image_sha256=image_digest(image), image_shape=list(image.shape), boxes=[], padding_px=3,
                approval=dict(status="draft", reviewed_by="", reviewed_at="", content_sha256=""))


def mask_boxes(image: np.ndarray, boxes: list, *, padding_px: int = 3) -> np.ndarray:
    """x,y,w,h 픽셀 박스; 바깥은 clamp, 유효하지 않은 영역은 전체 실패."""
    _validate_image(image)
    if type(padding_px) is not int or not 0 <= padding_px <= 100:
        raise PiiEgressBlocked("마스킹 여백은 0~100 픽셀 정수여야 합니다.")
    if not isinstance(boxes, list) or not boxes:
        raise PiiEgressBlocked("검토된 마스킹 영역이 필요합니다.")
    h, w = image.shape[:2]
    regions = []
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in box)):
            raise PiiEgressBlocked("잘못된 마스킹 좌표입니다.")
        x, y, bw, bh = box
        if bw <= 0 or bh <= 0 or x >= w or y >= h or x + bw <= 0 or y + bh <= 0:
            raise PiiEgressBlocked("마스킹 영역이 이미지와 겹치지 않습니다.")
        regions.append((max(0, math.floor(x) - padding_px), max(0, math.floor(y) - padding_px),
                        min(w, math.ceil(x + bw) + padding_px), min(h, math.ceil(y + bh) + padding_px)))
    masked = image.copy()
    for x0, y0, x1, y1 in regions:
        masked[y0:y1, x0:x1] = 0
        if masked.ndim == 3 and masked.shape[2] == 4:
            masked[y0:y1, x0:x1, 3] = 255
    return masked


class MaskPlan:
    def __init__(self, data: dict[str, Any]):
        try:
            self._data = json.loads(json.dumps(data, allow_nan=False))
            d = self._data
            approval = d["approval"]
            if (type(d["schema_version"]) is not int or d["schema_version"] != 1
                    or d["coordinate_system"] != "normalized_image_px"
                    or approval["status"] != "approved"
                    or any(not isinstance(approval.get(k), str) or not approval[k].strip()
                           for k in ("reviewed_by", "reviewed_at"))
                    or approval["content_sha256"] != plan_digest(d)):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise PiiEgressBlocked("마스킹 계획이 미승인이거나 검토 후 변경되었습니다.") from None

    @classmethod
    def load(cls, path: str | Path) -> MaskPlan:
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            raise PiiEgressBlocked("마스킹 계획 파일을 읽을 수 없습니다.") from None

    def apply(self, image: np.ndarray) -> np.ndarray:
        digest = image_digest(image)
        if self._data["image_shape"] != list(image.shape) or self._data["image_sha256"] != digest:
            raise PiiEgressBlocked("검토한 이미지와 현재 이미지가 다릅니다. 다시 검토하세요.")
        return mask_boxes(image, self._data["boxes"], padding_px=self._data["padding_px"])


class MaskedDetector:
    """같은 좌표계의 마스킹 사본만 탐지기에 전달. 실패 시 원본 재시도 없음."""
    def __init__(self, detector: Any, plan: MaskPlan):
        self._detector = detector
        self._plan = plan

    def detect(self, image: np.ndarray):
        masked = self._plan.apply(image)
        return self._detector.detect(masked)
