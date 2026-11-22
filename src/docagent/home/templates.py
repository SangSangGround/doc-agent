"""검토된 단일 페이지 양식 묶음. 승인 지문은 변경 감지용이며 신원 인증은 아니다."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def template_digest(manifest: dict[str, Any]) -> str:
    """승인 메타데이터를 제외한 양식의 변경 감지값."""
    return _digest({key: value for key, value in manifest.items() if key != "approval"})


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_manifest(data: dict[str, Any], *, demo: bool = False) -> None:
    """좌표·원문·출처·승인 및 선택지 수를 검증한다. 원문 공백을 수정하지 않는다."""
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("지원하지 않는 문서 양식입니다.")
    if data.get("purpose") not in ("demo", "production"):
        raise ValueError("문서 사용 목적이 필요합니다.")
    if data["purpose"] == "demo" and demo is not True:
        raise ValueError("합성 시연 양식은 데모 모드에서만 사용할 수 있습니다.")
    for key in ("document_id", "document_version", "title", "reference_image"):
        if not _text(data.get(key)):
            raise ValueError("문서 식별자·버전·제목·기준 이미지가 필요합니다.")
    if not _hash(data.get("reference_sha256")):
        raise ValueError("기준 이미지의 SHA-256이 필요합니다.")
    size = data.get("page_size_mm")
    if not isinstance(size, list) or len(size) != 2 or any(not _number(x) or x <= 0 for x in size):
        raise ValueError("문서 크기가 잘못되었습니다.")
    source = data.get("source")
    if not isinstance(source, dict) or not _text(source.get("reference")):
        raise ValueError("원문 출처가 필요합니다.")
    pages = data.get("pages")
    if (not isinstance(pages, list) or len(pages) != 1 or not isinstance(pages[0], dict)
            or type(pages[0].get("page")) is not int or pages[0]["page"] != 1
            or not _text(pages[0].get("source_text"))):
        raise ValueError("첫 번째 페이지의 검토된 전체 원문이 필요합니다.")
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("최소 하나의 선택 항목이 필요합니다.")
    ids: set[str] = set()
    boxes: list[list[float]] = []
    for group in groups:
        if (not isinstance(group, dict) or not _text(group.get("id")) or group["id"] in ids
                or not _text(group.get("source_text"))):
            raise ValueError("선택 항목의 식별자와 원문을 확인하세요.")
        ids.add(group["id"])
        if group["source_text"] not in pages[0]["source_text"]:
            raise ValueError("항목 원문은 검토된 페이지 원문의 정확한 일부여야 합니다.")
        options = group.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("선택지가 필요합니다.")
        minimum, maximum = group.get("min_selected"), group.get("max_selected")
        if (type(minimum) is not int or type(maximum) is not int
                or not 0 <= minimum <= maximum <= len(options) or maximum == 0):
            raise ValueError("선택 가능한 항목 수가 잘못되었습니다.")
        option_ids: set[str] = set()
        for option in options:
            if (not isinstance(option, dict) or not _text(option.get("id")) or option["id"] in option_ids
                    or not _text(option.get("label"))):
                raise ValueError("선택지 식별자와 원문 표시를 확인하세요.")
            option_ids.add(option["id"])
            box = option.get("box_mm")
            if (not isinstance(box, list) or len(box) != 4 or any(not _number(x) for x in box)
                    or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                    or box[0] + box[2] > size[0] or box[1] + box[3] > size[1]):
                raise ValueError("체크박스 좌표가 문서 범위를 벗어났습니다.")
            for previous in boxes:
                if (max(box[0], previous[0]) < min(box[0] + box[2], previous[0] + previous[2])
                        and max(box[1], previous[1]) < min(box[1] + box[3], previous[1] + previous[3])):
                    raise ValueError("체크박스 좌표가 서로 겹칩니다.")
            boxes.append(box)
    approval = data.get("approval")
    signatures = data.get("signatures", [])
    if not isinstance(signatures, list):
        raise ValueError("서명란 목록을 확인하세요.")
    signature_ids: set[str] = set()
    for signature in signatures:
        if (not isinstance(signature, dict) or not _text(signature.get("id"))
                or signature["id"] in signature_ids
                or signature.get("role") not in ("applicant", "representative", "staff", "unknown")):
            raise ValueError("서명란 식별자와 작성 주체를 확인하세요.")
        signature_ids.add(signature["id"])
        source_text = signature.get("source_text")
        if not _text(source_text) or source_text not in pages[0]["source_text"]:
            raise ValueError("서명란 근거는 페이지 원문에 있는 문장이어야 합니다.")
        role_words = {"applicant": ("신청인", "신청자"),
                      "representative": ("대리인", "법정대리인"),
                      "staff": ("담당자", "직원"), "unknown": ("",)}
        if ("서명" not in source_text
                or not any(word in source_text for word in role_words[signature["role"]])):
            raise ValueError("서명과 작성 주체를 뒷받침하는 원문이 필요합니다.")
        box = signature.get("box_mm")
        if (not isinstance(box, list) or len(box) != 4 or any(not _number(x) for x in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                or box[0] + box[2] > size[0] or box[1] + box[3] > size[1]):
            raise ValueError("서명란 좌표가 문서 범위를 벗어났습니다.")
        related = signature.get("before_signing_group_ids", [])
        if (not isinstance(related, list) or any(not isinstance(item, str) or item not in ids for item in related)
                or len(set(related)) != len(related)):
            raise ValueError("서명 전에 읽을 항목이 실제 원문 항목과 연결되어야 합니다.")
    if not isinstance(approval, dict) or approval.get("status") not in ("draft", "approved"):
        raise ValueError("문서 검토 상태가 필요합니다.")
    if data["purpose"] == "production" and approval["status"] != "approved":
        raise ValueError("실제 문서는 원문과 위치를 검토하고 승인해야 합니다.")
    if approval["status"] == "approved":
        if (not _text(approval.get("reviewed_by")) or not _text(approval.get("reviewed_at"))
                or approval.get("content_sha256") != template_digest(data)):
            raise ValueError("문서 내용이 승인 후 변경되었거나 승인 기록이 없습니다.")
    # JSON 직렬화 불가능 값·비유한 메타데이터도 허용하지 않는다.
    _digest(data)


@dataclass(frozen=True)
class HomeTemplate:
    manifest: dict[str, Any]
    reference: np.ndarray
    reference_path: Path
    manifest_sha256: str


def load_template(path: str | Path, *, demo: bool = False) -> HomeTemplate:
    """동일 디렉터리 하위의 기준 이미지만 읽고 원문·이미지 지문을 확인한다."""
    manifest_path = Path(path).resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(data, demo=demo)
    relative = Path(data["reference_image"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("기준 이미지는 문서 묶음 안의 상대 경로여야 합니다.")
    reference_path = (manifest_path.parent / relative).resolve()
    try:
        reference_path.relative_to(manifest_path.parent)
    except ValueError:
        raise ValueError("기준 이미지가 문서 묶음 밖을 가리킵니다.") from None
    blob = reference_path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != data["reference_sha256"]:
        raise ValueError("검토한 기준 이미지가 변경되었습니다.")
    reference = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
    if reference is None or min(reference.shape[:2]) < 8:
        raise ValueError("기준 이미지를 읽을 수 없습니다.")
    return HomeTemplate(data, reference, reference_path, _digest(data))
