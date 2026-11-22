"""로컬 OCR 좌표로 만드는 개인정보 마스킹 초안. 외부 전송 승인이 아니다."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from docagent.contracts import A4_PAGE_SIZE_MM, OcrWord
from docagent.pii.detectors import RegexPiiDetector
from docagent.pii.image_mask import image_digest, mask_boxes
from docagent.vision.ocr import TesseractOcr

_WARNINGS = ["limited_pattern_coverage", "unlabelled_names_not_verified", "handwriting_not_verified"]


def _base() -> dict[str, Any]:
    return {
        "schema_version": 1, "status": "blocked", "reason": "invalid_image", "boxes": [],
        "image_sha256": None, "image_shape": None, "coverage_verified": False,
        "external_transfer_allowed": False, "requires_review": True,
        "warnings": list(_WARNINGS), "candidate_count": 0,
        "instruction": "개인정보 마스킹 초안을 만들지 못했습니다. 외부 전송은 허용되지 않습니다.",
    }


def _validate_image(image: Any) -> None:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim not in (2, 3) or min(image.shape[:2], default=0) < 8
            or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("마스킹 입력은 uint8 회색조 또는 BGR 이미지여야 합니다.")


def _word_rect(word: OcrWord, shape: tuple, page_size: tuple) -> list[int]:
    box = word.box_mm
    values = (box.x_mm, box.y_mm, box.w_mm, box.h_mm, word.confidence)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("OCR 좌표가 유효하지 않습니다.")
    if (not 0 <= word.confidence <= 1 or box.w_mm <= 0 or box.h_mm <= 0
            or box.x_mm < 0 or box.y_mm < 0
            or box.right_mm > page_size[0] or box.bottom_mm > page_size[1]):
        raise ValueError("OCR 좌표가 이미지 범위를 벗어났습니다.")
    h, w = shape[:2]
    x0 = max(0, math.floor(box.x_mm * w / page_size[0]))
    y0 = max(0, math.floor(box.y_mm * h / page_size[1]))
    x1 = min(w, math.ceil(box.right_mm * w / page_size[0]))
    y1 = min(h, math.ceil(box.bottom_mm * h / page_size[1]))
    return [x0, y0, x1 - x0, y1 - y0]


def propose_redactions(image: Any, *, ocr: Any = None) -> dict[str, Any]:
    """로컬 OCR/정규식이 찾은 단어를 불투명 마스킹 후보로 반환한다.

    `ocr`는 `read(image)->list[OcrWord]`와 좌표계 `page_size_mm`(기본 A4)를
    갖는 로컬 어댑터다. OCR 원문/개인정보 값은 결과·오류에 넣지 않는다.
    후보 0개도 안전 판정이 아니다. 이름·주소·필기·OCR 누락은 사람 검토가
    필요하며 이 함수는 기존 MaskPlan 승인 절차를 대신하지 않는다.
    """
    result = _base()
    try:
        _validate_image(image)
        result.update(image_sha256=image_digest(image), image_shape=list(image.shape))
    except (TypeError, ValueError):
        return result
    try:
        reader = ocr if ocr is not None else TesseractOcr(config="--psm 11", min_confidence=0.0)
        page_size = getattr(reader, "page_size_mm", A4_PAGE_SIZE_MM)
        if (not isinstance(page_size, (list, tuple)) or len(page_size) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in page_size)):
            raise ValueError("OCR 페이지 좌표계를 확인해야 합니다.")
        words = list(reader.read(image))
        segments: list[tuple[int, int, list[int]]] = []
        chunks: list[str] = []
        offset = 0
        low_confidence = False
        for word in words:
            if not isinstance(word, OcrWord) or not isinstance(word.text, str):
                raise ValueError("OCR 단어 형식이 유효하지 않습니다.")
            if not word.text.strip():
                continue
            rect = _word_rect(word, image.shape, page_size)
            chunks.append(word.text)
            segments.append((offset, offset + len(word.text), rect))
            offset += len(word.text) + 1
            low_confidence |= word.confidence < .5
        if not chunks:
            result.update(reason="ocr_no_text", warnings=result["warnings"] + ["ocr_no_text"])
            return result
        spans = RegexPiiDetector().detect(" ".join(chunks))
        matched: dict[tuple[int, ...], set[str]] = {}
        for span in spans:
            covered = False
            for start, end, rect in segments:
                if start < span.end and end > span.start:
                    matched.setdefault(tuple(rect), set()).add(span.pii_type)
                    covered = True
            if not covered:
                raise ValueError("개인정보 구간을 이미지 좌표로 연결할 수 없습니다.")
        boxes = [{"box_px": list(rect), "types": sorted(types)} for rect, types in matched.items()]
        result.update(
            status="draft", reason="human_review_required", boxes=boxes,
            candidate_count=len(spans), word_count=len(chunks), padding_px=3,
            instruction="개인정보 마스킹 후보를 만들었습니다. 자동 검사만으로 모든 개인정보가 가려졌는지 확인할 수 없습니다.",
        )
        if not boxes:
            result["warnings"].append("no_candidates_does_not_mean_safe")
            result["instruction"] = "자동 검사에서 마스킹 후보를 찾지 못했습니다. 개인정보가 없다는 뜻은 아니며 추가 확인이 필요합니다."
        if low_confidence:
            result["warnings"].append("low_ocr_confidence")
        return result
    except Exception:
        # 어댑터/탐지기 오류 문자열에 OCR 원문이 포함될 수 있어 상세를 반환하지 않는다.
        result.update(reason="ocr_or_detection_failed", boxes=[], candidate_count=0)
        return result


def render_redaction_preview(image: np.ndarray, proposal: dict[str, Any]) -> np.ndarray:
    """같은 이미지의 초안만 미리보기로 가린다. 원본을 수정하지 않는다."""
    _validate_image(image)
    if (not isinstance(proposal, dict) or proposal.get("status") != "draft"
            or proposal.get("coverage_verified") is not False
            or proposal.get("external_transfer_allowed") is not False
            or proposal.get("image_shape") != list(image.shape)
            or proposal.get("image_sha256") != image_digest(image)):
        raise ValueError("현재 이미지에 대한 미승인 마스킹 초안이 필요합니다.")
    rows = proposal.get("boxes")
    if not isinstance(rows, list):
        raise ValueError("마스킹 좌표 목록이 필요합니다.")
    boxes = []
    for row in rows:
        box = row.get("box_px") if isinstance(row, dict) else None
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) is not int for v in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= 0 or box[3] <= 0
                or box[0] + box[2] > image.shape[1] or box[1] + box[3] > image.shape[0]):
            raise ValueError("이미지 내부의 마스킹 좌표가 필요합니다.")
        boxes.append(box)
    return mask_boxes(image, boxes, padding_px=3) if boxes else image.copy()
