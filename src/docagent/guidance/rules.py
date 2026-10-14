"""문서·버전·페이지를 제한한 결정론적 검색. 안내문 생성/요약 없음."""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FALLBACK = "해당 작성 영역을 정확하게 확인하기 어렵습니다. 다시 촬영하거나 확인해주세요."
CONTENT_KEYS = (
    "document_id", "document_version", "page", "field_id", "field_type",
    "surrounding_text", "keywords", "forbidden_keywords", "instruction",
)


def content_digest(entry: dict[str, Any]) -> str:
    """검토 이후 안내문/매칭 조건 변경을 탐지한다. 전자 서명은 아니다."""
    content = {key: entry[key] for key in CONTENT_KEYS}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _confidence(value: Any) -> bool:
    return (type(value) in (float, int) and math.isfinite(value) and 0 <= value <= 1)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


@dataclass(frozen=True)
class Observation:
    document_id: str
    document_version: str
    page: int
    field_type: str
    surrounding_text: str
    document_confidence: float
    detection_confidence: float
    ocr_confidence: float
    identity_verified: bool = False
    needs_review: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(**data)


@dataclass(frozen=True)
class GuidanceResult:
    status: str
    instruction: str = FALLBACK
    document_id: str | None = None
    document_version: str | None = None
    page: int | None = None
    field_id: str | None = None
    content_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class RuleCatalog:
    """신뢰하는 로컬 파일에서만 읽는 안내 목록. OCR 응답을 목록으로 사용하지 않는다."""

    def __init__(self, data: dict[str, Any]) -> None:
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise ValueError("안내 데이터 schema_version 은 1이어야 합니다.")
        if data.get("purpose") not in ("demo", "production"):
            raise ValueError("안내 데이터 purpose 는 demo 또는 production 이어야 합니다.")
        self.purpose = data["purpose"]
        # 호출자가 원본 dict를 수정하여 검토 검사를 우회하지 못하게 복사한다.
        self._entries = json.loads(json.dumps(data.get("fields"), allow_nan=False))
        if not isinstance(self._entries, list):
            raise ValueError("안내 데이터 fields 는 목록이어야 합니다.")
        seen = set()
        for e in self._entries:
            self._validate(e)
            key = (e["document_id"], e["document_version"], e["page"], e["field_id"])
            if key in seen:
                raise ValueError("문서·버전·페이지·항목 ID가 중복되었습니다.")
            seen.add(key)

    @staticmethod
    def _validate(e: dict[str, Any]) -> None:
        if not isinstance(e, dict) or not set(CONTENT_KEYS).issubset(e):
            raise ValueError("안내 항목 필수 정보가 누락되었습니다.")
        for key in ("document_id", "document_version", "field_id", "surrounding_text", "instruction"):
            if not _text(e[key]):
                raise ValueError(f"안내 항목 {key} 는 비어 있지 않은 문자열이어야 합니다.")
        if type(e["page"]) is not int or e["page"] < 1:
            raise ValueError("페이지는 1 이상의 정수여야 합니다.")
        if e["field_type"] not in ("checkbox", "signature"):
            raise ValueError("지원 항목 유형은 checkbox, signature 입니다.")
        for key in ("keywords", "forbidden_keywords"):
            if not isinstance(e[key], list) or any(not _text(x) for x in e[key]):
                raise ValueError(f"{key} 는 문자열 목록이어야 합니다.")
        if not e["keywords"]:
            raise ValueError("필수 키워드가 없는 안내는 등록할 수 없습니다.")
        approval = e.get("approval", {})
        if not isinstance(approval, dict) or approval.get("status") not in ("draft", "approved"):
            raise ValueError("안내 승인 상태를 명시해야 합니다.")
        if approval["status"] == "approved":
            if not all(_text(approval.get(k)) for k in ("reviewed_by", "reviewed_at")):
                raise ValueError("승인에는 검토자와 검토 시각이 필요합니다.")
            if approval.get("content_sha256") != content_digest(e):
                raise ValueError("승인 후 안내 내용이 변경되었습니다. 다시 검토해야 합니다.")

    @classmethod
    def load(cls, path: str | Path) -> RuleCatalog:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("안내 데이터는 JSON 객체여야 합니다.")
        return cls(data)

    def match(self, observed: Observation, *, demo: bool = False,
              minimum_confidence: float = 0.85) -> GuidanceResult:
        if not _confidence(minimum_confidence) or minimum_confidence <= 0:
            raise ValueError("최소 신뢰도는 0보다 크고 1 이하여야 합니다.")
        if self.purpose == "demo" and not demo:
            return GuidanceResult("demo_only")
        if (not all(_text(x) for x in (observed.document_id, observed.document_version))
                or type(observed.page) is not int or observed.page < 1
                or observed.field_type not in ("checkbox", "signature")
                or not isinstance(observed.surrounding_text, str)
                or type(observed.needs_review) is not bool):
            return GuidanceResult("invalid_input")
        if observed.identity_verified is not True:
            return GuidanceResult("unverified_document")
        confidences = (observed.document_confidence, observed.detection_confidence, observed.ocr_confidence)
        if any(not _confidence(c) or c < minimum_confidence for c in confidences):
            return GuidanceResult("low_confidence")
        if observed.needs_review:
            return GuidanceResult("needs_review")
        entries = [e for e in self._entries if (
            e["document_id"], e["document_version"], e["page"]
        ) == (observed.document_id, observed.document_version, observed.page)]
        if not entries:
            return GuidanceResult("unsupported_document")
        text = normalize_text(observed.surrounding_text)
        candidates = [e for e in entries if e["field_type"] == observed.field_type
                      and all(normalize_text(k) in text for k in e["keywords"])
                      and not any(normalize_text(k) in text for k in e["forbidden_keywords"])]
        if not candidates:
            return GuidanceResult("no_match")
        if len(candidates) != 1:
            return GuidanceResult("ambiguous")
        e = candidates[0]
        if e["approval"]["status"] != "approved":
            return GuidanceResult("unapproved")
        return GuidanceResult("matched", e["instruction"], e["document_id"],
                              e["document_version"], e["page"], e["field_id"], content_digest(e))
