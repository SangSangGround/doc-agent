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
SOURCE_KEYS = ("reading_mode", "source_text", "source")


def content_digest(entry: dict[str, Any]) -> str:
    """검토 이후 안내문/매칭 조건 변경을 탐지한다. 전자 서명은 아니다."""
    content = {key: entry[key] for key in CONTENT_KEYS}
    # 기존 가상 문서의 해시를 유지하되 원문 읽기 계약을 추가하면 함께 검토한다.
    content.update({key: entry[key] for key in SOURCE_KEYS if key in entry})
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def page_content_digest(page: dict[str, Any]) -> str:
    """원문 전체와 출처 정보를 묶는다. 승인 기록 자체는 해시에서 제외한다."""
    content = {key: value for key, value in page.items() if key != "approval"}
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_text(value: Any) -> bool:
    """낭독 경로에서 잘리거나 UTF-8 변환에 실패하는 원문은 수락하지 않는다."""
    if not _text(value) or "\x00" in value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


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
    reading_mode: str | None = None
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class RuleCatalog:
    """신뢰하는 로컬 파일에서만 읽는 안내 목록. OCR 응답을 목록으로 사용하지 않는다."""

    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise ValueError("안내 데이터는 JSON 객체여야 합니다.")
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
            self._validate_reading(e)
            key = (e["document_id"], e["document_version"], e["page"], e["field_id"])
            if key in seen:
                raise ValueError("문서·버전·페이지·항목 ID가 중복되었습니다.")
            seen.add(key)
        self._pages = json.loads(json.dumps(data.get("pages", []), allow_nan=False))
        if not isinstance(self._pages, list):
            raise ValueError("문서 원문 pages 는 목록이어야 합니다.")
        seen_pages = set()
        for page in self._pages:
            self._validate_page(page)
            key = (page["document_id"], page["document_version"], page["page"])
            if key in seen_pages:
                raise ValueError("문서·버전·페이지 원문이 중복되었습니다.")
            seen_pages.add(key)

    def _validate_reading(self, entry: dict[str, Any]) -> None:
        """실제 문서에는 원문 낭독 계약을 필수로 적용한다."""
        approved_production = self.purpose == "production" and entry["approval"]["status"] == "approved"
        if approved_production or any(key in entry for key in SOURCE_KEYS[:2]):
            if entry.get("reading_mode") != "verbatim" or not _source_text(entry.get("source_text")):
                raise ValueError("실제 문서 안내에는 verbatim 원문 source_text 가 필요합니다.")
            if entry["instruction"] != entry["source_text"]:
                raise ValueError("안내문은 원문 source_text 전체와 정확히 같아야 합니다.")
        if approved_production or "source" in entry:
            self._validate_source(entry.get("source"))

    @staticmethod
    def _validate_source(source: Any) -> None:
        if (not isinstance(source, dict) or not _text(source.get("reference"))
                or not isinstance(source.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", source["sha256"]) is None):
            raise ValueError("원문에는 출처와 원본 파일 SHA-256 이 필요합니다.")

    def _validate_page(self, page: dict[str, Any]) -> None:
        if (not isinstance(page, dict)
                or any(not _text(page.get(key)) for key in ("document_id", "document_version"))
                or not _source_text(page.get("source_text"))
                or type(page.get("page")) is not int or page["page"] < 1
                or page.get("reading_mode") != "verbatim"):
            raise ValueError("페이지 원문의 문서·버전·페이지·source_text 를 확인하세요.")
        self._validate_source(page.get("source"))
        if "page_count" in page and (type(page["page_count"]) is not int
                                     or page["page_count"] < page["page"]):
            raise ValueError("원문의 전체 페이지 수를 확인하세요.")
        approval = page.get("approval", {})
        if not isinstance(approval, dict) or approval.get("status") not in ("draft", "approved"):
            raise ValueError("페이지 원문의 승인 상태를 명시해야 합니다.")
        if approval["status"] == "approved":
            if not all(_text(approval.get(key)) for key in ("reviewed_by", "reviewed_at")):
                raise ValueError("원문 승인에는 검토자와 검토 시각이 필요합니다.")
            if approval.get("content_sha256") != page_content_digest(page):
                raise ValueError("승인 후 원문 또는 출처가 변경되었습니다. 다시 검토해야 합니다.")

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

    def supports(self, document_id: str, document_version: str, page: int) -> bool:
        """문서·버전·페이지에 승인된 안내가 하나 이상 등록돼 있는지 확인한다."""
        return any((e["document_id"], e["document_version"], e["page"]) ==
                   (document_id, document_version, page) and e["approval"]["status"] == "approved"
                   for e in self._entries)

    def read_page(self, document_id: str, document_version: str, page: int,
                  *, demo: bool = False) -> GuidanceResult:
        """별도 승인된 한 페이지 원문 전체를 반환한다. 항목들을 합쳐 원문을 추정하지 않는다.

        문서 식별과 촬영물 검증은 호출자가 수행한다. 이 메서드는 OCR 문자열을 받지 않는다.
        """
        if self.purpose == "demo" and not demo:
            return GuidanceResult("demo_only")
        if not _text(document_id) or not _text(document_version) or type(page) is not int or page < 1:
            return GuidanceResult("invalid_input")
        candidates = [p for p in self._pages if (p["document_id"], p["document_version"], p["page"])
                      == (document_id, document_version, page)]
        if not candidates:
            return GuidanceResult("unsupported_document")
        original = candidates[0]
        if original["approval"]["status"] != "approved":
            return GuidanceResult("unapproved")
        text = original["source_text"]
        return GuidanceResult("matched", text, document_id, document_version, page,
                              content_sha256=page_content_digest(original),
                              reading_mode="verbatim", source_text=text)

    def read_document(self, document_id: str, document_version: str,
                      *, demo: bool = False) -> GuidanceResult:
        """모든 페이지가 별도 승인된 경우에만 페이지 순서대로 원문 전체를 반환한다.

        각 페이지의 source_text 는 변경하지 않고 페이지 경계에만 줄바꿈 두 개를 넣는다.
        등록된 일부 페이지만으로 전체 문서를 추정하지 않도록 page_count 를 요구한다.
        """
        if self.purpose == "demo" and not demo:
            return GuidanceResult("demo_only")
        if not _text(document_id) or not _text(document_version):
            return GuidanceResult("invalid_input")
        pages = sorted((p for p in self._pages if (p["document_id"], p["document_version"])
                        == (document_id, document_version)), key=lambda p: p["page"])
        if not pages:
            return GuidanceResult("unsupported_document")
        count = pages[0].get("page_count")
        if (type(count) is not int or count < 1 or len(pages) != count
                or any(p.get("page_count") != count or p["page"] != index
                       for index, p in enumerate(pages, 1))):
            return GuidanceResult("incomplete_document")
        if any(p["approval"]["status"] != "approved" for p in pages):
            return GuidanceResult("unapproved")
        text = "\n\n".join(p["source_text"] for p in pages)
        content = {"document_id": document_id, "document_version": document_version,
                   "page_separator": "\n\n", "pages": [page_content_digest(p) for p in pages]}
        fingerprint = hashlib.sha256(json.dumps(content, ensure_ascii=False, sort_keys=True,
                                                separators=(",", ":")).encode()).hexdigest()
        return GuidanceResult("matched", text, document_id, document_version,
                              content_sha256=fingerprint, reading_mode="verbatim", source_text=text)

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
        text = e["source_text"] if e.get("reading_mode") == "verbatim" else e["instruction"]
        return GuidanceResult("matched", text, e["document_id"], e["document_version"],
                              e["page"], e["field_id"], content_digest(e),
                              e.get("reading_mode"), e.get("source_text"))
