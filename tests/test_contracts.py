"""계약 계층 테스트.

검증 범위:

1. 모든 dataclass 의 ``from_dict(x.to_dict()) == x`` 왕복 동일성(세션 복원 근거)
2. :class:`BoxMm` 기하 연산(center / contains / 넓이 / 경계)
3. Enum 직렬화 규약(문자열 value, 미정의 값은 예외)
4. 공개 정보 영역 / 개인정보 영역 분리(:meth:`DocumentStructure.public_payload`)
5. :class:`PiiSpan` 이 원문 값을 담지 않는다는 구조적 불변식
6. :mod:`docagent.contracts` 의 외부 의존 0 (표준 라이브러리만 import)
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from docagent import contracts as C
from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_PAGE_SIZE_MM,
    A4_WIDTH_MM,
    EXPLAIN_THRESHOLD,
    PARTIAL_THRESHOLD,
    VISION_TRUST_THRESHOLD,
    AgentTurn,
    BoxMm,
    BoxPx,
    Detection,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    OcrWord,
    Option,
    PiiSpan,
    Point,
    RetrievedChunk,
    SanitizedText,
    Sensitivity,
    ToolCall,
    ToolResult,
    VerificationResult,
)


# --------------------------------------------------------------------------
# 픽스처 성격의 샘플 생성기 (결정론적 — 난수를 쓰지 않는다)
# --------------------------------------------------------------------------


def _sample_field_consent() -> Field:
    """공개 정보 영역 샘플: 개인정보 수집·이용 동의 체크 항목."""
    return Field(
        id="consent_01",
        type=FieldType.CHECKBOX,
        title="개인정보 수집·이용 동의",
        role=FieldRole.APPLICANT,
        options=(
            Option(label="동의함", box_mm=BoxMm(30.0, 120.0, 5.0, 5.0), checked=None),
            Option(label="동의하지 않음", box_mm=BoxMm(60.0, 120.0, 5.0, 5.0), checked=False),
        ),
        required=True,
        sensitivity=Sensitivity.PUBLIC,
        box_mm=BoxMm(25.0, 112.0, 160.0, 18.0),
        clause_text="수집 항목: 성명, 연락처. 보유 기간: 1년.",
        order=1,
        confidence=0.93,
    )


def _sample_field_signature() -> Field:
    """개인정보 영역 샘플: 신청인 서명란."""
    return Field(
        id="signature_01",
        type=FieldType.SIGNATURE,
        title="신청인 서명",
        role=FieldRole.APPLICANT,
        options=(),
        required=True,
        sensitivity=Sensitivity.PRIVATE,
        box_mm=BoxMm(130.0, 250.0, 50.0, 15.0),
        clause_text="위 내용을 확인하였습니다.",
        order=2,
        confidence=0.88,
    )


def _sample_document() -> DocumentStructure:
    """공개 항목 1개 + 개인정보 항목 1개를 가진 문서 샘플."""
    return DocumentStructure(
        document_id="doc_0001",
        doc_title="예금계좌 개설 신청서",
        fields=(_sample_field_consent(), _sample_field_signature()),
        page_size_mm=A4_PAGE_SIZE_MM,
        source_image="samples/form_a4.png",
        warnings=("2번 항목의 경계가 흐릿합니다.",),
    )


#: to_dict/from_dict 왕복을 검사할 (클래스, 인스턴스) 전체 목록.
ROUNDTRIP_SAMPLES: list[tuple[type, Any]] = [
    (Point, Point(12.5, 30.25)),
    (BoxMm, BoxMm(10.0, 20.0, 30.0, 40.0)),
    (BoxPx, BoxPx(100, 200, 300, 400)),
    (
        Detection,
        Detection(
            type=FieldType.CHECKBOX,
            box_mm=BoxMm(30.0, 120.0, 5.0, 5.0),
            confidence=0.91,
            source="opencv_contour",
        ),
    ),
    (
        OcrWord,
        OcrWord(text="동의함", box_mm=BoxMm(37.0, 119.0, 12.0, 4.0), confidence=0.87),
    ),
    (
        Option,
        Option(label="동의함", box_mm=BoxMm(30.0, 120.0, 5.0, 5.0), checked=True),
    ),
    (Option, Option(label="미확인", box_mm=BoxMm(0.0, 0.0, 1.0, 1.0), checked=None)),
    (Field, _sample_field_consent()),
    (Field, _sample_field_signature()),
    (Field, Field(id="minimal_01")),
    (DocumentStructure, _sample_document()),
    (DocumentStructure, DocumentStructure(document_id="doc_empty")),
    (
        VerificationResult,
        VerificationResult(
            field_id="signature_01",
            written=True,
            ink_ratio_before=0.012,
            ink_ratio_after=0.184,
            confidence=0.95,
            reason="잉크 비율이 임계값 이상 증가했습니다.",
        ),
    ),
    (PiiSpan, PiiSpan(start=3, end=17, pii_type="rrn", raw_len=14, confidence=0.99)),
    (
        SanitizedText,
        SanitizedText(
            text="신청인 [주민등록번호] 확인 완료",
            spans=(PiiSpan(start=4, end=18, pii_type="rrn", raw_len=14),),
            blocked=False,
        ),
    ),
    (SanitizedText, SanitizedText(text="개인정보 없음")),
    (
        RetrievedChunk,
        RetrievedChunk(
            chunk_id="chunk_007",
            text="개인정보 보호법 제15조에 따른 수집·이용 동의",
            source="개인정보 보호법",
            score=0.72,
        ),
    ),
    (ToolCall, ToolCall(name="write_checkbox", arguments={"field_id": "consent_01"})),
    (ToolCall, ToolCall(name="home")),
    (
        ToolResult,
        ToolResult(
            ok=False,
            speech="죄송합니다. 서명란 위치를 확인하지 못했습니다.",
            state_patch={"stage": "verify"},
            error="신뢰도 미달",
            handoff=True,
        ),
    ),
    (ToolResult, ToolResult(ok=True)),
    (
        AgentTurn,
        AgentTurn(
            user_text="첫 번째 항목 다시 읽어 주세요",
            intent="explain_field",
            tool_calls=(ToolCall(name="read_field", arguments={"field_id": "consent_01"}),),
            speech="첫 번째 항목은 개인정보 수집·이용 동의입니다.",
            confidence=0.9,
            handoff_reason=None,
        ),
    ),
    (
        AgentTurn,
        AgentTurn(
            user_text="",
            intent="handoff",
            speech="담당자를 연결해 드리겠습니다.",
            confidence=0.4,
            handoff_reason="항목 해석 신뢰도가 낮습니다.",
        ),
    ),
]


# --------------------------------------------------------------------------
# 1. 왕복 동일성
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "sample"),
    ROUNDTRIP_SAMPLES,
    ids=[f"{cls.__name__}-{i}" for i, (cls, _) in enumerate(ROUNDTRIP_SAMPLES)],
)
def test_to_dict_from_dict_roundtrip(cls: type, sample: Any) -> None:
    """``from_dict(x.to_dict()) == x`` 가 모든 dataclass 에서 성립한다."""
    restored = cls.from_dict(sample.to_dict())
    assert restored == sample


@pytest.mark.parametrize(
    ("cls", "sample"),
    ROUNDTRIP_SAMPLES,
    ids=[f"{cls.__name__}-{i}" for i, (cls, _) in enumerate(ROUNDTRIP_SAMPLES)],
)
def test_json_roundtrip(cls: type, sample: Any) -> None:
    """``to_dict()`` 결과는 JSON 직렬화 가능하고, JSON 을 거쳐도 동일하게 복원된다."""
    text = json.dumps(sample.to_dict(), ensure_ascii=False)
    restored = cls.from_dict(json.loads(text))
    assert restored == sample


def test_all_contract_dataclasses_are_covered() -> None:
    """contracts 의 모든 dataclass 가 왕복 테스트 목록에 포함되어 있다."""
    declared = {
        getattr(C, name)
        for name in C.__all__
        if dataclasses.is_dataclass(getattr(C, name))
    }
    covered = {cls for cls, _ in ROUNDTRIP_SAMPLES}
    assert declared == covered, "왕복 테스트에서 빠진 dataclass 가 있습니다."


def test_document_structure_json_helpers() -> None:
    """fields JSON 문자열 왕복도 무손실이며 한글이 이스케이프되지 않는다."""
    document = _sample_document()
    text = document.to_json()
    assert "예금계좌 개설 신청서" in text  # ensure_ascii=False 확인
    assert DocumentStructure.from_json(text) == document


def test_document_structure_from_json_rejects_broken_input() -> None:
    """깨진 JSON 은 조용히 넘어가지 않고 한국어 메시지와 함께 예외가 된다."""
    with pytest.raises(ValueError, match="파싱"):
        DocumentStructure.from_json("{not json")
    with pytest.raises(ValueError, match="객체"):
        DocumentStructure.from_json("[1, 2, 3]")


def test_page_size_is_tuple_after_roundtrip() -> None:
    """``page_size_mm`` 은 JSON 에서 list 로 나갔다가 tuple 로 복원된다."""
    document = _sample_document()
    payload = document.to_dict()
    assert isinstance(payload["page_size_mm"], list)
    restored = DocumentStructure.from_dict(payload)
    assert isinstance(restored.page_size_mm, tuple)
    assert restored.page_size_mm == (A4_WIDTH_MM, A4_HEIGHT_MM)


def test_field_options_normalized_to_tuple() -> None:
    """``options`` 를 list 로 넘겨도 tuple 로 정규화되어 불변성이 유지된다."""
    field = Field(
        id="choice_01",
        options=[Option(label="예", box_mm=BoxMm(0.0, 0.0, 5.0, 5.0))],
    )
    assert isinstance(field.options, tuple)


# --------------------------------------------------------------------------
# 2. BoxMm 기하 연산
# --------------------------------------------------------------------------


def test_box_center() -> None:
    """중심점은 좌상단 + 변 길이의 절반이다."""
    assert BoxMm(10.0, 20.0, 30.0, 40.0).center() == Point(25.0, 40.0)


def test_box_edges_and_area() -> None:
    """오른쪽·아래쪽 경계와 넓이가 정의대로 계산된다."""
    box = BoxMm(10.0, 20.0, 30.0, 40.0)
    assert box.right_mm == 40.0
    assert box.bottom_mm == 60.0
    assert box.area_mm2 == 1200.0
    assert box.to_tuple() == (10.0, 20.0, 30.0, 40.0)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        (Point(25.0, 40.0), True),   # 내부
        (Point(10.0, 20.0), True),   # 좌상단 경계 포함
        (Point(40.0, 60.0), True),   # 우하단 경계 포함
        (Point(9.99, 40.0), False),  # 왼쪽 밖
        (Point(25.0, 60.01), False), # 아래쪽 밖 (y 는 아래 방향이 +)
    ],
)
def test_box_contains(point: Point, expected: bool) -> None:
    """``contains`` 는 경계를 포함하며 y 는 아래쪽이 + 방향이다."""
    assert BoxMm(10.0, 20.0, 30.0, 40.0).contains(point) is expected


def test_box_rejects_negative_size() -> None:
    """음수 크기는 생성 시점에 즉시 거부한다(조용한 실패 금지)."""
    with pytest.raises(ValueError, match="0 이상"):
        BoxMm(0.0, 0.0, -1.0, 10.0)
    with pytest.raises(ValueError, match="0 이상"):
        BoxPx(0, 0, 10, -1)


def test_zero_size_box_is_allowed() -> None:
    """폭·높이 0 은 유효하다(점 형태의 앵커)."""
    box = BoxMm(5.0, 5.0, 0.0, 0.0)
    assert box.center() == Point(5.0, 5.0)
    assert box.contains(Point(5.0, 5.0)) is True


def test_point_to_tuple() -> None:
    """``Point.to_tuple`` 은 ``(x_mm, y_mm)`` 순서를 유지한다."""
    assert Point(1.5, 2.5).to_tuple() == (1.5, 2.5)


def test_field_target_point() -> None:
    """``target_point`` 는 박스 중심이며, 박스가 없으면 None 이다."""
    assert _sample_field_signature().target_point() == Point(155.0, 257.5)
    assert Field(id="no_box").target_point() is None


# --------------------------------------------------------------------------
# 3. Enum 직렬화
# --------------------------------------------------------------------------


@pytest.mark.parametrize("enum_cls", [FieldType, FieldRole, Sensitivity])
def test_enum_values_are_strings(enum_cls: type) -> None:
    """모든 Enum 값은 JSON 에 그대로 담을 수 있는 문자열이다."""
    for member in enum_cls:
        assert isinstance(member.value, str)


def test_enum_member_sets_are_fixed() -> None:
    """Enum 구성원 집합은 계약이다. 변경 시 이 테스트가 먼저 깨져야 한다."""
    assert {m.value for m in FieldType} == {
        "signature",
        "checkbox",
        "choice",
        "text_input",
        "date",
        "unknown",
    }
    assert {m.value for m in FieldRole} == {
        "applicant",
        "representative",
        "official",
        "unknown",
    }
    assert {m.value for m in Sensitivity} == {"public", "private"}


def test_enum_serialized_as_value_string() -> None:
    """dict 에는 Enum 인스턴스가 아니라 문자열 value 가 담긴다."""
    payload = _sample_field_consent().to_dict()
    assert payload["type"] == "checkbox"
    assert payload["role"] == "applicant"
    assert payload["sensitivity"] == "public"


def test_from_dict_accepts_enum_instance_and_string() -> None:
    """``from_dict`` 는 문자열과 Enum 인스턴스를 모두 받아들인다."""
    from_string = Field.from_dict({"id": "f1", "type": "date"})
    from_member = Field.from_dict({"id": "f1", "type": FieldType.DATE})
    assert from_string.type is FieldType.DATE
    assert from_string == from_member


def test_from_dict_rejects_unknown_enum_value() -> None:
    """정의되지 않은 Enum 값은 한국어 메시지와 함께 거부된다."""
    with pytest.raises(ValueError, match="FieldType 에 정의되지 않은 값"):
        Field.from_dict({"id": "f1", "type": "barcode"})


def test_field_defaults_are_safe() -> None:
    """기본 민감도는 안전한 PRIVATE, 기본 유형·역할은 UNKNOWN 이다."""
    field = Field(id="f1")
    assert field.sensitivity is Sensitivity.PRIVATE
    assert field.type is FieldType.UNKNOWN
    assert field.role is FieldRole.UNKNOWN
    assert field.is_public() is False


# --------------------------------------------------------------------------
# 4. 공개 정보 영역 / 개인정보 영역 분리
# --------------------------------------------------------------------------


def test_public_payload_keeps_public_field_intact() -> None:
    """공개 항목은 라벨·약관 문구까지 그대로 포함된다."""
    payload = _sample_document().public_payload()
    consent = next(f for f in payload["fields"] if f["id"] == "consent_01")
    assert consent["title"] == "개인정보 수집·이용 동의"
    assert consent["clause_text"].startswith("수집 항목")
    assert len(consent["options"]) == 2
    assert "redacted" not in consent


def test_public_payload_drops_coordinates_from_public_field() -> None:
    """공개 항목도 좌표·신뢰도는 내보내지 않는다(§6 — 좌표는 로컬 코드 전용)."""
    payload = _sample_document().public_payload()
    consent = next(f for f in payload["fields"] if f["id"] == "consent_01")
    for removed in ("box_mm", "confidence"):
        assert removed not in consent
    for option in consent["options"]:
        assert set(option) == {"label"}
    assert "box_mm" not in json.dumps(payload, ensure_ascii=False)


def test_public_payload_redacts_private_field() -> None:
    """개인정보 항목은 구조 정보만 남고 제목·약관·선택지·좌표가 제거된다."""
    payload = _sample_document().public_payload()
    signature = next(f for f in payload["fields"] if f["id"] == "signature_01")
    assert signature["redacted"] is True
    for removed in ("title", "clause_text", "options", "box_mm", "confidence"):
        assert removed not in signature
    assert signature["type"] == "signature"
    assert signature["required"] is True
    assert payload["redacted_field_ids"] == ["signature_01"]


def test_public_payload_is_json_serializable() -> None:
    """공개 payload 는 그대로 LLM 요청 본문에 넣을 수 있어야 한다."""
    text = json.dumps(_sample_document().public_payload(), ensure_ascii=False)
    assert "신청인 서명" not in text  # 개인정보 항목의 제목이 새어나가지 않는다.
    assert "예금계좌 개설 신청서" in text


def test_public_payload_sorted_by_order() -> None:
    """공개 payload 의 항목은 낭독 순서대로 정렬된다."""
    document = DocumentStructure(
        document_id="doc_order",
        fields=(
            Field(id="b", order=2, sensitivity=Sensitivity.PUBLIC),
            Field(id="a", order=1, sensitivity=Sensitivity.PUBLIC),
        ),
    )
    assert [f["id"] for f in document.public_payload()["fields"]] == ["a", "b"]


# --------------------------------------------------------------------------
# 5. DocumentStructure 조회 헬퍼
# --------------------------------------------------------------------------


def test_required_fields_sorted_by_order() -> None:
    """필수 항목만 순서대로 반환한다."""
    document = DocumentStructure(
        document_id="doc_req",
        fields=(
            Field(id="opt", required=False, order=0),
            Field(id="req_b", required=True, order=5),
            Field(id="req_a", required=True, order=2),
        ),
    )
    assert [f.id for f in document.required_fields()] == ["req_a", "req_b"]


def test_field_by_id() -> None:
    """존재하면 항목을, 없으면 None 을 반환한다."""
    document = _sample_document()
    assert document.field_by_id("consent_01") is not None
    assert document.field_by_id("does_not_exist") is None


def test_duplicate_field_ids_rejected() -> None:
    """항목 id 중복은 생성 시점에 거부된다(Agent 의 항목 지시가 모호해지는 것을 막는다)."""
    with pytest.raises(ValueError, match="중복"):
        DocumentStructure(
            document_id="doc_dup",
            fields=(Field(id="same"), Field(id="same")),
        )


def test_empty_ids_rejected() -> None:
    """빈 식별자는 허용하지 않는다."""
    with pytest.raises(ValueError, match="Field.id"):
        Field(id="")
    with pytest.raises(ValueError, match="document_id"):
        DocumentStructure(document_id="")


# --------------------------------------------------------------------------
# 6. PII 불변식
# --------------------------------------------------------------------------


def test_pii_span_never_carries_raw_value() -> None:
    """:class:`PiiSpan` 의 필드 집합은 원문 값을 담을 수 없는 형태로 고정된다."""
    names = {f.name for f in dataclasses.fields(PiiSpan)}
    assert names == {"start", "end", "pii_type", "raw_len", "confidence"}
    forbidden = {"value", "raw", "raw_value", "text", "original", "matched"}
    assert not (names & forbidden)


def test_pii_span_dict_has_no_extra_keys() -> None:
    """직렬화 결과에도 원문 값이 끼어들 자리가 없다."""
    payload = PiiSpan(start=0, end=5, pii_type="phone", raw_len=13).to_dict()
    assert set(payload) == {"start", "end", "pii_type", "raw_len", "confidence"}


def test_pii_span_rejects_invalid_range() -> None:
    """역전되거나 빈 구간은 거부한다."""
    with pytest.raises(ValueError, match="start 보다 커야"):
        PiiSpan(start=5, end=5, pii_type="rrn", raw_len=0)
    with pytest.raises(ValueError, match="0 이상"):
        PiiSpan(start=-1, end=3, pii_type="rrn", raw_len=3)


def test_sanitized_text_has_pii_flag() -> None:
    """``has_pii`` 는 탐지 구간 존재 여부를 그대로 반영한다."""
    assert SanitizedText(text="깨끗함").has_pii is False
    assert (
        SanitizedText(
            text="[전화번호]",
            spans=(PiiSpan(start=0, end=13, pii_type="phone", raw_len=13),),
        ).has_pii
        is True
    )


# --------------------------------------------------------------------------
# 7. 신뢰도 임계값 · 파생 규칙
# --------------------------------------------------------------------------


def test_threshold_ordering() -> None:
    """임계값은 0 < PARTIAL < EXPLAIN <= 1 순서를 유지한다."""
    assert 0.0 < PARTIAL_THRESHOLD < EXPLAIN_THRESHOLD <= 1.0
    assert EXPLAIN_THRESHOLD == pytest.approx(0.85)
    assert PARTIAL_THRESHOLD == pytest.approx(0.70)
    assert VISION_TRUST_THRESHOLD == pytest.approx(0.85)


def test_motion_tolerance_is_defined_once() -> None:
    """펜 허용 오차는 계약 한 곳에서만 정의되고 나머지는 참조만 한다."""
    from docagent.agent.tools import MOTION_TOLERANCE_MM as tools_value
    from docagent.config import DocAgentConfig
    from docagent.contracts import MOTION_TOLERANCE_MM

    assert MOTION_TOLERANCE_MM == pytest.approx(1.0)
    assert tools_value is MOTION_TOLERANCE_MM
    assert DocAgentConfig().motion_tolerance_mm == MOTION_TOLERANCE_MM


def test_a4_constants() -> None:
    """A4 상수는 mm 기준 210 x 297 이다."""
    assert (A4_WIDTH_MM, A4_HEIGHT_MM) == (210.0, 297.0)
    assert A4_PAGE_SIZE_MM == (210.0, 297.0)


def test_agent_turn_needs_handoff() -> None:
    """사람 지원 판정은 handoff_reason 하나로만 결정된다."""
    assert AgentTurn(confidence=0.95).needs_handoff() is False
    assert AgentTurn(confidence=PARTIAL_THRESHOLD).needs_handoff() is False
    # 신뢰도만 낮은 턴은 핸드오프가 아니다(재질문 턴이 여기 해당한다).
    assert AgentTurn(confidence=0.0).needs_handoff() is False
    assert AgentTurn(confidence=0.69).needs_handoff() is False
    assert AgentTurn(confidence=0.99, handoff_reason="법적 판단 필요").needs_handoff() is True
    assert AgentTurn(confidence=0.0, handoff_reason="직원 연결").needs_handoff() is True


def test_confidence_range_validation() -> None:
    """신뢰도는 0.0~1.0 범위를 벗어날 수 없다."""
    with pytest.raises(ValueError, match="0.0~1.0"):
        Detection(type=FieldType.DATE, box_mm=BoxMm(0.0, 0.0, 1.0, 1.0), confidence=1.5, source="x")
    with pytest.raises(ValueError, match="0.0~1.0"):
        Field(id="f1", confidence=-0.1)
    with pytest.raises(ValueError, match="0.0~1.0"):
        AgentTurn(confidence=2.0)


def test_verification_ink_delta() -> None:
    """``ink_delta`` 는 기입 전후 잉크 비율의 차이다."""
    result = VerificationResult(
        field_id="signature_01",
        written=True,
        ink_ratio_before=0.010,
        ink_ratio_after=0.150,
        confidence=0.9,
    )
    assert result.ink_delta == pytest.approx(0.140)


# --------------------------------------------------------------------------
# 8. 의존성 규약
# --------------------------------------------------------------------------

#: contracts.py 가 import 해도 되는 표준 라이브러리 모듈.
ALLOWED_CONTRACT_IMPORTS = {"__future__", "json", "dataclasses", "enum", "typing"}


def test_contracts_module_has_no_external_dependency() -> None:
    """contracts.py 는 numpy 를 포함한 어떤 외부 패키지도 import 하지 않는다."""
    source = Path(C.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_CONTRACT_IMPORTS, (
        f"contracts.py 에 허용되지 않은 import 가 있습니다: "
        f"{sorted(imported - ALLOWED_CONTRACT_IMPORTS)}"
    )


def test_required_keys_error_is_explicit() -> None:
    """필수 키 누락은 조용히 넘어가지 않고 어떤 키가 없는지 알려 준다."""
    with pytest.raises(ValueError, match="x_mm"):
        Point.from_dict({"y_mm": 1.0})


def test_config_rejects_non_a4_page_size() -> None:
    """A4 이외의 용지 크기는 조용히 진행하지 않고 거부된다.

    탐지 좌표는 A4 기준으로만 환산되므로(geometry), 설정만 다른 용지로 바꾸면
    탐지 좌표와 펜 좌표가 어긋난 채 진행된다. 그 오차는 펜 도달 허용 오차
    (1.0mm)를 쉽게 넘는다.
    """
    from docagent.config import DocAgentConfig

    with pytest.raises(ValueError, match="A4"):
        DocAgentConfig(page_size_mm=(216.0, 279.0))
    assert DocAgentConfig().page_size_mm == A4_PAGE_SIZE_MM


def test_build_structure_rejects_non_a4_coords() -> None:
    """``build_structure(coords=...)`` 도 같은 규율을 받는다."""
    from docagent.vision.structuring import build_structure

    detections = [
        Detection(
            type=FieldType.CHECKBOX,
            box_mm=BoxMm(30.0, 120.0, 5.0, 5.0),
            confidence=0.9,
            source="test",
        )
    ]
    with pytest.raises(ValueError, match="A4"):
        build_structure(detections, [], (216.0, 279.0), "letter_doc")
