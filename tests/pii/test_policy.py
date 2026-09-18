"""정책 테스트 — 공개/개인정보 이원 분리가 화이트리스트로 동작하는가.

요구사항 7("build_public_payload 결과에 PRIVATE 필드가 없음")을
공용 픽스처 ``sample_structure`` 로 검증한다.
"""

from __future__ import annotations

import json

import pytest

from docagent.contracts import (
    BoxMm,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    Option,
    Sensitivity,
)
from docagent.pii.policy import (
    PUBLIC_FIELD_KEYS,
    PUBLIC_FIELD_ROLES,
    PUBLIC_FIELD_TYPES,
    REDACTED_FIELD_KEYS,
    build_public_payload,
    classify_field,
    is_llm_transmittable,
    public_texts,
)


def make_choice_field(**overrides: object) -> Field:
    """공개 후보가 되는 CHOICE 항목을 만드는 테스트 헬퍼."""
    defaults: dict[str, object] = {
        "id": "consent_01",
        "type": FieldType.CHOICE,
        "title": "개인정보 수집·이용 동의",
        "role": FieldRole.APPLICANT,
        "options": (
            Option(label="동의함", box_mm=BoxMm(10, 10, 5, 5)),
            Option(label="동의하지 않음", box_mm=BoxMm(30, 10, 5, 5)),
        ),
        "required": True,
        "sensitivity": Sensitivity.PUBLIC,
        "clause_text": "수집 항목은 성명, 주민등록번호, 주소, 연락처입니다.",
        "order": 0,
        "confidence": 1.0,
    }
    defaults.update(overrides)
    return Field(**defaults)  # type: ignore[arg-type]


class TestClassifyField:
    """:func:`classify_field` — 화이트리스트 판정."""

    def test_public_choice_is_public(self) -> None:
        """선언·유형·역할·내용이 모두 통과하면 공개다."""
        assert classify_field(make_choice_field()) is Sensitivity.PUBLIC

    def test_text_input_is_always_private(self) -> None:
        """이용자가 값을 적는 자리는 선언이 PUBLIC 이어도 개인정보다."""
        field = make_choice_field(
            id="applicant_name", type=FieldType.TEXT_INPUT, options=()
        )
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_signature_is_always_private(self) -> None:
        """서명란은 공개 영역이 될 수 없다."""
        field = make_choice_field(
            id="signature_applicant", type=FieldType.SIGNATURE, options=()
        )
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_date_is_private(self) -> None:
        """신청일자도 이용자가 적는 값이므로 개인정보다."""
        field = make_choice_field(id="apply_date", type=FieldType.DATE, options=())
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_unknown_type_is_private(self) -> None:
        """분류 실패(UNKNOWN)는 안전 기본값인 개인정보다."""
        field = make_choice_field(id="unknown_01", type=FieldType.UNKNOWN, options=())
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_unknown_role_is_private(self) -> None:
        """역할을 모르면 공개하지 않는다."""
        assert classify_field(make_choice_field(role=FieldRole.UNKNOWN)) is (
            Sensitivity.PRIVATE
        )

    def test_declared_private_is_respected(self) -> None:
        """선언이 PRIVATE 면 계산 결과와 무관하게 PRIVATE 이다."""
        field = make_choice_field(sensitivity=Sensitivity.PRIVATE)
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_pii_in_clause_downgrades_to_private(self) -> None:
        """공개 문구 자리에 개인정보가 섞이면 강등한다."""
        field = make_choice_field(clause_text="신청인 김철수의 주민등록번호 900101-1234567")
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_pii_in_title_downgrades_to_private(self) -> None:
        """제목에 개인정보가 있어도 강등한다."""
        assert classify_field(make_choice_field(title="성명 김철수 확인란")) is (
            Sensitivity.PRIVATE
        )

    def test_pii_in_option_label_downgrades_to_private(self) -> None:
        """선택지 라벨에 개인정보가 있어도 강등한다."""
        field = make_choice_field(
            options=(Option(label="010-1234-5678", box_mm=BoxMm(0, 0, 1, 1)),)
        )
        assert classify_field(field) is Sensitivity.PRIVATE

    def test_is_llm_transmittable_matches_classification(self) -> None:
        """전송 가능 여부는 분류 결과와 일치한다."""
        assert is_llm_transmittable(make_choice_field())
        assert not is_llm_transmittable(
            make_choice_field(type=FieldType.TEXT_INPUT, options=())
        )

    def test_whitelists_are_narrow(self) -> None:
        """화이트리스트가 실제로 좁게 유지되고 있다."""
        assert PUBLIC_FIELD_TYPES == {FieldType.CHOICE, FieldType.CHECKBOX}
        assert FieldRole.UNKNOWN not in PUBLIC_FIELD_ROLES

    def test_public_texts_collects_all_outgoing_strings(self) -> None:
        """검사 대상 문자열이 빠짐없이 수집된다."""
        texts = public_texts(make_choice_field())
        assert "개인정보 수집·이용 동의" in texts
        assert "동의함" in texts
        assert len(texts) == 4


class TestBuildPublicPayload:
    """:func:`build_public_payload` — 요구사항 7."""

    def test_private_fields_are_reduced_to_structure(
        self, sample_structure: DocumentStructure
    ) -> None:
        """PRIVATE 항목은 구조 정보만 남고 내용이 사라진다."""
        payload = build_public_payload(sample_structure)
        redacted = [f for f in payload["fields"] if f.get("redacted")]
        assert redacted, "강등된 항목이 하나도 없습니다."
        for item in redacted:
            assert set(item) == set(REDACTED_FIELD_KEYS)
            assert item["sensitivity"] == "private"
            assert "title" not in item
            assert "clause_text" not in item
            assert "options" not in item
            assert "box_mm" not in item

    def test_private_field_titles_do_not_leak(
        self, sample_structure: DocumentStructure
    ) -> None:
        """PRIVATE 항목의 제목이 그 항목의 payload 에 남지 않는다.

        (같은 단어가 공개 약관 본문에 인쇄되어 있는 것은 별개 문제다.
        약관 본문은 문서에 찍혀 있는 공개 정보이므로 그대로 전송해도 된다.)
        """
        payload = build_public_payload(sample_structure)
        redacted = [item for item in payload["fields"] if item.get("redacted")]
        rendered = json.dumps(redacted, ensure_ascii=False)
        private_titles = [
            field.title
            for field in sample_structure.fields
            if field.sensitivity is Sensitivity.PRIVATE and field.title
        ]
        assert private_titles, "픽스처에 PRIVATE 항목이 없습니다."
        for title in private_titles:
            assert title not in rendered, f"PRIVATE 제목 유출: {title}"

    def test_private_entries_carry_only_structure(
        self, sample_structure: DocumentStructure
    ) -> None:
        """강등 항목의 값은 전부 id·enum·bool·정수뿐이다(자유 문자열 없음)."""
        for item in build_public_payload(sample_structure)["fields"]:
            if not item.get("redacted"):
                continue
            assert isinstance(item["id"], str)
            assert item["type"] in {
                "signature",
                "checkbox",
                "choice",
                "text_input",
                "date",
                "unknown",
            }
            assert item["role"] in {
                "applicant",
                "representative",
                "official",
                "unknown",
            }
            assert isinstance(item["required"], bool)
            assert isinstance(item["order"], int)

    def test_redacted_ids_cover_every_private_field(
        self, sample_structure: DocumentStructure
    ) -> None:
        """강등된 id 목록이 PRIVATE 항목 전체를 담는다."""
        payload = build_public_payload(sample_structure)
        expected = {
            field.id
            for field in sample_structure.fields
            if field.sensitivity is Sensitivity.PRIVATE
        }
        assert set(payload["redacted_field_ids"]) == expected

    def test_public_consent_field_keeps_its_clause(
        self, sample_structure: DocumentStructure
    ) -> None:
        """공개 동의 항목은 약관 본문과 선택지 라벨을 유지한다."""
        payload = build_public_payload(sample_structure)
        consent = next(f for f in payload["fields"] if f["id"] == "consent_01")
        assert consent["sensitivity"] == "public"
        assert "개인정보를 수집·이용" in consent["clause_text"]
        assert [option["label"] for option in consent["options"]] == [
            "동의함",
            "동의하지 않음",
        ]
        assert consent["required"] is True

    def test_no_coordinates_anywhere(self, sample_structure: DocumentStructure) -> None:
        """좌표는 공개 항목에서도 제거된다(Act 단계는 로컬 코드가 읽는다)."""
        rendered = json.dumps(build_public_payload(sample_structure), ensure_ascii=False)
        assert "box_mm" not in rendered
        assert "x_mm" not in rendered
        assert "confidence" not in rendered
        assert "source_image" not in rendered

    def test_only_whitelisted_keys_appear(
        self, sample_structure: DocumentStructure
    ) -> None:
        """항목 dict 에 화이트리스트 밖 키가 없다."""
        allowed = set(PUBLIC_FIELD_KEYS) | set(REDACTED_FIELD_KEYS)
        for item in build_public_payload(sample_structure)["fields"]:
            assert set(item) <= allowed

    def test_fields_are_ordered(self, sample_structure: DocumentStructure) -> None:
        """항목이 order 순으로 정렬된다."""
        orders = [item["order"] for item in build_public_payload(sample_structure)["fields"]]
        assert orders == sorted(orders)

    def test_payload_is_json_serializable(
        self, sample_structure: DocumentStructure
    ) -> None:
        """payload 가 그대로 JSON 직렬화된다."""
        text = json.dumps(build_public_payload(sample_structure), ensure_ascii=False)
        assert json.loads(text)["document_id"] == sample_structure.document_id

    def test_payload_passes_the_egress_gate(
        self, sample_structure: DocumentStructure
    ) -> None:
        """만들어진 payload 는 유출 차단 게이트를 그대로 통과한다."""
        from docagent.pii.gate import LlmEgressGate

        gate = LlmEgressGate()
        payload = build_public_payload(sample_structure)
        gate.assert_clean(json.dumps(payload, ensure_ascii=False))

    def test_neither_payload_carries_coordinates(
        self, sample_structure: DocumentStructure
    ) -> None:
        """계약 기본 구현과 정책 구현 **양쪽 다** 좌표·신뢰도를 내보내지 않는다.

        규격서 §6 은 "좌표가 필요한 시점에는 LLM 이 아니라 로컬 코드가 원본
        구조에서 직접 읽는다"고 못 박는다. 두 구현 중 한쪽만 좌표를 지우면
        새 호출부가 무심코 좌표를 프롬프트에 실을 수 있으므로 같은 규율을 건다.
        """
        contract_payload = json.dumps(
            sample_structure.public_payload(), ensure_ascii=False
        )
        policy_payload = json.dumps(
            build_public_payload(sample_structure), ensure_ascii=False
        )
        for rendered in (contract_payload, policy_payload):
            assert "box_mm" not in rendered
            assert "confidence" not in rendered

    def test_policy_payload_is_stricter_on_content(self) -> None:
        """정책 구현은 계약 기본 구현과 달리 **내용**까지 탐지기로 재판정한다."""
        structure = DocumentStructure(
            document_id="doc_leak_2",
            doc_title="테스트 문서",
            fields=(make_choice_field(clause_text="신청인 김철수 900101-1234567"),),
        )
        contract_payload = json.dumps(structure.public_payload(), ensure_ascii=False)
        policy_payload = json.dumps(
            build_public_payload(structure), ensure_ascii=False
        )
        assert "900101-1234567" in contract_payload
        assert "900101-1234567" not in policy_payload

    def test_non_structure_input_raises(self) -> None:
        """DocumentStructure 가 아니면 TypeError."""
        with pytest.raises(TypeError, match="DocumentStructure"):
            build_public_payload({"document_id": "x"})  # type: ignore[arg-type]

    def test_field_with_leaked_value_is_downgraded_in_payload(self) -> None:
        """공개로 선언됐어도 값이 새어 들어온 항목은 payload 에서 강등된다."""
        structure = DocumentStructure(
            document_id="doc_leak",
            doc_title="테스트 문서",
            fields=(
                make_choice_field(clause_text="신청인 김철수 900101-1234567"),
            ),
        )
        payload = build_public_payload(structure)
        assert payload["redacted_field_ids"] == ["consent_01"]
        assert "김철수" not in json.dumps(payload, ensure_ascii=False)
