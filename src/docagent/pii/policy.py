"""공개 정보 영역 / 개인정보 영역 이원 분리 정책 — 화이트리스트 방식.

로드맵의 "공개 정보 영역 / 사용자 개인정보 영역" 구조를 코드로 명문화한 모듈이다.

판정 원칙(순서대로 적용)
------------------------
1. **화이트리스트가 먼저다.** "개인정보가 아니니 통과"가 아니라
   ":data:`PUBLIC_FIELD_TYPES` 에 있으니 통과"로 판단한다.
   목록에 없는 유형(:attr:`~docagent.contracts.FieldType.UNKNOWN` 포함)은 무조건
   :attr:`~docagent.contracts.Sensitivity.PRIVATE` 다.
2. **역할도 화이트리스트다.** 이용자 본인이 값을 적어 넣는 자리는 공개될 수 없다.
3. **선언값과 계산값 중 더 보수적인 쪽을 택한다.**
   ``Field.sensitivity`` 가 이미 ``PRIVATE`` 면 계산 결과와 무관하게 ``PRIVATE``.
4. **내용을 실제로 검사한다.** 화이트리스트를 통과했더라도 ``title`` ·
   ``clause_text`` · 선택지 라벨에서 개인정보가 탐지되면 ``PRIVATE`` 로 강등한다.
   (OCR 오인식으로 이용자가 적은 값이 공개 필드에 섞여 들어오는 사고 대비.)

:func:`build_public_payload` 는 :meth:`DocumentStructure.public_payload` 보다
**더 엄격하다.** 좌표(``box_mm``)와 신뢰도까지 제거해 LLM 에는
"무엇을 묻고 무엇을 고를 수 있는가"만 남긴다. 좌표가 필요한 Act 단계에서는
LLM 이 아니라 로컬 코드가 원본 :class:`DocumentStructure` 에서 직접 읽는다.
"""

from __future__ import annotations

from typing import Any, Final, Sequence

from docagent.contracts import (
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    Sensitivity,
)
from docagent.pii.detectors import RegexPiiDetector

__all__ = [
    "PUBLIC_FIELD_TYPES",
    "PUBLIC_FIELD_ROLES",
    "PUBLIC_FIELD_KEYS",
    "REDACTED_FIELD_KEYS",
    "classify_field",
    "is_llm_transmittable",
    "build_public_payload",
    "public_texts",
]

#: 공개 정보 영역이 될 수 **있는** 기입란 유형(화이트리스트).
#:
#: 선택형 항목의 제목·약관 본문·선택지 라벨은 모두 **문서에 인쇄된 내용**이므로
#: 공개 정보다. 반대로 TEXT_INPUT / SIGNATURE / DATE 는 이용자가 값을 적어 넣는
#: 자리이므로 어떤 경우에도 공개 영역이 아니다.
PUBLIC_FIELD_TYPES: Final[frozenset[FieldType]] = frozenset(
    {FieldType.CHOICE, FieldType.CHECKBOX}
)

#: 공개 정보 영역이 될 수 **있는** 역할(화이트리스트).
#: :attr:`FieldRole.UNKNOWN` 은 분류 실패이므로 제외한다(안전 기본값).
PUBLIC_FIELD_ROLES: Final[frozenset[FieldRole]] = frozenset(
    {FieldRole.APPLICANT, FieldRole.REPRESENTATIVE, FieldRole.OFFICIAL}
)

#: 공개 항목에서 payload 로 내보내는 키(화이트리스트).
PUBLIC_FIELD_KEYS: Final[tuple[str, ...]] = (
    "id",
    "type",
    "role",
    "required",
    "order",
    "sensitivity",
    "title",
    "clause_text",
    "options",
)

#: 강등된 개인정보 항목에서 남기는 키(구조 정보만).
REDACTED_FIELD_KEYS: Final[tuple[str, ...]] = (
    "id",
    "type",
    "role",
    "required",
    "order",
    "sensitivity",
    "redacted",
)

#: 정책 판정에 쓰는 기본 탐지기(상태 없음 → 모듈 수준 공유 가능).
_DEFAULT_DETECTOR: Final[RegexPiiDetector] = RegexPiiDetector()


def public_texts(field: Field) -> tuple[str, ...]:
    """공개 후보 항목이 외부로 내보내려는 문자열 전체를 모은다.

    :param field: 검사 대상 기입란.
    :returns: ``title`` · ``clause_text`` · 선택지 라벨로 이루어진 튜플
        (빈 문자열은 제외).
    """
    texts: list[str] = []
    if field.title:
        texts.append(field.title)
    if field.clause_text:
        texts.append(field.clause_text)
    for option in field.options:
        if option.label:
            texts.append(option.label)
    return tuple(texts)


def classify_field(field: Field, *, detector: Any | None = None) -> Sensitivity:
    """기입란의 민감도를 화이트리스트 기준으로 판정한다.

    :param field: 판정 대상 기입란.
    :param detector: 내용 검사에 쓸 탐지기. ``None`` 이면 기본 정규식 탐지기.
    :returns: :attr:`Sensitivity.PUBLIC` 또는 :attr:`Sensitivity.PRIVATE`.
        판단이 조금이라도 불확실하면 항상 ``PRIVATE`` 다.
    :raises docagent.errors.PiiEgressBlocked: 없음(이 함수는 차단하지 않고 강등한다).
    """
    if field.sensitivity is not Sensitivity.PUBLIC:
        # 선언값이 이미 보수적이면 그대로 존중한다(더 완화하지 않는다).
        return Sensitivity.PRIVATE
    if field.type not in PUBLIC_FIELD_TYPES:
        return Sensitivity.PRIVATE
    if field.role not in PUBLIC_FIELD_ROLES:
        return Sensitivity.PRIVATE

    engine = detector if detector is not None else _DEFAULT_DETECTOR
    for text in public_texts(field):
        if engine.detect(text):
            # 인쇄된 공개 문구여야 할 자리에 개인정보가 섞였다 → 강등.
            return Sensitivity.PRIVATE
    return Sensitivity.PUBLIC


def is_llm_transmittable(field: Field, *, detector: Any | None = None) -> bool:
    """이 기입란의 내용을 외부 LLM 으로 보내도 되는지 판정한다.

    :param field: 판정 대상 기입란.
    :param detector: 내용 검사에 쓸 탐지기. ``None`` 이면 기본 탐지기.
    :returns: 공개 정보 영역이면 True.
    """
    return classify_field(field, detector=detector) is Sensitivity.PUBLIC


def _public_field_payload(field: Field) -> dict[str, Any]:
    """공개 항목의 화이트리스트 payload 를 만든다(좌표·신뢰도 제외).

    :param field: 공개로 판정된 기입란.
    :returns: :data:`PUBLIC_FIELD_KEYS` 만 담은 dict.
    """
    return {
        "id": field.id,
        "type": field.type.value,
        "role": field.role.value,
        "required": field.required,
        "order": field.order,
        "sensitivity": Sensitivity.PUBLIC.value,
        "title": field.title,
        "clause_text": field.clause_text,
        "options": [{"label": option.label} for option in field.options],
    }


def _redacted_field_payload(field: Field) -> dict[str, Any]:
    """개인정보 항목의 구조 정보만 담은 축약 payload 를 만든다.

    :param field: 개인정보로 판정된 기입란.
    :returns: :data:`REDACTED_FIELD_KEYS` 만 담은 dict.
    """
    return {
        "id": field.id,
        "type": field.type.value,
        "role": field.role.value,
        "required": field.required,
        "order": field.order,
        "sensitivity": Sensitivity.PRIVATE.value,
        "redacted": True,
    }


def build_public_payload(
    structure: DocumentStructure, *, detector: Any | None = None
) -> dict[str, Any]:
    """외부 LLM 에 보낼 수 있는 것만 담은 payload 를 만든다.

    포함되는 것: 약관 본문, 항목 제목, 선택지 라벨, 필수 여부, 항목 id/유형/역할/순서.
    **제외되는 것**: 좌표(``box_mm``), 신뢰도, 이용자가 기입한 값,
    OCR 원문 전체, 원본 이미지 경로.

    :param structure: Vision 이 만든 문서 구조.
    :param detector: 내용 검사에 쓸 탐지기. ``None`` 이면 기본 탐지기.
    :returns: JSON 직렬화 가능한 dict.
        키는 ``document_id`` / ``doc_title`` / ``fields`` /
        ``redacted_field_ids`` / ``warnings``.
    :raises TypeError: ``structure`` 가 :class:`DocumentStructure` 가 아닌 경우.
    """
    if not isinstance(structure, DocumentStructure):
        raise TypeError(
            "build_public_payload 는 DocumentStructure 만 받습니다: "
            f"{type(structure).__name__}"
        )

    engine = detector if detector is not None else _DEFAULT_DETECTOR
    fields: list[dict[str, Any]] = []
    redacted_ids: list[str] = []

    for field in sorted(structure.fields, key=lambda f: (f.order, f.id)):
        if classify_field(field, detector=engine) is Sensitivity.PUBLIC:
            fields.append(_public_field_payload(field))
        else:
            redacted_ids.append(field.id)
            fields.append(_redacted_field_payload(field))

    return {
        "document_id": structure.document_id,
        "doc_title": structure.doc_title,
        "fields": fields,
        "redacted_field_ids": redacted_ids,
        "warnings": list(structure.warnings),
    }


def _assert_key_whitelist(payload: Sequence[dict[str, Any]]) -> None:
    """payload 항목이 화이트리스트 키만 갖는지 확인한다(자체 검증용).

    :param payload: ``build_public_payload`` 가 만든 ``fields`` 목록.
    :raises ValueError: 허용되지 않은 키가 있는 경우.
    """
    allowed = set(PUBLIC_FIELD_KEYS) | set(REDACTED_FIELD_KEYS)
    for item in payload:
        extra = set(item) - allowed
        if extra:
            raise ValueError(
                f"공개 payload 에 허용되지 않은 키가 있습니다: {', '.join(sorted(extra))}"
            )
