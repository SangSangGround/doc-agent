"""정답(ground truth)을 아는 합성 한국어 신청서 이미지 생성기.

실제 관공서 서식 데이터셋이 없는 상태에서 Vision 파이프라인 전체
(See → Understand → Verify)를 검증하기 위한 결정론적 픽스처 생성기다.
문서에 인쇄되는 모든 값은 **가상의 예시**이며 실존 인물의 정보를 담지 않는다.

핵심 개념
---------
* :class:`FormSpec` — 렌더링 파라미터(변형·화질·해상도). frozen dataclass.
* :class:`FormLayout` — mm 좌표로 확정된 문서 배치. 그리기와 정답 생성이
  **같은 레이아웃 객체**를 공유하므로 정답과 그림이 어긋날 수 없다.
* :class:`SyntheticForm` — 렌더링 결과(변형 적용 이미지 + 정답 구조 + 메타).
* :func:`make_application_form` — 진입점.
* :func:`build_truth` — 이미지를 렌더링하지 않고 정답 :class:`DocumentStructure`
  만 빠르게 만든다(Agent·PII 테스트용).
* :func:`render_written` — 체크·서명이 기입된 이미지를 **동일한 기하 변형**으로
  렌더링한다. 원본과 픽셀 정렬이 보장되므로 Verify 단계의 전/후 비교에 쓸 수 있다.

좌표 규약
---------
:attr:`SyntheticForm.truth` 가 담는 좌표는 **변형 이전의 이상적인 A4 mm 좌표**다.
즉 Vision 파이프라인이 기울기 보정·원근 보정·여백 제거를 마친 뒤 산출해야 하는
목표값이다. 변형이 걸린 :attr:`SyntheticForm.image` 의 픽셀 좌표와는 직접
대응하지 않는다. 변형이 전혀 없는 사양(:meth:`FormSpec.clean`)에서는
:meth:`SyntheticForm.flat_box_px` 로 mm ↔ px 를 직접 환산할 수 있다.

결정론
------
같은 :class:`FormSpec` 은 항상 **바이트 단위로 동일한** 이미지를 만든다.
난수는 전부 ``numpy.random.default_rng(spec.seed)`` 로만 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from docagent.contracts import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    BoxMm,
    BoxPx,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    Option,
    Sensitivity,
)
from docagent.testing.fonts import korean_font_available, load_font

__all__ = [
    "MM_PER_INCH",
    "DEFAULT_DPI",
    "DEFAULT_SEED",
    "DEFAULT_DOC_TITLE",
    "CONSENT_CLAUSE_TEXT",
    "CONSENT_FIELD_ID",
    "DATE_FIELD_ID",
    "APPLICANT_SIGNATURE_FIELD_ID",
    "REPRESENTATIVE_SIGNATURE_FIELD_ID",
    "AGREE_LABEL",
    "DISAGREE_LABEL",
    "REQUIRED_MARKER",
    "PERSON_ROW_SPECS",
    "FormSpec",
    "FormLayout",
    "SyntheticForm",
    "px_per_mm",
    "mm_to_px",
    "page_size_px",
    "build_layout",
    "build_truth",
    "make_application_form",
    "render_written",
    "expected_written_truth",
    "iter_spec_grid",
]


# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------

#: 1 인치의 밀리미터 길이.
MM_PER_INCH: float = 25.4
#: 기본 렌더링 해상도(dpi).
DEFAULT_DPI: int = 200
#: 기본 난수 시드. 프로젝트 전역에서 이 값을 기본으로 쓴다.
DEFAULT_SEED: int = 20260909

#: 기본 문서 제목(가상 서식).
DEFAULT_DOC_TITLE: str = "○○지원금 지급 신청서"

#: 개인정보 수집·이용 동의 설명문(가상 문안).
CONSENT_CLAUSE_TEXT: str = (
    "본인은 ○○지원금 지급 신청과 관련하여 아래와 같이 개인정보를 수집·이용하는 것에 동의합니다. "
    "수집 항목은 성명, 주민등록번호, 주소, 연락처이며, 수집 목적은 지급 자격 확인과 지급 처리입니다. "
    "보유 및 이용 기간은 지급 완료일부터 5년입니다. "
    "귀하는 이 동의를 거부할 권리가 있으며, 동의하지 않는 경우 지원금 지급 신청이 제한될 수 있습니다."
)

#: 동의 항목 field id.
CONSENT_FIELD_ID: str = "consent_01"
#: 신청인 서명란 field id.
APPLICANT_SIGNATURE_FIELD_ID: str = "signature_applicant"
#: 대리인 서명란 field id.
REPRESENTATIVE_SIGNATURE_FIELD_ID: str = "signature_representative"
#: 신청일자 field id.
DATE_FIELD_ID: str = "apply_date"

#: 동의 선택지 라벨.
AGREE_LABEL: str = "동의함"
#: 미동의 선택지 라벨.
DISAGREE_LABEL: str = "동의하지 않음"

#: 인적사항 표의 행 정의 ``(field_id, 라벨, 가상 예시 값, 필수 여부)``.
#:
#: 값은 모두 **명백한 가상 예시**다. 실존 인물·실제 계좌·실제 주민등록번호가 아니다.
PERSON_ROW_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    ("applicant_name", "성명", "홍길동", True),
    ("applicant_rrn", "주민등록번호", "900101-1234567", True),
    ("applicant_address", "주소", "서울특별시 중구 세종대로 110", False),
    ("applicant_phone", "연락처", "010-1234-5678", True),
)

#: 필수 항목 표기 마커.
REQUIRED_MARKER: str = "[필수]"

# 페이지 배치 상수(mm) — 테스트가 참조할 수 있도록 모듈 수준에 둔다.
_MARGIN_LEFT_MM: float = 20.0
_MARGIN_RIGHT_MM: float = 190.0
_CONTENT_WIDTH_MM: float = _MARGIN_RIGHT_MM - _MARGIN_LEFT_MM

_TITLE_BOX_MM = BoxMm(_MARGIN_LEFT_MM, 20.0, _CONTENT_WIDTH_MM, 12.0)
_NOTICE_Y_MM: float = 36.0
_TABLE_TOP_MM: float = 46.0
_TABLE_ROW_H_MM: float = 11.0
_TABLE_LABEL_W_MM: float = 40.0
_CLAUSE_HEADER_Y_MM: float = 98.0
_CLAUSE_BODY_TOP_MM: float = 106.0
_CLAUSE_LINE_H_MM: float = 5.6
_CLAUSE_MAX_LINES: int = 5
_OPTION_ROW_Y_MM: float = 139.0
_OPTION_SIDE_MM: float = 6.0
_OPTION_X_MM: tuple[float, float] = (26.0, 86.0)
_STATEMENT_Y_MM: float = 168.0
_DATE_LABEL_Y_MM: float = 178.0
_DATE_BOX_MM = BoxMm(48.0, 176.0, 60.0, 9.0)
_APPLICANT_SIGN_Y_MM: float = 200.0
_REPRESENTATIVE_SIGN_Y_MM: float = 222.0
_SIGN_LINE_X0_MM: float = 48.0
_SIGN_LINE_X1_MM: float = 118.0
_SIGN_BOX_H_MM: float = 10.0
_SIGN_SUFFIX: str = "(서명 또는 인)"

_INK: tuple[int, int, int] = (20, 20, 24)
_PAPER: tuple[int, int, int] = (255, 255, 255)


# --------------------------------------------------------------------------
# 단위 변환
# --------------------------------------------------------------------------


def px_per_mm(dpi: int) -> float:
    """1mm 당 픽셀 수를 반환한다.

    :param dpi: 해상도(dots per inch, 1 이상).
    :returns: 픽셀/mm 비율.
    :raises ValueError: ``dpi`` 가 1 미만인 경우.
    """
    if dpi < 1:
        raise ValueError(f"dpi 는 1 이상이어야 합니다: {dpi}")
    return dpi / MM_PER_INCH


def mm_to_px(value_mm: float, dpi: int) -> int:
    """밀리미터 길이를 반올림된 픽셀 값으로 변환한다.

    :param value_mm: 변환할 길이(mm).
    :param dpi: 해상도(dpi).
    :returns: 반올림된 픽셀 값(int).
    """
    return int(round(value_mm * px_per_mm(dpi)))


def page_size_px(dpi: int) -> tuple[int, int]:
    """A4 페이지의 픽셀 크기 ``(가로, 세로)`` 를 반환한다.

    :param dpi: 해상도(dpi).
    :returns: ``(width_px, height_px)``.
    """
    return (mm_to_px(A4_WIDTH_MM, dpi), mm_to_px(A4_HEIGHT_MM, dpi))


# --------------------------------------------------------------------------
# 사양(Spec)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FormSpec:
    """합성 신청서 렌더링 사양.

    모든 변형은 :attr:`seed` 로 결정론적으로 재현된다. 같은 사양은 항상
    바이트 단위로 동일한 이미지를 만든다.

    :param document_id: 정답 :class:`DocumentStructure` 의 문서 식별자.
    :param doc_title: 문서 제목.
    :param dpi: 렌더링 해상도. 로드맵 데이터셋 계획은 150~300 을 쓴다.
    :param rotation_deg: 회전각(도). 양수는 반시계 방향. -45~45 허용.
    :param perspective_strength: 원근 왜곡 강도(0.0~1.0). 0 이면 왜곡 없음.
    :param margin_px: 문서를 배경 위에 얹을 때 사방에 두는 여백(px, 0 이상).
        윤곽선 기반 문서 검출 테스트에 필수다.
    :param background_gray: 배경(책상) 밝기(0~255).
    :param noise_sigma: 가우시안 잡음 표준편차(0 이상).
    :param blur_ksize: 가우시안 블러 커널 크기. 1 이면 블러 없음. **홀수**여야 한다.
    :param jpeg_quality: JPEG 재압축 품질(1~100). 100 이면 재압축하지 않는다.
    :param illumination_gradient: 조명 불균일 강도(0.0~1.0). 0 이면 균일.
    :param include_representative: 대리인 서명란 포함 여부(role 분류 테스트용).
    :param fill_example_values: 인적사항 표에 가상 예시 값을 인쇄할지 여부.
    :param seed: 난수 시드.
    :raises ValueError: 파라미터가 허용 범위를 벗어난 경우(한국어 메시지).
    """

    document_id: str = "synthetic_form_0001"
    doc_title: str = DEFAULT_DOC_TITLE
    dpi: int = DEFAULT_DPI
    rotation_deg: float = 0.0
    perspective_strength: float = 0.0
    margin_px: int = 0
    background_gray: int = 190
    noise_sigma: float = 0.0
    blur_ksize: int = 1
    jpeg_quality: int = 100
    illumination_gradient: float = 0.0
    include_representative: bool = False
    fill_example_values: bool = True
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("FormSpec.document_id 는 빈 문자열일 수 없습니다.")
        if not 72 <= self.dpi <= 1200:
            raise ValueError(f"FormSpec.dpi 는 72~1200 이어야 합니다: {self.dpi}")
        if not -45.0 <= self.rotation_deg <= 45.0:
            raise ValueError(
                f"FormSpec.rotation_deg 는 -45~45 도여야 합니다: {self.rotation_deg}"
            )
        if not 0.0 <= self.perspective_strength <= 1.0:
            raise ValueError(
                "FormSpec.perspective_strength 는 0.0~1.0 이어야 합니다: "
                f"{self.perspective_strength}"
            )
        if self.margin_px < 0:
            raise ValueError(f"FormSpec.margin_px 는 0 이상이어야 합니다: {self.margin_px}")
        if not 0 <= self.background_gray <= 255:
            raise ValueError(
                f"FormSpec.background_gray 는 0~255 이어야 합니다: {self.background_gray}"
            )
        if self.noise_sigma < 0:
            raise ValueError(
                f"FormSpec.noise_sigma 는 0 이상이어야 합니다: {self.noise_sigma}"
            )
        if self.blur_ksize < 1 or self.blur_ksize % 2 == 0:
            raise ValueError(
                f"FormSpec.blur_ksize 는 1 이상의 홀수여야 합니다: {self.blur_ksize}"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                f"FormSpec.jpeg_quality 는 1~100 이어야 합니다: {self.jpeg_quality}"
            )
        if not 0.0 <= self.illumination_gradient <= 1.0:
            raise ValueError(
                "FormSpec.illumination_gradient 는 0.0~1.0 이어야 합니다: "
                f"{self.illumination_gradient}"
            )

    @property
    def has_geometry_change(self) -> bool:
        """기하 변형(여백·원근·회전)이 하나라도 걸려 있으면 True."""
        return (
            self.margin_px > 0
            or self.perspective_strength > 0.0
            or abs(self.rotation_deg) > 1e-9
        )

    @classmethod
    def clean(cls, **overrides: Any) -> "FormSpec":
        """변형이 전혀 없는 기준 사양을 만든다.

        :param overrides: 덮어쓸 필드.
        :returns: 변형 없는 :class:`FormSpec`.
        """
        return cls(**overrides)

    def with_(self, **overrides: Any) -> "FormSpec":
        """일부 필드만 바꾼 새 사양을 반환한다(frozen 이므로 복제).

        :param overrides: 덮어쓸 필드.
        :returns: 새 :class:`FormSpec`.
        """
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# 레이아웃
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _PersonRow:
    """인적사항 표의 한 행(내부 레이아웃 단위)."""

    field_id: str
    label: str
    example: str
    required: bool
    label_box_mm: BoxMm
    value_box_mm: BoxMm


@dataclass(frozen=True)
class _OptionSlot:
    """체크박스 선택지 하나(내부 레이아웃 단위)."""

    label: str
    box_mm: BoxMm
    label_x_mm: float


@dataclass(frozen=True)
class _SignatureSlot:
    """서명란 하나(내부 레이아웃 단위)."""

    field_id: str
    label: str
    role: FieldRole
    box_mm: BoxMm
    line_y_mm: float


@dataclass(frozen=True)
class FormLayout:
    """mm 좌표로 확정된 문서 배치.

    그리기(:func:`_draw_flat`)와 정답 생성(:func:`build_truth`)이 같은 객체를
    참조하므로, 인쇄된 그림과 정답 좌표가 구조적으로 어긋날 수 없다.

    :param doc_title: 문서 제목.
    :param title_box_mm: 제목 영역.
    :param person_rows: 인적사항 표의 행 목록.
    :param clause_text: 동의 약관 본문.
    :param clause_box_mm: 약관 본문 영역.
    :param options: 체크박스 선택지 목록.
    :param date_slot: 신청일자 기입 영역.
    :param signatures: 서명란 목록(신청인, 선택적으로 대리인).
    """

    doc_title: str
    title_box_mm: BoxMm
    person_rows: tuple[_PersonRow, ...]
    clause_text: str
    clause_box_mm: BoxMm
    options: tuple[_OptionSlot, ...]
    date_slot: BoxMm
    signatures: tuple[_SignatureSlot, ...]

    @property
    def table_box_mm(self) -> BoxMm:
        """인적사항 표 전체 외곽 사각형."""
        top = min(row.label_box_mm.y_mm for row in self.person_rows)
        bottom = max(row.label_box_mm.bottom_mm for row in self.person_rows)
        return BoxMm(_MARGIN_LEFT_MM, top, _CONTENT_WIDTH_MM, bottom - top)

    def option_box(self, label: str) -> BoxMm:
        """선택지 라벨에 해당하는 체크박스 사각형을 반환한다.

        :param label: 선택지 라벨(예: ``"동의함"``).
        :returns: 체크박스 :class:`BoxMm`.
        :raises ValueError: 정의되지 않은 라벨인 경우.
        """
        for slot in self.options:
            if slot.label == label:
                return slot.box_mm
        allowed = ", ".join(repr(slot.label) for slot in self.options)
        raise ValueError(f"정의되지 않은 선택지 라벨입니다: {label!r}. 허용 값: {allowed}")

    def signature_slot(self, field_id: str) -> _SignatureSlot:
        """``field_id`` 에 해당하는 서명란을 반환한다.

        :param field_id: 서명 항목 id.
        :returns: :class:`_SignatureSlot`.
        :raises ValueError: 해당 서명란이 없는 경우.
        """
        for slot in self.signatures:
            if slot.field_id == field_id:
                return slot
        allowed = ", ".join(repr(slot.field_id) for slot in self.signatures)
        raise ValueError(f"정의되지 않은 서명란 id 입니다: {field_id!r}. 허용 값: {allowed}")


def build_layout(spec: FormSpec) -> FormLayout:
    """사양으로부터 mm 좌표 배치를 만든다(렌더링 없음).

    :param spec: 렌더링 사양.
    :returns: :class:`FormLayout`.
    """
    rows: list[_PersonRow] = []
    for index, (field_id, label, example, required) in enumerate(PERSON_ROW_SPECS):
        top = _TABLE_TOP_MM + index * _TABLE_ROW_H_MM
        rows.append(
            _PersonRow(
                field_id=field_id,
                label=label,
                example=example,
                required=required,
                label_box_mm=BoxMm(
                    _MARGIN_LEFT_MM, top, _TABLE_LABEL_W_MM, _TABLE_ROW_H_MM
                ),
                value_box_mm=BoxMm(
                    _MARGIN_LEFT_MM + _TABLE_LABEL_W_MM,
                    top,
                    _CONTENT_WIDTH_MM - _TABLE_LABEL_W_MM,
                    _TABLE_ROW_H_MM,
                ),
            )
        )

    options = (
        _OptionSlot(
            label=AGREE_LABEL,
            box_mm=BoxMm(
                _OPTION_X_MM[0], _OPTION_ROW_Y_MM, _OPTION_SIDE_MM, _OPTION_SIDE_MM
            ),
            label_x_mm=_OPTION_X_MM[0] + _OPTION_SIDE_MM + 3.0,
        ),
        _OptionSlot(
            label=DISAGREE_LABEL,
            box_mm=BoxMm(
                _OPTION_X_MM[1], _OPTION_ROW_Y_MM, _OPTION_SIDE_MM, _OPTION_SIDE_MM
            ),
            label_x_mm=_OPTION_X_MM[1] + _OPTION_SIDE_MM + 3.0,
        ),
    )

    signatures: list[_SignatureSlot] = [
        _SignatureSlot(
            field_id=APPLICANT_SIGNATURE_FIELD_ID,
            label="신청인",
            role=FieldRole.APPLICANT,
            box_mm=BoxMm(
                _SIGN_LINE_X0_MM,
                _APPLICANT_SIGN_Y_MM,
                _SIGN_LINE_X1_MM - _SIGN_LINE_X0_MM,
                _SIGN_BOX_H_MM,
            ),
            line_y_mm=_APPLICANT_SIGN_Y_MM + _SIGN_BOX_H_MM,
        )
    ]
    if spec.include_representative:
        signatures.append(
            _SignatureSlot(
                field_id=REPRESENTATIVE_SIGNATURE_FIELD_ID,
                label="대리인",
                role=FieldRole.REPRESENTATIVE,
                box_mm=BoxMm(
                    _SIGN_LINE_X0_MM,
                    _REPRESENTATIVE_SIGN_Y_MM,
                    _SIGN_LINE_X1_MM - _SIGN_LINE_X0_MM,
                    _SIGN_BOX_H_MM,
                ),
                line_y_mm=_REPRESENTATIVE_SIGN_Y_MM + _SIGN_BOX_H_MM,
            )
        )

    return FormLayout(
        doc_title=spec.doc_title,
        title_box_mm=_TITLE_BOX_MM,
        person_rows=tuple(rows),
        clause_text=CONSENT_CLAUSE_TEXT,
        clause_box_mm=BoxMm(
            _MARGIN_LEFT_MM,
            _CLAUSE_BODY_TOP_MM,
            _CONTENT_WIDTH_MM,
            _CLAUSE_LINE_H_MM * _CLAUSE_MAX_LINES,
        ),
        options=options,
        date_slot=_DATE_BOX_MM,
        signatures=tuple(signatures),
    )


# --------------------------------------------------------------------------
# 정답 구조
# --------------------------------------------------------------------------


def build_truth(spec: FormSpec, *, layout: FormLayout | None = None) -> DocumentStructure:
    """이미지를 렌더링하지 않고 정답 :class:`DocumentStructure` 만 만든다.

    Agent·PII 테스트처럼 이미지가 필요 없는 경우에 쓴다(수 밀리초).

    :param spec: 렌더링 사양.
    :param layout: 미리 계산한 배치. ``None`` 이면 내부에서 만든다.
    :returns: 모든 좌표가 **변형 이전 A4 mm 정답값**인 :class:`DocumentStructure`.
    """
    layout = layout if layout is not None else build_layout(spec)
    fields: list[Field] = []
    order = 0

    for row in layout.person_rows:
        fields.append(
            Field(
                id=row.field_id,
                type=FieldType.TEXT_INPUT,
                title=row.label,
                role=FieldRole.APPLICANT,
                required=row.required,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=row.value_box_mm,
                clause_text="",
                order=order,
                confidence=1.0,
            )
        )
        order += 1

    fields.append(
        Field(
            id=CONSENT_FIELD_ID,
            type=FieldType.CHOICE,
            title="개인정보 수집·이용 동의",
            role=FieldRole.APPLICANT,
            options=tuple(
                Option(label=slot.label, box_mm=slot.box_mm, checked=None)
                for slot in layout.options
            ),
            required=True,
            sensitivity=Sensitivity.PUBLIC,
            box_mm=BoxMm(
                _MARGIN_LEFT_MM,
                _CLAUSE_HEADER_Y_MM,
                _CONTENT_WIDTH_MM,
                (_OPTION_ROW_Y_MM + _OPTION_SIDE_MM) - _CLAUSE_HEADER_Y_MM,
            ),
            clause_text=layout.clause_text,
            order=order,
            confidence=1.0,
        )
    )
    order += 1

    fields.append(
        Field(
            id=DATE_FIELD_ID,
            type=FieldType.DATE,
            title="신청일자",
            role=FieldRole.APPLICANT,
            required=True,
            sensitivity=Sensitivity.PRIVATE,
            box_mm=layout.date_slot,
            order=order,
            confidence=1.0,
        )
    )
    order += 1

    for slot in layout.signatures:
        fields.append(
            Field(
                id=slot.field_id,
                type=FieldType.SIGNATURE,
                title=f"{slot.label} 서명",
                role=slot.role,
                required=slot.role is FieldRole.APPLICANT,
                sensitivity=Sensitivity.PRIVATE,
                box_mm=slot.box_mm,
                order=order,
                confidence=1.0,
            )
        )
        order += 1

    return DocumentStructure(
        document_id=spec.document_id,
        doc_title=layout.doc_title,
        fields=tuple(fields),
        page_size_mm=(A4_WIDTH_MM, A4_HEIGHT_MM),
        source_image=None,
        warnings=(
            ()
            if korean_font_available()
            else ("한글 폰트를 찾지 못해 텍스트가 기본 폰트로 렌더링되었습니다.",)
        ),
    )


# --------------------------------------------------------------------------
# 렌더링 결과
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SyntheticForm:
    """합성 신청서 렌더링 결과.

    :param image: 변형·화질 저하가 모두 적용된 최종 이미지.
        ``(H, W, 3)`` uint8 **BGR**(OpenCV 호환).
    :param truth: 정답 문서 구조. 좌표는 **변형 이전 A4 mm** 기준이다.
    :param flat_image: 변형 이전의 깨끗한 페이지 이미지(``(H, W, 3)`` uint8 BGR).
        :func:`render_written` 이 기입 표시를 덧그리는 원본이다.
    :param layout: mm 좌표 배치.
    :param spec: 이 결과를 만든 사양.
    :param dpi: 렌더링 해상도.
    :param page_px: 변형 이전 페이지 크기 ``(width_px, height_px)``.
    """

    image: np.ndarray
    truth: DocumentStructure
    flat_image: np.ndarray
    layout: FormLayout
    spec: FormSpec
    dpi: int
    page_px: tuple[int, int]

    @property
    def image_size_px(self) -> tuple[int, int]:
        """최종 이미지 크기 ``(width_px, height_px)``."""
        return (int(self.image.shape[1]), int(self.image.shape[0]))

    @property
    def px_per_mm(self) -> float:
        """1mm 당 픽셀 수."""
        return px_per_mm(self.dpi)

    def field(self, field_id: str) -> Field:
        """정답에서 항목 하나를 꺼낸다.

        :param field_id: 항목 id.
        :returns: :class:`Field`.
        :raises ValueError: 해당 id 가 없는 경우.
        """
        found = self.truth.field_by_id(field_id)
        if found is None:
            allowed = ", ".join(item.id for item in self.truth.fields)
            raise ValueError(f"정답에 없는 항목 id 입니다: {field_id!r}. 허용 값: {allowed}")
        return found

    def option_box(self, label: str) -> BoxMm:
        """선택지 라벨의 체크박스 mm 좌표를 반환한다.

        :param label: 선택지 라벨.
        :returns: :class:`BoxMm`.
        :raises ValueError: 정의되지 않은 라벨인 경우.
        """
        return self.layout.option_box(label)

    def flat_box_px(self, box_mm: BoxMm) -> BoxPx:
        """mm 사각형을 **변형 이전** 픽셀 사각형으로 환산한다.

        변형이 없는 사양(:attr:`FormSpec.has_geometry_change` 가 False)에서는
        :attr:`image` 위의 좌표와 그대로 일치한다.

        :param box_mm: 변환할 mm 사각형.
        :returns: :class:`BoxPx`.
        """
        scale = self.px_per_mm
        return BoxPx(
            x=int(round(box_mm.x_mm * scale)),
            y=int(round(box_mm.y_mm * scale)),
            w=int(round(box_mm.w_mm * scale)),
            h=int(round(box_mm.h_mm * scale)),
        )


# --------------------------------------------------------------------------
# 텍스트 그리기 헬퍼
# --------------------------------------------------------------------------


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: Any) -> float:
    """텍스트의 렌더링 폭(px)을 잰다.

    :param draw: 대상 ``ImageDraw``.
    :param text: 측정할 문자열.
    :param font: Pillow 폰트 객체.
    :returns: 폭(px). 측정 불가 시 근사값.
    """
    try:
        return float(draw.textlength(text, font=font))
    except (AttributeError, TypeError, OSError):
        try:
            return float(font.getlength(text))
        except (AttributeError, TypeError, OSError):
            return float(len(text) * 8)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy_px: tuple[float, float],
    text: str,
    font: Any,
    *,
    fill: tuple[int, int, int] = _INK,
) -> None:
    """좌상단 기준으로 텍스트를 그린다.

    Pillow 기본 비트맵 폰트는 일부 한글 글리프를 갖고 있지 않다. 이때 예외가
    나더라도 **도형 기반 검증은 유효해야 하므로** 해당 문자열만 건너뛴다.
    (이 폴백은 폰트 부재 상황에 한정된 의도적 열화이며, 다른 예외는 그대로 전파한다.)

    :param draw: 대상 ``ImageDraw``.
    :param xy_px: 좌상단 좌표(px).
    :param text: 그릴 문자열.
    :param font: Pillow 폰트 객체.
    :param fill: 글자 색(RGB).
    :returns: ``None``.
    """
    if not text:
        return
    try:
        draw.text(xy_px, text, font=font, fill=fill)
    except (UnicodeEncodeError, ValueError):
        # 기본 비트맵 폰트에서 한글 글리프를 못 찾는 경우에만 도달한다.
        return


def _wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font: Any, max_width_px: float
) -> list[str]:
    """한국어 문장을 폭 기준으로 줄바꿈한다(공백 우선, 없으면 글자 단위).

    :param draw: 대상 ``ImageDraw``.
    :param text: 원문.
    :param font: Pillow 폰트 객체.
    :param max_width_px: 한 줄 최대 폭(px).
    :returns: 줄 문자열 목록.
    """
    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if current and _text_width(draw, candidate, font) > max_width_px:
            break_at = candidate.rfind(" ")
            if break_at > len(current) * 0.5:
                lines.append(candidate[:break_at].rstrip())
                current = candidate[break_at + 1 :]
            else:
                lines.append(current.rstrip())
                current = char
        else:
            current = candidate
    if current.strip():
        lines.append(current.rstrip())
    return lines


# --------------------------------------------------------------------------
# 페이지 그리기
# --------------------------------------------------------------------------


def _mm(value_mm: float, scale: float) -> float:
    """mm 값을 px 실수 좌표로 바꾼다(내부 헬퍼)."""
    return value_mm * scale


def _rect_px(box: BoxMm, scale: float) -> tuple[float, float, float, float]:
    """mm 사각형을 ``(x0, y0, x1, y1)`` px 좌표로 바꾼다(내부 헬퍼)."""
    return (
        box.x_mm * scale,
        box.y_mm * scale,
        box.right_mm * scale,
        box.bottom_mm * scale,
    )


def _draw_flat(layout: FormLayout, spec: FormSpec) -> Image.Image:
    """변형 없는 A4 페이지를 그린다.

    :param layout: mm 좌표 배치.
    :param spec: 렌더링 사양.
    :returns: RGB Pillow 이미지.
    """
    width_px, height_px = page_size_px(spec.dpi)
    scale = px_per_mm(spec.dpi)
    image = Image.new("RGB", (width_px, height_px), _PAPER)
    draw = ImageDraw.Draw(image)

    font_title = load_font(max(1, mm_to_px(7.0, spec.dpi)), bold=True)
    font_head = load_font(max(1, mm_to_px(4.2, spec.dpi)), bold=True)
    font_body = load_font(max(1, mm_to_px(3.6, spec.dpi)))
    font_small = load_font(max(1, mm_to_px(3.0, spec.dpi)))

    line_w = max(1, int(round(0.35 * scale)))
    thick_w = max(2, int(round(0.6 * scale)))

    # 제목(가운데 정렬) + 밑줄
    title_width = _text_width(draw, layout.doc_title, font_title)
    title_x = (width_px - title_width) / 2.0
    _draw_text(draw, (title_x, _mm(layout.title_box_mm.y_mm, scale)), layout.doc_title, font_title)
    draw.line(
        [
            (title_x, _mm(layout.title_box_mm.bottom_mm - 1.0, scale)),
            (title_x + title_width, _mm(layout.title_box_mm.bottom_mm - 1.0, scale)),
        ],
        fill=_INK,
        width=line_w,
    )

    # 안내 문구
    _draw_text(
        draw,
        (_mm(_MARGIN_LEFT_MM, scale), _mm(_NOTICE_Y_MM, scale)),
        f"※ {REQUIRED_MARKER} 표시 항목은 반드시 기입하여야 합니다.",
        font_small,
    )

    # 인적사항 표
    table = layout.table_box_mm
    draw.rectangle(_rect_px(table, scale), outline=_INK, width=thick_w)
    for row in layout.person_rows:
        draw.rectangle(_rect_px(row.label_box_mm, scale), outline=_INK, width=line_w)
        draw.rectangle(_rect_px(row.value_box_mm, scale), outline=_INK, width=line_w)
        label = f"{row.label} {REQUIRED_MARKER}" if row.required else row.label
        _draw_text(
            draw,
            (
                _mm(row.label_box_mm.x_mm + 3.0, scale),
                _mm(row.label_box_mm.y_mm + 3.0, scale),
            ),
            label,
            font_body,
        )
        if spec.fill_example_values:
            _draw_text(
                draw,
                (
                    _mm(row.value_box_mm.x_mm + 4.0, scale),
                    _mm(row.value_box_mm.y_mm + 3.0, scale),
                ),
                row.example,
                font_body,
            )

    # 약관 제목 + 본문
    _draw_text(
        draw,
        (_mm(_MARGIN_LEFT_MM, scale), _mm(_CLAUSE_HEADER_Y_MM, scale)),
        "* 개인정보 수집·이용 동의",
        font_head,
    )
    max_line_px = _mm(_CONTENT_WIDTH_MM, scale)
    for index, line in enumerate(
        _wrap_text(draw, layout.clause_text, font_body, max_line_px)[:_CLAUSE_MAX_LINES]
    ):
        _draw_text(
            draw,
            (
                _mm(_MARGIN_LEFT_MM, scale),
                _mm(_CLAUSE_BODY_TOP_MM + index * _CLAUSE_LINE_H_MM, scale),
            ),
            line,
            font_body,
        )

    # 체크박스 선택지(사각 테두리를 실제로 그린다)
    for slot in layout.options:
        draw.rectangle(_rect_px(slot.box_mm, scale), outline=_INK, width=thick_w)
        _draw_text(
            draw,
            (_mm(slot.label_x_mm, scale), _mm(slot.box_mm.y_mm + 0.6, scale)),
            slot.label,
            font_body,
        )

    # 확인 문장
    _draw_text(
        draw,
        (_mm(_MARGIN_LEFT_MM, scale), _mm(_STATEMENT_Y_MM, scale)),
        "위와 같이 ○○지원금 지급을 신청합니다.",
        font_body,
    )

    # 신청일자
    _draw_text(
        draw,
        (_mm(_MARGIN_LEFT_MM, scale), _mm(_DATE_LABEL_Y_MM, scale)),
        f"신청일자 {REQUIRED_MARKER}",
        font_body,
    )
    draw.line(
        [
            (_mm(layout.date_slot.x_mm, scale), _mm(layout.date_slot.bottom_mm, scale)),
            (_mm(layout.date_slot.right_mm, scale), _mm(layout.date_slot.bottom_mm, scale)),
        ],
        fill=_INK,
        width=line_w,
    )

    # 서명란
    for slot in layout.signatures:
        _draw_text(
            draw,
            (_mm(_MARGIN_LEFT_MM, scale), _mm(slot.box_mm.y_mm + 3.0, scale)),
            slot.label,
            font_body,
        )
        draw.line(
            [
                (_mm(_SIGN_LINE_X0_MM, scale), _mm(slot.line_y_mm, scale)),
                (_mm(_SIGN_LINE_X1_MM, scale), _mm(slot.line_y_mm, scale)),
            ],
            fill=_INK,
            width=thick_w,
        )
        _draw_text(
            draw,
            (_mm(_SIGN_LINE_X1_MM + 4.0, scale), _mm(slot.box_mm.y_mm + 3.0, scale)),
            _SIGN_SUFFIX,
            font_small,
        )

    return image


def _draw_marks(
    image: Image.Image,
    layout: FormLayout,
    spec: FormSpec,
    *,
    checked_option: str | None,
    sign: bool,
    sign_representative: bool,
) -> Image.Image:
    """작성 표시(체크·서명)를 덧그린 새 이미지를 반환한다.

    :param image: 원본(변형 이전) 페이지 이미지.
    :param layout: mm 좌표 배치.
    :param spec: 렌더링 사양.
    :param checked_option: 체크할 선택지 라벨. ``None`` 이면 체크하지 않는다.
    :param sign: 신청인 서명 여부.
    :param sign_representative: 대리인 서명 여부(대리인 서명란이 있을 때만).
    :returns: 표시가 덧그려진 새 Pillow 이미지.
    :raises ValueError: 정의되지 않은 선택지 라벨이거나, 없는 서명란에 서명을 요청한 경우.
    """
    scale = px_per_mm(spec.dpi)
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    stroke = max(2, int(round(0.7 * scale)))

    if checked_option is not None:
        box = layout.option_box(checked_option)  # 미정의 라벨이면 여기서 ValueError
        x0, y0, x1, y1 = _rect_px(box, scale)
        pad = (x1 - x0) * 0.2
        # V 자 체크 표시
        draw.line(
            [
                (x0 + pad, y0 + (y1 - y0) * 0.5),
                (x0 + (x1 - x0) * 0.42, y1 - pad),
                (x1 - pad, y0 + pad),
            ],
            fill=_INK,
            width=stroke,
            joint="curve",
        )

    targets: list[str] = []
    if sign:
        targets.append(APPLICANT_SIGNATURE_FIELD_ID)
    if sign_representative:
        targets.append(REPRESENTATIVE_SIGNATURE_FIELD_ID)

    for field_id in targets:
        slot = layout.signature_slot(field_id)  # 없는 서명란이면 ValueError
        _draw_signature(draw, slot, scale, stroke)

    return marked


def _draw_signature(
    draw: ImageDraw.ImageDraw, slot: _SignatureSlot, scale: float, stroke: int
) -> None:
    """서명란 위에 손글씨 느낌의 곡선을 그린다(결정론적 사인 곡선 합성).

    :param draw: 대상 ``ImageDraw``.
    :param slot: 서명란 배치.
    :param scale: px/mm 비율.
    :param stroke: 선 굵기(px).
    :returns: ``None``.
    """
    box = slot.box_mm
    x0 = (box.x_mm + 4.0) * scale
    x1 = (box.right_mm - 4.0) * scale
    center_y = (box.y_mm + box.h_mm * 0.62) * scale
    amplitude = box.h_mm * 0.42 * scale
    samples = 96
    points: list[tuple[float, float]] = []
    for index in range(samples + 1):
        t = index / samples
        x = x0 + (x1 - x0) * t
        y = center_y - amplitude * (
            np.sin(t * 6.6) * 0.62 + np.sin(t * 15.1 + 0.9) * 0.28
        )
        points.append((float(x), float(y)))
    draw.line(points, fill=_INK, width=stroke, joint="curve")


# --------------------------------------------------------------------------
# 기하·화질 변형
# --------------------------------------------------------------------------


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Pillow RGB 이미지를 OpenCV BGR ndarray 로 바꾼다.

    :param image: RGB Pillow 이미지.
    :returns: ``(H, W, 3)`` uint8 BGR 배열.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _pad_background(image: np.ndarray, spec: FormSpec) -> np.ndarray:
    """문서를 배경 위에 얹어 사방 여백을 만든다(윤곽선 검출용).

    :param image: ``(H, W, 3)`` uint8 BGR 문서 이미지.
    :param spec: 렌더링 사양.
    :returns: 여백이 추가된 이미지.
    """
    margin = spec.margin_px
    value = float(spec.background_gray)
    return cv2.copyMakeBorder(
        image,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=(value, value, value),
    )


def _apply_perspective(
    image: np.ndarray, spec: FormSpec, rng: np.random.Generator
) -> np.ndarray:
    """원근 왜곡을 적용한다(네 꼭짓점을 중심 방향으로 결정론적으로 당긴다).

    :param image: 입력 이미지.
    :param spec: 렌더링 사양.
    :param rng: 시드 고정 난수 생성기.
    :returns: 왜곡된 이미지(크기 동일).
    """
    height, width = image.shape[:2]
    src = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    max_shift = spec.perspective_strength * 0.15 * float(min(width, height))
    weights = rng.uniform(0.2, 1.0, size=4).astype(np.float32)
    dst = np.empty_like(src)
    for index in range(4):
        direction = center - src[index]
        norm = float(np.linalg.norm(direction))
        unit = direction / norm if norm > 0 else direction
        dst[index] = src[index] + unit * (max_shift * weights[index])
    matrix = cv2.getPerspectiveTransform(src, dst)
    value = float(spec.background_gray)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(value, value, value),
    )


def _apply_rotation(image: np.ndarray, spec: FormSpec) -> np.ndarray:
    """이미지를 회전한다(잘림이 없도록 캔버스를 확장한다).

    :param image: 입력 이미지.
    :param spec: 렌더링 사양. ``rotation_deg`` 양수는 반시계 방향.
    :returns: 회전된 이미지.
    """
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, spec.rotation_deg, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int(round(height * sin + width * cos))
    new_height = int(round(height * cos + width * sin))
    matrix[0, 2] += new_width / 2.0 - center[0]
    matrix[1, 2] += new_height / 2.0 - center[1]
    value = float(spec.background_gray)
    return cv2.warpAffine(
        image,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(value, value, value),
    )


def _apply_illumination(image: np.ndarray, spec: FormSpec) -> np.ndarray:
    """대각 방향 조명 불균일을 적용한다.

    :param image: 입력 이미지.
    :param spec: 렌더링 사양.
    :returns: 밝기 기울기가 적용된 이미지.
    """
    height, width = image.shape[:2]
    strength = spec.illumination_gradient
    ramp_x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ramp_y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    field = (ramp_y[:, None] + ramp_x[None, :]) / 2.0
    gain = (1.0 + strength * 0.5) - strength * field
    result = image.astype(np.float32) * gain[:, :, None]
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _apply_noise(
    image: np.ndarray, spec: FormSpec, rng: np.random.Generator
) -> np.ndarray:
    """가우시안 잡음을 더한다.

    :param image: 입력 이미지.
    :param spec: 렌더링 사양.
    :param rng: 시드 고정 난수 생성기.
    :returns: 잡음이 더해진 이미지.
    """
    noise = rng.normal(0.0, spec.noise_sigma, size=image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)


def _apply_jpeg(image: np.ndarray, spec: FormSpec) -> np.ndarray:
    """JPEG 재압축 아티팩트를 적용한다.

    :param image: 입력 이미지.
    :param spec: 렌더링 사양.
    :returns: 재압축 후 디코딩된 이미지.
    :raises RuntimeError: JPEG 인코딩·디코딩에 실패한 경우(조용한 실패 금지).
    """
    ok, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(spec.jpeg_quality)]
    )
    if not ok:
        raise RuntimeError("합성 이미지 JPEG 인코딩에 실패했습니다.")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("합성 이미지 JPEG 디코딩에 실패했습니다.")
    return decoded


def _degrade(flat_bgr: np.ndarray, spec: FormSpec) -> np.ndarray:
    """기하 변형과 화질 저하를 순서대로 적용한다(결정론적).

    적용 순서: 여백 → 원근 → 회전 → 조명 → 블러 → 잡음 → JPEG.
    난수는 매 호출마다 ``spec.seed`` 로 새로 초기화하므로, 같은 사양이면
    입력 그림이 달라도(원본/작성본) **기하 변형이 완전히 동일**하다.

    :param flat_bgr: 변형 이전 ``(H, W, 3)`` uint8 BGR 페이지 이미지.
    :param spec: 렌더링 사양.
    :returns: 변형이 적용된 ``(H, W, 3)`` uint8 BGR 이미지.
    """
    rng = np.random.default_rng(spec.seed)
    image = flat_bgr
    if spec.margin_px > 0:
        image = _pad_background(image, spec)
    if spec.perspective_strength > 0.0:
        image = _apply_perspective(image, spec, rng)
    if abs(spec.rotation_deg) > 1e-9:
        image = _apply_rotation(image, spec)
    if spec.illumination_gradient > 0.0:
        image = _apply_illumination(image, spec)
    if spec.blur_ksize > 1:
        image = cv2.GaussianBlur(image, (spec.blur_ksize, spec.blur_ksize), 0)
    if spec.noise_sigma > 0.0:
        image = _apply_noise(image, spec, rng)
    if spec.jpeg_quality < 100:
        image = _apply_jpeg(image, spec)
    return np.ascontiguousarray(image, dtype=np.uint8)


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------


def make_application_form(spec: FormSpec | None = None) -> SyntheticForm:
    """정답을 아는 합성 한국어 신청서를 렌더링한다.

    :param spec: 렌더링 사양. ``None`` 이면 변형 없는 기본 사양을 쓴다.
    :returns: :class:`SyntheticForm`.
        ``image`` 는 변형 적용 BGR 배열, ``truth`` 는 변형 이전 mm 정답 구조다.
    :raises ValueError: 사양 파라미터가 허용 범위를 벗어난 경우.
    """
    spec = spec if spec is not None else FormSpec()
    layout = build_layout(spec)
    flat_pil = _draw_flat(layout, spec)
    flat_bgr = _pil_to_bgr(flat_pil)
    image = _degrade(flat_bgr, spec)
    truth = build_truth(spec, layout=layout)
    return SyntheticForm(
        image=image,
        truth=truth,
        flat_image=flat_bgr,
        layout=layout,
        spec=spec,
        dpi=spec.dpi,
        page_px=page_size_px(spec.dpi),
    )


def render_written(
    form: SyntheticForm,
    checked_option: str | None = AGREE_LABEL,
    sign: bool = True,
    *,
    sign_representative: bool = False,
) -> np.ndarray:
    """체크·서명이 기입된 이미지를 **동일한 기하 변형**으로 렌더링한다.

    반환 이미지는 :attr:`SyntheticForm.image` 와 픽셀 단위로 정렬되어 있으므로,
    Verify 단계가 두 이미지의 잉크 비율을 같은 좌표에서 비교할 수 있다.

    :param form: :func:`make_application_form` 결과.
    :param checked_option: 체크할 선택지 라벨(예: ``"동의함"``). ``None`` 이면 미체크.
    :param sign: 신청인 서명 곡선을 그릴지 여부.
    :param sign_representative: 대리인 서명 곡선을 그릴지 여부.
        대리인 서명란이 없는 사양에서 True 를 주면 :class:`ValueError`.
    :returns: ``(H, W, 3)`` uint8 BGR 배열. ``form.image`` 와 크기가 같다.
    :raises ValueError: 정의되지 않은 선택지 라벨이거나 없는 서명란을 요청한 경우.
    """
    flat_pil = Image.fromarray(
        np.ascontiguousarray(form.flat_image[:, :, ::-1]), mode="RGB"
    )
    marked = _draw_marks(
        flat_pil,
        form.layout,
        form.spec,
        checked_option=checked_option,
        sign=sign,
        sign_representative=sign_representative,
    )
    return _degrade(_pil_to_bgr(marked), form.spec)


def expected_written_truth(
    form: SyntheticForm,
    checked_option: str | None = AGREE_LABEL,
    sign: bool = True,
    *,
    sign_representative: bool = False,
) -> DocumentStructure:
    """작성 후 상태에 대응하는 **기대 정답** 구조를 만든다.

    :func:`render_written` 과 같은 인자를 주면, Verify 모듈이 산출해야 할
    정답(선택지 ``checked`` 값)이 채워진 구조를 얻는다.

    :param form: :func:`make_application_form` 결과.
    :param checked_option: 체크된 선택지 라벨. ``None`` 이면 모든 선택지가 ``False``.
    :param sign: 신청인 서명 여부. 현재 구조 표현에는 영향이 없고 문서화 목적이다.
    :param sign_representative: 대리인 서명 여부. 동일.
    :returns: ``options[].checked`` 가 확정된 :class:`DocumentStructure`.
    :raises ValueError: 정의되지 않은 선택지 라벨인 경우.
    """
    if checked_option is not None:
        form.layout.option_box(checked_option)  # 라벨 검증(미정의면 ValueError)
    updated_fields: list[Field] = []
    for item in form.truth.fields:
        if item.id == CONSENT_FIELD_ID and item.options:
            updated_fields.append(
                Field(
                    id=item.id,
                    type=item.type,
                    title=item.title,
                    role=item.role,
                    options=tuple(
                        Option(
                            label=option.label,
                            box_mm=option.box_mm,
                            checked=(option.label == checked_option),
                        )
                        for option in item.options
                    ),
                    required=item.required,
                    sensitivity=item.sensitivity,
                    box_mm=item.box_mm,
                    clause_text=item.clause_text,
                    order=item.order,
                    confidence=item.confidence,
                )
            )
        else:
            updated_fields.append(item)
    return DocumentStructure(
        document_id=form.truth.document_id,
        doc_title=form.truth.doc_title,
        fields=tuple(updated_fields),
        page_size_mm=form.truth.page_size_mm,
        source_image=form.truth.source_image,
        warnings=form.truth.warnings,
    )


def iter_spec_grid(
    *,
    angles: Sequence[float] = (0.0, 10.0, -10.0, 20.0, -20.0),
    dpis: Sequence[int] = (150, 200, 300),
    count: int = 10,
    base: FormSpec | None = None,
    seed: int = DEFAULT_SEED,
) -> Iterable[FormSpec]:
    """데이터셋 생성을 위한 사양 조합을 결정론적으로 순회한다.

    각도·해상도를 순환 조합하고, 잡음·블러·JPEG·조명은 시드 고정 난수로 흔든다.

    :param angles: 회전각 후보(도).
    :param dpis: 해상도 후보(dpi).
    :param count: 생성할 사양 개수(1 이상).
    :param base: 기준 사양. ``None`` 이면 기본 :class:`FormSpec`.
    :param seed: 난수 시드.
    :returns: :class:`FormSpec` 제너레이터.
    :raises ValueError: ``count`` 가 1 미만이거나 후보 목록이 비어 있는 경우.
    """
    if count < 1:
        raise ValueError(f"생성 개수는 1 이상이어야 합니다: {count}")
    if not angles or not dpis:
        raise ValueError("각도·해상도 후보 목록이 비어 있습니다.")
    base_spec = base if base is not None else FormSpec()
    rng = np.random.default_rng(seed)
    for index in range(count):
        yield base_spec.with_(
            document_id=f"{base_spec.document_id}_{index:04d}",
            dpi=int(dpis[index % len(dpis)]),
            rotation_deg=float(angles[index % len(angles)]),
            perspective_strength=float(round(rng.uniform(0.0, 0.35), 4)),
            margin_px=int(rng.integers(20, 90)),
            background_gray=int(rng.integers(150, 225)),
            noise_sigma=float(round(rng.uniform(0.0, 6.0), 3)),
            blur_ksize=int(rng.choice([1, 1, 3, 5])),
            jpeg_quality=int(rng.integers(60, 101)),
            illumination_gradient=float(round(rng.uniform(0.0, 0.45), 4)),
            include_representative=bool(index % 3 == 0),
            seed=int(seed + index),
        )
