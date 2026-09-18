"""OCR 유틸(:mod:`docagent.vision.ocr`)과 구조화(:mod:`docagent.vision.structuring`) 테스트.

합성 신청서의 **레이아웃(mm 좌표)** 으로부터 실제 OCR 이 낼 법한 단어 목록을
결정론적으로 만들어 :class:`StubOcr` 에 주입한다. 따라서 ``pytesseract`` 나
Tesseract 실행 파일 없이도 전 항목이 통과한다.

핵심 단언은 "구조화 결과가 :func:`build_truth` 의 정답과 일치하는가" 다.
개수·유형·필수 여부·역할·민감도·제목·순서를 정답과 직접 대조한다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import ModuleType

import pytest

from docagent.contracts import (
    A4_PAGE_SIZE_MM,
    BoxMm,
    Detection,
    DocumentStructure,
    FieldRole,
    FieldType,
    OcrWord,
    Sensitivity,
)
from docagent.errors import AdapterUnavailable, DocumentNotFoundError
from docagent.testing.synthetic import FormLayout
from docagent.vision import ocr as ocr_module
from docagent.vision.ocr import (
    StubOcr,
    TesseractOcr,
    group_words_to_lines,
    lines_to_paragraphs,
    sort_reading_order,
    union_box,
)
from docagent.vision.structuring import (
    build_structure,
    classify_sensitivity,
    classify_signature_role,
    strip_markers,
)

# 합성 문서의 글자 높이(mm) — synthetic 의 load_font 호출과 같은 값을 쓴다.
TITLE_H_MM = 7.0
HEAD_H_MM = 4.2
BODY_H_MM = 3.6
SMALL_H_MM = 3.0

#: 탐지기 신뢰도(합성 탐지 결과에 부여).
DETECTION_CONFIDENCE = 0.92
#: 약관 본문 한 줄에 담을 최대 글자 수.
CLAUSE_WRAP_CHARS = 46


# --------------------------------------------------------------------------
# 합성 OCR 단어 생성
# --------------------------------------------------------------------------


def _char_width_mm(char: str, height_mm: float) -> float:
    """글자 하나의 렌더링 폭을 근사한다(한글 전각 / ASCII 반각).

    :param char: 글자 하나.
    :param height_mm: 글자 높이(mm).
    :returns: 폭(mm).
    """
    return height_mm * (0.95 if ord(char) > 0x2000 else 0.5)


def _words(
    text: str, x_mm: float, y_mm: float, height_mm: float, *, confidence: float = 0.95
) -> list[OcrWord]:
    """한 줄의 문자열을 공백 기준 단어 박스로 쪼갠다.

    :param text: 원문 한 줄.
    :param x_mm: 줄 시작 x(mm).
    :param y_mm: 줄 위쪽 y(mm).
    :param height_mm: 글자 높이(mm).
    :param confidence: 각 단어에 부여할 신뢰도.
    :returns: :class:`OcrWord` 목록.
    """
    space_mm = height_mm * 0.5
    result: list[OcrWord] = []
    cursor = x_mm
    for token in text.split(" "):
        if not token:
            cursor += space_mm
            continue
        width = sum(_char_width_mm(char, height_mm) for char in token)
        result.append(
            OcrWord(
                text=token,
                box_mm=BoxMm(cursor, y_mm, width, height_mm),
                confidence=confidence,
            )
        )
        cursor += width + space_mm
    return result


def _wrap(text: str, max_chars: int) -> list[str]:
    """공백 단위 그리디 줄바꿈(단어를 쪼개지 않는다).

    :param text: 원문.
    :param max_chars: 한 줄 최대 글자 수.
    :returns: 줄 목록. 공백 하나로 다시 이으면 원문과 같아진다.
    """
    lines: list[str] = []
    current = ""
    for token in text.split(" "):
        candidate = f"{current} {token}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def make_words(layout: FormLayout, sm: ModuleType) -> list[OcrWord]:
    """합성 신청서 레이아웃으로부터 OCR 단어 목록을 만든다.

    :func:`docagent.testing.synthetic._draw_flat` 이 실제로 인쇄하는 문자열과
    같은 내용·같은 mm 위치를 쓰므로, 실제 OCR 결과의 대역(代役)으로 충분하다.

    :param layout: :class:`docagent.testing.synthetic.FormLayout`.
    :param sm: :mod:`docagent.testing.synthetic` 모듈(배치 상수 참조용).
    :returns: :class:`OcrWord` 목록.
    :raises AssertionError: 약관 본문이 선택지 행을 침범할 만큼 길어진 경우.
    """
    words: list[OcrWord] = []

    # 제목(가운데 정렬 근사)
    words += _words(layout.doc_title, 70.0, layout.title_box_mm.y_mm, TITLE_H_MM)
    # 필수 마커 안내문
    words += _words(
        f"※ {sm.REQUIRED_MARKER} 표시 항목은 반드시 기입하여야 합니다.",
        sm._MARGIN_LEFT_MM,
        sm._NOTICE_Y_MM,
        SMALL_H_MM,
    )
    # 인적사항 표(라벨 + 인쇄된 예시 값)
    for row in layout.person_rows:
        label = f"{row.label} {sm.REQUIRED_MARKER}" if row.required else row.label
        words += _words(
            label, row.label_box_mm.x_mm + 3.0, row.label_box_mm.y_mm + 3.0, BODY_H_MM
        )
        words += _words(
            row.example,
            row.value_box_mm.x_mm + 4.0,
            row.value_box_mm.y_mm + 3.0,
            BODY_H_MM,
        )
    # 약관 소제목 + 본문
    words += _words(
        "* 개인정보 수집·이용 동의", sm._MARGIN_LEFT_MM, sm._CLAUSE_HEADER_Y_MM, HEAD_H_MM
    )
    clause_lines = _wrap(layout.clause_text, CLAUSE_WRAP_CHARS)
    assert len(clause_lines) <= 6, "약관 본문이 선택지 행을 침범할 만큼 길어졌습니다."
    for index, line in enumerate(clause_lines):
        words += _words(
            line,
            sm._MARGIN_LEFT_MM,
            sm._CLAUSE_BODY_TOP_MM + index * sm._CLAUSE_LINE_H_MM,
            BODY_H_MM,
        )
    # 선택지 라벨
    for slot in layout.options:
        words += _words(slot.label, slot.label_x_mm, slot.box_mm.y_mm + 0.6, BODY_H_MM)
    # 확인 문장
    words += _words(
        "위와 같이 ○○지원금 지급을 신청합니다.",
        sm._MARGIN_LEFT_MM,
        sm._STATEMENT_Y_MM,
        BODY_H_MM,
    )
    # 신청일자 라벨
    words += _words(
        f"신청일자 {sm.REQUIRED_MARKER}",
        sm._MARGIN_LEFT_MM,
        sm._DATE_LABEL_Y_MM,
        BODY_H_MM,
    )
    # 서명란 라벨 + 안내
    for slot in layout.signatures:
        words += _words(
            slot.label, sm._MARGIN_LEFT_MM, slot.box_mm.y_mm + 3.0, BODY_H_MM
        )
        words += _words(
            "(서명 또는 인)", sm._SIGN_LINE_X1_MM + 4.0, slot.box_mm.y_mm + 3.0, SMALL_H_MM
        )
    return words


def make_detections(
    layout: FormLayout, *, date_as_text: bool = True
) -> list[Detection]:
    """합성 신청서 레이아웃으로부터 탐지 결과를 만든다.

    :param layout: :class:`FormLayout`.
    :param date_as_text: True 면 신청일자란을 :attr:`FieldType.TEXT_INPUT` 으로 내보내
        구조화 단계의 DATE 재분류 규칙을 시험한다.
    :returns: :class:`Detection` 목록.
    """
    detections: list[Detection] = []
    for row in layout.person_rows:
        detections.append(
            Detection(
                type=FieldType.TEXT_INPUT,
                box_mm=row.value_box_mm,
                confidence=DETECTION_CONFIDENCE,
                source="opencv_contour",
            )
        )
    for slot in layout.options:
        detections.append(
            Detection(
                type=FieldType.CHECKBOX,
                box_mm=slot.box_mm,
                confidence=DETECTION_CONFIDENCE,
                source="opencv_contour",
            )
        )
    detections.append(
        Detection(
            type=FieldType.TEXT_INPUT if date_as_text else FieldType.DATE,
            box_mm=layout.date_slot,
            confidence=DETECTION_CONFIDENCE,
            source="opencv_contour",
        )
    )
    for slot in layout.signatures:
        detections.append(
            Detection(
                type=FieldType.SIGNATURE,
                box_mm=slot.box_mm,
                confidence=DETECTION_CONFIDENCE,
                source="opencv_contour",
            )
        )
    return detections


@pytest.fixture(scope="module")
def layout(synthetic_module: ModuleType) -> FormLayout:
    """대리인 서명란을 포함한 합성 신청서 배치."""
    spec = synthetic_module.FormSpec(
        document_id="structuring_fixture", include_representative=True
    )
    return synthetic_module.build_layout(spec)


@pytest.fixture(scope="module")
def truth(synthetic_module: ModuleType) -> DocumentStructure:
    """``layout`` 과 짝을 이루는 정답 구조."""
    spec = synthetic_module.FormSpec(
        document_id="structuring_fixture", include_representative=True
    )
    return synthetic_module.build_truth(spec)


@pytest.fixture(scope="module")
def stub_words(layout: FormLayout, synthetic_module: ModuleType) -> list[OcrWord]:
    """합성 OCR 단어 목록."""
    return make_words(layout, synthetic_module)


@pytest.fixture(scope="module")
def structure(layout: FormLayout, stub_words: list[OcrWord]) -> DocumentStructure:
    """구조화 결과(정답 대조 대상)."""
    engine = StubOcr(stub_words)
    return build_structure(
        make_detections(layout),
        engine.read(),
        A4_PAGE_SIZE_MM,
        "structuring_fixture",
    )


# --------------------------------------------------------------------------
# OCR 유틸
# --------------------------------------------------------------------------


class TestOcrModule:
    """:mod:`docagent.vision.ocr` 의 어댑터와 그룹핑 유틸."""

    def test_optional_package_is_not_imported_at_module_level(self) -> None:
        """모듈 최상단에 ``pytesseract`` import 가 없어야 한다(선택적 어댑터 규약)."""
        source = Path(ocr_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.append(node.module)
        assert not any("pytesseract" in name for name in top_level)
        for forbidden in ("torch", "ultralytics", "anthropic"):
            assert not any(forbidden in name for name in top_level)

    def test_tesseract_adapter_reports_missing_package_in_korean(self) -> None:
        """``pytesseract`` 미설치 시 한국어 설치 안내가 담긴 예외를 던진다."""
        engine = TesseractOcr()
        dummy = [[0] * 8 for _ in range(8)]

        class _Fake:
            shape = (8, 8)

        with pytest.raises(AdapterUnavailable) as info:
            engine.read(_Fake())
        message = str(info.value)
        assert "pytesseract" in message
        assert "설치" in message
        assert ".venv" in message
        assert dummy is not None  # 사용하지 않는 더미 데이터 경고 방지

    def test_tesseract_adapter_validates_arguments(self) -> None:
        """생성자 인자 검증은 한국어 ValueError 로 즉시 실패한다."""
        with pytest.raises(ValueError):
            TesseractOcr(lang="")
        with pytest.raises(ValueError):
            TesseractOcr(min_confidence=1.5)
        with pytest.raises(ValueError):
            TesseractOcr(page_size_mm=(0.0, 297.0))

    def test_stub_ocr_is_deterministic_and_ordered(self, stub_words: list[OcrWord]) -> None:
        """StubOcr 는 항상 같은 결과를 읽기 순서로 돌려준다."""
        engine = StubOcr(stub_words)
        first = engine.read(None)
        second = engine.read(None)
        assert first == second
        assert first is not second  # 방어적 복사
        tops = [word.box_mm.y_mm for word in first]
        assert tops == sorted(tops) or all(
            tops[i] <= tops[i + 1] + 3.0 for i in range(len(tops) - 1)
        )

    def test_stub_ocr_rejects_invalid_words(self) -> None:
        """OcrWord 가 아닌 값이나 범위를 벗어난 신뢰도는 거부한다."""
        with pytest.raises(ValueError):
            StubOcr(["성명"])  # type: ignore[list-item]
        with pytest.raises(ValueError):
            StubOcr([OcrWord("성명", BoxMm(0, 0, 5, 4), 1.4)])
        with pytest.raises(ValueError):
            StubOcr.from_texts([("성명", BoxMm(0, 0, 5, 4))], confidence=-0.1)

    def test_from_texts_builds_equivalent_engine(self) -> None:
        """``from_texts`` 는 OcrWord 를 직접 주입한 것과 같은 결과를 만든다."""
        entries = [("성명", BoxMm(20, 46, 8, 4)), ("주소", BoxMm(20, 68, 8, 4))]
        engine = StubOcr.from_texts(entries, confidence=0.8)
        assert [word.text for word in engine.read()] == ["성명", "주소"]
        assert all(word.confidence == 0.8 for word in engine.read())

    def test_group_words_to_lines_keeps_row_together(self, stub_words: list[OcrWord]) -> None:
        """같은 행의 단어는 한 줄로 묶이고 x 오름차순으로 이어진다."""
        lines = group_words_to_lines(stub_words)
        texts = [line.text for line in lines]
        assert "* 개인정보 수집·이용 동의" in texts
        assert any(text.startswith("성명 [필수]") for text in texts)
        header = next(line for line in lines if line.text.startswith("*"))
        xs = [word.box_mm.x_mm for word in header.words]
        assert xs == sorted(xs)

    def test_lines_to_paragraphs_splits_heading_from_body(
        self, stub_words: list[OcrWord], synthetic_module: ModuleType
    ) -> None:
        """약관 소제목과 본문은 서로 다른 문단으로 갈린다."""
        lines = group_words_to_lines(stub_words)
        clause_lines = [
            line
            for line in lines
            if synthetic_module._CLAUSE_BODY_TOP_MM - 1.0
            <= line.box_mm.y_mm
            <= synthetic_module._OPTION_ROW_Y_MM
        ]
        paragraphs = lines_to_paragraphs(clause_lines)
        assert len(paragraphs) == 1
        assert paragraphs[0].text == synthetic_module.CONSENT_CLAUSE_TEXT

    def test_grouping_utilities_validate_arguments(self) -> None:
        """음수 허용오차는 한국어 ValueError."""
        with pytest.raises(ValueError):
            group_words_to_lines([], y_tol_mm=-1.0)
        with pytest.raises(ValueError):
            lines_to_paragraphs([], gap_mm=-1.0)
        with pytest.raises(ValueError):
            union_box([])

    def test_empty_input_returns_empty_containers(self) -> None:
        """빈 입력은 빈 결과를 돌려준다(예외 아님)."""
        assert group_words_to_lines([]) == []
        assert lines_to_paragraphs([]) == []
        assert sort_reading_order([]) == []


# --------------------------------------------------------------------------
# 구조화 — 정답 대조
# --------------------------------------------------------------------------


class TestBuildStructureAgainstTruth:
    """구조화 결과를 :func:`build_truth` 정답과 직접 대조한다."""

    def test_field_count_matches(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """항목 개수가 정답과 같다."""
        assert len(structure.fields) == len(truth.fields) == 8

    def test_field_types_match_in_reading_order(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """읽기 순서대로 유형이 정답과 일치한다(신청일자란은 DATE 로 재분류)."""
        assert [field.type for field in structure.fields] == [
            field.type for field in truth.fields
        ]

    def test_titles_match(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """항목명이 정답과 일치한다(마커 제거 포함)."""
        assert [field.title for field in structure.fields] == [
            field.title for field in truth.fields
        ]

    def test_required_flags_match(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """필수 여부가 정답과 일치한다(마커 · 별표 규칙 · 서명 역할 규칙)."""
        assert [field.required for field in structure.fields] == [
            field.required for field in truth.fields
        ]

    def test_roles_match(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """기입 주체가 정답과 일치한다(신청인 / 대리인 구분 포함)."""
        assert [field.role for field in structure.fields] == [
            field.role for field in truth.fields
        ]

    def test_sensitivity_matches(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """민감도가 정답과 일치한다(약관만 PUBLIC)."""
        assert [field.sensitivity for field in structure.fields] == [
            field.sensitivity for field in truth.fields
        ]
        public = [f for f in structure.fields if f.sensitivity is Sensitivity.PUBLIC]
        assert len(public) == 1
        assert public[0].type is FieldType.CHOICE

    def test_boxes_are_close_to_truth(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """좌표가 정답과 3mm 이내로 일치한다."""
        for got, expected in zip(structure.fields, truth.fields):
            assert got.box_mm is not None and expected.box_mm is not None
            assert abs(got.box_mm.y_mm - expected.box_mm.y_mm) <= 3.0
            assert abs(got.box_mm.x_mm - expected.box_mm.x_mm) <= 3.0

    def test_order_is_dense_and_sorted(self, structure: DocumentStructure) -> None:
        """order 는 0 부터 빈틈없이 읽기 순서로 매겨진다."""
        assert [field.order for field in structure.fields] == list(
            range(len(structure.fields))
        )
        tops = [field.box_mm.y_mm for field in structure.fields]
        assert tops == sorted(tops)

    def test_ids_are_deterministic_by_type(self, structure: DocumentStructure) -> None:
        """id 는 타입별 접두사 + 일련번호로 결정론적이다."""
        assert [field.id for field in structure.fields] == [
            "text_01",
            "text_02",
            "text_03",
            "text_04",
            "consent_01",
            "date_01",
            "signature_01",
            "signature_02",
        ]

    def test_doc_title_is_inferred_from_top_line(self, structure: DocumentStructure, truth: DocumentStructure) -> None:
        """문서 제목은 최상단 행에서 유추하며 ``○○`` 를 줄머리로 오인하지 않는다."""
        assert structure.doc_title == truth.doc_title

    def test_choice_options_and_clause(self, structure: DocumentStructure, synthetic_module: ModuleType) -> None:
        """선택지 라벨 2개와 약관 본문이 그대로 복원된다."""
        choice = next(f for f in structure.fields if f.type is FieldType.CHOICE)
        assert [option.label for option in choice.options] == [
            synthetic_module.AGREE_LABEL,
            synthetic_module.DISAGREE_LABEL,
        ]
        assert all(option.checked is None for option in choice.options)
        assert choice.clause_text == synthetic_module.CONSENT_CLAUSE_TEXT
        assert choice.required is True

    def test_signature_roles_are_distinguished(self, structure: DocumentStructure) -> None:
        """서명란 둘의 역할이 각각 신청인 / 대리인으로 갈린다."""
        signatures = [f for f in structure.fields if f.type is FieldType.SIGNATURE]
        assert [f.role for f in signatures] == [
            FieldRole.APPLICANT,
            FieldRole.REPRESENTATIVE,
        ]
        assert signatures[0].required is True
        assert signatures[1].required is False

    def test_no_warnings_on_clean_document(self, structure: DocumentStructure) -> None:
        """모든 항목을 확신할 수 있으면 경고가 남지 않는다."""
        assert structure.warnings == ()
        assert all(field.confidence >= 0.85 for field in structure.fields)

    def test_json_roundtrip(self, structure: DocumentStructure) -> None:
        """fields JSON 왕복이 무손실이다(세션 복원 근거)."""
        assert DocumentStructure.from_json(structure.to_json()) == structure


class TestPublicPayloadIsolation:
    """개인정보 항목이 외부 전송 payload 로 새지 않는지 확인한다."""

    def test_private_fields_are_redacted(self, structure: DocumentStructure) -> None:
        """PRIVATE 항목은 구조 정보만 남고 제목·좌표가 제거된다."""
        payload = structure.public_payload()
        private_ids = [
            field.id
            for field in structure.fields
            if field.sensitivity is Sensitivity.PRIVATE
        ]
        assert payload["redacted_field_ids"] == private_ids
        for entry in payload["fields"]:
            if entry.get("redacted"):
                assert set(entry) == {
                    "id",
                    "type",
                    "role",
                    "required",
                    "order",
                    "sensitivity",
                    "redacted",
                }

    def test_private_titles_do_not_leak(self, structure: DocumentStructure) -> None:
        """개인정보 항목의 제목·예시 값이 payload 문자열에 나타나지 않는다."""
        payload = json.dumps(structure.public_payload(), ensure_ascii=False)
        for leaked in ("신청인 서명", "대리인 서명", "홍길동", "900101-1234567"):
            assert leaked not in payload

    def test_public_clause_is_kept(self, structure: DocumentStructure, synthetic_module: ModuleType) -> None:
        """공개 영역인 약관 본문은 그대로 남아 RAG 근거로 쓸 수 있다."""
        payload = json.dumps(structure.public_payload(), ensure_ascii=False)
        assert synthetic_module.CONSENT_CLAUSE_TEXT in payload


class TestClassificationRules:
    """개별 분류 규칙의 경계 동작."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("* 개인정보 수집·이용 동의", "개인정보 수집·이용 동의"),
            ("성명 [필수]", "성명"),
            ("주소", "주소"),
            ("○○지원금 지급 신청서", "○○지원금 지급 신청서"),
            ("※ 안내", "안내"),
        ],
    )
    def test_strip_markers(self, text: str, expected: str) -> None:
        """마커 제거는 낱말 안의 기호를 건드리지 않는다."""
        assert strip_markers(text) == expected

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("신청인", FieldRole.APPLICANT),
            ("본인", FieldRole.APPLICANT),
            ("법정대리인", FieldRole.REPRESENTATIVE),
            ("보호자", FieldRole.REPRESENTATIVE),
            ("담당자", FieldRole.OFFICIAL),
            ("", FieldRole.UNKNOWN),
            ("서명란", FieldRole.UNKNOWN),
        ],
    )
    def test_signature_role_keywords(self, label: str, expected: FieldRole) -> None:
        """역할 키워드표대로 분류된다."""
        role, _ = classify_signature_role(label)
        assert role is expected

    def test_conflicting_role_keywords_are_ambiguous(self) -> None:
        """서로 다른 역할 키워드가 함께 있으면 UNKNOWN + 모호 판정이다."""
        role, ambiguous = classify_signature_role("신청인 또는 대리인")
        assert role is FieldRole.UNKNOWN
        assert ambiguous is True

    @pytest.mark.parametrize(
        ("field_type", "title", "expected"),
        [
            (FieldType.CHOICE, "개인정보 수집·이용 동의", Sensitivity.PUBLIC),
            (FieldType.CHECKBOX, "마케팅 활용 동의", Sensitivity.PUBLIC),
            (FieldType.CHOICE, "계좌 선택", Sensitivity.PRIVATE),
            (FieldType.TEXT_INPUT, "무해한 제목", Sensitivity.PRIVATE),
            (FieldType.UNKNOWN, "", Sensitivity.PRIVATE),
        ],
    )
    def test_sensitivity_is_fail_closed(
        self, field_type: FieldType, title: str, expected: Sensitivity
    ) -> None:
        """분류가 애매하면 PRIVATE 로 떨어진다."""
        assert classify_sensitivity(field_type, title) is expected

    def test_private_field_keeps_its_printed_clause_text(self) -> None:
        """PRIVATE 로 분류된 항목도 인쇄된 조항 본문(clause_text)을 잃지 않는다.

        clause_text 는 문서에 인쇄된 공개 정보이므로 구조화 단계에서 지우면
        로컬 '원문 듣기' 경로까지 함께 막힌다. 외부 전송 차단은
        ``public_payload()`` 와 PiiGate 의 몫이다.
        """
        detections = (
            Detection(
                type=FieldType.CHECKBOX,
                box_mm=BoxMm(30.0, 130.0, 5.0, 5.0),
                confidence=0.95,
                source="test",
            ),
        )

        def line(text: str, x: float, y: float) -> OcrWord:
            return OcrWord(
                text=text,
                box_mm=BoxMm(x, y, len(text) * 3.0, 4.0),
                confidence=0.95,
            )

        words = (
            line("1. 주민등록번호 수집·이용 동의", 25.0, 110.0),
            line("보유 기간은 5년입니다.", 25.0, 120.0),
        )
        structure = build_structure(
            detections, words, A4_PAGE_SIZE_MM, "doc_private_clause"
        )
        target = [f for f in structure.fields if f.sensitivity is Sensitivity.PRIVATE]
        assert target, "PRIVATE 항목이 만들어지지 않아 회귀를 검증할 수 없습니다."
        assert any(item.clause_text for item in target), (
            "PRIVATE 항목의 clause_text 가 비어 있습니다 — 원문 낭독이 불가능해집니다."
        )
        # 외부 전송용 payload 에서는 여전히 강등된다(이중 방어는 여기서 담당).
        for entry in structure.public_payload()["fields"]:
            if entry.get("redacted"):
                assert "clause_text" not in entry


class TestAmbiguityAndFailureModes:
    """저신뢰·모호 상황에서 조용히 넘어가지 않는지 확인한다."""

    def test_unlabeled_signature_raises_warning(self, layout: FormLayout, stub_words: list[OcrWord]) -> None:
        """라벨 없는 서명란은 역할 미확정 + 경고 + 낮은 신뢰도로 남는다."""
        words = [
            word
            for word in stub_words
            if word.text not in ("신청인", "대리인")
        ]
        structure = build_structure(
            make_detections(layout), words, A4_PAGE_SIZE_MM, "ambiguous_doc"
        )
        signatures = [f for f in structure.fields if f.type is FieldType.SIGNATURE]
        assert all(f.role is FieldRole.UNKNOWN for f in signatures)
        assert any("기입 주체를 확정하지 못했습니다" in w for w in structure.warnings)
        assert any("직원 연결" in w for w in structure.warnings)
        assert all(f.confidence < 0.85 for f in signatures)

    def test_duplicated_signature_roles_are_reported(self, layout: FormLayout, stub_words: list[OcrWord]) -> None:
        """서명란 두 곳이 같은 역할로 읽히면 경고를 남긴다."""
        words = [
            OcrWord("신청인", word.box_mm, word.confidence)
            if word.text == "대리인"
            else word
            for word in stub_words
        ]
        structure = build_structure(
            make_detections(layout), words, A4_PAGE_SIZE_MM, "duplicated_doc"
        )
        assert any("역할이 중복되었습니다" in w for w in structure.warnings)

    def test_missing_labels_produce_warnings(self, layout: FormLayout) -> None:
        """OCR 결과가 전혀 없으면 항목명 경고가 항목 수만큼 쌓인다."""
        structure = build_structure(
            make_detections(layout), [], A4_PAGE_SIZE_MM, "no_text_doc"
        )
        assert structure.warnings
        assert any("항목명을 찾지 못했습니다" in w for w in structure.warnings)

    def test_empty_input_raises_document_not_found(self) -> None:
        """탐지·OCR 이 모두 비면 조용히 빈 구조를 돌려주지 않는다."""
        with pytest.raises(DocumentNotFoundError):
            build_structure([], [], A4_PAGE_SIZE_MM, "empty_doc")

    def test_invalid_arguments_are_rejected(self, layout: FormLayout, stub_words: list[OcrWord]) -> None:
        """document_id · coords 검증은 한국어 ValueError."""
        detections = make_detections(layout)
        with pytest.raises(ValueError):
            build_structure(detections, stub_words, A4_PAGE_SIZE_MM, "")
        with pytest.raises(ValueError):
            build_structure(detections, stub_words, (0.0, 297.0), "bad_coords")

    def test_single_checkbox_becomes_checkbox_field(self, layout: FormLayout, stub_words: list[OcrWord]) -> None:
        """체크박스가 하나뿐이면 CHOICE 가 아니라 CHECKBOX 로 둔다."""
        detections = [
            item
            for item in make_detections(layout)
            if item.type is not FieldType.CHECKBOX
        ]
        detections.append(
            Detection(
                type=FieldType.CHECKBOX,
                box_mm=layout.options[0].box_mm,
                confidence=DETECTION_CONFIDENCE,
                source="opencv_contour",
            )
        )
        structure = build_structure(
            detections, stub_words, A4_PAGE_SIZE_MM, "single_checkbox"
        )
        single = next(f for f in structure.fields if f.type is FieldType.CHECKBOX)
        assert len(single.options) == 1
        assert single.id.startswith("checkbox_")

    def test_date_detection_is_preserved(self, layout: FormLayout, stub_words: list[OcrWord]) -> None:
        """탐지기가 이미 DATE 로 준 경우에도 DATE 로 남는다."""
        structure = build_structure(
            make_detections(layout, date_as_text=False),
            stub_words,
            A4_PAGE_SIZE_MM,
            "date_doc",
        )
        date_field = next(f for f in structure.fields if f.type is FieldType.DATE)
        assert date_field.title == "신청일자"
        assert date_field.required is True


from docagent.contracts import VISION_TRUST_THRESHOLD  # noqa: E402


class TestOfficialOnlyFieldsAreFlagged:
    """직원 전용 칸이 이용자 몫으로 안내되지 않아야 한다.

    역할 판정이 서명란에만 걸려 있으면 '담당자 확인란' 같은 입력란·체크박스가
    신뢰도 0.95 짜리 APPLICANT 항목으로 확정되어, 시각장애인 이용자가 직원
    전용 칸을 자기 칸으로 안내받는다.
    """

    def _structure_with_label(self, label: str) -> DocumentStructure:
        """라벨 왼쪽에 두고 입력란 하나만 있는 구조를 만든다."""
        detections = [
            Detection(
                type=FieldType.TEXT_INPUT,
                box_mm=BoxMm(60.0, 50.0, 50.0, 8.0),
                confidence=0.95,
                source="test",
            )
        ]
        words = [
            OcrWord(text=label, box_mm=BoxMm(20.0, 50.0, 30.0, 6.0), confidence=0.95)
        ]
        return build_structure(detections, words, A4_PAGE_SIZE_MM, "official_doc")

    def test_official_input_is_not_labelled_applicant(self) -> None:
        """'담당자' 라벨이 붙은 입력란은 APPLICANT 로 확정되지 않는다."""
        structure = self._structure_with_label("담당자 확인란")
        field = structure.fields[0]
        assert field.role is FieldRole.OFFICIAL

    def test_official_input_is_warned_and_distrusted(self) -> None:
        """직원 전용 칸은 경고로 남고 신뢰도가 신뢰 하한 아래로 내려간다."""
        structure = self._structure_with_label("담당자 확인란")
        field = structure.fields[0]
        assert field.confidence < VISION_TRUST_THRESHOLD
        assert any("직원" in warning for warning in structure.warnings)

    def test_applicant_input_is_unchanged(self) -> None:
        """신청인 칸은 종전대로 APPLICANT 이고 신뢰도도 깎이지 않는다."""
        structure = self._structure_with_label("신청인 성명")
        field = structure.fields[0]
        assert field.role is FieldRole.APPLICANT
        assert field.confidence >= VISION_TRUST_THRESHOLD

    def test_unlabelled_input_keeps_applicant_default(self) -> None:
        """역할 키워드가 없으면 기존 기본값(APPLICANT)을 유지한다."""
        structure = self._structure_with_label("연락처")
        assert structure.fields[0].role is FieldRole.APPLICANT
