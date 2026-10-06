"""탐지 결과 + OCR 결과 → :class:`DocumentStructure` (= fields JSON) 구조화.

이 모듈이 만드는 :meth:`DocumentStructure.to_dict` 결과가 **Vision 과 Agent
사이의 유일한 공식 인터페이스**다. 여기서 잘못 분류된 항목은 이후 단계 전체를
오염시키므로, 모든 판단 규칙을 모듈 상수로 노출하고 애매하면 **보수적으로**
(개인정보 쪽 / 역할 미확정 쪽으로) 떨어뜨린다.

입력
----
``detections``
    :class:`~docagent.contracts.Detection` 목록. 좌표는 이미 **A4 mm** 로
    변환되어 있어야 한다(픽셀 좌표는 모듈 경계를 넘지 못한다).
``words``
    :class:`~docagent.contracts.OcrWord` 목록. 좌표 규약은 동일하다.
``coords``
    페이지 크기 ``(가로_mm, 세로_mm)``. 새 계약 타입을 만들지 않기 위해
    기존 ``tuple[float, float]`` 을 그대로 쓴다(기본 A4).
``document_id``
    세션 복원 키가 되는 문서 인스턴스 식별자.

판단 규칙 요약
--------------
1. **체크박스 그룹핑** — 같은 행(y 중심 차이 ≤ :data:`CHECKBOX_ROW_TOL_MM`,
   가로 간격 ≤ :data:`CHECKBOX_ROW_MAX_GAP_MM`) 또는 인접 행(가로로 겹치고
   세로 간격 ≤ :data:`CHECKBOX_STACK_GAP_MM`)의 체크박스를 하나로 묶는다.
   2개 이상이면 :attr:`FieldType.CHOICE`, 1개뿐이면 :attr:`FieldType.CHECKBOX`.
2. **선택지 라벨** — 같은 행의 좌/우 배치를 비교하고 문구를 중복 할당하지 않는다.
   배치가 모호하면 빈 라벨로 남긴다. 허용 간격은 :data:`OPTION_LABEL_MAX_GAP_MM`.
3. **제목·약관** — 그룹 위쪽으로 행을 거슬러 올라가며
   :data:`CLAUSE_MAX_LINE_GAP_MM` 이내로 이어지는 문단을 ``clause_text`` 로 모으고,
   소제목 패턴(:data:`HEADING_PREFIXES`)을 만나면 그 행을 ``title`` 로 확정하고 멈춘다.
   소제목이 없으면 선택지 라벨로 제목을 유추한다.
4. **필수 여부** — :data:`EXPLICIT_REQUIRED_MARKERS` 는 항상 필수로 본다.
   ``*`` 는 문서 상단에 마커 안내문(:data:`STAR_NOTICE_PATTERN`)이 있을 때만
   필수 마커로 활성화한다. 서명란은 :data:`SIGNATURE_REQUIRED_ROLES` 규칙을 따른다
   (신청인 본인 서명이 없으면 서식 자체가 성립하지 않으므로 기본 필수).
5. **서명란 역할** — 서명 영역의 왼쪽/위쪽 근접 텍스트를 :data:`ROLE_KEYWORDS`
   와 대조한다. 서로 다른 역할 키워드가 동시에 잡히거나 아무것도 못 찾으면
   :attr:`FieldRole.UNKNOWN` 으로 두고 신뢰도를 낮춘 뒤 ``warnings`` 에 남긴다.
   서명란이 둘 이상인데 하나를 확실히 고르지 못하는 상황은 별도 경고 문구로 남긴다.
6. **입력란 라벨** — 입력 영역 왼쪽에서 :data:`LABEL_MAX_GAP_MM` 이내의 단어들을
   모아 라벨로 삼는다(오른쪽에 인쇄된 예시 값은 라벨에 섞이지 않는다).
   라벨에 :data:`DATE_KEYWORDS` 가 있으면 :attr:`FieldType.DATE` 로 재분류한다.
7. **민감도** — :data:`PUBLIC_FIELD_TYPES` (선택형·약관)만 :attr:`Sensitivity.PUBLIC`,
   나머지는 전부 :attr:`Sensitivity.PRIVATE` 다. 선택형이라도 제목이
   :data:`PRIVATE_LABEL_KEYWORDS` 에 걸리면 PRIVATE 로 내린다(fail-closed).
8. **id** — 타입별 접두사 + 일련번호(``consent_01`` / ``signature_01`` /
   ``text_01`` / ``date_01`` / ``checkbox_01`` / ``unknown_01``). 읽기 순서 기준이라 결정론적이다.
9. **order** — 읽기 순서(y → x) 정수. 0 부터 빈틈없이 매긴다.

이 규칙들의 귀결로 :meth:`DocumentStructure.public_payload` 는 개인정보 항목의
제목·약관·선택지·좌표를 담지 않는다. 좌표가 필요한 Act 단계는 payload 가 아니라
원본 구조를 로컬 코드에서 직접 읽는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from docagent.contracts import (
    A4_PAGE_SIZE_MM,
    VISION_TRUST_THRESHOLD,
    BoxMm,
    Detection,
    DocumentStructure,
    Field,
    FieldRole,
    FieldType,
    OcrWord,
    Option,
    Sensitivity,
)
from docagent.errors import DocumentNotFoundError
from docagent.vision.ocr import (
    DEFAULT_LINE_Y_TOL_MM,
    TextLine,
    group_words_to_lines,
    union_box,
)

__all__ = [
    "CHECKBOX_ROW_TOL_MM",
    "CHECKBOX_ROW_MAX_GAP_MM",
    "CHECKBOX_STACK_GAP_MM",
    "OPTION_LABEL_MAX_GAP_MM",
    "LABEL_MAX_GAP_MM",
    "CLAUSE_MAX_LINE_GAP_MM",
    "ROW_OVERLAP_MIN_MM",
    "TITLE_MAX_CHARS",
    "EXPLICIT_REQUIRED_MARKERS",
    "STAR_MARKERS",
    "STAR_NOTICE_PATTERN",
    "HEADING_PREFIXES",
    "HEADING_SUFFIXES",
    "ROLE_KEYWORDS",
    "SIGNATURE_REQUIRED_ROLES",
    "DATE_KEYWORDS",
    "PRIVATE_LABEL_KEYWORDS",
    "PUBLIC_FIELD_TYPES",
    "ID_PREFIXES",
    "build_structure",
    "classify_sensitivity",
    "classify_role",
    "classify_signature_role",
    "strip_markers",
]


# --------------------------------------------------------------------------
# 거리·크기 임계값 (전부 mm)
# --------------------------------------------------------------------------

#: 같은 행의 체크박스로 볼 y 중심 허용오차(mm).
CHECKBOX_ROW_TOL_MM: float = 4.0
#: 같은 행에서 하나의 선택 그룹으로 묶을 최대 가로 간격(mm).
CHECKBOX_ROW_MAX_GAP_MM: float = 65.0
#: 세로로 쌓인 선택지를 하나의 그룹으로 묶을 최대 세로 간격(mm).
CHECKBOX_STACK_GAP_MM: float = 12.0
#: 체크박스와 선택지 라벨 사이 허용 간격(mm).
OPTION_LABEL_MAX_GAP_MM: float = 14.0
#: 입력란·서명란과 그 라벨 사이 허용 간격(mm).
#: 서식의 라벨 칸 폭이 보통 30~40mm 이고, 짧은 라벨("주소")은 칸 왼쪽에 붙어
#: 인쇄되므로 입력 칸까지 30mm 넘게 벌어질 수 있다. 같은 행에서 오른쪽부터
#: 연속으로만 훑기 때문에 값이 커도 다른 항목의 텍스트가 섞이지 않는다.
LABEL_MAX_GAP_MM: float = 40.0
#: 약관 본문을 위로 이어 붙일 때 허용하는 행 간 세로 간격(mm).
CLAUSE_MAX_LINE_GAP_MM: float = 10.0
#: 같은 행으로 볼 최소 세로 겹침 길이(mm).
ROW_OVERLAP_MIN_MM: float = 0.8
#: 제목으로 채택할 최대 글자 수. 이보다 길면 본문으로 본다.
TITLE_MAX_CHARS: int = 40
#: 좌우 라벨 간격 차이가 이 값 이하면 위치만으로 방향을 확정하지 않는다.
_LABEL_DIRECTION_MARGIN_MM: float = 2.0
#: 서명 상자 바로 위 라벨의 최대 세로 간격.
_SIGNATURE_ABOVE_GAP_MM: float = 8.0


# --------------------------------------------------------------------------
# 텍스트 규칙표
# --------------------------------------------------------------------------

#: 항상 필수로 보는 마커.
EXPLICIT_REQUIRED_MARKERS: tuple[str, ...] = ("[필수]", "(필수)", "필수항목", "필수기재")
#: 안내문이 있을 때만 필수 마커로 활성화되는 별표류.
STAR_MARKERS: tuple[str, ...] = ("*", "＊", "✱")
#: 별표 규칙을 켜는 문서 상단 안내문 패턴.
STAR_NOTICE_PATTERN: re.Pattern[str] = re.compile(r"(표시|표기).{0,15}(필수|반드시)")
#: 소제목으로 보는 줄머리 기호.
HEADING_PREFIXES: tuple[str, ...] = ("*", "＊", "※", "■", "□", "○", "●", "◆", "◇", "▶", "·", "-", "1.", "2.", "3.")
#: 소제목으로 보는 줄끝 표현.
HEADING_SUFFIXES: tuple[str, ...] = ("동의", "동의서", "여부", "확인", "선택", "사항", "안내")

#: 서명란 역할 판정 키워드표. 앞에 있는 항목부터 검사한다.
ROLE_KEYWORDS: tuple[tuple[FieldRole, tuple[str, ...]], ...] = (
    (FieldRole.REPRESENTATIVE, ("법정대리인", "대리인", "보호자", "후견인", "위임인", "대리")),
    (FieldRole.OFFICIAL, ("담당자", "접수자", "확인자", "취급자", "검토자", "처리자", "접수부서")),
    (FieldRole.APPLICANT, ("신청인", "본인", "작성자", "가입자", "예금주", "신청자", "청구인")),
)
#: 서명란이 기본 필수인 역할. 신청인 본인 서명이 없으면 서식이 성립하지 않는다.
SIGNATURE_REQUIRED_ROLES: tuple[FieldRole, ...] = (FieldRole.APPLICANT,)

#: 라벨에 있으면 :attr:`FieldType.DATE` 로 재분류하는 키워드.
DATE_KEYWORDS: tuple[str, ...] = ("일자", "날짜", "생년월일", "년월일", "작성일", "신청일")

#: 개인정보 값을 담는 라벨 키워드(민감도 판정용).
PRIVATE_LABEL_KEYWORDS: tuple[str, ...] = (
    "성명", "이름", "성함", "주민등록번호", "주민번호", "생년월일", "외국인등록번호",
    "여권번호", "운전면허", "주소", "거주지", "연락처", "전화", "휴대폰", "이메일",
    "계좌", "예금주", "카드번호", "사업자등록번호", "서명", "날인",
)
#: 공개 정보 영역으로 볼 수 있는 기입란 유형(약관·선택지는 문서 자체의 내용이다).
PUBLIC_FIELD_TYPES: tuple[FieldType, ...] = (FieldType.CHOICE, FieldType.CHECKBOX)

#: 타입별 field id 접두사.
ID_PREFIXES: dict[FieldType, str] = {
    FieldType.CHOICE: "consent",
    FieldType.CHECKBOX: "checkbox",
    FieldType.SIGNATURE: "signature",
    FieldType.TEXT_INPUT: "text",
    FieldType.DATE: "date",
    FieldType.UNKNOWN: "unknown",
}

#: 라벨을 못 찾았을 때 깎는 신뢰도.
_PENALTY_NO_LABEL: float = 0.20
#: 제목을 못 찾았을 때 깎는 신뢰도.
_PENALTY_NO_TITLE: float = 0.15
#: 서명란 역할을 확정하지 못했을 때 깎는 신뢰도.
_PENALTY_UNKNOWN_ROLE: float = 0.25
#: 직원 전용(OFFICIAL) 칸으로 판정했을 때 깎는 신뢰도.
#:
#: 이용자에게 "당신이 적을 칸"으로 안내되면 안 되는 항목이므로, 신뢰도를 낮춰
#: :data:`~docagent.contracts.VISION_TRUST_THRESHOLD` 분기(재확인·직원 연결)에
#: 반드시 걸리게 한다.
_PENALTY_OFFICIAL_ROLE: float = 0.30

#: 입력 영역으로 취급해 약관 수집을 중단시키는 탐지 유형.
_INPUT_DETECTION_TYPES: tuple[FieldType, ...] = (
    FieldType.TEXT_INPUT,
    FieldType.DATE,
    FieldType.SIGNATURE,
)


# --------------------------------------------------------------------------
# 내부 자료구조
# --------------------------------------------------------------------------


@dataclass
class _Draft:
    """id·order 확정 전의 항목 초안(모듈 내부 전용).

    :param type: 기입란 유형.
    :param title: 항목명 후보.
    :param role: 기입 주체.
    :param options: 선택지 목록.
    :param required: 필수 여부.
    :param box_mm: 기입 영역.
    :param clause_text: 약관 본문.
    :param confidence: 종합 신뢰도.
    :param label_text: 라벨 원문(디버깅·경고 메시지용).
    """

    type: FieldType
    title: str
    role: FieldRole
    options: tuple[Option, ...]
    required: bool
    box_mm: BoxMm
    clause_text: str
    confidence: float
    label_text: str


# --------------------------------------------------------------------------
# 텍스트 헬퍼
# --------------------------------------------------------------------------


def _leading_marker(text: str) -> str:
    """줄머리 기호를 찾는다. **뒤에 공백이 따라올 때만** 줄머리로 인정한다.

    ``"○○지원금 지급 신청서"`` 처럼 기호가 낱말의 일부인 경우를 줄머리로
    오인해 제목을 훼손하는 일을 막기 위한 규칙이다.

    :param text: 검사할 문자열.
    :returns: 발견한 줄머리 기호. 없으면 빈 문자열.
    """
    for prefix in HEADING_PREFIXES:
        if text.startswith(prefix) and text[len(prefix) :][:1].isspace():
            return prefix
    return ""


def strip_markers(text: str) -> str:
    """필수 마커와 줄머리 기호를 떼어 낸 제목 문자열을 만든다.

    :param text: 원문(예: ``"* 개인정보 수집·이용 동의"``, ``"성명 [필수]"``).
    :returns: 정리된 문자열(예: ``"개인정보 수집·이용 동의"``, ``"성명"``).
    """
    cleaned = text
    for marker in EXPLICIT_REQUIRED_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    cleaned = cleaned.strip()
    while cleaned:
        prefix = _leading_marker(cleaned)
        if not prefix:
            break
        cleaned = cleaned[len(prefix) :].lstrip()
    cleaned = cleaned.rstrip(" :：·-")
    return re.sub(r"\s+", " ", cleaned).strip()


def _compact_ocr_text(text: str) -> str:
    """키워드 판정에서만 OCR 이 삽입한 공백을 무시한다. 원문은 보존한다."""
    return re.sub(r"\s+", "", text)


def _has_explicit_required_marker(text: str) -> bool:
    """명시적 필수 마커가 있으면 True."""
    return any(marker in _compact_ocr_text(text) for marker in EXPLICIT_REQUIRED_MARKERS)


def _has_star_marker(text: str) -> bool:
    """별표류 마커가 있으면 True."""
    return any(marker in text for marker in STAR_MARKERS)


def _is_heading(line: TextLine) -> bool:
    """행이 소제목처럼 보이면 True.

    줄머리 기호로 시작하거나, 짧으면서 :data:`HEADING_SUFFIXES` 로 끝나면 제목으로 본다.

    :param line: 검사할 행.
    :returns: 소제목 여부.
    """
    text = line.text.strip()
    if not text:
        return False
    if _leading_marker(text):
        return True
    if len(text) <= TITLE_MAX_CHARS:
        stripped = strip_markers(text)
        return any(stripped.endswith(suffix) for suffix in HEADING_SUFFIXES)
    return False


def _star_rule_enabled(lines: Sequence[TextLine]) -> bool:
    """문서에 별표 안내문이 있어 ``*`` 를 필수 마커로 써도 되는지 판정한다.

    :param lines: 문서 전체 행 목록.
    :returns: 별표 규칙 활성화 여부.
    """
    return any(STAR_NOTICE_PATTERN.search(line.text) for line in lines)


def _is_required(text: str, *, star_rule: bool) -> bool:
    """라벨 문자열이 필수 항목을 뜻하는지 판정한다.

    :param text: 라벨 또는 제목 원문.
    :param star_rule: 별표 규칙 활성화 여부.
    :returns: 필수 여부.
    """
    if _has_explicit_required_marker(text):
        return True
    return star_rule and _has_star_marker(text)


# --------------------------------------------------------------------------
# 기하 헬퍼
# --------------------------------------------------------------------------


def _vertical_overlap(a: BoxMm, b: BoxMm) -> float:
    """두 사각형의 세로 겹침 길이(mm). 겹치지 않으면 0 이하."""
    return min(a.bottom_mm, b.bottom_mm) - max(a.y_mm, b.y_mm)


def _horizontal_overlap(a: BoxMm, b: BoxMm) -> float:
    """두 사각형의 가로 겹침 길이(mm). 겹치지 않으면 0 이하."""
    return min(a.right_mm, b.right_mm) - max(a.x_mm, b.x_mm)


def _same_row(a: BoxMm, b: BoxMm) -> bool:
    """두 사각형이 같은 행으로 볼 만큼 세로로 겹치면 True."""
    return _vertical_overlap(a, b) >= ROW_OVERLAP_MIN_MM


def _words_left_of(
    words: Sequence[OcrWord], box: BoxMm, *, max_gap_mm: float
) -> list[OcrWord]:
    """상자 **왼쪽**에서 ``max_gap_mm`` 이내로 붙어 있는 단어들을 모은다.

    오른쪽으로부터 연속으로 이어지는 단어만 취하므로, 상자 안쪽에 인쇄된
    예시 값이나 멀리 떨어진 다른 항목의 텍스트가 섞이지 않는다.

    :param words: 후보 단어 목록.
    :param box: 기준 상자.
    :param max_gap_mm: 허용 간격(mm).
    :returns: x 오름차순 단어 목록. 없으면 빈 리스트.
    """
    candidates = [
        word
        for word in words
        if _same_row(word.box_mm, box) and word.box_mm.right_mm <= box.x_mm + 0.5
    ]
    if not candidates:
        return []
    candidates.sort(key=lambda w: w.box_mm.x_mm, reverse=True)
    picked: list[OcrWord] = []
    edge = box.x_mm
    for word in candidates:
        if edge - word.box_mm.right_mm > max_gap_mm:
            break
        picked.append(word)
        edge = word.box_mm.x_mm
    picked.reverse()
    return picked


def _words_right_of(
    words: Sequence[OcrWord], box: BoxMm, *, max_gap_mm: float, stop_x_mm: float | None
) -> list[OcrWord]:
    """상자 **오른쪽**에서 ``max_gap_mm`` 이내로 붙어 있는 단어들을 모은다.

    :param words: 후보 단어 목록.
    :param box: 기준 상자.
    :param max_gap_mm: 허용 간격(mm).
    :param stop_x_mm: 이 x 이상은 다른 항목 영역이므로 수집을 멈춘다. ``None`` 이면 제한 없음.
    :returns: x 오름차순 단어 목록. 없으면 빈 리스트.
    """
    candidates = [
        word
        for word in words
        if _same_row(word.box_mm, box) and word.box_mm.x_mm >= box.right_mm - 0.5
    ]
    if stop_x_mm is not None:
        candidates = [word for word in candidates if word.box_mm.x_mm < stop_x_mm]
    candidates.sort(key=lambda w: w.box_mm.x_mm)
    picked: list[OcrWord] = []
    edge = box.right_mm
    for word in candidates:
        if word.box_mm.x_mm - edge > max_gap_mm:
            break
        picked.append(word)
        edge = word.box_mm.right_mm
    return picked


def _join(words: Sequence[OcrWord]) -> str:
    """단어들을 공백 하나로 이어 붙인다."""
    return " ".join(word.text for word in words).strip()


# --------------------------------------------------------------------------
# 체크박스 그룹핑
# --------------------------------------------------------------------------


def _connected(a: BoxMm, b: BoxMm) -> bool:
    """두 체크박스를 같은 선택 그룹으로 묶을 수 있으면 True.

    같은 행이면서 가로 간격이 좁거나, 가로로 겹치면서 세로 간격이 좁으면 연결한다.

    :param a: 체크박스 A.
    :param b: 체크박스 B.
    :returns: 연결 여부.
    """
    row_aligned = abs((a.y_mm + a.h_mm / 2) - (b.y_mm + b.h_mm / 2)) <= CHECKBOX_ROW_TOL_MM
    if row_aligned:
        gap = max(a.x_mm, b.x_mm) - min(a.right_mm, b.right_mm)
        if gap <= CHECKBOX_ROW_MAX_GAP_MM:
            return True
    if _horizontal_overlap(a, b) > 0:
        gap = max(a.y_mm, b.y_mm) - min(a.bottom_mm, b.bottom_mm)
        if gap <= CHECKBOX_STACK_GAP_MM:
            return True
    return False


def _group_checkboxes(detections: Sequence[Detection]) -> list[list[Detection]]:
    """체크박스 탐지들을 선택 그룹으로 묶는다(연결 요소 탐색).

    :param detections: :attr:`FieldType.CHECKBOX` 탐지 목록.
    :returns: 그룹 목록. 각 그룹은 읽기 순서(y → x)로 정렬된다.
    """
    items = sorted(
        detections, key=lambda d: (d.box_mm.y_mm, d.box_mm.x_mm)
    )
    parent = list(range(len(items)))

    def find(index: int) -> int:
        """연결 요소의 대표 인덱스를 찾는다(경로 압축)."""
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        """두 인덱스를 같은 연결 요소로 합친다."""
        root_l, root_r = find(left), find(right)
        if root_l != root_r:
            parent[max(root_l, root_r)] = min(root_l, root_r)

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _connected(items[i].box_mm, items[j].box_mm):
                union(i, j)

    buckets: dict[int, list[Detection]] = {}
    for index, item in enumerate(items):
        buckets.setdefault(find(index), []).append(item)
    groups = list(buckets.values())
    for group in groups:
        group.sort(key=lambda d: (d.box_mm.y_mm, d.box_mm.x_mm))
    groups.sort(key=lambda g: (g[0].box_mm.y_mm, g[0].box_mm.x_mm))
    return groups


def _checkbox_label_words(
    group: Sequence[Detection], words: Sequence[OcrWord]
) -> list[list[OcrWord]]:
    """행별 좌/우 배치를 비교해 라벨을 할당한다. 같은 단어를 재사용하지 않는다.

    양쪽이 똑같이 그럴듯하면 빈 라벨로 남겨 확인을 요청한다.
    누락된 라벨을 옆 체크박스의 문구로 채우지 않는다.
    """
    labels: list[list[OcrWord]] = [[] for _ in group]
    remaining = set(range(len(group)))
    while remaining:
        first = min(remaining)
        row = sorted(
            [i for i in remaining if i == first or _same_row(group[first].box_mm, group[i].box_mm)],
            key=lambda i: group[i].box_mm.x_mm,
        )
        remaining.difference_update(row)
        lefts: list[list[OcrWord]] = []
        rights: list[list[OcrWord]] = []
        for position, index in enumerate(row):
            box = group[index].box_mm
            previous = group[row[position - 1]].box_mm if position else None
            following = group[row[position + 1]].box_mm if position + 1 < len(row) else None
            left_candidates = [
                word for word in words
                if previous is None or word.box_mm.x_mm >= previous.right_mm
            ]
            lefts.append(_words_left_of(left_candidates, box, max_gap_mm=OPTION_LABEL_MAX_GAP_MM))
            rights.append(_words_right_of(
                words, box, max_gap_mm=OPTION_LABEL_MAX_GAP_MM,
                stop_x_mm=following.x_mm if following else None,
            ))
        left_count, right_count = sum(map(bool, lefts)), sum(map(bool, rights))
        if left_count != right_count:
            chosen = lefts if left_count > right_count else rights
        else:
            left_gap = sum(group[i].box_mm.x_mm - ls[-1].box_mm.right_mm
                           for i, ls in zip(row, lefts) if ls)
            right_gap = sum(rs[0].box_mm.x_mm - group[i].box_mm.right_mm
                            for i, rs in zip(row, rights) if rs)
            if abs(left_gap - right_gap) <= _LABEL_DIRECTION_MARGIN_MM * max(left_count, 1):
                chosen = [[] for _ in row]
            else:
                chosen = lefts if left_gap < right_gap else rights
        for index, selected in zip(row, chosen):
            labels[index] = selected
    return labels


def _title_and_clause(
    group_box: BoxMm,
    lines: Sequence[TextLine],
    detections: Sequence[Detection],
) -> tuple[str, str]:
    """선택 그룹 위쪽에서 제목과 약관 본문을 추출한다.

    그룹 바로 위 행부터 위로 거슬러 올라가며 :data:`CLAUSE_MAX_LINE_GAP_MM`
    이내로 이어지는 행을 약관 본문에 모은다. 소제목 행을 만나면 제목으로
    확정하고 멈추며, 다른 기입 영역과 겹치는 행을 만나도 멈춘다.

    :param group_box: 선택 그룹 전체 경계 상자.
    :param lines: 문서 전체 행 목록.
    :param detections: 문서 전체 탐지 목록(다른 기입 영역 경계 판정용).
    :returns: ``(title, clause_text)``. 못 찾으면 빈 문자열.
    """
    input_boxes = [
        item.box_mm for item in detections if item.type in _INPUT_DETECTION_TYPES
    ]
    above = [line for line in lines if line.box_mm.bottom_mm <= group_box.y_mm + 0.5]
    above.sort(key=lambda line: line.box_mm.y_mm, reverse=True)

    collected: list[TextLine] = []
    title = ""
    edge = group_box.y_mm
    for line in above:
        if edge - line.box_mm.bottom_mm > CLAUSE_MAX_LINE_GAP_MM:
            break
        if any(_vertical_overlap(line.box_mm, box) > 0 for box in input_boxes):
            break
        if _is_heading(line):
            title = strip_markers(line.text)
            break
        collected.append(line)
        edge = line.box_mm.y_mm

    collected.reverse()
    clause_text = " ".join(line.text for line in collected).strip()
    return (title, clause_text)


def _heading_line_for(
    group_box: BoxMm, lines: Sequence[TextLine], title: str
) -> TextLine | None:
    """``title`` 을 만들어 낸 소제목 행을 되찾는다(항목 상자 확장을 위해).

    :param group_box: 선택 그룹 경계 상자.
    :param lines: 문서 전체 행 목록.
    :param title: :func:`_title_and_clause` 가 돌려준 제목.
    :returns: 해당 행. 없으면 ``None``.
    """
    if not title:
        return None
    candidates = [
        line
        for line in lines
        if line.box_mm.bottom_mm <= group_box.y_mm + 0.5 and strip_markers(line.text) == title
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda line: line.box_mm.y_mm)


# --------------------------------------------------------------------------
# 분류 규칙
# --------------------------------------------------------------------------


def classify_role(label_text: str) -> tuple[FieldRole, bool]:
    """라벨 텍스트에서 기입 주체를 판정한다(서명란 전용이 아니다).

    체크박스·입력란도 "담당자 확인란"처럼 **직원 전용 칸**일 수 있다. 역할을
    서명란에서만 따지면 그런 칸이 신청인 몫으로 안내되어, 이용자가 직원 칸에
    기입하게 된다.

    :param label_text: 기입란 근접 텍스트(예: ``"신청인"``, ``"담당자 확인"``).
    :returns: ``(역할, 모호 여부)``. 서로 다른 역할 키워드가 동시에 잡히거나
        아무것도 없으면 ``(FieldRole.UNKNOWN, True)`` 를 돌려준다.
    """
    if not label_text.strip():
        return (FieldRole.UNKNOWN, True)
    label_text = _compact_ocr_text(label_text)
    matched: list[FieldRole] = []
    for role, keywords in ROLE_KEYWORDS:
        if any(keyword in label_text for keyword in keywords):
            matched.append(role)
    if len(matched) == 1:
        return (matched[0], False)
    return (FieldRole.UNKNOWN, True)


def classify_signature_role(label_text: str) -> tuple[FieldRole, bool]:
    """서명란 라벨에서 기입 주체를 판정한다(:func:`classify_role` 과 같은 규칙).

    :param label_text: 서명란 근접 텍스트.
    :returns: ``(역할, 모호 여부)``.
    """
    return classify_role(label_text)


def classify_sensitivity(field_type: FieldType, title: str) -> Sensitivity:
    """기입란 유형과 제목으로 민감도를 판정한다(fail-closed).

    :param field_type: 기입란 유형.
    :param title: 항목명.
    :returns: :attr:`Sensitivity.PUBLIC` 또는 :attr:`Sensitivity.PRIVATE`.
    """
    if field_type not in PUBLIC_FIELD_TYPES:
        return Sensitivity.PRIVATE
    if any(keyword in _compact_ocr_text(title) for keyword in PRIVATE_LABEL_KEYWORDS):
        return Sensitivity.PRIVATE
    return Sensitivity.PUBLIC


def _clamp(value: float) -> float:
    """신뢰도를 0.0~1.0 으로 자른다."""
    return max(0.0, min(1.0, value))


# --------------------------------------------------------------------------
# 초안 생성
# --------------------------------------------------------------------------


def _draft_choice(
    group: Sequence[Detection],
    words: Sequence[OcrWord],
    lines: Sequence[TextLine],
    detections: Sequence[Detection],
    *,
    star_rule: bool,
) -> _Draft:
    """체크박스 그룹 하나를 선택형 항목 초안으로 만든다."""
    boxes = [item.box_mm for item in group]
    label_words = _checkbox_label_words(group, words)
    options = [
        Option(label=strip_markers(_join(matched)), box_mm=item.box_mm, checked=None)
        for item, matched in zip(group, label_words)
    ]

    group_box = union_box(boxes)
    title, clause_text = _title_and_clause(group_box, lines, detections)
    heading = _heading_line_for(group_box, lines, title)

    if not title:
        labels = [option.label for option in options if option.label]
        title = " / ".join(labels) + " 선택" if labels else ""

    top = heading.box_mm.y_mm if heading is not None else group_box.y_mm
    field_box = BoxMm(
        x_mm=min(group_box.x_mm, heading.box_mm.x_mm if heading else group_box.x_mm),
        y_mm=top,
        w_mm=max(
            group_box.right_mm, heading.box_mm.right_mm if heading else group_box.right_mm
        )
        - min(group_box.x_mm, heading.box_mm.x_mm if heading else group_box.x_mm),
        h_mm=group_box.bottom_mm - top,
    )

    required = _is_required(
        f"{heading.text if heading else ''} {title}", star_rule=star_rule
    )
    field_type = FieldType.CHOICE if len(group) >= 2 else FieldType.CHECKBOX

    confidence = min(item.confidence for item in group)
    matched_words = [word for matched in label_words for word in matched]
    if heading:
        matched_words.extend(heading.words)
    if clause_text:
        matched_words.extend(word for line in lines
                             if line.text in clause_text for word in line.words)
    if matched_words:
        confidence = min(confidence, min(word.confidence for word in matched_words))
    if not heading:
        confidence -= _PENALTY_NO_TITLE
    if any(not option.label for option in options):
        confidence -= _PENALTY_NO_LABEL

    label_text = heading.text if heading else ""
    role = _role_for_input(label_text)
    if role is FieldRole.OFFICIAL:
        # 직원 전용 칸을 이용자 몫으로 안내하지 않도록 신뢰도를 깎아
        # VISION_TRUST_THRESHOLD 분기(재확인·직원 연결)에 걸리게 한다.
        confidence -= _PENALTY_OFFICIAL_ROLE

    return _Draft(
        type=field_type,
        title=title,
        role=role,
        options=tuple(options),
        required=required,
        box_mm=field_box,
        clause_text=clause_text,
        confidence=_clamp(confidence),
        label_text=label_text,
    )


def _role_for_input(label_text: str) -> FieldRole:
    """체크박스·입력란의 기입 주체를 정한다.

    역할 키워드가 명확히 하나 잡히면 그 역할을, 아무것도 없거나 모호하면
    기존 기본값인 :attr:`FieldRole.APPLICANT` 를 쓴다. 기본값을 유지하는 이유는
    대다수 신청서 칸이 신청인 몫이고, 여기서 UNKNOWN 을 남발하면 정상 항목까지
    직원 연결로 밀려나기 때문이다. 중요한 것은 **직원 전용 칸을 놓치지 않는
    것**이다.

    :param label_text: 기입란 근접 라벨 텍스트.
    :returns: :class:`~docagent.contracts.FieldRole`.
    """
    role, ambiguous = classify_role(label_text)
    if ambiguous:
        return FieldRole.APPLICANT
    return role


def _is_date_line(detection: Detection, words: Sequence[OcrWord]) -> bool:
    """서명란으로 탐지된 밑줄이 실제로는 **날짜 기입선**인지 판정한다.

    기하학적 탐지기는 "신청일자 ______" 의 기입선과 "신청인 ______ (서명 또는 인)"
    의 서명선을 구분할 수 없다(:mod:`docagent.vision.heuristic` 모듈 독스트링 참조).
    의미 판별은 이 구조화 단계의 책임이므로, 왼쪽 라벨이 날짜를 가리키고
    서명·날인 키워드가 전혀 없을 때만 날짜 기입선으로 되돌린다.

    이 판정이 없으면 날짜선이 주체 미상 서명란으로 남아, 서명란이 둘 이상인
    서식에서 항상 "어느 것이 신청인 서명란인지 확정 불가" 경고와 직원 연결이
    발생한다(연결 검증에서 실제로 관측된 오분류).

    :param detection: :attr:`FieldType.SIGNATURE` 로 탐지된 결과.
    :param words: mm 좌표 OCR 단어 목록.
    :returns: 날짜 기입선으로 보이면 True.
    """
    left = _words_left_of(words, detection.box_mm, max_gap_mm=LABEL_MAX_GAP_MM)
    label = _compact_ocr_text(strip_markers(_join(left)))
    if not label:
        return False
    if any(keyword in label for keyword in ("서명", "날인", "서명란")):
        return False
    return any(keyword in label for keyword in DATE_KEYWORDS)


def _draft_signature(
    detection: Detection, words: Sequence[OcrWord], *, star_rule: bool
) -> tuple[_Draft, bool]:
    """서명 탐지 하나를 항목 초안으로 만든다.

    :returns: ``(초안, 역할 모호 여부)``.
    """
    left = _words_left_of(words, detection.box_mm, max_gap_mm=LABEL_MAX_GAP_MM)
    if not left:
        above = [line for line in group_words_to_lines(words)
                 if 0 <= detection.box_mm.y_mm - line.box_mm.bottom_mm <= _SIGNATURE_ABOVE_GAP_MM
                 and _horizontal_overlap(line.box_mm, detection.box_mm) > 0]
        if above:
            left = list(max(above, key=lambda line: line.box_mm.bottom_mm).words)
    label_text = _join(left)
    role, ambiguous = classify_signature_role(label_text)

    title = strip_markers(label_text)
    if not _compact_ocr_text(title).endswith("서명"):
        title = f"{title} 서명" if title else "서명"
    required = _is_required(label_text, star_rule=star_rule) or (
        role in SIGNATURE_REQUIRED_ROLES
    )

    confidence = min(detection.confidence, min((word.confidence for word in left), default=1.0))
    if not label_text:
        confidence -= _PENALTY_NO_LABEL
    if ambiguous:
        confidence -= _PENALTY_UNKNOWN_ROLE

    draft = _Draft(
        type=FieldType.SIGNATURE,
        title=title,
        role=role,
        options=(),
        required=required,
        box_mm=detection.box_mm,
        clause_text="",
        confidence=_clamp(confidence),
        label_text=label_text,
    )
    return (draft, ambiguous)


def _draft_input(
    detection: Detection, words: Sequence[OcrWord], *, star_rule: bool
) -> _Draft:
    """입력란(문자·날짜) 탐지 하나를 항목 초안으로 만든다."""
    left = _words_left_of(words, detection.box_mm, max_gap_mm=LABEL_MAX_GAP_MM)
    label_text = _join(left)
    title = strip_markers(label_text)

    field_type = detection.type
    if field_type in (
        FieldType.TEXT_INPUT,
        FieldType.UNKNOWN,
        FieldType.SIGNATURE,
    ) and any(keyword in _compact_ocr_text(title) for keyword in DATE_KEYWORDS):
        field_type = FieldType.DATE

    confidence = min(detection.confidence, min((word.confidence for word in left), default=1.0))
    if not label_text:
        confidence -= _PENALTY_NO_LABEL

    role = _role_for_input(label_text)
    if role is FieldRole.OFFICIAL:
        confidence -= _PENALTY_OFFICIAL_ROLE

    return _Draft(
        type=field_type,
        title=title,
        role=role,
        options=(),
        required=_is_required(label_text, star_rule=star_rule),
        box_mm=detection.box_mm,
        clause_text="",
        confidence=_clamp(confidence),
        label_text=label_text,
    )


# --------------------------------------------------------------------------
# 공개 API
# --------------------------------------------------------------------------


def build_structure(
    detections: Sequence[Detection],
    words: Sequence[OcrWord],
    coords: tuple[float, float] = A4_PAGE_SIZE_MM,
    document_id: str = "document_0001",
    *,
    doc_title: str = "",
    source_image: str | None = None,
    y_tol_mm: float = DEFAULT_LINE_Y_TOL_MM,
) -> DocumentStructure:
    """탐지·OCR 결과를 :class:`DocumentStructure` (fields JSON)로 구조화한다.

    :param detections: mm 좌표 :class:`Detection` 목록.
    :param words: mm 좌표 :class:`OcrWord` 목록.
    :param coords: 페이지 크기 ``(가로_mm, 세로_mm)``. 기본 A4.
    :param document_id: 문서 인스턴스 식별자(빈 문자열 불가).
    :param doc_title: 문서 제목. 빈 문자열이면 페이지 최상단 행에서 유추한다.
    :param source_image: 원본 이미지 경로·식별자.
    :param y_tol_mm: 행 그룹핑 허용오차(mm).
    :returns: 항목이 읽기 순서로 정렬되고 id·order 가 확정된 :class:`DocumentStructure`.
    :raises ValueError: ``document_id`` 가 비었거나 ``coords`` 가 올바르지 않은 경우.
    :raises docagent.errors.DocumentNotFoundError: 탐지와 OCR 결과가 **모두** 비어
        문서로 볼 근거가 전혀 없는 경우(조용히 빈 구조를 돌려주지 않는다).
    """
    if not document_id:
        raise ValueError("build_structure 의 document_id 는 빈 문자열일 수 없습니다.")
    page = tuple(float(v) for v in coords)
    if len(page) != 2 or page[0] <= 0 or page[1] <= 0:
        raise ValueError(f"coords 는 0 보다 큰 (가로_mm, 세로_mm) 이어야 합니다: {coords!r}")
    # 탐지 좌표는 A4 기준으로만 환산된다(geometry.coordinate_system_from_image).
    # 여기서 다른 페이지 크기를 받아들이면 좌표계가 둘로 갈라진 채 진행되므로
    # 이번 범위에서는 A4 만 허용한다.
    if page != tuple(float(v) for v in A4_PAGE_SIZE_MM):
        raise ValueError(
            "이번 범위는 A4 문서만 지원합니다. coords 는 "
            f"{A4_PAGE_SIZE_MM} 이어야 합니다: {coords!r}."
        )
    if not detections and not words:
        raise DocumentNotFoundError(
            "탐지 결과와 OCR 결과가 모두 비어 있어 문서 구조를 만들 수 없습니다.",
            source=source_image,
        )

    lines = group_words_to_lines(words, y_tol_mm=y_tol_mm)
    star_rule = _star_rule_enabled(lines)
    warnings: list[str] = []

    drafts: list[_Draft] = []
    ambiguous_signatures = 0
    signature_count = 0

    checkbox_groups = _group_checkboxes(
        [item for item in detections if item.type is FieldType.CHECKBOX]
    )
    for group in checkbox_groups:
        drafts.append(
            _draft_choice(group, words, lines, detections, star_rule=star_rule)
        )

    for detection in detections:
        if detection.type is FieldType.CHECKBOX:
            continue
        if detection.type is FieldType.SIGNATURE and not _is_date_line(detection, words):
            signature_count += 1
            draft, ambiguous = _draft_signature(detection, words, star_rule=star_rule)
            if ambiguous:
                ambiguous_signatures += 1
            drafts.append(draft)
        else:
            drafts.append(_draft_input(detection, words, star_rule=star_rule))

    drafts.sort(key=lambda d: (d.box_mm.y_mm, d.box_mm.x_mm))

    counters: dict[str, int] = {}
    fields: list[Field] = []
    for order, draft in enumerate(drafts):
        prefix = ID_PREFIXES.get(draft.type, "unknown")
        counters[prefix] = counters.get(prefix, 0) + 1
        field_id = f"{prefix}_{counters[prefix]:02d}"
        sensitivity = classify_sensitivity(draft.type, draft.title)
        fields.append(
            Field(
                id=field_id,
                type=draft.type,
                title=draft.title,
                role=draft.role,
                options=draft.options,
                required=draft.required,
                sensitivity=sensitivity,
                box_mm=draft.box_mm,
                # clause_text 는 **문서에 인쇄된 공개 정보**이므로 민감도와 무관하게
                # 보존한다. 여기서 지우면 제목에 개인정보 낱말이 들어간 동의 조항의
                # 본문을 '원문 듣기' 로도 영영 들을 수 없다(로컬 낭독 경로까지 막힌다).
                # 외부 전송 차단은 DocumentStructure.public_payload() 의 강등 규칙과
                # PiiGate 가 담당한다.
                clause_text=draft.clause_text,
                order=order,
                confidence=draft.confidence,
            )
        )
        if any(not option.label for option in draft.options):
            warnings.append(
                f"[{field_id}] 선택지 문구가 없거나 좌우 매칭이 불명확합니다. 확인이 필요합니다."
            )
        if not draft.title:
            warnings.append(
                f"[{field_id}] 항목명을 찾지 못했습니다. 근처 텍스트가 인식되지 않았을 수 있습니다."
            )
        if draft.confidence < VISION_TRUST_THRESHOLD:
            warnings.append(
                f"[{field_id}] 해석 신뢰도가 낮습니다({draft.confidence:.2f}). "
                "사용자에게 재확인하거나 직원 연결을 검토하십시오."
            )
        if draft.role is FieldRole.OFFICIAL:
            warnings.append(
                f"[{field_id}] 담당 직원이 적는 칸으로 보입니다(라벨: '{draft.title}'). "
                "이용자에게 기입을 안내하지 마십시오."
            )

    if ambiguous_signatures:
        warnings.append(
            f"서명란 {ambiguous_signatures}개의 기입 주체를 확정하지 못했습니다. "
            "누가 서명해야 하는지 사용자에게 확인하거나 직원 연결이 필요합니다."
        )
    if signature_count >= 2 and ambiguous_signatures:
        warnings.append(
            f"서명란이 {signature_count}개 있는데 그중 하나를 확실히 고를 수 없습니다. "
            "잘못된 칸에 서명하지 않도록 직원 연결을 권장합니다."
        )
    roles = [field.role for field in fields if field.type is FieldType.SIGNATURE]
    duplicated = sorted({role.value for role in roles if roles.count(role) > 1})
    if duplicated:
        warnings.append(
            f"서명란 역할이 중복되었습니다(역할: {', '.join(duplicated)}). "
            "서명 대상 칸을 사용자에게 재확인해야 합니다."
        )

    resolved_title = doc_title
    if not resolved_title and lines:
        resolved_title = strip_markers(lines[0].text)

    return DocumentStructure(
        document_id=document_id,
        doc_title=resolved_title,
        fields=tuple(fields),
        page_size_mm=page,
        source_image=source_image,
        warnings=tuple(warnings),
    )
