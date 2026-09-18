"""개인정보 탐지 패턴 정의와 유니코드 정규화(오프셋 보존).

이 모듈은 **탐지 규칙의 단일 원천**이다. :mod:`docagent.pii.detectors` 가
여기 정의된 :class:`PatternSpec` 목록을 순회하며 실제 탐지를 수행한다.

설계 원칙
---------
1. **정규화 후 탐지, 원문 좌표로 보고.**
   회피 변형(구분자 없음 / 공백 / 점 / 전각 숫자 ``０-９`` / 전각·en dash 하이픈 /
   줄바꿈 삽입 / 제로폭 문자)에 강해지려면 NFKC 정규화와 공백 축약이 필요하다.
   그러나 :class:`~docagent.contracts.PiiSpan` 의 인덱스는 **원문 기준**이어야
   하므로, :func:`normalize` 가 정규화 문자열과 함께 **오프셋 매핑**을 돌려준다.

1-b. **정규화 사본을 여러 벌 만들어 모두 탐지한다.**
   단일 정규화만으로는 숫자 **사이**에 끼어든 공백·줄바꿈(OCR 줄바꿈, STT 띄어쓰기)과
   OCR 동형이의 문자(``O``↔``0``, ``l``·``I``↔``1``, ``S``↔``5``, ``B``↔``8``)를 흡수하지
   못한다. 그래서 :func:`normalize_variants` 가 기본 정규화 외에 **숫자 사이 구분자를
   제거한 사본**과 **동형이의 문자를 접은 사본**(그리고 둘을 함께 적용한 사본)을 만들고,
   탐지기가 사본 전부를 훑는다. 모든 사본이 같은 원문 오프셋 매핑을 유지하므로
   보고되는 좌표는 언제나 원문 기준이다.

2. **원문 값을 담지 않는다.** 이 모듈의 어떤 반환값도 탐지된 문자열 자체를
   보관하지 않는다. 위치·유형·길이·신뢰도만 다룬다.

3. **라벨 앵커 패턴은 값만 잡는다.** ``"성명 김철수"`` 같은 문맥 기반 규칙은
   라벨까지 함께 정규식에 쓰되, 명명 그룹 ``v`` 로 **실제 개인정보 구간만**
   포착한다. 라벨은 문서에 인쇄된 공개 정보이므로 마스킹 대상이 아니다.

4. **거짓양성 억제.** 접수번호·문서번호·금액·일반 날짜·조문 번호 같은 무해한
   숫자열은 :data:`DENY_PATTERNS` 로 "탐지 금지 구간"을 만들어 배제한다.
   다만 라벨이 명시된(문맥 앵커) 탐지와 신뢰도가 매우 높은 유형
   (:data:`DENY_EXEMPT_TYPES`) 은 이 억제를 적용받지 않는다.

주민등록번호 검증 범위에 대한 주석
----------------------------------
구(舊) 주민등록번호 체계의 **마지막 자리 검증숫자(check digit) 규칙은
2020년 10월 주민등록법 시행규칙 개정으로 폐지**되었다. 그 이후 발급분은
뒷 6자리가 임의값이므로 검증숫자 계산을 강제하면 **정상 번호를 놓치는
치명적 오탐 누락(false negative)** 이 발생한다. 이 프로젝트는 안전 KPI를
"개인정보 LLM 전송 0건"으로 두므로 놓치는 쪽이 훨씬 나쁘다.
따라서 여기서는 **형식(13자리) + 생년월일 유효성 + 성별코드(1~8)** 만으로
판정하고 검증숫자는 요구하지 않는다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Callable, Final, Mapping, Pattern

__all__ = [
    "NormalizedText",
    "normalize",
    "normalize_variants",
    "OCR_HOMOGLYPHS",
    "PatternSpec",
    "PATTERN_SPECS",
    "DENY_PATTERNS",
    "DENY_EXEMPT_TYPES",
    "PII_TYPE_LABELS",
    "TOKEN_NAMES",
    "NAME_STOPWORDS",
    "luhn_ok",
    "rrn_digits_valid",
    "business_number_ok",
    "digits_only",
]


# --------------------------------------------------------------------------
# 유니코드 정규화 (오프셋 보존)
# --------------------------------------------------------------------------

#: 하이픈으로 통일할 대시류 문자.
#: en dash / em dash / minus sign / 전각 하이픈 / 물결 대체 등 회피 변형을 흡수한다.
_DASHES: Final[frozenset[str]] = frozenset(
    "-"  # HYPHEN-MINUS
    "‐‑‒–—―"  # hyphen ~ horizontal bar
    "⁃"  # hyphen bullet
    "−"  # minus sign
    "˗"  # modifier letter minus
    "﹘﹣－"  # small/fullwidth forms
    "ー"  # katakana-hiragana prolonged sound mark
)

#: 탐지 전에 제거하는 제로폭·서식 문자(대표적인 회피 수단).
_ZERO_WIDTH: Final[frozenset[str]] = frozenset(
    "​‌‍⁠﻿­᠎"
)


@dataclass(frozen=True)
class NormalizedText:
    """정규화된 텍스트와 원문 오프셋 매핑.

    :param text: 정규화 결과 문자열(NFKC + 대시 통일 + 공백 축약 + 제로폭 제거).
    :param starts: ``text[i]`` 를 만들어 낸 원문 구간의 **시작 인덱스**.
    :param ends: ``text[i]`` 를 만들어 낸 원문 구간의 **끝 인덱스(제외)**.
    :param source_len: 원문 길이(문자 수).
    :raises ValueError: 매핑 길이가 ``text`` 길이와 다른 경우.
    """

    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    source_len: int

    def __post_init__(self) -> None:
        if len(self.starts) != len(self.text) or len(self.ends) != len(self.text):
            raise ValueError(
                "NormalizedText 의 오프셋 매핑 길이가 정규화 텍스트 길이와 다릅니다: "
                f"text={len(self.text)}, starts={len(self.starts)}, ends={len(self.ends)}"
            )

    def to_source_span(self, start: int, end: int) -> tuple[int, int]:
        """정규화 좌표 구간 ``[start, end)`` 를 원문 좌표 구간으로 되돌린다.

        :param start: 정규화 텍스트 기준 시작 인덱스(포함).
        :param end: 정규화 텍스트 기준 끝 인덱스(제외). ``start`` 보다 커야 한다.
        :returns: ``(원문 시작, 원문 끝)`` 튜플.
        :raises ValueError: 구간이 비었거나 범위를 벗어난 경우.
        """
        if end <= start:
            raise ValueError(
                f"정규화 구간이 비어 있습니다: start={start}, end={end}"
            )
        if start < 0 or end > len(self.text):
            raise ValueError(
                f"정규화 구간이 범위를 벗어났습니다: start={start}, end={end}, "
                f"길이={len(self.text)}"
            )
        return (self.starts[start], self.ends[end - 1])


def normalize(text: str) -> NormalizedText:
    """탐지용 정규화를 수행하고 원문 오프셋 매핑을 함께 반환한다.

    수행 내용:

    * 연속 공백(줄바꿈·탭·전각 공백 포함)을 **공백 1개**로 축약한다.
    * 대시류 문자를 ASCII 하이픈 ``-`` 로 통일한다.
    * 제로폭·소프트하이픈 문자를 제거한다.
    * 나머지 문자는 **문자 단위 NFKC** 정규화한다(전각 숫자 ``９`` → ``9`` 등).

    :param text: 원문 문자열.
    :returns: :class:`NormalizedText`.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우(조용한 실패 금지).
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize 는 문자열만 받습니다: {type(text).__name__}")

    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    length = len(text)

    while index < length:
        ch = text[index]

        if ch in _ZERO_WIDTH:
            index += 1
            continue

        if ch.isspace():
            run_end = index
            while run_end < length and text[run_end].isspace():
                run_end += 1
            chars.append(" ")
            starts.append(index)
            ends.append(run_end)
            index = run_end
            continue

        if ch in _DASHES:
            chars.append("-")
            starts.append(index)
            ends.append(index + 1)
            index += 1
            continue

        folded = unicodedata.normalize("NFKC", ch)
        for out in folded:
            chars.append("-" if out in _DASHES else out)
            starts.append(index)
            ends.append(index + 1)
        index += 1

    return NormalizedText(
        text="".join(chars),
        starts=tuple(starts),
        ends=tuple(ends),
        source_len=length,
    )


#: OCR 동형이의 문자 접기 표. 숫자와 맞닿아 있을 때만 적용한다.
#:
#: 문서 스캔·촬영에서 흔히 뒤바뀌는 글자만 담았다. 무조건 접으면 일반 영문 단어가
#: 숫자열로 둔갑해 거짓양성이 폭증하므로, :func:`normalize_variants` 는 **앞뒤 중
#: 한쪽이 숫자인 위치**에서만 이 표를 적용한다.
OCR_HOMOGLYPHS: Final[Mapping[str, str]] = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
    "i": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "Z": "2",
}

#: 숫자 사이에서 제거 대상이 되는 구분자(정규화 후 기준).
_SQUASHABLE: Final[frozenset[str]] = frozenset(" -.")


def _derive(base: NormalizedText, kept: list[tuple[str, int, int]]) -> NormalizedText:
    """문자·오프셋 목록으로 파생 :class:`NormalizedText` 를 만든다.

    :param base: 원본 정규화 결과(원문 길이 참조용).
    :param kept: ``(문자, 원문 시작, 원문 끝)`` 목록.
    :returns: 같은 원문을 가리키는 파생 :class:`NormalizedText`.
    """
    return NormalizedText(
        text="".join(item[0] for item in kept),
        starts=tuple(item[1] for item in kept),
        ends=tuple(item[2] for item in kept),
        source_len=base.source_len,
    )


def _squash_digit_separators(base: NormalizedText) -> NormalizedText:
    """숫자와 숫자 사이에 낀 구분자(공백·하이픈·점)를 제거한 사본을 만든다.

    ``"9001
011234567"`` / ``"900101-123 4567"`` / ``"900101 - - 1234567"``
    처럼 숫자 중간이 끊긴 회피 변형을 흡수한다.

    :param base: 기본 정규화 결과.
    :returns: 구분자가 제거된 파생 :class:`NormalizedText`.
    """
    text = base.text
    length = len(text)
    kept: list[tuple[str, int, int]] = []
    for index, ch in enumerate(text):
        if ch in _SQUASHABLE:
            before = index - 1
            while before >= 0 and text[before] in _SQUASHABLE:
                before -= 1
            after = index + 1
            while after < length and text[after] in _SQUASHABLE:
                after += 1
            if (
                before >= 0
                and after < length
                and text[before].isdigit()
                and text[after].isdigit()
            ):
                continue
        kept.append((ch, base.starts[index], base.ends[index]))
    return _derive(base, kept)


def _fold_homoglyphs(base: NormalizedText) -> NormalizedText:
    """숫자와 맞닿은 OCR 동형이의 문자를 숫자로 접은 사본을 만든다.

    ``"9OO1O1-1234567"`` / ``"90010l-1234567"`` 처럼 OCR 이 글자를 잘못 읽어
    정규식이 무너지는 경우를 흡수한다.

    :param base: 기본 정규화 결과.
    :returns: 동형이의 문자가 접힌 파생 :class:`NormalizedText`.
    """
    text = base.text
    length = len(text)
    kept: list[tuple[str, int, int]] = []
    for index, ch in enumerate(text):
        folded = ch
        replacement = OCR_HOMOGLYPHS.get(ch)
        if replacement is not None:
            prev_ch = text[index - 1] if index > 0 else ""
            next_ch = text[index + 1] if index + 1 < length else ""
            neighbours_digit = prev_ch.isdigit() or next_ch.isdigit()
            neighbours_foldable = (
                prev_ch in OCR_HOMOGLYPHS or next_ch in OCR_HOMOGLYPHS
            )
            if neighbours_digit or neighbours_foldable:
                folded = replacement
        kept.append((folded, base.starts[index], base.ends[index]))
    return _derive(base, kept)


def normalize_variants(text: str) -> tuple[NormalizedText, ...]:
    """탐지에 쓸 정규화 사본을 전부 만든다(기본 + 회피 변형 흡수 사본).

    반환 순서는 결정론적이며 ``(기본, 동형이의 접기, 구분자 제거, 둘 다)`` 다.
    내용이 같은 사본은 한 벌만 남긴다.

    :param text: 원문 문자열.
    :returns: :class:`NormalizedText` 튜플. 모두 **같은 원문 오프셋**을 가리킨다.
    :raises TypeError: ``text`` 가 문자열이 아닌 경우.
    """
    base = normalize(text)
    variants: list[NormalizedText] = [base]
    folded = _fold_homoglyphs(base)
    squashed = _squash_digit_separators(base)
    both = _squash_digit_separators(folded)
    seen = {base.text}
    for candidate in (folded, squashed, both):
        if candidate.text not in seen:
            seen.add(candidate.text)
            variants.append(candidate)
    return tuple(variants)


# --------------------------------------------------------------------------
# 검증 헬퍼
# --------------------------------------------------------------------------


def digits_only(value: str) -> str:
    """문자열에서 숫자만 뽑아 잇는다.

    :param value: 대상 문자열.
    :returns: 숫자만 남은 문자열.
    """
    return "".join(ch for ch in value if ch.isdigit())


def luhn_ok(digits: str) -> bool:
    """Luhn 체크섬을 만족하면 True.

    :param digits: 숫자로만 이루어진 문자열.
    :returns: 체크섬 통과 여부. 비어 있거나 숫자가 아니면 False.
    """
    if not digits or not digits.isdigit():
        return False
    total = 0
    for position, ch in enumerate(reversed(digits)):
        value = int(ch)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


#: 성별코드 → 출생 세기. 1·2(내국인 1900년대), 3·4(내국인 2000년대),
#: 5·6(외국인 1900년대), 7·8(외국인 2000년대).
_GENDER_CENTURY: Final[Mapping[int, int]] = {
    1: 1900,
    2: 1900,
    3: 2000,
    4: 2000,
    5: 1900,
    6: 1900,
    7: 2000,
    8: 2000,
}

#: 외국인등록번호로 판정하는 성별코드.
FOREIGN_GENDER_CODES: Final[frozenset[int]] = frozenset({5, 6, 7, 8})


def rrn_digits_valid(digits: str) -> bool:
    """주민등록번호/외국인등록번호 13자리의 형식·생년월일·성별코드를 검증한다.

    검증숫자(마지막 자리) 규칙은 2020년 폐지되었으므로 **요구하지 않는다**
    (모듈 독스트링 참조).

    :param digits: 숫자 13자리 문자열.
    :returns: 생년월일이 실재하고 성별코드가 1~8이면 True.
    """
    if len(digits) != 13 or not digits.isdigit():
        return False
    gender = int(digits[6])
    century = _GENDER_CENTURY.get(gender)
    if century is None:
        return False
    year = century + int(digits[0:2])
    month = int(digits[2:4])
    day = int(digits[4:6])
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


def business_number_ok(digits: str) -> bool:
    """사업자등록번호 10자리 체크섬을 검증한다.

    가중치 ``(1,3,7,1,3,7,1,3,5)`` 를 앞 9자리에 곱해 더하고,
    9번째 자리 × 5 의 십의 자리를 더한 뒤 ``(10 - 합%10) % 10`` 이
    마지막 자리와 같아야 한다.

    :param digits: 숫자 10자리 문자열.
    :returns: 체크섬 통과 여부.
    """
    if len(digits) != 10 or not digits.isdigit():
        return False
    weights = (1, 3, 7, 1, 3, 7, 1, 3, 5)
    total = sum(int(digits[i]) * weights[i] for i in range(9))
    total += (int(digits[8]) * 5) // 10
    return (10 - (total % 10)) % 10 == int(digits[9])


# --------------------------------------------------------------------------
# 패턴 사양
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternSpec:
    """개인정보 탐지 규칙 1개.

    :param pii_type: :class:`~docagent.contracts.PiiSpan` 에 기록할 유형 식별자.
    :param pattern: 컴파일된 정규식. 명명 그룹 ``v`` 가 있으면 그 구간만
        개인정보로 본다(라벨은 공개 정보이므로 제외).
    :param priority: 겹침 병합 시 우선순위(클수록 우선).
    :param confidence: 탐지 신뢰도(0.0~1.0).
    :param validator: 추가 검증 함수. ``None`` 이면 정규식 일치만으로 채택한다.
    :param context_anchored: 라벨 문맥에 앵커된 규칙이면 True.
        True 인 규칙의 탐지 결과는 :data:`DENY_PATTERNS` 억제를 받지 않는다.
    :param description: 사람이 읽는 규칙 설명(한국어).
    """

    pii_type: str
    pattern: Pattern[str]
    priority: int
    confidence: float
    validator: Callable[[re.Match[str]], bool] | None = None
    context_anchored: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.pii_type:
            raise ValueError("PatternSpec.pii_type 은 빈 문자열일 수 없습니다.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"PatternSpec.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )

    def value_span(self, match: re.Match[str]) -> tuple[int, int]:
        """매치에서 **개인정보 값 구간**(정규화 좌표)을 뽑는다.

        :param match: 정규식 매치 객체.
        :returns: ``(start, end)`` 정규화 좌표 구간.
        """
        if "v" in self.pattern.groupindex:
            return match.span("v")
        return match.span()


# 구분자: 하이픈 / 공백 / 점. 정규화 후이므로 전각·en dash 는 이미 '-' 로 바뀌어 있다.
_SEP = r"[-\s.]{0,3}"

#: 라벨과 값 사이에 끼어드는 한국어 조사.
#:
#: ``"건강보험증번호는 12345678901"`` 처럼 조사가 한 글자만 붙어도 라벨 앵커
#: 규칙이 통째로 무력화되던 회피 경로를 막는다. 긴 조사를 먼저 시도하도록
#: 정렬해 두었다(교대 선택은 왼쪽 우선이므로 순서가 의미를 가진다).
_JOSA = r"(?:이랑|께서|에게|은|는|이|가|을|를|의|와|과|랑|도|만)"

#: 라벨 → 값 사이 구분자. 공백·조사·콜론이 어떤 순서로 섞여도 흡수한다.
#:
#: 조사는 값이 한글로 시작할 때 값의 첫 글자를 삼킬 수 있으나(예: ``"성명 은수"``),
#: 뒤따르는 값 패턴이 실패하면 정규식 역추적으로 조사 없이 다시 맞춰지므로
#: 안전하다.
_LABEL_SEP = rf"\s*{_JOSA}?\s*[:：]?\s*"

#: 성명 뒤에 붙을 수 있는 조사·호칭·서술어 어미.
#:
#: 성명 값 구간(그룹 ``v``)에서 **제외**하기 위한 꼬리다. 이것이 없으면
#: ``"김철수입니다"`` 처럼 어미가 붙었을 때 2~4자 한글 규칙이 통째로 실패해
#: 성명이 마스킹되지 않은 채 나간다.
_NAME_TAIL = (
    rf"(?:{_JOSA}|입니다|이에요|예요|이라고|라고|이라는|이며|이고|씨|님)"
)

# 이름 라벨 — 뒤에 오는 한글 2~4자를 성명으로 본다.
_NAME_LABELS = (
    r"성\s*명|이\s*름|신\s*청\s*인|대\s*리\s*인|예\s*금\s*주|"
    r"수\s*급\s*자|보\s*호\s*자|대\s*표\s*자|성명\(한글\)|환\s*자\s*명"
)

#: 성명 오탐 억제용 불용어. 라벨 뒤에 오더라도 성명으로 채택하지 않는다.
NAME_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "성명",
        "이름",
        "주소",
        "주민",
        "등록",
        "번호",
        "연락",
        "연락처",
        "전화",
        "서명",
        "날인",
        "동의",
        "필수",
        "선택",
        "신청",
        "신청인",
        "대리인",
        "보호자",
        "예금주",
        "대표자",
        "본인",
        "상기",
        "위와",
        "확인",
        "없음",
        "해당",
        "기재",
        "작성",
        "미상",
        "생년월일",
        "휴대폰",
        "휴대전화",
        "이메일",
        "직인",
        "또는",
        "기타",
        "정보",
        "개인",
        "개인정보",
        "수집",
        "이용",
        "제공",
        "목적",
        "기간",
        "항목",
    }
)

_SIDO = (
    r"서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충청북도|충남|충청남도|"
    r"전북|전라북도|전남|전라남도|경북|경상북도|경남|경상남도|제주"
)

_BANKS = (
    r"국민|신한|우리|하나|농협|기업|산업|수협|씨티|SC제일|케이뱅크|카카오뱅크|"
    r"토스뱅크|새마을금고|우체국|부산은행|대구은행|광주은행|전북은행|경남은행|신협"
)


#: 값 뒤에 붙은 조사를 떼어 내기 위한 정규식(불용어 비교용).
_TRAILING_JOSA_RE: Final[Pattern[str]] = re.compile(rf"{_JOSA}$")


def _strip_josa(value: str) -> str:
    """값 뒤에 붙은 조사 1개를 떼어 낸 어간을 돌려준다.

    :param value: 후보 문자열(예: ``"성명은"``).
    :returns: 조사를 뗀 어간(예: ``"성명"``). 뗄 조사가 없으면 원본 그대로.
    """
    stem = _TRAILING_JOSA_RE.sub("", value)
    return stem if len(stem) >= 2 else value


def _name_ok(match: re.Match[str]) -> bool:
    """성명 후보가 불용어가 아니면 True.

    라벨에 조사가 붙어 ``"성명은"`` 처럼 잡히는 경우를 대비해 **조사를 뗀
    어간**으로도 불용어를 대조한다. 라벨을 값으로 오인해 마스킹하고 정작
    뒤따르는 실제 성명을 흘려보내는 사고를 막기 위한 이중 방어다.
    """
    value = match.group("v")
    return value not in NAME_STOPWORDS and _strip_josa(value) not in NAME_STOPWORDS


def _rrn_ok(match: re.Match[str]) -> bool:
    """주민등록번호 후보(내국인 성별코드)를 검증한다."""
    digits = digits_only(match.group("v"))
    return rrn_digits_valid(digits) and int(digits[6]) not in FOREIGN_GENDER_CODES


def _frn_ok(match: re.Match[str]) -> bool:
    """외국인등록번호 후보(성별코드 5~8)를 검증한다."""
    digits = digits_only(match.group("v"))
    return rrn_digits_valid(digits) and int(digits[6]) in FOREIGN_GENDER_CODES


def _card_ok(match: re.Match[str]) -> bool:
    """카드번호 후보가 13~19자리이고 Luhn 을 통과하면 True."""
    digits = digits_only(match.group("v"))
    return 13 <= len(digits) <= 19 and luhn_ok(digits)


def _biz_ok(match: re.Match[str]) -> bool:
    """사업자등록번호 후보의 체크섬을 검증한다."""
    return business_number_ok(digits_only(match.group("v")))


def _account_ok(match: re.Match[str]) -> bool:
    """계좌번호 후보의 숫자 길이가 은행 계좌 범위(9~16)인지 확인한다."""
    return 9 <= len(digits_only(match.group("v"))) <= 16


def _bare_account_ok(match: re.Match[str]) -> bool:
    """라벨 없는 계좌번호 후보를 좁은 자리수 범위로만 채택한다.

    라벨(계좌·은행명)이 없는 하이픈 묶음 숫자는 사업자등록번호(10자리)·
    카드번호(16자리)와 형식이 겹치므로, 그 둘을 피한 **11~14자리**만 계좌로 본다.
    두 유형은 각자의 전용 규칙(체크섬·Luhn)이 더 높은 우선순위로 먼저 잡는다.
    """
    return 11 <= len(digits_only(match.group("v"))) <= 14


#: 탐지 규칙 전체 목록. 우선순위 내림차순으로 정렬해 두었다.
PATTERN_SPECS: Final[tuple[PatternSpec, ...]] = (
    PatternSpec(
        pii_type="rrn",
        pattern=re.compile(
            rf"(?<![0-9])(?P<v>\d{{6}}{_SEP}[1-8]\d{{6}})(?![0-9])"
        ),
        priority=100,
        confidence=0.99,
        validator=_rrn_ok,
        description="주민등록번호(형식 + 생년월일 + 성별코드 1~4)",
    ),
    PatternSpec(
        pii_type="frn",
        pattern=re.compile(
            rf"(?<![0-9])(?P<v>\d{{6}}{_SEP}[1-8]\d{{6}})(?![0-9])"
        ),
        priority=99,
        confidence=0.97,
        validator=_frn_ok,
        description="외국인등록번호(성별코드 5~8)",
    ),
    PatternSpec(
        pii_type="email",
        pattern=re.compile(
            r"(?P<v>[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+)"
        ),
        priority=96,
        confidence=0.97,
        description="이메일 주소",
    ),
    PatternSpec(
        pii_type="card",
        pattern=re.compile(
            r"(?<![0-9])(?P<v>\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,7})(?![0-9])"
        ),
        priority=90,
        confidence=0.93,
        validator=_card_ok,
        description="신용·체크카드 번호(Luhn 검증)",
    ),
    PatternSpec(
        pii_type="account",
        pattern=re.compile(
            r"(?:계\s*좌\s*번\s*호|계\s*좌|입\s*금\s*계\s*좌|출\s*금\s*계\s*좌|"
            rf"예\s*금\s*주\s*계\s*좌){_LABEL_SEP}"
            r"(?=(?P<v>\d{2,6}(?:[-\s]\d{2,6}){1,3})(?![0-9]))"
        ),
        priority=86,
        confidence=0.93,
        validator=_account_ok,
        context_anchored=True,
        description="계좌번호(라벨 문맥)",
    ),
    PatternSpec(
        pii_type="account",
        pattern=re.compile(
            rf"(?:{_BANKS})\s*(?:은행)?{_LABEL_SEP}"
            r"(?=(?P<v>\d{2,6}(?:[-\s]\d{2,6}){1,3})(?![0-9]))"
        ),
        priority=85,
        confidence=0.9,
        validator=_account_ok,
        context_anchored=True,
        description="계좌번호(은행명 문맥)",
    ),
    PatternSpec(
        pii_type="passport",
        pattern=re.compile(
            r"(?<![A-Za-z0-9])(?P<v>[MSRODGmsrodg]\d{8}|[A-Z]\d{3}[A-Z]\d{4})"
            r"(?![A-Za-z0-9])"
        ),
        priority=80,
        confidence=0.85,
        description="여권번호(대한민국 발급 형식)",
    ),
    PatternSpec(
        pii_type="driver_license",
        pattern=re.compile(
            r"(?<![0-9])(?P<v>\d{2}-\d{2}-\d{6}-\d{2})(?![0-9])"
        ),
        priority=78,
        confidence=0.88,
        description="운전면허번호(구분자 포함 형식)",
    ),
    PatternSpec(
        pii_type="driver_license",
        pattern=re.compile(
            rf"(?:운\s*전\s*면\s*허(?:\s*번\s*호)?|면\s*허\s*번\s*호){_LABEL_SEP}"
            r"(?=(?P<v>\d{2}[-\s]?\d{2}[-\s]?\d{6}[-\s]?\d{2})(?![0-9]))"
        ),
        priority=77,
        confidence=0.93,
        context_anchored=True,
        description="운전면허번호(라벨 문맥)",
    ),
    PatternSpec(
        pii_type="business_number",
        pattern=re.compile(
            r"(?<![0-9])(?P<v>\d{3}-?\d{2}-?\d{5})(?![0-9])"
        ),
        priority=75,
        confidence=0.9,
        validator=_biz_ok,
        description="사업자등록번호(체크섬 검증)",
    ),
    PatternSpec(
        pii_type="health_insurance",
        pattern=re.compile(
            r"(?:건\s*강\s*보\s*험\s*증?\s*(?:번\s*호)?|보\s*험\s*증\s*번\s*호|"
            rf"건\s*강\s*보\s*험\s*증){_LABEL_SEP}"
            r"(?=(?P<v>\d{8,11})(?![0-9]))"
        ),
        priority=70,
        confidence=0.9,
        context_anchored=True,
        description="건강보험증번호(라벨 문맥)",
    ),
    PatternSpec(
        pii_type="phone_mobile",
        pattern=re.compile(
            r"(?<![0-9\-])(?P<v>(?:\+82[-\s.]?)?0?1[016789][-\s.]{0,2}"
            r"\d{3,4}[-\s.]{0,2}\d{4})(?![0-9])"
        ),
        priority=65,
        confidence=0.95,
        description="휴대전화번호",
    ),
    PatternSpec(
        pii_type="phone_landline",
        pattern=re.compile(
            r"(?<![0-9\-])(?P<v>0(?:2|31|32|33|41|42|43|44|51|52|53|54|55|"
            r"61|62|63|64|70)[-\s.]{0,2}\d{3,4}[-\s.]{0,2}\d{4})(?![0-9])"
        ),
        priority=60,
        confidence=0.9,
        description="유선전화번호(지역번호 포함)",
    ),
    PatternSpec(
        pii_type="address",
        pattern=re.compile(
            rf"(?P<v>(?:{_SIDO})"
            r"(?:특별자치시|특별자치도|특별시|광역시|도|시)?"
            r"(?:\s*[가-힣0-9]+(?:시|군|구))?"
            r"[가-힣0-9\s]{0,20}?"
            r"(?:[가-힣0-9]+(?:대로|로|길)|[가-힣0-9]+(?:동|읍|면|리))"
            r"\s*\d+(?:-\d+)?(?:번지)?"
            r"(?:\s*,?\s*(?:[가-힣0-9]+동\s*)?\d+호)?)"
        ),
        priority=50,
        confidence=0.85,
        description="도로명·지번 주소",
    ),
    PatternSpec(
        pii_type="birthdate",
        pattern=re.compile(
            r"(?:생\s*년\s*월\s*일|생\s*일|출\s*생\s*일(?:자)?|출\s*생\s*년\s*월\s*일)"
            rf"{_LABEL_SEP}"
            r"(?=(?P<v>\d{4}\s*[년.\-/]\s*\d{1,2}\s*[월.\-/]\s*\d{1,2}\s*일?"
            r"|\d{8}|\d{6})(?![0-9]))"
        ),
        priority=45,
        confidence=0.9,
        context_anchored=True,
        description="생년월일(라벨 문맥)",
    ),
    PatternSpec(
        pii_type="name",
        pattern=re.compile(
            rf"(?:{_NAME_LABELS}){_LABEL_SEP}"
            rf"(?=(?P<v>[가-힣]{{2,4}}?){_NAME_TAIL}{{0,2}}(?![가-힣]))"
        ),
        priority=30,
        confidence=0.8,
        validator=_name_ok,
        context_anchored=True,
        description="성명(라벨 문맥 + 한글 2~4자)",
    ),
    PatternSpec(
        pii_type="name",
        pattern=re.compile(
            r"(?:저\s*는|제\s*이\s*름\s*은|제\s*성\s*함\s*은|본\s*인\s*은)\s*"
            r"(?P<v>[가-힣]{2,4})"
            r"(?=이?라고|입니다|이에요|예요|이고|인데|이며|이라|라고|이라는)"
        ),
        priority=29,
        confidence=0.7,
        validator=_name_ok,
        context_anchored=True,
        description="성명(자기소개 문맥 — 라벨이 없는 발화)",
    ),
    PatternSpec(
        pii_type="account",
        pattern=re.compile(
            r"(?<![0-9\-])(?P<v>\d{2,6}(?:-\d{2,6}){2,3})(?![0-9\-])"
        ),
        priority=28,
        confidence=0.6,
        validator=_bare_account_ok,
        description="계좌번호(라벨 없는 하이픈 묶음 11~14자리, 저신뢰)",
    ),
)


#: 탐지 금지 구간(거짓양성 억제). 이 구간과 겹치는 탐지는 버린다.
#: 단, 문맥 앵커 규칙과 :data:`DENY_EXEMPT_TYPES` 유형은 예외다.
DENY_PATTERNS: Final[tuple[Pattern[str], ...]] = (
    # 접수·문서·처리번호 등 행정 일련번호
    re.compile(
        r"(?:접\s*수|문\s*서|처\s*리|민\s*원|상\s*담|일\s*련|고\s*지|공\s*고|공\s*문|"
        r"정\s*수|정\s*책|승\s*인|결\s*재)\s*번\s*호\s*[:：]?\s*제?\s*[0-9A-Za-z\-]+\s*호?"
    ),
    # 금액
    re.compile(r"\d{1,3}(?:,\d{3})+\s*원"),
    re.compile(r"\d+\s*원"),
    # 일반 날짜(생년월일 라벨이 붙지 않은 것)
    re.compile(r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일"),
    re.compile(r"(?<![0-9])\d{4}[-./]\d{1,2}[-./]\d{1,2}(?![0-9])"),
    # 법령 조문
    re.compile(r"제\s*\d+\s*조(?:\s*제?\s*\d+\s*항)?(?:\s*제?\s*\d+\s*호)?"),
    # 수량·기간 등 단위가 붙은 숫자
    re.compile(r"\d+\s*(?:쪽|페이지|건|명|개|년|개월|일간|시간|분간|%|퍼센트)"),
)

#: 탐지 금지 구간의 억제를 **받지 않는** 유형.
#: 이 유형들은 형식 자체가 매우 특이해 거짓양성 위험보다 누락 위험이 크다.
DENY_EXEMPT_TYPES: Final[frozenset[str]] = frozenset({"rrn", "frn", "email"})


#: 유형 → 사용자 안내용 한국어 명칭.
PII_TYPE_LABELS: Final[Mapping[str, str]] = {
    "rrn": "주민등록번호",
    "frn": "외국인등록번호",
    "email": "이메일 주소",
    "card": "카드번호",
    "account": "계좌번호",
    "passport": "여권번호",
    "driver_license": "운전면허번호",
    "business_number": "사업자등록번호",
    "health_insurance": "건강보험증번호",
    "phone_mobile": "휴대전화번호",
    "phone_landline": "유선전화번호",
    "address": "주소",
    "birthdate": "생년월일",
    "name": "성명",
}

#: 유형 → TYPE_TOKEN 마스킹에 쓰는 토큰 이름.
TOKEN_NAMES: Final[Mapping[str, str]] = {
    "rrn": "RRN",
    "frn": "FRN",
    "email": "EMAIL",
    "card": "CARD",
    "account": "ACCOUNT",
    "passport": "PASSPORT",
    "driver_license": "DRIVER",
    "business_number": "BIZNO",
    "health_insurance": "HEALTH",
    "phone_mobile": "PHONE",
    "phone_landline": "TEL",
    "address": "ADDRESS",
    "birthdate": "BIRTH",
    "name": "NAME",
}
