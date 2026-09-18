"""가드레일 — 에이전트가 답하면 안 되는 질문을 규칙으로 막는다.

로드맵이 명시한 정지 조건을 규칙 기반으로 구현한다. 이 계층은 LLM 이전에 실행되며,
막아야 할 질문이 모델까지 도달하지 않게 하는 것이 목적이다.

차단 범주
---------
=================================== =====================================================
범주                                 왜 막는가
=================================== =====================================================
:attr:`GuardCategory.ELIGIBILITY`    자격·지급액 판단은 행정청의 처분 영역이다.
:attr:`GuardCategory.LEGAL`          법적 효력·책임·불이익 판단은 법률 자문에 해당한다.
:attr:`GuardCategory.FINANCIAL`      투자·금리 유불리 판단은 금융 자문에 해당한다.
:attr:`GuardCategory.SPECULATION`    근거 없는 의견·추천 요구는 할루시네이션의 입구다.
:attr:`GuardCategory.PII`            질문에 개인정보가 섞이면 외부 LLM 경로를 막는다.
=================================== =====================================================

**PII 범주의 처리 방침**: 개인정보가 섞인 질문은 "LLM 경로 차단"이 목적이며,
차단 후에는 원문을 그대로 읽어 주는 로컬 처리로 안내한다. 이 안내 문구를 담아
:class:`~docagent.errors.HandoffRequired` 를 던지므로, ``Guard.check`` 만 호출하는
호출자(오케스트레이터)와 :meth:`Guard.inspect` 를 쓰는 호출자가 같은 결론에 이른다.

할루시네이션 차단
-----------------
:func:`answer_grounding_check` 는 답변에 등장하는 숫자·금액·기간·고유명사 가운데
근거 청크에 없는 것을 찾아 돌려준다. 하나라도 있으면 그 답변은 폐기해야 한다.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from docagent.contracts import Field, RetrievedChunk, SanitizedText
from docagent.errors import HandoffRequired, PiiEgressBlocked
from docagent.interfaces import PiiGate

__all__ = [
    "DISCLAIMER",
    "GuardCategory",
    "GuardVerdict",
    "Guard",
    "NullPiiGate",
    "BLOCK_RULES",
    "UNEXPLAINED_DIGIT_RUN_MIN",
    "answer_grounding_check",
]

_LOGGER = logging.getLogger(__name__)


#: 모든 설명 응답 끝에 붙이는 고지 문구. **이 프로젝트의 유일한 고지 정의다.**
#:
#: 쉬운 설명은 요약일 뿐이고 법적 효력을 가지는 것은 원문이라는 사실을 매번 알린다
#: (로드맵의 법적 리스크·UX 원칙). :data:`docagent.agent.tools.ORIGINAL_NOTICE` 는
#: 이 상수를 그대로 참조한다 — 같은 취지의 문구를 두 곳에 두면 음성 전용 UI 에서
#: 같은 고지가 두 번, 그것도 호출 방법을 서로 다르게 안내하며 낭독된다.
#:
#: 호출 방법은 **음성으로 실제 통하는 표현**(:mod:`docagent.agent.intent` 가
#: 인식하는 "원문 읽어줘")으로 적는다.
DISCLAIMER: str = (
    "쉽게 설명한 내용입니다. 이 설명은 원문을 대신하지 않으니, "
    "원문을 그대로 들으시려면 '원문 읽어줘'라고 말씀해 주세요."
)


class GuardCategory(Enum):
    """가드레일 차단 범주."""

    #: 자격 판단("제가 받을 수 있나요").
    ELIGIBILITY = "eligibility"
    #: 법적 판단("법적으로 유효한가요").
    LEGAL = "legal"
    #: 세무·금융 조언("어디에 투자할까요").
    FINANCIAL = "financial"
    #: 근거 없는 추측·추천 요구("네 생각은 어때").
    SPECULATION = "speculation"
    #: 질문에 개인정보가 포함됨.
    PII = "pii"


@dataclass(frozen=True)
class GuardVerdict:
    """가드레일 판정 결과.

    :param blocked: 차단해야 하면 True.
    :param category: 차단 범주. 통과면 ``None``.
    :param reason: 사용자에게 그대로 낭독 가능한 한국어 사유. 통과면 빈 문자열.
    :param trigger: 판정 근거가 된 표현. 통과면 빈 문자열.
    :param local_only: 외부 LLM 경로만 막고 로컬 원문 처리로 우회해야 하면 True
        (:attr:`GuardCategory.PII` 에서만 True).
    """

    blocked: bool
    category: GuardCategory | None = None
    reason: str = ""
    trigger: str = ""
    local_only: bool = False


#: 범주별 차단 트리거 표현. 부분 문자열(공백 제거 후)로 검사한다.
#:
#: 정규식이 아니라 표현 목록으로 둔 이유: 규칙을 사람이 읽고 검토할 수 있어야 하고,
#: 워크숍에서 표현을 추가·삭제하는 일이 잦기 때문이다.
BLOCK_RULES: dict[GuardCategory, tuple[str, ...]] = {
    GuardCategory.ELIGIBILITY: (
        "받을수있나요",
        "받을수있어요",
        "받을수있습니까",
        "받을수있을까요",
        "자격이되나요",
        "자격이돼요",
        "자격이있나요",
        "대상이되나요",
        "대상인가요",
        "해당되나요",
        "신청가능한가요",
        "신청할수있나요",
        "얼마나나와요",
        "얼마나받아요",
        "얼마받나요",
        "얼마나주나요",
        "몇원나와요",
    ),
    GuardCategory.LEGAL: (
        "법적으로유효",
        "법적효력",
        "법적으로문제",
        "법적책임",
        "소송",
        "고소",
        "불이익이있나요",
        "불이익을받나요",
        "책임은누가",
        "책임져야",
        "처벌받나요",
        "위법인가요",
        "합법인가요",
    ),
    GuardCategory.FINANCIAL: (
        "어디에투자",
        "투자해도되나요",
        "투자할까요",
        "이자가유리",
        "이자유리",
        "금리가유리",
        "어느상품이좋",
        "수익률이좋",
        "절세",
        "세금은얼마",
    ),
    GuardCategory.SPECULATION: (
        "네생각은어때",
        "너생각은어때",
        "네생각에는",
        "추천해줘",
        "추천해주세요",
        "골라줘",
        "대신정해줘",
        "알아서해줘",
        "아무거나해줘",
    ),
}

#: 범주별 한국어 차단 사유(사용자 낭독용).
_CATEGORY_REASON: dict[GuardCategory, str] = {
    GuardCategory.ELIGIBILITY: (
        "지급 자격이나 금액을 판단하는 것은 제가 할 수 없는 일입니다. "
        "서류에 적힌 내용은 그대로 읽어 드릴 수 있고, 자격 여부는 담당 직원에게 연결해 드리겠습니다."
    ),
    GuardCategory.LEGAL: (
        "법적인 효력이나 책임을 판단하는 것은 제가 할 수 없는 일입니다. "
        "서류 원문은 그대로 읽어 드릴 수 있고, 판단이 필요한 부분은 담당 직원에게 연결해 드리겠습니다."
    ),
    GuardCategory.FINANCIAL: (
        "어떤 상품이 유리한지 같은 금융·세무 조언은 제가 드릴 수 없습니다. "
        "상품 설명 원문은 읽어 드릴 수 있고, 상담은 담당 직원에게 연결해 드리겠습니다."
    ),
    GuardCategory.SPECULATION: (
        "근거 문서에 없는 내용을 추측해서 말씀드릴 수 없습니다. "
        "서류에 적힌 내용을 그대로 읽어 드리거나, 담당 직원에게 연결해 드리겠습니다."
    ),
    GuardCategory.PII: (
        "질문에 개인정보로 보이는 내용이 들어 있어 외부 설명 기능을 사용하지 않았습니다. "
        "서류 원문을 그대로 읽어 드리거나, 담당 직원에게 연결해 드리겠습니다."
    ),
}

#: 사용자 발화에서 "설명되지 않은 숫자열"로 보는 최소 자리수.
#:
#: 블록리스트가 놓친 형태의 개인정보까지 막기 위한 allowlist 방어선이다.
#: 발화 안에서 구분자를 지운 뒤 이 자리수 이상 연속되는 숫자는, 탐지 규칙이
#: 아무것도 찾지 못했더라도 개인정보로 보고 외부 설명 경로를 막는다.
#: (연도·조문·수량은 4자리 이하이므로 걸리지 않고, 전화번호·계좌번호·
#: 주민등록번호·카드번호는 모두 걸린다.)
UNEXPLAINED_DIGIT_RUN_MIN: int = 7

#: 숫자 사이 구분자를 지우는 정규식(회피 변형 흡수).
_DIGIT_JOIN_RE = re.compile(r"(?<=\d)[\s.\-]+(?=\d)")

#: 설명되지 않은 숫자열 정규식.
_LONG_DIGITS_RE = re.compile(rf"\d{{{UNEXPLAINED_DIGIT_RUN_MIN},}}")


class NullPiiGate:
    """개인정보 검사를 **의도적으로** 끄는 게이트(테스트·단위 검증 전용).

    :class:`Guard` 는 게이트가 없으면 조용히 통과시키지 않고 예외를 던진다.
    차단 규칙만 따로 시험하고 싶을 때는 이 클래스를 **명시적으로** 주입해,
    "검사를 껐다"는 사실이 호출부에 드러나게 한다. 운영 경로에서는 쓰지 않는다.
    """

    def sanitize(self, text: str) -> SanitizedText:
        """아무것도 탐지하지 않은 :class:`~docagent.contracts.SanitizedText` 를 돌려준다.

        :param text: 원문 문자열.
        :returns: 원문 그대로인 :class:`~docagent.contracts.SanitizedText`.
        """
        return SanitizedText(text=text, spans=(), blocked=False)

    def assert_clean(self, text: str) -> None:
        """아무것도 검사하지 않고 통과시킨다(:class:`PiiGate` 프로토콜 준수).

        검사를 **의도적으로 끈** 게이트이므로 어떤 문자열도 막지 않는다.
        그 사실은 :attr:`Guard.has_pii_gate` 가 False 로 드러낸다. 이 메서드가
        없으면 ``isinstance(gate, PiiGate)`` 가 False 가 되고, 계약대로
        ``assert_clean`` 을 부르는 호출자에서 :class:`AttributeError` 가 난다.

        :param text: 검사 대상(무시한다).
        :returns: ``None``.
        """
        del text


def _resolve_default_pii_gate() -> PiiGate:
    """기본 개인정보 게이트를 지연 import 로 확보한다(fail-closed).

    :mod:`docagent.agent.llm` 의 ``_resolve_default_gate`` 와 같은 규율이다.
    게이트를 주입하지 않은 호출자가 **조용히 검사 없이** 통과하는 일을 막는다.

    :returns: :class:`~docagent.interfaces.PiiGate` 구현.
    :raises docagent.errors.PiiEgressBlocked: 게이트 구현을 확보하지 못한 경우.
    """
    try:
        from docagent.pii.gate import build_default_gate
    except ImportError as exc:  # pragma: no cover - 패키지가 온전하면 발생하지 않는다.
        raise PiiEgressBlocked(
            "PII 게이트 구현(docagent.pii.gate)을 찾지 못해 가드레일을 만들 수 없습니다. "
            "pii_gate 인자로 게이트를 직접 주입하거나, 검사를 끄려면 "
            "NullPiiGate 를 명시적으로 주입하십시오."
        ) from exc
    return build_default_gate()


#: 공백·문장부호를 제거하는 정규식(트리거 표현을 띄어쓰기와 무관하게 맞추기 위함).
_SQUASH_RE = re.compile(r"[\s.,!?~·…\"'()\[\]{}]+")


def _squash(text: str) -> str:
    """공백과 문장부호를 제거해 트리거 비교용 문자열을 만든다.

    :param text: 원문.
    :returns: 공백·문장부호가 제거된 문자열.
    """
    return _SQUASH_RE.sub("", text)


def _strip_benign_numbers(text: str) -> str:
    """접수번호·금액·날짜·조문 같은 정당한 숫자를 지운 사본을 만든다.

    문서에서 읽어 온 문구(항목명·약관 전문)에는 접수번호·조문 번호처럼 긴 숫자가
    정상적으로 등장한다. 사용자 발화보다 느슨한 이 기준을 **문서 텍스트에만**
    적용해 정상 문서가 통째로 핸드오프되는 일을 막는다.

    :param text: 검사 대상 문자열.
    :returns: 정당한 숫자를 지운 사본. 게이트 모듈을 확보하지 못하면 원본을
        그대로 돌려주어 **더 엄격한** 쪽으로 판정되게 한다.
    """
    try:
        from docagent.pii.gate import strip_benign_numbers
    except ImportError:  # pragma: no cover - 패키지가 온전하면 발생하지 않는다.
        return text
    return strip_benign_numbers(text)


def _has_unexplained_digits(text: str) -> bool:
    """발화에 설명되지 않은 긴 숫자열이 남아 있으면 True.

    구분자(공백·줄바꿈·하이픈·점)로 끊긴 숫자를 먼저 이어 붙인 뒤
    :data:`UNEXPLAINED_DIGIT_RUN_MIN` 자리 이상 연속 숫자가 있는지 본다.

    :param text: 사용자 발화 원문.
    :returns: 설명되지 않은 숫자열 존재 여부.
    """
    return bool(_LONG_DIGITS_RE.search(_DIGIT_JOIN_RE.sub("", text)))


class Guard:
    """질문을 검사해 답하면 안 되는 경우를 차단한다.

    :param pii_gate: 개인정보 탐지 게이트(:class:`docagent.interfaces.PiiGate`).
        ``None`` 이면 :func:`_resolve_default_pii_gate` 로 기본 게이트를 확보한다.
        **검사를 건너뛰는 기본값은 없다.** 검사를 끄려면 :class:`NullPiiGate` 를
        명시적으로 주입해야 한다.
    :param rules: 차단 규칙. ``None`` 이면 :data:`BLOCK_RULES`.
    :raises docagent.errors.PiiEgressBlocked: 기본 게이트를 확보하지 못한 경우.
    """

    def __init__(
        self,
        pii_gate: PiiGate | None = None,
        *,
        rules: dict[GuardCategory, tuple[str, ...]] | None = None,
    ) -> None:
        self._pii_gate: PiiGate = (
            pii_gate if pii_gate is not None else _resolve_default_pii_gate()
        )
        self._rules = dict(rules) if rules is not None else dict(BLOCK_RULES)

    @property
    def has_pii_gate(self) -> bool:
        """실제 개인정보 검사가 이루어지면 True.

        :class:`NullPiiGate` 를 주입해 검사를 끈 경우에만 False 다.
        """
        return not isinstance(self._pii_gate, NullPiiGate)

    def inspect(self, user_text: str, field: Field | None = None) -> GuardVerdict:
        """질문을 검사해 판정만 돌려준다(예외를 던지지 않는다).

        :param user_text: 사용자 발화. 빈 문자열이면 통과 판정.
        :param field: 질문이 가리키는 항목. 현재 규칙에서는 쓰지 않지만
            향후 항목별 예외 규칙을 위해 인터페이스에 남겨 둔다.
        :returns: :class:`GuardVerdict`.
        """
        del field  # 현재 규칙은 항목에 의존하지 않는다.
        if not user_text or not user_text.strip():
            return GuardVerdict(blocked=False)

        squashed = _squash(user_text)
        for category, triggers in self._rules.items():
            for trigger in triggers:
                if trigger in squashed:
                    return GuardVerdict(
                        blocked=True,
                        category=category,
                        reason=_CATEGORY_REASON[category],
                        trigger=trigger,
                    )

        sanitized = self._pii_gate.sanitize(user_text)
        if sanitized.has_pii or sanitized.blocked:
            types = sorted({span.pii_type for span in sanitized.spans})
            return GuardVerdict(
                blocked=True,
                category=GuardCategory.PII,
                reason=_CATEGORY_REASON[GuardCategory.PII],
                trigger=", ".join(types) if types else "pii",
                local_only=True,
            )

        # allowlist 방어선 — 탐지 규칙이 아무것도 찾지 못했더라도, 발화에 설명되지
        # 않은 긴 숫자열이 남아 있으면 막는다. 블록리스트만으로는 "규칙이 모르는
        # 형태의 개인정보"를 보증할 수 없기 때문이다.
        if self.has_pii_gate and _has_unexplained_digits(user_text):
            return GuardVerdict(
                blocked=True,
                category=GuardCategory.PII,
                reason=_CATEGORY_REASON[GuardCategory.PII],
                trigger="unexplained_digits",
                local_only=True,
            )
        return GuardVerdict(blocked=False)

    def check(self, user_text: str, field: Field | None = None) -> None:
        """질문을 검사하고, 답하면 안 되는 경우 예외를 던진다.

        :param user_text: 사용자 발화.
        :param field: 질문이 가리키는 항목(선택).
        :returns: ``None``. 통과 시 아무것도 반환하지 않는다.
        :raises docagent.errors.HandoffRequired: 차단 대상인 경우.
            ``reason`` 은 사용자에게 그대로 낭독할 수 있는 한국어 문장이다.
        """
        verdict = self.inspect(user_text, field)
        if verdict.blocked:
            _LOGGER.info(
                "가드레일 차단: category=%s trigger=%s",
                verdict.category.value if verdict.category else "-",
                verdict.trigger,
            )
            raise HandoffRequired(
                verdict.reason,
                field_id=None if field is None else field.id,
            )

    def inspect_outbound(self, text: str) -> GuardVerdict:
        """**외부로 나갈 문자열**(문서에서 읽어 온 문구 포함)을 검사한다.

        :meth:`inspect` 는 사용자 발화만 본다. 그러나 프롬프트에는 OCR 로 읽은
        항목명·약관 전문도 함께 실리므로, 이 텍스트도 같은 규율을 받아야 한다.
        ``sensitivity=PUBLIC`` 판정을 신뢰하지 않고 여기서 한 번 더 확인한다.

        차단 규칙(자격·법률 판단 등)은 적용하지 않는다. 문서 원문에는 "법적
        효력" 같은 표현이 정상적으로 등장하기 때문이다. 검사하는 것은
        **개인정보 유입 여부**뿐이다.

        :param text: 검사 대상 문자열(질의·프롬프트 등).
        :returns: :class:`GuardVerdict`. 문제가 없으면 ``blocked=False``.
        """
        if not text or not text.strip():
            return GuardVerdict(blocked=False)

        sanitized = self._pii_gate.sanitize(text)
        if sanitized.has_pii or sanitized.blocked:
            types = sorted({span.pii_type for span in sanitized.spans})
            return GuardVerdict(
                blocked=True,
                category=GuardCategory.PII,
                reason=_CATEGORY_REASON[GuardCategory.PII],
                trigger=", ".join(types) if types else "pii",
                local_only=True,
            )

        # allowlist 방어선 — 탐지 규칙이 못 본 형태까지 막는다. 문서 문구에는
        # 접수번호·조문 번호가 정상적으로 등장하므로 그 숫자는 먼저 지운다.
        if self.has_pii_gate and _has_unexplained_digits(_strip_benign_numbers(text)):
            return GuardVerdict(
                blocked=True,
                category=GuardCategory.PII,
                reason=_CATEGORY_REASON[GuardCategory.PII],
                trigger="unexplained_digits",
                local_only=True,
            )
        return GuardVerdict(blocked=False)

    def check_outbound(self, text: str, field: Field | None = None) -> None:
        """외부로 나갈 문자열을 검사하고, 개인정보가 섞였으면 예외를 던진다.

        :param text: 검사 대상 문자열(질의·프롬프트 등).
        :param field: 대상 항목(선택). 예외에 항목 id 를 남기는 데만 쓴다.
        :returns: ``None``. 통과 시 아무것도 반환하지 않는다.
        :raises docagent.errors.HandoffRequired: 개인정보가 섞여 외부 경로를
            쓸 수 없는 경우.
        """
        verdict = self.inspect_outbound(text)
        if verdict.blocked:
            _LOGGER.info(
                "외부 전송 전 가드레일 차단: trigger=%s", verdict.trigger
            )
            raise HandoffRequired(
                verdict.reason,
                field_id=None if field is None else field.id,
            )


# --------------------------------------------------------------------------
# 할루시네이션(근거 밖 사실) 차단
# --------------------------------------------------------------------------

#: 숫자·금액·기간 표현 추출 정규식.
_NUMBER_RE = re.compile(
    r"\d[\d,]*(?:\.\d+)?\s*(?:퍼센트|개월|만원|억원|천원|원|년|일|월|주|회|건|명|세|%)?"
)

#: 고유명사 후보 — 기관·법령을 가리키는 접미사로 끝나는 한글 낱말.
_PROPER_NOUN_RE = re.compile(
    r"[가-힣]{2,}(?:법률|법|시행령|시행규칙|위원회|공단|공사|은행|보험|청|부|처|원|국|과|센터|재단|협회)"
)

#: 라틴 문자 고유명사 후보(대문자로 시작하는 2자 이상 낱말).
_LATIN_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z]{1,}\b")

#: 고유명사 판정에서 제외할 일반 명사(접미사 규칙의 오탐 방지).
_PROPER_NOUN_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "지원",
        "신청",
        "확인",
        "이용",
        "사용",
        "적용",
        "제공",
        "결정",
        "판단",
        "부분",
        "전부",
        "일부",
        "본인",
        "직원",
        "기관",
        "내용",
        "경우",
        "이유",
        "방법",
        "정보",
        "동의",
        "서류",
        "서명",
        "설명",
        "권리",
        "의무",
        "기간",
        "목적",
        "항목",
        "대상",
    }
)


def _normalize_number(token: str) -> str:
    """숫자 표현을 비교용으로 정규화한다(쉼표·공백 제거).

    :param token: 숫자 표현(예: ``"5 년"``, ``"1,000원"``).
    :returns: 정규화 문자열(예: ``"5년"``, ``"1000원"``).
    """
    return token.replace(",", "").replace(" ", "").strip()


def answer_grounding_check(
    answer: str,
    chunks: Sequence[RetrievedChunk],
) -> list[str]:
    """답변에서 근거 청크에 없는 숫자·고유명사를 찾아 돌려준다.

    반환값이 비어 있지 않으면 그 답변은 **근거 밖 사실을 만들어 낸 것**이므로
    폐기해야 한다. 값 자체가 아니라 "근거에 없다"는 사실이 판단 기준이다.

    :param answer: 검사할 생성 답변.
    :param chunks: 답변의 근거로 쓰인 청크 목록.
    :returns: 근거에서 확인되지 않은 표현 목록(등장 순서, 중복 제거).
        답변이 비어 있거나 문제가 없으면 빈 리스트.

    사용 예::

        unsupported = answer_grounding_check(answer, chunks)
        if unsupported:
            raise HandoffRequired("근거에 없는 내용이 포함되었습니다.")
    """
    if not answer or not answer.strip():
        return []
    evidence = " ".join(chunk.text for chunk in chunks)
    evidence_squashed = _squash(evidence)

    unsupported: list[str] = []
    seen: set[str] = set()

    for match in _NUMBER_RE.findall(answer):
        token = _normalize_number(match)
        if not token or not any(ch.isdigit() for ch in token):
            continue
        if token in seen:
            continue
        seen.add(token)
        if token not in evidence_squashed:
            unsupported.append(token)

    for match in _PROPER_NOUN_RE.findall(answer):
        token = match.strip()
        if token in _PROPER_NOUN_EXCEPTIONS or token in seen:
            continue
        seen.add(token)
        if token not in evidence:
            unsupported.append(token)

    for match in _LATIN_PROPER_RE.findall(answer):
        token = match.strip()
        if token in seen:
            continue
        seen.add(token)
        if token not in evidence:
            unsupported.append(token)

    return unsupported
