"""한국어 규칙 기반 의도 분류기 — LLM 없이 결정론적으로 동작한다.

기본 경로(:class:`RuleIntentClassifier`)는 정규식 규칙만 쓴다. 네트워크·API 키가
전혀 없는 환경에서도 100% 동작해야 하기 때문이다. LLM 분류기는
:class:`LlmIntentClassifier` 로 분리해 두었고 기본 경로에서는 사용하지 않는다.

분류 원칙
---------
1. **추측 금지.** 규칙에 걸리지 않거나 신뢰도가 기준에 못 미치면
   :attr:`Intent.UNKNOWN` 으로 떨어뜨린다. 오케스트레이터가 되묻게 하는 편이
   틀린 항목에 체크하는 것보다 안전하다.
2. **부정이 긍정보다 먼저.** "동의 안 해" 가 "동의" 로 잘못 읽히면
   사용자가 원치 않는 동의에 체크된다. 규칙 순서로 이 사고를 구조적으로 막는다.
3. **번복이 최우선.** "아까 동의한 거 취소할래" 는 :attr:`Intent.REVISE` 이지
   :attr:`Intent.AGREE` 도 :attr:`Intent.CANCEL` 도 아니다.
4. **띄어쓰기 무시.** 음성 인식 결과의 띄어쓰기는 불안정하므로 공백을 제거한
   문자열(compact)에 규칙을 적용한다.

규칙 우선순위(위에서부터 먼저 검사)
-----------------------------------
CALL_STAFF → REVISE → READ_ORIGINAL → EXPLAIN → REPEAT → PREVIOUS → NEXT
→ (현재 항목 선택지 라벨 일치) SELECT_OPTION → (서수·선택 표현) SELECT_OPTION
→ DISAGREE → AGREE → CONFIRM → CANCEL → QUESTION → UNKNOWN
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any, Mapping

from docagent.contracts import DocumentStructure, Field, FieldType
from docagent.errors import AdapterUnavailable, PiiEgressBlocked
from docagent.agent.state import SessionState

__all__ = [
    "Intent",
    "IntentResult",
    "RuleIntentClassifier",
    "LlmIntentClassifier",
    "DEFAULT_MIN_CONFIDENCE",
    "AGREE_KEYWORDS",
    "DISAGREE_KEYWORDS",
]


#: 규칙 신뢰도가 이 값 미만이면 :attr:`Intent.UNKNOWN` 으로 강등한다.
DEFAULT_MIN_CONFIDENCE: float = 0.6

#: 선택지 라벨이 "긍정" 쪽인지 판별할 때 쓰는 키워드.
AGREE_KEYWORDS: tuple[str, ...] = ("동의함", "동의", "예", "네", "찬성", "허용", "승인")
#: 선택지 라벨이 "부정" 쪽인지 판별할 때 쓰는 키워드.
DISAGREE_KEYWORDS: tuple[str, ...] = (
    "동의하지 않음",
    "동의하지않음",
    "미동의",
    "비동의",
    "거부",
    "아니오",
    "아니요",
    "반대",
)


class Intent(Enum):
    """사용자 발화의 의도."""

    AGREE = "agree"
    DISAGREE = "disagree"
    EXPLAIN = "explain"
    READ_ORIGINAL = "read_original"
    NEXT = "next"
    PREVIOUS = "previous"
    REPEAT = "repeat"
    SELECT_OPTION = "select_option"
    REVISE = "revise"
    CALL_STAFF = "call_staff"
    QUESTION = "question"
    CONFIRM = "confirm"
    CANCEL = "cancel"
    UNKNOWN = "unknown"


def _as_intent(value: Any) -> Intent:
    """문자열 또는 :class:`Intent` 를 :class:`Intent` 로 변환한다.

    :param value: Enum 인스턴스이거나 ``value`` 문자열.
    :returns: :class:`Intent`.
    :raises ValueError: 정의되지 않은 값인 경우.
    """
    if isinstance(value, Intent):
        return value
    try:
        return Intent(value)
    except ValueError as exc:
        allowed = ", ".join(repr(member.value) for member in Intent)
        raise ValueError(
            f"Intent 에 정의되지 않은 값입니다: {value!r}. 허용 값: {allowed}"
        ) from exc


@dataclass(frozen=True)
class IntentResult:
    """의도 분류 결과.

    :param intent: 분류된 의도.
    :param confidence: 0.0~1.0 신뢰도. :attr:`Intent.UNKNOWN` 은 항상 0.0 이다.
    :param slots: 부가 정보. 사용하는 키는 다음과 같다.

        * ``target_field_id`` — :attr:`Intent.REVISE` 가 되돌릴 항목 id.
        * ``option_label`` — :attr:`Intent.SELECT_OPTION` 이 고른 선택지 라벨.
        * ``option_index`` — 서수 표현("첫 번째")으로 고른 경우의 0 기반 인덱스.
        * ``legal_judgment`` — 자격·법적 판단을 묻는 질문이면 ``True``.
        * ``matched`` — 실제로 걸린 정규식 패턴(디버깅·감사용).
    :raises ValueError: ``confidence`` 가 0.0~1.0 범위를 벗어난 경우.
    """

    intent: Intent
    confidence: float = 0.0
    slots: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"IntentResult.confidence 는 0.0~1.0 이어야 합니다: {self.confidence}"
            )

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "slots": dict(self.slots),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntentResult":
        """:meth:`to_dict` 결과로부터 인스턴스를 복원한다.

        :param data: :meth:`to_dict` 가 만든 매핑.
        :returns: :class:`IntentResult`.
        :raises ValueError: 필수 키 누락 또는 미정의 의도 값.
        """
        if "intent" not in data:
            raise ValueError("IntentResult.from_dict 에 필수 키가 없습니다: intent")
        return cls(
            intent=_as_intent(data["intent"]),
            confidence=float(data.get("confidence", 0.0)),
            slots=dict(data.get("slots", {})),
        )


# --------------------------------------------------------------------------
# 규칙 정의
# --------------------------------------------------------------------------

#: 그 자체로 긍정인 짧은 응답(공백 제거·소문자 변환 후 완전 일치로만 판단한다).
_EXACT_AGREE: frozenset[str] = frozenset(
    {
        "네", "넵", "네네", "예", "옙", "응", "웅", "어", "그래", "그래요",
        "좋아", "좋아요", "좋습니다", "알겠어", "알겠어요", "알겠습니다",
        "오케이", "ok", "okay", "yes", "해줘", "해주세요", "그렇게해줘",
        "그렇게해주세요", "동의", "동의해", "동의해요",
    }
)

#: 그 자체로 부정인 짧은 응답.
_EXACT_DISAGREE: frozenset[str] = frozenset(
    {
        "아니", "아니요", "아뇨", "아니오", "싫어", "싫어요", "싫다",
        "안돼", "안돼요", "별로", "no", "노",
    }
)

#: ``(의도, 정규식, 신뢰도)`` 규칙표. **순서가 곧 우선순위**다.
_RULES: tuple[tuple[Intent, str, float], ...] = (
    (
        Intent.CALL_STAFF,
        r"직원|사람연결|사람좀|사람불러|사람을불러|상담원|담당자|도와주세요|도와주실|"
        r"도움이필요|도와주시|안내원|창구",
        0.95,
    ),
    (
        Intent.REVISE,
        r"(아까|방금|앞에|앞서|먼저|이전에|조금전).{0,14}(취소|바꾸|바꿔|번복|다시선택|"
        r"다시하|다시고|되돌|고치|변경|잘못)|"
        r"취소할래|취소하고싶|취소해줘|바꿀래|바꾸고싶|바꿔줄래|바꿔주세요|다시바꾸|"
        r"잘못눌|잘못선택|잘못골|잘못했|잘못체크|번복|되돌리|무르고싶|무를래",
        0.92,
    ),
    (
        Intent.READ_ORIGINAL,
        r"원문|전문|그대로읽|그대로들|있는그대로|원래문구|원래대로읽|본문읽|조항읽|"
        r"약관읽|약관그대로",
        0.93,
    ),
    (
        Intent.EXPLAIN,
        r"쉽게설명|쉽게말|설명해|설명좀|설명부탁|설명해주|무슨뜻|뭔뜻|뜻이뭐|뜻이무|"
        r"무슨말|무슨내용|어떤내용|무슨의미|뭐예요|뭐에요|뭐야|뭐죠|뭡니까|무엇입니까|"
        r"풀어서|이해가안|어려워요|어렵네|쉬운말",
        0.90,
    ),
    (
        Intent.REPEAT,
        r"다시들려|다시말해|다시읽어|다시안내|다시한번|한번더|한번만더|또읽어|"
        r"못들었|안들렸|재생",
        0.90,
    ),
    (
        Intent.PREVIOUS,
        r"이전항목|이전거|이전것|이전으로|뒤로|앞으로돌아|앞항목|아까그거|아까거|"
        r"직전|되돌아가|이전",
        0.88,
    ),
    (
        Intent.NEXT,
        r"다음|넘어가|넘길래|건너뛰|건너뛸|스킵|진행해|계속해|계속가|다음거|다음것",
        0.90,
    ),
    (
        Intent.SELECT_OPTION,
        r"첫번째|두번째|세번째|네번째|1번|2번|3번|4번|번째로|선택할래|선택해|"
        r"골라줘|고를래|고르겠",
        0.85,
    ),
    (
        Intent.DISAGREE,
        r"동의안|동의를안|동의하지않|동의하지마|동의못|동의는안|비동의|미동의|"
        r"거부|거절|반대|안할래|안하겠|하지않을래|하지않겠|싫어|원치않",
        0.92,
    ),
    (
        Intent.AGREE,
        r"동의(할|하|합|해|함|은|는|요|$)|동의하겠|동의합니다|동의할게|동의할래|"
        r"찬성|허락|그렇게해|그렇게하겠|맞아요|맞아|그래주세요",
        0.90,
    ),
    (
        Intent.CONFIRM,
        r"확인|다썼|다쓴|다적|다기입|다했|다됐|다되었|완료|끝났|끝냈|체크했|"
        r"서명했|표시했|맞습니다|제대로",
        0.88,
    ),
    (
        Intent.CANCEL,
        r"취소|그만|중단|중지|됐어요|됐습니다|하지마|안할게|관두",
        0.85,
    ),
    (
        Intent.QUESTION,
        r"까요|나요|인가요|는가요|을까|ㄹ수있|받을수|자격|가능한가|가능해|해당되|"
        r"어떻게해야|어떻게하나|왜|누가|언제|얼마",
        0.80,
    ),
)

#: 자격·법적 판단이 필요한 질문 판별 패턴. 걸리면 사람 지원으로 넘긴다.
_LEGAL_PATTERN: str = (
    r"자격|받을수|받을자격|해당되|대상인가|대상이|신청가능|법적|위법|처벌|불이익|"
    r"권리|의무|손해|책임|보상|환급|얼마|금액|세금|소송|계약해지"
)

#: 서수 표현 → 0 기반 인덱스.
_ORDINALS: tuple[tuple[str, int], ...] = (
    ("첫번째", 0),
    ("첫째", 0),
    ("1번", 0),
    ("일번", 0),
    ("두번째", 1),
    ("둘째", 1),
    ("2번", 1),
    ("이번째", 1),
    ("세번째", 2),
    ("셋째", 2),
    ("3번", 2),
    ("네번째", 3),
    ("넷째", 3),
    ("4번", 3),
)

_COMPILED_RULES: tuple[tuple[Intent, re.Pattern[str], float], ...] = tuple(
    (intent, re.compile(pattern), confidence)
    for intent, pattern, confidence in _RULES
)
_COMPILED_LEGAL: re.Pattern[str] = re.compile(_LEGAL_PATTERN)


def _compact(text: str) -> str:
    """분류용 정규화 문자열을 만든다.

    공백·문장부호를 제거하고 소문자로 낮춘다. 음성 인식 결과의 띄어쓰기가
    불안정하므로 공백 차이로 규칙이 어긋나지 않게 한다.

    :param text: 원문 발화.
    :returns: 공백·문장부호가 제거된 소문자 문자열.
    """
    lowered = text.strip().lower()
    return re.sub(r"[\s.,!?~·\"'()\[\]{}<>:;/\\-]+", "", lowered)


def _looks_like_agree_label(label: str) -> bool:
    """선택지 라벨이 "긍정" 쪽이면 True.

    :param label: 선택지 라벨(예: ``"동의함"``).
    :returns: 긍정 여부.
    """
    compact = _compact(label)
    if any(_compact(word) in compact for word in DISAGREE_KEYWORDS):
        return False
    return any(_compact(word) in compact for word in AGREE_KEYWORDS)


def _looks_like_disagree_label(label: str) -> bool:
    """선택지 라벨이 "부정" 쪽이면 True.

    :param label: 선택지 라벨(예: ``"동의하지 않음"``).
    :returns: 부정 여부.
    """
    compact = _compact(label)
    return any(_compact(word) in compact for word in DISAGREE_KEYWORDS)


class RuleIntentClassifier:
    """정규식 규칙만으로 한국어 발화를 분류하는 기본 분류기.

    외부 패키지·네트워크를 전혀 쓰지 않으며, 같은 입력에 항상 같은 결과를 낸다.

    :param min_confidence: 이 값 미만의 규칙 신뢰도는 :attr:`Intent.UNKNOWN` 으로
        강등한다. 기본값 :data:`DEFAULT_MIN_CONFIDENCE`.
    """

    def __init__(self, *, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence 는 0.0~1.0 이어야 합니다: {min_confidence}"
            )
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        *,
        state: SessionState | None = None,
        structure: DocumentStructure | None = None,
    ) -> IntentResult:
        """발화를 의도로 분류한다.

        :param text: 사용자 발화(STT 결과). 빈 문자열이면 :attr:`Intent.UNKNOWN`.
        :param state: 현재 세션 상태. 번복 대상 추출·선택지 매칭에 쓴다. 없어도 동작한다.
        :param structure: 문서 구조. 선택지 라벨·항목 제목 매칭에 쓴다. 없어도 동작한다.
        :returns: :class:`IntentResult`. 규칙에 걸리지 않으면 신뢰도 0.0 의
            :attr:`Intent.UNKNOWN` 을 돌려주며, 절대 임의로 추측하지 않는다.
        """
        compact = _compact(text)
        if not compact:
            return IntentResult(Intent.UNKNOWN, 0.0, {"reason": "빈 발화"})

        current = self._current_field(state, structure)

        for intent, pattern, confidence in _COMPILED_RULES:
            # 선택지 라벨 직접 호명은 서수 규칙보다 먼저 확인한다.
            if intent is Intent.SELECT_OPTION:
                label_hit = self._match_option_label(compact, current)
                if label_hit is not None:
                    return self._finalize(
                        Intent.SELECT_OPTION,
                        0.9,
                        {"option_label": label_hit, "matched": "option_label"},
                    )
            match = pattern.search(compact)
            if match is None:
                continue
            slots: dict[str, Any] = {"matched": match.group(0)}
            if intent is Intent.REVISE:
                target = self._resolve_revise_target(compact, state, structure)
                if target is not None:
                    slots["target_field_id"] = target
            elif intent is Intent.SELECT_OPTION:
                index = self._match_ordinal(compact)
                if index is not None:
                    slots["option_index"] = index
                    label = self._label_at(current, index)
                    if label is not None:
                        slots["option_label"] = label
                elif "option_label" not in slots:
                    # "선택해줘" 만으로는 무엇을 고를지 알 수 없다. 되묻게 한다.
                    return IntentResult(
                        Intent.UNKNOWN,
                        0.0,
                        {"reason": "선택 대상이 지정되지 않았습니다."},
                    )
            elif intent is Intent.QUESTION:
                if _COMPILED_LEGAL.search(compact) is not None:
                    slots["legal_judgment"] = True
            return self._finalize(intent, confidence, slots)

        # 규칙에 걸리지 않은 짧은 응답을 마지막으로 확인한다.
        if compact in _EXACT_DISAGREE:
            return self._finalize(Intent.DISAGREE, 0.9, {"matched": compact})
        if compact in _EXACT_AGREE:
            return self._finalize(Intent.AGREE, 0.9, {"matched": compact})

        return IntentResult(
            Intent.UNKNOWN, 0.0, {"reason": "일치하는 규칙이 없습니다."}
        )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _finalize(
        self, intent: Intent, confidence: float, slots: dict[str, Any]
    ) -> IntentResult:
        """신뢰도 하한을 적용해 결과를 확정한다.

        :param intent: 후보 의도.
        :param confidence: 규칙 신뢰도.
        :param slots: 부가 정보.
        :returns: 하한 미달이면 :attr:`Intent.UNKNOWN`, 아니면 그대로.
        """
        if confidence < self.min_confidence:
            merged = dict(slots)
            merged["reason"] = "신뢰도가 기준에 미달하여 UNKNOWN 으로 강등했습니다."
            merged["rejected_intent"] = intent.value
            return IntentResult(Intent.UNKNOWN, 0.0, merged)
        return IntentResult(intent, confidence, slots)

    @staticmethod
    def _current_field(
        state: SessionState | None, structure: DocumentStructure | None
    ) -> Field | None:
        """현재 안내 중인 항목을 찾는다.

        :param state: 세션 상태.
        :param structure: 문서 구조.
        :returns: :class:`~docagent.contracts.Field` 또는 ``None``.
        """
        if state is None or structure is None or state.current_field_id is None:
            return None
        return structure.field_by_id(state.current_field_id)

    @staticmethod
    def _match_option_label(compact: str, current: Field | None) -> str | None:
        """발화에 현재 항목의 선택지 라벨이 직접 등장하면 그 라벨을 반환한다.

        :param compact: 정규화된 발화.
        :param current: 현재 항목.
        :returns: 선택지 라벨 또는 ``None``.
        """
        if current is None or not current.options:
            return None
        # 긴 라벨을 먼저 본다("동의하지 않음" 이 "동의" 보다 우선).
        for option in sorted(
            current.options, key=lambda o: len(_compact(o.label)), reverse=True
        ):
            label_compact = _compact(option.label)
            if label_compact and label_compact in compact:
                return option.label
        return None

    @staticmethod
    def _match_ordinal(compact: str) -> int | None:
        """서수 표현을 0 기반 인덱스로 바꾼다.

        :param compact: 정규화된 발화.
        :returns: 인덱스 또는 ``None``.
        """
        for token, index in _ORDINALS:
            if token in compact:
                return index
        return None

    @staticmethod
    def _label_at(current: Field | None, index: int) -> str | None:
        """현재 항목의 ``index`` 번째 선택지 라벨을 반환한다.

        :param current: 현재 항목.
        :param index: 0 기반 인덱스.
        :returns: 라벨 또는 범위를 벗어나면 ``None``.
        """
        if current is None or index < 0 or index >= len(current.options):
            return None
        return current.options[index].label

    @staticmethod
    def _resolve_revise_target(
        compact: str,
        state: SessionState | None,
        structure: DocumentStructure | None,
    ) -> str | None:
        """번복 대상 항목 id 를 추출한다.

        해석 순서:

        1. 발화에 항목 제목이 직접 등장하면 그 항목.
        2. "동의 / 선택 / 체크" 를 언급했으면 **가장 최근에 완료된 선택형(CHOICE) 항목**.
        3. 그 외에는 가장 최근에 완료된 항목.
        4. 완료된 항목이 없으면 선택 기록이 있는 마지막 항목, 그것도 없으면 현재 항목.

        :param compact: 정규화된 발화.
        :param state: 세션 상태.
        :param structure: 문서 구조.
        :returns: 항목 id 또는 판단 불가 시 ``None``.
        """
        if state is None:
            return None
        if structure is not None:
            for item in structure.fields:
                title_compact = _compact(item.title)
                if len(title_compact) >= 2 and title_compact in compact:
                    return item.id
        mentions_choice = bool(re.search(r"동의|선택|체크|고른|골랐", compact))
        if mentions_choice and structure is not None:
            for fid in reversed(state.completed_fields):
                item = structure.field_by_id(fid)
                if item is not None and item.type is FieldType.CHOICE:
                    return fid
            for fid in reversed(state.completed_fields):
                item = structure.field_by_id(fid)
                if item is not None and item.options:
                    return fid
        if state.completed_fields:
            return state.completed_fields[-1]
        if state.selected_options:
            return list(state.selected_options)[-1]
        return state.current_field_id


class LlmIntentClassifier:
    """LLM 기반 의도 분류기 — **선택적 어댑터**. 기본 경로에서는 쓰지 않는다.

    규칙 분류기가 :attr:`Intent.UNKNOWN` 을 낸 발화만 넘겨 보조로 쓰는 것을 전제한다.
    LLM 응답이 정의된 의도가 아니면 조용히 추측하지 않고 :attr:`Intent.UNKNOWN` 을
    그대로 돌려준다.

    :param client: :class:`docagent.interfaces.LlmClient` 를 만족하는 객체.
        ``None`` 이면 :meth:`classify` 시점에 ``anthropic`` 패키지를 지연 import 하려
        시도하고, 없으면 :class:`~docagent.errors.AdapterUnavailable` 을 던진다.
    :param fallback: 파싱 실패 시 사용할 규칙 분류기. ``None`` 이면 새로 만든다.
    :param gate: 개인정보 유출 차단 게이트. ``None`` 이면 기본 게이트를 지연 import
        로 확보한다. 확보하지 못하면 :class:`~docagent.errors.PiiEgressBlocked` 를
        던지고 **게이트 없이는 호출하지 않는다.**

    이 분류기는 사용자 **발화 원문**을 프롬프트로 보내므로,
    :func:`docagent.agent.llm.build_llm` 과 똑같이 게이트를 강제한다.
    주입된 클라이언트는 :class:`docagent.pii.gate.GatedLlmClient` 로 감싸인 뒤에만
    호출되며, 마스킹 + 마스킹 후 재검사 + 감사 기록을 모두 거친다.
    """

    #: LLM 에 보내는 시스템 프롬프트. **공개 정보만** 담는다.
    SYSTEM_PROMPT: str = (
        "당신은 시각장애인용 문서작성 보조 에이전트의 의도 분류기입니다. "
        "사용자 발화를 다음 중 하나의 식별자로만 답하십시오: "
        + ", ".join(member.value for member in Intent)
        + ". 다른 말은 절대 덧붙이지 마십시오."
    )

    def __init__(
        self,
        client: Any | None = None,
        *,
        fallback: RuleIntentClassifier | None = None,
        gate: Any | None = None,
    ) -> None:
        self._client = client
        self._gate = gate
        self._gated: Any | None = None
        self.fallback = fallback if fallback is not None else RuleIntentClassifier()

    def bind_gate(self, gate: Any) -> None:
        """이 분류기가 쓸 게이트를 교체한다(세션 조립용).

        :func:`docagent.pipeline.build_session` 이 세션 게이트를 물려 주어,
        분류기 호출도 세션 감사 로그에 함께 남게 한다.

        :param gate: 사용할 게이트.
        :returns: ``None``.
        """
        self._gate = gate
        self._gated = None

    def classify(
        self,
        text: str,
        *,
        state: SessionState | None = None,
        structure: DocumentStructure | None = None,
    ) -> IntentResult:
        """LLM 으로 발화를 분류한다.

        :param text: 사용자 발화. 게이트가 마스킹한 뒤에만 LLM 으로 나간다.
        :param state: 현재 세션 상태(번복 대상 추출용).
        :param structure: 문서 구조.
        :returns: :class:`IntentResult`. 응답을 해석하지 못하면 규칙 분류기 결과.
        :raises docagent.errors.AdapterUnavailable: LLM 클라이언트가 없고
            ``anthropic`` 패키지도 설치되어 있지 않은 경우.
        """
        client = self._resolve_client()
        raw = client.complete(self.SYSTEM_PROMPT, text, 32)
        token = _compact(str(raw))
        for member in Intent:
            if _compact(member.value) == token:
                if member is Intent.UNKNOWN:
                    break
                slots: dict[str, Any] = {"source": "llm"}
                if member is Intent.REVISE:
                    target = RuleIntentClassifier._resolve_revise_target(
                        _compact(text), state, structure
                    )
                    if target is not None:
                        slots["target_field_id"] = target
                return IntentResult(member, 0.8, slots)
        return self.fallback.classify(text, state=state, structure=structure)

    def _resolve_client(self) -> Any:
        """게이트로 감싼 LLM 클라이언트를 확보한다(지연 import).

        :returns: ``complete(system, user, max_tokens)`` 를 가진 **게이트 래퍼**.
        :raises docagent.errors.AdapterUnavailable: ``anthropic`` 미설치.
        :raises docagent.errors.PiiEgressBlocked: 게이트를 확보하지 못한 경우.
        """
        if self._client is not None:
            if self._gated is None:
                self._gated = self._wrap(self._client)
            return self._gated
        try:  # 선택적 패키지: 모듈 최상단이 아니라 여기서만 import 한다.
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise AdapterUnavailable(
                "anthropic",
                feature="LLM 기반 의도 분류",
                extra="llm",
            ) from exc
        raise AdapterUnavailable(
            "anthropic",
            feature="LLM 기반 의도 분류(클라이언트 주입 필요)",
            extra="llm",
        )

    def _wrap(self, client: Any) -> Any:
        """클라이언트를 게이트 래퍼로 감싼다(이미 감싸여 있으면 그대로 쓴다).

        :param client: 감쌀 원본 클라이언트.
        :returns: :class:`docagent.pii.gate.GatedLlmClient`.
        :raises docagent.errors.PiiEgressBlocked: 게이트 구현을 확보하지 못한 경우.
        """
        try:  # 지연 import — agent 계층은 pii 를 최상단에서 import 하지 않는다.
            from docagent.pii.gate import GatedLlmClient, build_default_gate
        except ImportError as exc:  # pragma: no cover - 패키지가 온전하면 없다.
            raise PiiEgressBlocked(
                "PII 게이트 구현(docagent.pii.gate)을 찾지 못해 "
                "LLM 의도 분류 호출을 차단했습니다. gate 인자로 게이트를 주입하십시오."
            ) from exc
        if isinstance(client, GatedLlmClient):
            return client
        gate = self._gate if self._gate is not None else build_default_gate()
        return GatedLlmClient(inner=client, gate=gate, name="LlmIntentClassifier")
