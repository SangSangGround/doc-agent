"""응답 신뢰도 산출과 3단계 분기.

로드맵의 3단계 분기를 그대로 코드화한다.

======================================= ================================================
밴드                                     동작
======================================= ================================================
:attr:`ConfidenceBand.EXPLAIN`           설명만 제공한다.
:attr:`ConfidenceBand.EXPLAIN_WITH_ORIGINAL` 설명과 함께 원문 낭독을 병행 제안한다.
:attr:`ConfidenceBand.HANDOFF`           생성 텍스트를 버리고 원문 안내·직원 연결로 넘긴다.
======================================= ================================================

경계값은 :data:`docagent.contracts.EXPLAIN_THRESHOLD` (0.85) 와
:data:`docagent.contracts.PARTIAL_THRESHOLD` (0.70) 를 **import 해서** 쓴다.
이 모듈에 숫자를 다시 적어 두지 않는다(계약 중복 금지).

점수 구성
---------
네 가지 요소의 가중합이며 각 요소는 0.0~1.0 으로 정규화된다.

1. ``similarity`` — 최상위 근거 청크의 코사인 유사도.
2. ``distinctiveness`` — 상위 k 청크 점수의 분산(최상위와 나머지 평균의 상대 격차).
   1등이 뚜렷하면 높고, 여러 근거가 비슷하게 걸리면(= 질의가 모호하면) 낮다.
3. ``term_overlap`` — 질의 어휘가 근거에 얼마나 등장하는가.
4. ``coverage`` — 답변 문장 중 근거로 뒷받침되는 비율(근거 커버리지).

가중치는 이 모듈의 ``W_*`` 상수이며, 프로젝트 코퍼스 기준으로 보정한 값이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from docagent.agent.rag import char_ngrams, word_tokens
from docagent.contracts import (
    EXPLAIN_THRESHOLD,
    PARTIAL_THRESHOLD,
    RetrievedChunk,
)

try:  # pragma: no cover - 파이썬 3.11+ 에서는 StrEnum 이 있지만 3.10 호환을 유지한다.
    from enum import StrEnum as _StrEnumBase  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from enum import Enum as _EnumBase

    class _StrEnumBase(str, _EnumBase):  # type: ignore[no-redef]
        """``str`` 과 호환되는 Enum 기반 클래스(파이썬 3.10 폴백)."""


__all__ = [
    "ConfidenceBand",
    "ConfidenceReport",
    "QUESTION_STOPWORDS",
    "SIMILARITY_REFERENCE",
    "SENTENCE_SUPPORT_RATIO",
    "W_SIMILARITY",
    "W_DISTINCTIVENESS",
    "W_TERM_OVERLAP",
    "W_COVERAGE",
    "band_for",
    "split_sentences",
    "score_answer",
]


class ConfidenceBand(_StrEnumBase):
    """신뢰도 3단계 분기 밴드.

    ``str`` 서브클래스이므로 ``band == "explain"`` 비교와 JSON 직렬화가 그대로 된다.
    """

    #: 신뢰도가 :data:`~docagent.contracts.EXPLAIN_THRESHOLD` 이상. 설명을 제공한다.
    EXPLAIN = "explain"
    #: :data:`~docagent.contracts.PARTIAL_THRESHOLD` 이상 ~ 미만. 설명과 원문을 병행한다.
    EXPLAIN_WITH_ORIGINAL = "explain_with_original"
    #: :data:`~docagent.contracts.PARTIAL_THRESHOLD` 미만. 원문 안내 또는 직원 연결.
    HANDOFF = "handoff"


# --------------------------------------------------------------------------
# 보정 상수
# --------------------------------------------------------------------------

#: 코사인 유사도 만점 기준값.
#:
#: 짧은 질의와 500자 청크 사이의 TF-IDF 코사인은 구조적으로 낮게 나온다
#: (본 프로젝트 코퍼스에서 정답 청크가 0.15~0.25 구간). 이 값 이상이면
#: ``similarity`` 요소를 1.0 으로 본다.
SIMILARITY_REFERENCE: float = 0.20

#: 답변 문장 하나가 "근거로 뒷받침된다"고 인정받기 위한 최소 어휘 일치 비율.
SENTENCE_SUPPORT_RATIO: float = 0.60

#: 어휘 겹침 계산에서 제외하는 의문·요청 표현.
#:
#: "동의가 뭐예요?" 의 ``뭐예요`` 처럼 질문을 만드는 껍데기 어휘는 근거 문서에
#: 등장하지 않는 것이 정상이다. 이런 토큰을 분모에 넣으면 잘 답한 질의까지
#: 겹침률이 깎이므로 불용어로 제외한다.
QUESTION_STOPWORDS: frozenset[str] = frozenset(
    {
        "뭐예요",
        "뭐야",
        "뭔가요",
        "무엇",
        "무슨",
        "어떤",
        "어떻게",
        "언제",
        "어디",
        "누가",
        "누구",
        "인가요",
        "입니까",
        "건가요",
        "인지",
        "건지",
        "알려줘",
        "말해줘",
        "해줘",
        "주세요",
        "합니까",
        "하나요",
        "있나요",
        "될까요",
        "이거",
        "이건",
        "그거",
        "그건",
        "저거",
        "여기",
        "지금",
        "제가",
        "저는",
        "저도",
        "그리고",
        "그런데",
    }
)

#: 가중치 — 최상위 유사도.
W_SIMILARITY: float = 0.20
#: 가중치 — 상위 청크 점수 분산(1등의 뚜렷함).
W_DISTINCTIVENESS: float = 0.10
#: 가중치 — 질의·근거 어휘 겹침률.
W_TERM_OVERLAP: float = 0.30
#: 가중치 — 답변의 근거 커버리지.
W_COVERAGE: float = 0.40


#: 문장 분리 정규식(한국어 종결어미 '다.' 와 일반 문장부호를 함께 본다).
_SENTENCE_RE = re.compile(r"(?<=다\.)\s+|(?<=[.!?])\s+|\n+")

#: 어휘 겹침·커버리지 계산에 쓸 최소 토큰 길이(조사·접속어 잡음 제거).
_MIN_CONTENT_LEN: int = 2


@dataclass(frozen=True)
class ConfidenceReport:
    """신뢰도 산출 결과.

    :param score: 종합 신뢰도(0.0~1.0).
    :param band: 3단계 분기 밴드.
    :param reasons: 판단 근거를 한국어로 적은 문장들(사용자 낭독 가능·감사 추적용).
    :param components: ``(요소 이름, 값)`` 쌍. 튜닝·디버깅용이며 기본값은 빈 튜플이다.
    :raises ValueError: ``score`` 가 0.0~1.0 범위를 벗어난 경우.
    """

    score: float
    band: ConfidenceBand
    reasons: tuple[str, ...] = ()
    components: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(
                f"ConfidenceReport.score 는 0.0~1.0 이어야 합니다: {self.score}"
            )
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(
            self, "components", tuple((str(k), float(v)) for k, v in self.components)
        )

    @property
    def needs_handoff(self) -> bool:
        """사람 지원(또는 원문 안내)으로 넘겨야 하면 True."""
        return self.band is ConfidenceBand.HANDOFF

    def component(self, name: str) -> float:
        """요소 값을 이름으로 읽는다.

        :param name: 요소 이름(``similarity`` / ``distinctiveness`` /
            ``term_overlap`` / ``coverage``).
        :returns: 해당 요소 값.
        :raises KeyError: 없는 요소 이름인 경우.
        """
        for key, value in self.components:
            if key == name:
                return value
        raise KeyError(f"신뢰도 요소가 없습니다: {name}")

    def to_dict(self) -> dict[str, Any]:
        """JSON 직렬화 가능한 dict 를 반환한다."""
        return {
            "score": self.score,
            "band": self.band.value,
            "reasons": list(self.reasons),
            "components": {key: value for key, value in self.components},
        }


def band_for(score: float) -> ConfidenceBand:
    """점수를 3단계 밴드로 바꾼다.

    :param score: 0.0~1.0 신뢰도.
    :returns: :class:`ConfidenceBand`.
    :raises ValueError: ``score`` 가 0.0~1.0 범위를 벗어난 경우.
    """
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score 는 0.0~1.0 이어야 합니다: {score}")
    if score >= EXPLAIN_THRESHOLD:
        return ConfidenceBand.EXPLAIN
    if score >= PARTIAL_THRESHOLD:
        return ConfidenceBand.EXPLAIN_WITH_ORIGINAL
    return ConfidenceBand.HANDOFF


def split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 나눈다.

    :param text: 대상 텍스트.
    :returns: 공백만 남는 문장을 제외한 문장 목록. 입력이 비면 빈 리스트.
    """
    if not text or not text.strip():
        return []
    return [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]


def _content_tokens(text: str) -> list[str]:
    """의미 있는 내용어 토큰만 뽑는다(길이 2 이상, 의문 표현 제외).

    :param text: 대상 텍스트.
    :returns: 내용어 토큰 목록(중복 유지).
    """
    return [
        token
        for token in word_tokens(text)
        if len(token) >= _MIN_CONTENT_LEN and token not in QUESTION_STOPWORDS
    ]


def _is_supported(token: str, evidence_words: set[str], evidence_text: str) -> bool:
    """토큰이 근거에 뒷받침되는지 판정한다.

    완전 일치를 먼저 보고, 없으면 문자 bigram 절반 이상이 근거 본문에 나타나는지로
    부분 점수를 준다. ``물어봐`` 와 근거의 ``물어볼`` 처럼 활용형이 달라도
    같은 어간이면 인정하기 위해서다.

    :param token: 검사할 토큰.
    :param evidence_words: 근거의 어절 토큰 집합.
    :param evidence_text: 근거 본문 전체(부분 일치 검사용).
    :returns: 뒷받침되면 True.
    """
    if token in evidence_words:
        return True
    grams = char_ngrams(token, 2)
    if not grams:
        return False
    hits = sum(1 for gram in grams if gram in evidence_text)
    return hits / len(grams) >= 0.5


def score_answer(
    chunks: Sequence[RetrievedChunk],
    question: str,
    answer: str,
) -> ConfidenceReport:
    """근거·질의·답변을 종합해 신뢰도와 밴드를 산출한다.

    :param chunks: 검색된 근거 청크(점수 내림차순 가정). 비어 있으면 점수 0.0.
    :param question: 사용자의 질문(또는 설명 요청 문구).
    :param answer: 생성된 설명 문장.
    :returns: :class:`ConfidenceReport`.

    사용 예::

        report = score_answer(chunks, "개인정보 수집 동의가 뭐예요?", answer)
        if report.band is ConfidenceBand.HANDOFF:
            ...
    """
    if not chunks:
        return ConfidenceReport(
            score=0.0,
            band=ConfidenceBand.HANDOFF,
            reasons=("근거 문서를 찾지 못했습니다.",),
            components=(
                ("similarity", 0.0),
                ("distinctiveness", 0.0),
                ("term_overlap", 0.0),
                ("coverage", 0.0),
            ),
        )

    reasons: list[str] = []

    # 1) 최상위 유사도 --------------------------------------------------
    top_score = float(chunks[0].score)
    similarity = min(1.0, max(0.0, top_score / SIMILARITY_REFERENCE))
    reasons.append(
        f"최상위 근거 유사도 {top_score:.3f} (기준 {SIMILARITY_REFERENCE:.2f})."
    )

    # 2) 상위 청크 점수 분산 --------------------------------------------
    if len(chunks) < 2 or top_score <= 0.0:
        distinctiveness = 0.5  # 비교 대상이 없으면 중립값.
        reasons.append("비교할 하위 근거가 없어 변별력은 중립으로 두었습니다.")
    else:
        rest = [float(chunk.score) for chunk in chunks[1:]]
        rest_mean = sum(rest) / len(rest)
        distinctiveness = min(1.0, max(0.0, (top_score - rest_mean) / top_score))
        reasons.append(
            f"1순위 근거가 나머지 평균보다 {distinctiveness * 100:.0f}% 앞섭니다."
        )

    # 3) 질의-근거 어휘 겹침률 -------------------------------------------
    evidence_text = " ".join(chunk.text for chunk in chunks)
    evidence_words = set(word_tokens(evidence_text))
    question_tokens = list(dict.fromkeys(_content_tokens(question)))
    if question_tokens:
        matched = [
            token
            for token in question_tokens
            if _is_supported(token, evidence_words, evidence_text)
        ]
        term_overlap = len(matched) / len(question_tokens)
        missing = [token for token in question_tokens if token not in matched]
        if missing:
            reasons.append(
                "근거에서 확인하지 못한 질문 어휘: " + ", ".join(missing[:5]) + "."
            )
        else:
            reasons.append("질문 어휘가 모두 근거에서 확인되었습니다.")
    else:
        term_overlap = 0.5  # 질의에서 내용어를 못 뽑으면 중립값.
        reasons.append("질문에서 내용어를 찾지 못해 겹침률은 중립으로 두었습니다.")

    # 4) 답변의 근거 커버리지 --------------------------------------------
    sentences = split_sentences(answer)
    if sentences:
        supported = 0
        for sentence in sentences:
            tokens = list(dict.fromkeys(_content_tokens(sentence)))
            if not tokens:
                supported += 1  # 내용어가 없는 짧은 연결 문장은 위험하지 않다.
                continue
            hits = sum(
                1
                for token in tokens
                if _is_supported(token, evidence_words, evidence_text)
            )
            if hits / len(tokens) >= SENTENCE_SUPPORT_RATIO:
                supported += 1
        coverage = supported / len(sentences)
        reasons.append(
            f"답변 {len(sentences)}문장 중 {supported}문장이 근거로 뒷받침됩니다."
        )
    else:
        coverage = 0.0
        reasons.append("답변이 비어 있어 근거 커버리지를 0 으로 두었습니다.")

    score = (
        W_SIMILARITY * similarity
        + W_DISTINCTIVENESS * distinctiveness
        + W_TERM_OVERLAP * term_overlap
        + W_COVERAGE * coverage
    )
    score = round(min(1.0, max(0.0, score)), 6)
    band = band_for(score)
    reasons.append(f"종합 신뢰도 {score:.3f} → {_BAND_LABEL[band]}.")

    return ConfidenceReport(
        score=score,
        band=band,
        reasons=tuple(reasons),
        components=(
            ("similarity", round(similarity, 6)),
            ("distinctiveness", round(distinctiveness, 6)),
            ("term_overlap", round(term_overlap, 6)),
            ("coverage", round(coverage, 6)),
        ),
    )


#: 밴드별 한국어 라벨(로그·낭독용).
_BAND_LABEL: Mapping[ConfidenceBand, str] = {
    ConfidenceBand.EXPLAIN: "설명 제공",
    ConfidenceBand.EXPLAIN_WITH_ORIGINAL: "설명과 원문 병행",
    ConfidenceBand.HANDOFF: "원문 안내 또는 직원 연결",
}
