"""신뢰도 산출과 3단계 분기 테스트."""

from __future__ import annotations

import pytest

from docagent.agent.confidence import (
    QUESTION_STOPWORDS,
    SIMILARITY_REFERENCE,
    ConfidenceBand,
    ConfidenceReport,
    band_for,
    score_answer,
    split_sentences,
)
from docagent.contracts import EXPLAIN_THRESHOLD, PARTIAL_THRESHOLD, RetrievedChunk

CONSENT_TEXT = (
    "개인정보 수집·이용 동의는 기관이나 회사가 나의 개인정보를 모아서 쓰는 것을 "
    "내가 스스로 허락한다는 뜻입니다. 동의를 받는 쪽은 수집 목적과 수집 항목, "
    "보유 기간을 미리 알려 주어야 합니다."
)
RIGHTS_TEXT = (
    "본인은 자기 개인정보를 어떤 곳이 어떻게 가지고 있는지 보여 달라고 요구할 수 있습니다."
)


def chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    """테스트용 근거 청크를 만든다.

    :param chunk_id: 청크 id.
    :param text: 본문.
    :param score: 유사도 점수.
    :returns: :class:`RetrievedChunk`.
    """
    return RetrievedChunk(chunk_id=chunk_id, text=text, source=f"{chunk_id}.md", score=score)


class TestBandBoundaries:
    """밴드 경계는 계약 상수에서만 온다."""

    def test_uses_contract_thresholds(self) -> None:
        """임계값은 contracts 의 상수를 그대로 쓴다(하드코딩 중복 금지)."""
        assert band_for(EXPLAIN_THRESHOLD) is ConfidenceBand.EXPLAIN
        assert band_for(PARTIAL_THRESHOLD) is ConfidenceBand.EXPLAIN_WITH_ORIGINAL
        assert band_for(PARTIAL_THRESHOLD - 0.0001) is ConfidenceBand.HANDOFF

    @pytest.mark.parametrize(
        ("score", "band"),
        [
            (1.0, ConfidenceBand.EXPLAIN),
            (0.90, ConfidenceBand.EXPLAIN),
            (0.80, ConfidenceBand.EXPLAIN_WITH_ORIGINAL),
            (0.70, ConfidenceBand.EXPLAIN_WITH_ORIGINAL),
            (0.50, ConfidenceBand.HANDOFF),
            (0.0, ConfidenceBand.HANDOFF),
        ],
    )
    def test_band_for(self, score: float, band: ConfidenceBand) -> None:
        """3단계 분기가 로드맵 구간과 일치한다."""
        assert band_for(score) is band

    @pytest.mark.parametrize("score", [-0.01, 1.01])
    def test_band_for_rejects_out_of_range(self, score: float) -> None:
        """범위를 벗어난 점수는 조용히 넘기지 않고 ValueError."""
        with pytest.raises(ValueError, match="0.0~1.0"):
            band_for(score)

    def test_band_is_str_comparable(self) -> None:
        """밴드는 문자열과 비교 가능해 오케스트레이터가 그대로 직렬화할 수 있다."""
        assert ConfidenceBand.EXPLAIN == "explain"
        assert ConfidenceBand.HANDOFF.value == "handoff"


class TestScoreAnswer:
    """종합 점수 산출."""

    def test_no_chunks_is_handoff(self) -> None:
        """근거가 없으면 점수 0.0, 핸드오프."""
        report = score_answer([], "동의가 뭐예요?", "설명입니다.")
        assert report.score == 0.0
        assert report.band is ConfidenceBand.HANDOFF
        assert report.needs_handoff is True

    def test_grounded_answer_scores_high(self) -> None:
        """근거 그대로 답하고 질의 어휘가 근거에 있으면 EXPLAIN 밴드에 든다."""
        chunks = [
            chunk("c1", CONSENT_TEXT, 0.30),
            chunk("c2", RIGHTS_TEXT, 0.08),
        ]
        report = score_answer(
            chunks,
            "개인정보 수집 동의가 뭐예요?",
            "개인정보 수집·이용 동의는 기관이 나의 개인정보를 모아서 쓰는 것을 허락한다는 뜻입니다.",
        )
        assert report.band is ConfidenceBand.EXPLAIN
        assert report.score >= EXPLAIN_THRESHOLD

    def test_ungrounded_answer_lowers_coverage(self) -> None:
        """근거에 없는 문장으로 채운 답변은 커버리지가 떨어져 점수가 낮아진다."""
        chunks = [chunk("c1", CONSENT_TEXT, 0.30)]
        grounded = score_answer(
            chunks, "동의 수집 목적", "동의를 받는 쪽은 수집 목적과 수집 항목을 알려야 합니다."
        )
        fabricated = score_answer(
            chunks,
            "동의 수집 목적",
            "우주선 발사 일정은 화요일이며 티켓 가격이 급등했습니다. 잠수함 정비는 무료입니다.",
        )
        assert fabricated.component("coverage") < grounded.component("coverage")
        assert fabricated.score < grounded.score

    def test_unrelated_question_is_handoff(self) -> None:
        """질의 어휘가 근거와 겹치지 않으면 핸드오프 밴드로 떨어진다."""
        chunks = [chunk("c1", CONSENT_TEXT, 0.05)]
        report = score_answer(chunks, "오늘 축구 경기 결과 알려줘", "축구 결과입니다.")
        assert report.band is ConfidenceBand.HANDOFF

    def test_distinctiveness_rewards_clear_winner(self) -> None:
        """1순위가 뚜렷할수록 변별력 요소가 커진다."""
        clear = score_answer(
            [chunk("c1", CONSENT_TEXT, 0.30), chunk("c2", RIGHTS_TEXT, 0.05)],
            "동의",
            CONSENT_TEXT,
        )
        muddy = score_answer(
            [chunk("c1", CONSENT_TEXT, 0.30), chunk("c2", RIGHTS_TEXT, 0.29)],
            "동의",
            CONSENT_TEXT,
        )
        assert clear.component("distinctiveness") > muddy.component("distinctiveness")

    def test_similarity_saturates_at_reference(self) -> None:
        """기준값 이상의 유사도는 1.0 으로 포화한다."""
        report = score_answer(
            [chunk("c1", CONSENT_TEXT, SIMILARITY_REFERENCE * 2)], "동의", CONSENT_TEXT
        )
        assert report.component("similarity") == pytest.approx(1.0)

    def test_question_stopwords_excluded(self) -> None:
        """의문 표현은 겹침률 분모에서 빠져 정상 질의가 손해 보지 않는다."""
        chunks = [chunk("c1", CONSENT_TEXT, 0.30)]
        plain = score_answer(chunks, "개인정보 수집 동의", CONSENT_TEXT)
        asked = score_answer(chunks, "개인정보 수집 동의가 뭐예요?", CONSENT_TEXT)
        assert asked.component("term_overlap") == pytest.approx(
            plain.component("term_overlap")
        )
        assert "뭐예요" in QUESTION_STOPWORDS

    def test_reasons_are_korean_and_nonempty(self) -> None:
        """판단 근거가 한국어 문장으로 남는다(감사 추적)."""
        report = score_answer([chunk("c1", CONSENT_TEXT, 0.30)], "동의", CONSENT_TEXT)
        assert report.reasons
        assert all(reason.endswith(".") for reason in report.reasons)

    def test_is_deterministic(self) -> None:
        """같은 입력은 항상 같은 점수."""
        chunks = [chunk("c1", CONSENT_TEXT, 0.3), chunk("c2", RIGHTS_TEXT, 0.1)]
        first = score_answer(chunks, "동의가 뭐예요?", CONSENT_TEXT)
        second = score_answer(chunks, "동의가 뭐예요?", CONSENT_TEXT)
        assert first == second

    def test_empty_answer_zero_coverage(self) -> None:
        """빈 답변은 커버리지 0."""
        report = score_answer([chunk("c1", CONSENT_TEXT, 0.3)], "동의", "")
        assert report.component("coverage") == 0.0


class TestConfidenceReport:
    """리포트 자료구조."""

    def test_score_range_validated(self) -> None:
        """범위를 벗어난 점수는 생성 시 거부한다."""
        with pytest.raises(ValueError, match="0.0~1.0"):
            ConfidenceReport(score=1.5, band=ConfidenceBand.EXPLAIN)

    def test_to_dict_is_json_ready(self) -> None:
        """dict 로 바꿀 때 Enum 은 문자열 value 로 나간다."""
        report = ConfidenceReport(
            score=0.9,
            band=ConfidenceBand.EXPLAIN,
            reasons=("근거 충분.",),
            components=(("similarity", 1.0),),
        )
        payload = report.to_dict()
        assert payload["band"] == "explain"
        assert payload["components"]["similarity"] == 1.0

    def test_unknown_component_raises(self) -> None:
        """없는 요소 이름은 조용히 0 을 주지 않고 KeyError."""
        report = ConfidenceReport(score=0.5, band=ConfidenceBand.HANDOFF)
        with pytest.raises(KeyError):
            report.component("없는요소")


class TestSentenceSplit:
    """문장 분리."""

    def test_splits_korean_endings(self) -> None:
        """한국어 종결어미 '다.' 기준으로 나눈다."""
        assert split_sentences("첫 문장입니다. 둘째 문장입니다.") == [
            "첫 문장입니다.",
            "둘째 문장입니다.",
        ]

    def test_empty(self) -> None:
        """빈 입력은 빈 리스트."""
        assert split_sentences("   ") == []
