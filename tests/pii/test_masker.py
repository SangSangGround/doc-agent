"""마스킹 테스트 — 원문 흔적이 남지 않는가, 복원 맵이 새지 않는가.

등장하는 개인정보는 모두 가상 예시값이다.
"""

from __future__ import annotations

import copy
import json
import pickle

import pytest

from docagent.contracts import PiiSpan, SanitizedText
from docagent.pii.detectors import detect_pii
from docagent.pii.masker import (
    DEFAULT_PARTIAL_KEEP,
    FULL_MASK_TOKEN,
    MASK_CHAR,
    MaskStrategy,
    RestoreMap,
    mask,
    mask_with_restore,
    token_for,
)

SAMPLE_DOC = (
    "성명 김철수\n"
    "주민등록번호 900101-1234567\n"
    "주소 서울특별시 중구 세종대로 110\n"
    "전화 010-1234-5678\n"
)

#: 마스킹 결과에 절대 남아 있으면 안 되는 원문 조각.
FORBIDDEN_FRAGMENTS = (
    "김철수",
    "900101-1234567",
    "9001011234567",
    "1234567",
    "서울특별시 중구 세종대로 110",
    "010-1234-5678",
    "01012345678",
)


class TestFullStrategy:
    """``FULL`` 전략 — 외부 전송 기본값."""

    def test_roadmap_document_leaves_no_original_digits(self) -> None:
        """예시 문서를 마스킹하면 원문 숫자열이 하나도 남지 않는다."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.FULL)
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in result.text, f"원문 유출: {fragment}"

    def test_no_standalone_digit_run_of_six_or_more(self) -> None:
        """FULL 마스킹 후 6자리 이상 연속 숫자가 남지 않는다."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.FULL)
        runs: list[str] = []
        current = ""
        for ch in result.text:
            if ch.isdigit():
                current += ch
            else:
                runs.append(current)
                current = ""
        runs.append(current)
        assert max((len(run) for run in runs), default=0) < 6

    def test_labels_survive_masking(self) -> None:
        """라벨(인쇄된 공개 문구)은 그대로 남는다."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.FULL)
        assert "성명" in result.text
        assert "주민등록번호" in result.text
        assert result.text.count(FULL_MASK_TOKEN) == 4

    def test_spans_keep_source_indices(self) -> None:
        """``SanitizedText.spans`` 인덱스는 마스킹 전 원문 기준을 유지한다."""
        spans = detect_pii(SAMPLE_DOC)
        result = mask(SAMPLE_DOC, spans, MaskStrategy.FULL)
        assert result.spans == spans
        assert result.has_pii
        assert SAMPLE_DOC[result.spans[0].start : result.spans[0].end] == "김철수"

    def test_default_strategy_is_full(self) -> None:
        """전략을 생략하면 가장 안전한 FULL 이 쓰인다."""
        spans = detect_pii(SAMPLE_DOC)
        assert mask(SAMPLE_DOC, spans).text == mask(
            SAMPLE_DOC, spans, MaskStrategy.FULL
        ).text

    def test_text_without_pii_is_unchanged(self) -> None:
        """탐지 결과가 없으면 원문이 그대로 유지된다."""
        text = "접수번호 제2026-0001호"
        result = mask(text, detect_pii(text))
        assert result.text == text
        assert not result.has_pii


class TestPartialStrategy:
    """``PARTIAL`` 전략 — 로컬 확인 전용."""

    def test_rrn_partial_matches_specification(self) -> None:
        """주민등록번호는 ``900101-1******`` 형태로 가려진다."""
        text = "900101-1234567"
        result = mask(text, detect_pii(text), MaskStrategy.PARTIAL)
        assert result.text == "900101-1******"

    def test_partial_preserves_length(self) -> None:
        """PARTIAL 은 문자 수를 바꾸지 않는다(촉각·음성 위치 안내용)."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.PARTIAL)
        assert len(result.text) == len(SAMPLE_DOC)

    def test_partial_masks_the_tail(self) -> None:
        """뒷부분은 전부 가림 문자로 바뀐다."""
        text = "전화 010-1234-5678"
        result = mask(text, detect_pii(text), MaskStrategy.PARTIAL)
        assert result.text.startswith("전화 010-")
        assert result.text.endswith(MASK_CHAR * 9)

    def test_default_keep_is_used_for_unknown_type(self) -> None:
        """유형별 설정이 없으면 기본 노출 길이를 쓴다."""
        text = "abcdefgh"
        span = PiiSpan(start=0, end=8, pii_type="mystery", raw_len=8, confidence=1.0)
        result = mask(text, [span], MaskStrategy.PARTIAL)
        assert result.text == "ab" + MASK_CHAR * 6
        assert DEFAULT_PARTIAL_KEEP == 2


class TestTypeTokenStrategy:
    """``TYPE_TOKEN`` 전략과 복원 맵."""

    def test_tokens_are_numbered_per_type(self) -> None:
        """같은 유형이 여러 개면 1부터 번호가 붙는다."""
        text = "연락처 010-1234-5678 과 010-9876-5432"
        result = mask(text, detect_pii(text), MaskStrategy.TYPE_TOKEN)
        assert "[PII:PHONE_1]" in result.text
        assert "[PII:PHONE_2]" in result.text

    def test_token_format(self) -> None:
        """토큰 형식이 ``[PII:RRN_1]`` 규격을 따른다."""
        assert token_for("rrn", 1) == "[PII:RRN_1]"
        assert token_for("name", 3) == "[PII:NAME_3]"

    def test_token_index_must_be_positive(self) -> None:
        """0 이하 일련번호는 ValueError."""
        with pytest.raises(ValueError, match="1 이상"):
            token_for("rrn", 0)

    def test_type_token_leaves_no_original_value(self) -> None:
        """토큰 치환 후에도 원문 조각이 남지 않는다."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.TYPE_TOKEN)
        for fragment in FORBIDDEN_FRAGMENTS:
            assert fragment not in result.text

    def test_restore_round_trip(self) -> None:
        """복원 맵으로 원문을 되돌릴 수 있다."""
        sanitized, restore = mask_with_restore(
            SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.TYPE_TOKEN
        )
        assert restore.restore(sanitized.text) == SAMPLE_DOC

    def test_restore_handles_double_digit_indices(self) -> None:
        """토큰이 10개를 넘어도 접두 충돌 없이 복원된다."""
        numbers = [f"010-1234-{5000 + i:04d}" for i in range(12)]
        text = ", ".join(numbers)
        sanitized, restore = mask_with_restore(
            text, detect_pii(text), MaskStrategy.TYPE_TOKEN
        )
        assert "[PII:PHONE_12]" in sanitized.text
        assert restore.restore(sanitized.text) == text


class TestRestoreMapSecrecy:
    """복원 맵은 프로세스 메모리를 벗어나지 못한다."""

    def build(self) -> RestoreMap:
        """테스트용 복원 맵을 만든다."""
        _sanitized, restore = mask_with_restore(
            SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.TYPE_TOKEN
        )
        return restore

    def test_repr_hides_values(self) -> None:
        """repr 에 원문 값이 없다."""
        text = repr(self.build())
        assert "김철수" not in text
        assert "900101" not in text
        assert "RestoreMap" in text

    def test_str_hides_values(self) -> None:
        """str 에도 원문 값이 없다."""
        assert "900101" not in str(self.build())

    def test_pickle_is_blocked(self) -> None:
        """pickle 직렬화가 차단된다."""
        with pytest.raises(TypeError, match="직렬화"):
            pickle.dumps(self.build())

    def test_deepcopy_is_blocked(self) -> None:
        """deepcopy 도 __reduce__ 를 거치므로 차단된다."""
        with pytest.raises(TypeError, match="직렬화"):
            copy.deepcopy(self.build())

    def test_json_serialization_is_impossible(self) -> None:
        """json 직렬화 대상이 될 수 없다."""
        with pytest.raises(TypeError):
            json.dumps(self.build())

    def test_iteration_exposes_tokens_only(self) -> None:
        """순회하면 토큰만 나온다(값은 노출되지 않는다)."""
        restore = self.build()
        for token in restore:
            assert token.startswith("[PII:")
        assert len(restore.tokens()) == len(restore)

    def test_full_strategy_produces_empty_restore_map(self) -> None:
        """FULL 전략은 복원 맵을 만들지 않는다(원리적으로 복원 불가)."""
        _sanitized, restore = mask_with_restore(
            SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.FULL
        )
        assert len(restore) == 0

    def test_mask_discards_restore_map(self) -> None:
        """``mask()`` 는 복원 맵을 만들지 않고 버린다."""
        result = mask(SAMPLE_DOC, detect_pii(SAMPLE_DOC), MaskStrategy.TYPE_TOKEN)
        assert isinstance(result, SanitizedText)


class TestMaskValidation:
    """잘못된 입력은 조용히 넘어가지 않는다."""

    def test_overlapping_spans_raise(self) -> None:
        """겹치는 구간은 ValueError."""
        spans = [
            PiiSpan(start=0, end=5, pii_type="rrn", raw_len=5),
            PiiSpan(start=3, end=8, pii_type="name", raw_len=5),
        ]
        with pytest.raises(ValueError, match="겹칩니다"):
            mask("0123456789", spans)

    def test_span_out_of_range_raises(self) -> None:
        """텍스트 범위를 벗어난 구간은 ValueError."""
        spans = [PiiSpan(start=0, end=99, pii_type="rrn", raw_len=99)]
        with pytest.raises(ValueError, match="범위"):
            mask("짧은 문자열", spans)

    def test_non_string_text_raises(self) -> None:
        """문자열이 아닌 입력은 TypeError."""
        with pytest.raises(TypeError):
            mask(12345, [])  # type: ignore[arg-type]

    def test_unordered_spans_are_accepted(self) -> None:
        """구간 순서가 뒤섞여 들어와도 정렬 후 처리한다."""
        text = "AA900101-1234567BB010-1234-5678"
        spans = list(detect_pii(text))
        result = mask(text, list(reversed(spans)))
        assert result.text == f"AA{FULL_MASK_TOKEN}BB{FULL_MASK_TOKEN}"

    def test_empty_spans_returns_original(self) -> None:
        """구간이 없으면 원문 그대로."""
        assert mask("안녕하세요", []).text == "안녕하세요"
