"""탐지기 테스트 — 회피 변형에 강한가, 그리고 무해한 숫자를 잡지 않는가.

이 파일에 등장하는 모든 개인정보 예시값은 **명백한 허구**다.
주민등록번호 ``900101-1234567`` 은 합성 문서 생성기가 쓰는 것과 동일한
가상 예시값이며 실재하는 사람의 정보가 아니다.
"""

from __future__ import annotations

import pytest

from docagent.contracts import PiiSpan
from docagent.pii.detectors import (
    PiiDetectionError,
    RegexPiiDetector,
    contains_pii,
    detect_pii,
    summarize_types,
)
from docagent.pii.patterns import (
    business_number_ok,
    digits_only,
    luhn_ok,
    normalize,
    rrn_digits_valid,
)

# 로드맵 예시 문서와 같은 구성의 가상 개인정보 문장.
SAMPLE_DOC = (
    "성명 김철수\n"
    "주민등록번호 900101-1234567\n"
    "주소 서울특별시 중구 세종대로 110\n"
    "전화 010-1234-5678\n"
)

# 무해한 숫자만 들어 있는 문장(거짓양성 테스트용).
BENIGN_DOC = (
    "접수번호 제2026-0001호로 접수되었으며, 지급 금액은 1,234,567원입니다. "
    "처리 기한은 2026년 9월 30일이고 관련 근거는 제12조 제3항입니다. "
    "총 3건, 12쪽 분량입니다."
)


def types_of(text: str) -> set[str]:
    """텍스트에서 탐지된 개인정보 유형 집합을 돌려주는 테스트 헬퍼.

    :param text: 검사 대상.
    :returns: 유형 식별자 집합.
    """
    return {span.pii_type for span in detect_pii(text)}


def raw_of(text: str) -> list[str]:
    """탐지 구간이 가리키는 원문 조각 목록(테스트 검증용).

    :param text: 검사 대상.
    :returns: 원문 조각 리스트.
    """
    return [text[span.start : span.end] for span in detect_pii(text)]


class TestNormalize:
    """유니코드 정규화와 오프셋 매핑."""

    def test_fullwidth_digits_are_folded(self) -> None:
        """전각 숫자가 ASCII 숫자로 접힌다."""
        assert normalize("９００１０１").text == "900101"

    def test_whitespace_runs_collapse_to_single_space(self) -> None:
        """연속 공백·줄바꿈이 공백 하나로 축약된다."""
        assert normalize("가  \n\t 나").text == "가 나"

    def test_dash_variants_unify_to_ascii_hyphen(self) -> None:
        """en dash·em dash·전각 하이픈이 ASCII 하이픈으로 통일된다."""
        assert normalize("a–b—c－d−e").text == "a-b-c-d-e"

    def test_zero_width_characters_are_removed(self) -> None:
        """제로폭 문자가 제거된다."""
        assert normalize("900101​-1234567").text == "900101-1234567"

    def test_offset_mapping_points_back_to_source(self) -> None:
        """정규화 좌표를 원문 좌표로 되돌릴 수 있다."""
        source = "번호는  ９００１０１ 입니다"
        result = normalize(source)
        index = result.text.index("900101")
        start, end = result.to_source_span(index, index + 6)
        assert source[start:end] == "９００１０１"

    def test_empty_span_raises(self) -> None:
        """빈 구간 요청은 조용히 넘어가지 않고 ValueError."""
        result = normalize("abc")
        with pytest.raises(ValueError, match="비어"):
            result.to_source_span(1, 1)

    def test_non_string_input_raises(self) -> None:
        """문자열이 아닌 입력은 TypeError."""
        with pytest.raises(TypeError):
            normalize(12345)  # type: ignore[arg-type]


class TestValidators:
    """검증 헬퍼."""

    def test_luhn_accepts_known_test_card(self) -> None:
        """널리 쓰이는 테스트 카드번호가 Luhn 을 통과한다."""
        assert luhn_ok("4111111111111111")

    def test_luhn_rejects_broken_card(self) -> None:
        """한 자리를 바꾼 번호는 Luhn 에서 걸린다."""
        assert not luhn_ok("4111111111111112")

    def test_rrn_rejects_impossible_date(self) -> None:
        """13월 32일 같은 불가능한 생년월일은 거부된다."""
        assert not rrn_digits_valid("9013321234567")

    def test_rrn_rejects_gender_code_nine(self) -> None:
        """성별코드 9(1800년대)는 이번 범위에서 채택하지 않는다."""
        assert not rrn_digits_valid("9001019234567")

    def test_rrn_accepts_valid_sample(self) -> None:
        """가상 예시값은 형식·날짜·성별코드를 모두 만족한다."""
        assert rrn_digits_valid("9001011234567")

    def test_rrn_does_not_require_legacy_check_digit(self) -> None:
        """2020년 폐지된 검증숫자 규칙을 강제하지 않는다.

        뒷자리 마지막 숫자만 바꿔도 여전히 탐지 대상이어야 한다
        (놓치는 쪽이 훨씬 위험하다).
        """
        assert rrn_digits_valid("9001011234560")
        assert rrn_digits_valid("9001011234561")

    def test_business_number_checksum(self) -> None:
        """사업자등록번호 체크섬이 동작한다."""
        assert business_number_ok("1234567891")
        assert not business_number_ok("1234567890")

    def test_digits_only(self) -> None:
        """구분자를 제거하고 숫자만 남긴다."""
        assert digits_only("900101-1234567") == "9001011234567"


class TestRrnEvasion:
    """주민등록번호 회피 변형 — 전부 탐지되어야 한다."""

    EVASIONS = (
        "900101-1234567",
        "9001011234567",
        "900101 1234567",
        "900101–1234567",  # en dash
        "900101—1234567",  # em dash
        "900101－1234567",  # 전각 하이픈
        "900101.1234567",
        "900101 - 1234567",
        "900101-\n1234567",
        "900101​-1234567",  # 제로폭 공백 삽입
        "９００１０１-１２３４５６７",  # 전각 숫자
        "９００１０１１２３４５６７",  # 전각 숫자 + 구분자 없음
        "9001\n011234567",  # 숫자 중간 줄바꿈(OCR 줄바꿈)
        "900101-123 4567",  # 숫자 중간 공백(STT 띄어쓰기)
        "900101 1234 567",  # 숫자 중간 공백 2회
        "900101 - - 1234567",  # 구분자 3자 초과
        "9OO1O1-1234567",  # OCR O/0 혼동
        "90010l-1234567",  # OCR l/1 혼동
        "9001O1-l234567",  # OCR 혼동 혼합
    )

    @pytest.mark.parametrize("payload", EVASIONS)
    def test_every_evasion_is_detected(self, payload: str) -> None:
        """구분자·전각·줄바꿈·제로폭·숫자중간분할·OCR 동형이의 변형이 모두 탐지된다."""
        assert "rrn" in types_of(payload), f"탐지 실패: {payload!r}"

    @pytest.mark.parametrize("payload", EVASIONS)
    def test_evasion_detected_without_label(self, payload: str) -> None:
        """라벨 없이 문장 속에 섞여 있어도 탐지된다."""
        sentence = f"아래 칸에 {payload} 라고 적혀 있습니다."
        assert "rrn" in types_of(sentence)

    @pytest.mark.parametrize("payload", EVASIONS)
    def test_span_covers_the_whole_original_number(self, payload: str) -> None:
        """탐지 구간이 원문 번호 전체(구분자 포함)를 덮는다."""
        spans = [s for s in detect_pii(payload) if s.pii_type == "rrn"]
        assert len(spans) == 1
        span = spans[0]
        assert payload[span.start : span.end] == payload

    def test_unlabeled_account_number_is_detected(self) -> None:
        """라벨(계좌·은행명) 없이 적힌 계좌번호도 탐지한다."""
        assert "account" in types_of("110-123-456789 로 보내주세요")

    def test_unlabeled_account_keeps_business_number_untouched(self) -> None:
        """10자리 사업자등록번호 형식은 라벨 없는 계좌 규칙이 가로채지 않는다."""
        assert "account" not in types_of("등록 123-45-67890 입니다")

    @pytest.mark.parametrize(
        "sentence",
        ["저는 김철수라고 합니다", "저는 김철수인데 여기 적나요?", "제 이름은 김철수입니다"],
    )
    def test_self_introduction_name_is_detected(self, sentence: str) -> None:
        """라벨 없이 자기소개로 말한 성명도 탐지한다."""
        assert "name" in types_of(sentence)

    def test_foreign_registration_number_is_separate_type(self) -> None:
        """성별코드 5~8은 외국인등록번호로 구분한다."""
        assert types_of("외국인등록번호 900101-5234567") == {"frn"}

    def test_invalid_date_is_not_detected(self) -> None:
        """생년월일이 불가능하면 13자리여도 주민등록번호로 보지 않는다."""
        assert "rrn" not in types_of("일련번호 9099991234567 입니다")


class TestOtherPiiTypes:
    """나머지 유형별 탐지."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("연락처 010-1234-5678", "phone_mobile"),
            ("연락처 01012345678", "phone_mobile"),
            ("사무실 02-123-4567", "phone_landline"),
            ("사무실 031-123-4567", "phone_landline"),
            ("메일 hong.gildong@example.co.kr 로 보내주세요", "email"),
            ("카드 4111-1111-1111-1111", "card"),
            ("카드 4111111111111111", "card"),
            ("계좌번호 110-123-456789", "account"),
            ("국민은행 123456-78-901234", "account"),
            ("여권번호 M12345678", "passport"),
            ("운전면허 11-22-123456-01", "driver_license"),
            ("사업자등록번호 123-45-67891", "business_number"),
            ("건강보험증번호 1234567890", "health_insurance"),
            ("주소 경기도 성남시 분당구 판교로 235", "address"),
            ("주소 서울 강남구 역삼동 123-45", "address"),
            ("생년월일 1990-01-01", "birthdate"),
            ("생년월일 900101", "birthdate"),
            ("성명 김철수", "name"),
            ("신청인 이영희 (서명)", "name"),
        ],
    )
    def test_type_is_detected(self, text: str, expected: str) -> None:
        """대표 예시가 기대한 유형으로 탐지된다."""
        assert expected in types_of(text), f"{expected} 탐지 실패: {text!r}"

    def test_card_requires_luhn(self) -> None:
        """Luhn 을 통과하지 못하는 16자리는 카드로 보지 않는다."""
        assert "card" not in types_of("일련 1234-5678-9012-3456")

    def test_business_number_requires_checksum(self) -> None:
        """체크섬이 틀린 10자리는 사업자등록번호로 보지 않는다."""
        assert "business_number" not in types_of("코드 123-45-67890")


class TestFalsePositiveSuppression:
    """무해한 숫자열을 차단하지 않는다(거짓양성 억제)."""

    def test_benign_document_has_no_pii(self) -> None:
        """접수번호·금액·날짜·조문만 있는 문장은 탐지 결과가 없다."""
        assert detect_pii(BENIGN_DOC) == ()
        assert not contains_pii(BENIGN_DOC)

    @pytest.mark.parametrize(
        "text",
        [
            "접수번호 제2026-0001호",
            "문서번호 제2026-123456호",
            "지급 금액은 1,234,567원입니다.",
            "지급 금액 12345678901234원",
            "처리 기한은 2026년 9월 30일입니다.",
            "기준일 2026-09-30",
            "산업안전보건법 제38조 제1항",
            "총 12쪽, 3건, 5명",
            "보유 기간은 5년입니다.",
        ],
    )
    def test_benign_fragment_is_not_flagged(self, text: str) -> None:
        """무해한 숫자 표현 각각이 개별적으로도 탐지되지 않는다."""
        assert detect_pii(text) == (), f"거짓양성: {text!r} → {raw_of(text)}"

    def test_label_words_alone_are_not_names(self) -> None:
        """약관 본문의 '성명, 주민등록번호' 나열은 성명으로 보지 않는다."""
        clause = "수집 항목은 성명, 주민등록번호, 주소, 연락처이며 목적은 자격 확인입니다."
        assert detect_pii(clause) == ()

    def test_signature_label_is_not_a_name(self) -> None:
        """'신청인 서명' 의 '서명' 은 불용어라 성명으로 채택하지 않는다."""
        assert detect_pii("신청인 서명") == ()
        assert detect_pii("대리인 (서명 또는 인)") == ()


class TestSpanContract:
    """:class:`PiiSpan` 계약 준수."""

    def test_spans_never_carry_raw_values(self) -> None:
        """탐지 결과 dict 에 원문 값을 담는 키가 없다."""
        for span in detect_pii(SAMPLE_DOC):
            payload = span.to_dict()
            assert set(payload) == {
                "start",
                "end",
                "pii_type",
                "raw_len",
                "confidence",
            }
            for value in payload.values():
                assert "900101" not in str(value)
                assert "김철수" not in str(value)

    def test_spans_are_pii_span_instances(self) -> None:
        """반환 타입이 계약 타입이다."""
        assert all(isinstance(s, PiiSpan) for s in detect_pii(SAMPLE_DOC))

    def test_raw_len_matches_span_width(self) -> None:
        """``raw_len`` 이 구간 길이와 일치한다."""
        for span in detect_pii(SAMPLE_DOC):
            assert span.raw_len == span.end - span.start

    def test_spans_never_overlap_and_are_sorted(self) -> None:
        """구간은 겹치지 않고 시작 위치 오름차순이다."""
        spans = detect_pii(SAMPLE_DOC)
        previous_end = 0
        for span in spans:
            assert span.start >= previous_end
            previous_end = span.end

    def test_sample_document_detects_four_types(self) -> None:
        """로드맵 예시 문서에서 성명·주민등록번호·주소·전화가 모두 잡힌다."""
        assert types_of(SAMPLE_DOC) == {"name", "rrn", "address", "phone_mobile"}

    def test_summarize_types_is_sorted_and_unique(self) -> None:
        """유형 요약은 중복 없이 정렬된다."""
        assert summarize_types(detect_pii(SAMPLE_DOC)) == (
            "address",
            "name",
            "phone_mobile",
            "rrn",
        )


class TestDetectorObject:
    """:class:`RegexPiiDetector` 객체 인터페이스."""

    def test_detector_matches_module_function(self) -> None:
        """클래스와 모듈 함수의 결과가 같다."""
        assert RegexPiiDetector().detect(SAMPLE_DOC) == detect_pii(SAMPLE_DOC)

    def test_repr_does_not_leak_content(self) -> None:
        """repr 이 규칙 개수만 노출한다."""
        assert "규칙" in repr(RegexPiiDetector())

    def test_empty_text_returns_empty(self) -> None:
        """빈 문자열은 빈 결과."""
        assert detect_pii("") == ()

    def test_non_string_raises_domain_error(self) -> None:
        """잘못된 입력은 조용히 넘어가지 않고 도메인 예외."""
        with pytest.raises(PiiDetectionError):
            detect_pii(None)  # type: ignore[arg-type]

    def test_detection_is_deterministic(self) -> None:
        """같은 입력은 항상 같은 결과를 낸다."""
        first = detect_pii(SAMPLE_DOC)
        second = detect_pii(SAMPLE_DOC)
        assert first == second


class TestSyntheticFixtureIsClean:
    """합성 문서의 공개 문구에는 개인정보가 없어야 한다."""

    def test_consent_clause_has_no_pii(self, sample_structure) -> None:  # type: ignore[no-untyped-def]
        """동의 약관 본문에서 개인정보가 탐지되지 않는다."""
        consent = sample_structure.field_by_id("consent_01")
        assert consent is not None
        assert detect_pii(consent.clause_text) == ()

    def test_document_title_has_no_pii(self, sample_structure) -> None:  # type: ignore[no-untyped-def]
        """문서 제목에서 개인정보가 탐지되지 않는다."""
        assert detect_pii(sample_structure.doc_title) == ()


class TestJosaEvasion:
    """라벨 뒤에 한국어 조사가 붙어도 라벨 앵커 규칙이 무너지지 않는다.

    조사 한 글자로 문맥 앵커 규칙 전체가 무력화되어 원문이 게이트를 통과하던
    회귀를 막는다(적대적 감사 지적 사항).
    """

    #: 검사할 조사 목록(6종).
    JOSA = ("은", "는", "이", "가", "을", "를")

    @pytest.mark.parametrize("josa", JOSA)
    @pytest.mark.parametrize(
        ("label", "value", "pii_type"),
        [
            ("건강보험증번호", "12345678901", "health_insurance"),
            ("생년월일", "1990-01-01", "birthdate"),
            ("운전면허번호", "11-22-334455-66", "driver_license"),
            ("성명", "김철수", "name"),
            ("계좌번호", "352-0123-4567-89", "account"),
        ],
    )
    def test_label_with_josa_still_detected(
        self, label: str, value: str, pii_type: str, josa: str
    ) -> None:
        """라벨 + 조사 + 값 형태에서도 값이 탐지된다."""
        text = f"{label}{josa} {value} 입니다"
        assert pii_type in types_of(text)

    def test_spaced_variant_with_josa_detected(self) -> None:
        """구분자를 공백으로 바꾼 회피 변형도 조사와 함께 잡힌다."""
        assert "driver_license" in types_of("운전면허번호는 11 22 334455 66")

    def test_natural_date_after_josa_detected(self) -> None:
        """'생년월일은 1990년 1월 1일' 같은 자연어 표현도 잡힌다."""
        assert "birthdate" in types_of("생년월일은 1990년 1월 1일")


class TestNameLabelIsNotMistakenForValue:
    """라벨을 값으로 오인해 마스킹하고 실제 성명을 흘리는 일이 없어야 한다."""

    @pytest.mark.parametrize(
        "text",
        [
            "신청인 성명은 김철수",
            "성명 : 이름 김철수",
            "예금주 성명 박영희",
            "성명은 김철수",
            "이름 남궁철수",
        ],
    )
    def test_real_name_is_the_detected_span(self, text: str) -> None:
        """탐지 구간이 라벨이 아니라 실제 성명을 가리킨다."""
        detected = raw_of(text)
        assert detected, f"성명을 전혀 탐지하지 못했습니다: {text!r}"
        for fragment in detected:
            assert "성명" not in fragment
            assert "이름" not in fragment
            assert "신청인" not in fragment
            assert "예금주" not in fragment

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            ("신청인 성명은 김철수", "김철수"),
            ("성명 : 이름 김철수", "김철수"),
            ("예금주 성명 박영희", "박영희"),
            ("성명은 김철수입니다", "김철수"),
        ],
    )
    def test_masked_text_does_not_contain_name(self, text: str, name: str) -> None:
        """마스킹 결과에 원문 성명이 부분문자열로 남지 않는다."""
        from docagent.pii.masker import MaskStrategy, mask

        masked = mask(text, detect_pii(text), MaskStrategy.FULL)
        assert name not in masked.text

    def test_stopword_label_alone_is_not_detected(self) -> None:
        """값 없이 라벨만 있는 문장에서는 성명을 만들어 내지 않는다."""
        assert "name" not in types_of("성명 : ")
        assert "name" not in types_of("성명 없음")
