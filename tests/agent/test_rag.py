"""RAG 검색기 테스트 — 토큰화·청킹·TF-IDF·캐시·선택적 어댑터."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docagent.agent.rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    CorpusError,
    FaissRetriever,
    LocalTfidfRetriever,
    build_index,
    char_ngrams,
    chunk_documents,
    default_corpus_dir,
    strip_particle,
    tokenize,
    word_tokens,
)
from docagent.contracts import RetrievedChunk
from docagent.errors import AdapterUnavailable
from docagent.interfaces import Retriever


@pytest.fixture(scope="module")
def corpus_dir() -> Path:
    """저장소 기본 코퍼스 디렉터리."""
    directory = default_corpus_dir()
    assert directory.is_dir(), f"코퍼스 디렉터리가 없습니다: {directory}"
    return directory


@pytest.fixture(scope="module")
def retriever(corpus_dir: Path) -> LocalTfidfRetriever:
    """기본 코퍼스로 만든 검색기(모듈 단위 캐시)."""
    return LocalTfidfRetriever.from_corpus(corpus_dir)


class TestTokenizer:
    """한국어 토큰화 규칙."""

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            ("동의를", "동의"),
            ("동의가", "동의"),
            ("개인정보는", "개인정보"),
            ("주민번호를", "주민번호"),
            ("서식에서", "서식"),
            ("기관으로", "기관"),
        ],
    )
    def test_strip_particle_removes_josa(self, word: str, expected: str) -> None:
        """어절 끝 조사가 제거되어 같은 어간으로 모인다."""
        assert strip_particle(word) == expected

    @pytest.mark.parametrize("word", ["병", "의", "가", "동의", "Consent", "2026"])
    def test_strip_particle_keeps_short_or_nonhangul(self, word: str) -> None:
        """1음절 명사·짧은 어간·비한글은 그대로 둔다(과잉 절단 방지)."""
        assert strip_particle(word) == word

    def test_tokenize_mixes_words_and_ngrams(self) -> None:
        """어절 토큰과 문자 n-gram(2,3)이 함께 나온다."""
        tokens = tokenize("개인정보 동의를")
        assert "개인정보" in tokens  # 어절
        assert "동의" in tokens  # 조사 제거된 어절
        assert "개인" in tokens  # 2-gram
        assert "개인정" in tokens  # 3-gram

    def test_tokenize_is_deterministic(self) -> None:
        """같은 입력은 항상 같은 토큰 목록을 낸다."""
        text = "주민등록번호는 고유식별정보입니다."
        assert tokenize(text) == tokenize(text)

    def test_tokenize_empty(self) -> None:
        """빈 문자열은 빈 리스트."""
        assert tokenize("") == []

    def test_word_tokens_excludes_ngrams(self) -> None:
        """``word_tokens`` 는 n-gram 을 섞지 않는다."""
        tokens = word_tokens("개인정보 동의를")
        assert tokens == ["개인정보", "동의"]

    def test_char_ngrams(self) -> None:
        """문자 n-gram 생성과 짧은 단어 처리."""
        assert char_ngrams("주민번호", 2) == ["주민", "민번", "번호"]
        assert char_ngrams("가", 2) == ["가"]
        with pytest.raises(ValueError, match="1 이상"):
            char_ngrams("가나", 0)


class TestChunking:
    """코퍼스 청킹."""

    def test_chunk_documents_fills_source(self, corpus_dir: Path) -> None:
        """모든 청크에 파일명과 조항 표시가 담긴 출처가 채워진다."""
        chunks = chunk_documents(sorted(corpus_dir.glob("*.md")))
        assert chunks
        for chunk in chunks:
            assert chunk.source.strip(), f"출처 없는 청크: {chunk.chunk_id}"
            assert chunk.source.endswith(".md") or ".md ·" in chunk.source
            assert chunk.text.strip()

    def test_chunk_size_respected(self, corpus_dir: Path) -> None:
        """청크 길이가 상한을 넘지 않는다."""
        chunks = chunk_documents(sorted(corpus_dir.glob("*.md")))
        assert all(len(chunk.text) <= DEFAULT_CHUNK_SIZE for chunk in chunks)

    def test_chunk_ids_unique(self, corpus_dir: Path) -> None:
        """청크 id 는 문서 안에서 중복되지 않는다."""
        chunks = chunk_documents(sorted(corpus_dir.glob("*.md")))
        ids = [chunk.chunk_id for chunk in chunks]
        assert len(ids) == len(set(ids))

    def test_disclaimer_lines_excluded(self, corpus_dir: Path) -> None:
        """모든 문서에 반복되는 고지 문구(``>`` 줄)는 본문에서 빠진다."""
        chunks = chunk_documents(sorted(corpus_dir.glob("*.md")))
        joined = " ".join(chunk.text for chunk in chunks)
        assert "법적 자문이 아닙니다" not in joined

    def test_chunk_documents_is_deterministic(self, corpus_dir: Path) -> None:
        """입력 순서를 바꿔도 같은 결과가 나온다(파일명 기준 재정렬)."""
        paths = sorted(corpus_dir.glob("*.md"))
        first = chunk_documents(paths)
        second = chunk_documents(list(reversed(paths)))
        assert [c.to_dict() for c in first] == [c.to_dict() for c in second]

    @pytest.mark.parametrize(
        ("chunk_size", "overlap", "message"),
        [(0, 50, "chunk_size"), (500, -1, "overlap"), (100, 100, "overlap")],
    )
    def test_chunk_documents_rejects_bad_params(
        self, corpus_dir: Path, chunk_size: int, overlap: int, message: str
    ) -> None:
        """잘못된 파라미터는 한국어 ValueError 로 거부한다."""
        with pytest.raises(ValueError, match=message):
            chunk_documents(sorted(corpus_dir.glob("*.md")), chunk_size, overlap)

    def test_long_paragraph_is_split(self, tmp_path: Path) -> None:
        """문단 하나가 상한을 넘으면 문장 경계로 다시 쪼갠다."""
        sentence = "이것은 아주 긴 설명 문장이며 반복해서 이어집니다. "
        path = tmp_path / "long.md"
        path.write_text(
            "# 긴 문서\n\n## 절\n\n" + sentence * 40 + "\n", encoding="utf-8"
        )
        chunks = chunk_documents([path], chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(chunk.text) <= 200 for chunk in chunks)

    def test_missing_title_raises(self, tmp_path: Path) -> None:
        """H1 제목이 없으면 CorpusError."""
        path = tmp_path / "no_title.md"
        path.write_text("본문만 있습니다.\n", encoding="utf-8")
        with pytest.raises(CorpusError, match="H1"):
            chunk_documents([path])


class TestLocalTfidfRetriever:
    """TF-IDF 검색 동작."""

    def test_satisfies_retriever_protocol(self, retriever: LocalTfidfRetriever) -> None:
        """:class:`docagent.interfaces.Retriever` 프로토콜을 만족한다."""
        assert isinstance(retriever, Retriever)

    def test_search_returns_relevant_chunk(self, retriever: LocalTfidfRetriever) -> None:
        """주민등록번호 질의는 고유식별정보 문서를 1순위로 찾는다."""
        results = retriever.search("주민등록번호를 왜 적어야 하나요", k=3)
        assert results
        assert "unique_identifier" in results[0].chunk_id

    def test_search_scores_sorted_and_bounded(
        self, retriever: LocalTfidfRetriever
    ) -> None:
        """점수는 내림차순이고 0.0~1.0 범위(코사인)다."""
        results = retriever.search("개인정보 수집 이용 동의", k=5)
        scores = [chunk.score for chunk in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 < score <= 1.0 for score in scores)

    def test_search_respects_k(self, retriever: LocalTfidfRetriever) -> None:
        """반환 개수는 k 이하다."""
        assert len(retriever.search("동의", k=2)) <= 2

    def test_search_unknown_query_returns_empty(
        self, retriever: LocalTfidfRetriever
    ) -> None:
        """코퍼스와 겹치는 어휘가 없으면 빈 리스트를 돌려준다(억지 근거 금지)."""
        assert retriever.search("zzzzq wwwwx", k=3) == []

    def test_search_is_deterministic(self, retriever: LocalTfidfRetriever) -> None:
        """같은 질의는 항상 같은 결과."""
        first = retriever.search("서명은 무슨 뜻인가요", k=4)
        second = retriever.search("서명은 무슨 뜻인가요", k=4)
        assert [c.to_dict() for c in first] == [c.to_dict() for c in second]

    def test_search_rejects_bad_k(self, retriever: LocalTfidfRetriever) -> None:
        """k 가 1 미만이면 ValueError."""
        with pytest.raises(ValueError, match="k 는 1 이상"):
            retriever.search("동의", k=0)

    def test_empty_index_rejected(self) -> None:
        """청크가 없으면 색인을 만들 수 없다."""
        with pytest.raises(ValueError, match="근거 청크가 없습니다"):
            LocalTfidfRetriever([])

    def test_chunk_without_source_rejected(self) -> None:
        """출처 없는 근거는 색인 단계에서 거부한다."""
        with pytest.raises(ValueError, match="출처"):
            LocalTfidfRetriever(
                [RetrievedChunk(chunk_id="c1", text="본문", source="  ")]
            )

    def test_from_corpus_missing_dir(self, tmp_path: Path) -> None:
        """빈 디렉터리는 CorpusError."""
        with pytest.raises(CorpusError, match=".md"):
            LocalTfidfRetriever.from_corpus(tmp_path)


class TestIndexCache:
    """디스크 색인 캐시."""

    def test_cache_roundtrip(self, tmp_path: Path, corpus_dir: Path) -> None:
        """캐시를 쓰고 다시 읽어도 같은 검색 결과가 나온다."""
        cache = tmp_path / "index.json"
        first = build_index(corpus_dir, cache_path=cache)
        assert cache.is_file()
        second = build_index(corpus_dir, cache_path=cache)
        query = "개인정보 보유 기간"
        assert [c.to_dict() for c in first.search(query)] == [
            c.to_dict() for c in second.search(query)
        ]

    def test_cache_is_byte_identical(self, tmp_path: Path, corpus_dir: Path) -> None:
        """같은 코퍼스로 두 번 만들면 캐시 파일이 바이트 단위로 같다."""
        one = tmp_path / "a.json"
        two = tmp_path / "b.json"
        build_index(corpus_dir, cache_path=one)
        build_index(corpus_dir, cache_path=two)
        assert one.read_bytes() == two.read_bytes()

    def test_cache_is_utf8_readable(self, tmp_path: Path, corpus_dir: Path) -> None:
        """캐시는 한글이 이스케이프되지 않은 UTF-8 JSON 이다."""
        cache = tmp_path / "index.json"
        build_index(corpus_dir, cache_path=cache)
        payload = json.loads(cache.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert any("동의" in item["text"] for item in payload["chunks"])

    def test_broken_cache_raises(self, tmp_path: Path, corpus_dir: Path) -> None:
        """깨진 캐시는 조용히 무시하지 않고 CorpusError 로 올린다."""
        cache = tmp_path / "index.json"
        cache.write_text("{not json", encoding="utf-8")
        with pytest.raises(CorpusError, match="캐시"):
            build_index(corpus_dir, cache_path=cache)

    def test_refresh_overwrites(self, tmp_path: Path, corpus_dir: Path) -> None:
        """refresh=True 면 기존 캐시를 무시하고 다시 만든다."""
        cache = tmp_path / "index.json"
        cache.write_text('{"version": 1, "chunks": []}', encoding="utf-8")
        retriever = build_index(corpus_dir, cache_path=cache, refresh=True)
        assert len(retriever) > 0

    def test_default_chunk_constants(self) -> None:
        """로드맵 수치(500/50)를 기본값으로 쓴다."""
        assert DEFAULT_CHUNK_SIZE == 500
        assert DEFAULT_CHUNK_OVERLAP == 50


class TestFaissAdapter:
    """선택적 FAISS 어댑터."""

    def test_faiss_unavailable(self) -> None:
        """faiss 미설치 환경에서는 한국어 설치 안내와 함께 AdapterUnavailable."""
        with pytest.raises(AdapterUnavailable) as info:
            FaissRetriever([RetrievedChunk(chunk_id="c1", text="본문", source="s")])
        message = str(info.value)
        assert "설치" in message
        assert ".venv" in message
