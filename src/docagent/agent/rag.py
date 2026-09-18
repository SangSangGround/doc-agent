"""근거 문서 검색(RAG) — 외부 벡터DB 없이 동작하는 순수 검색기.

설계 원칙
---------
* **오프라인 결정론**: numpy 만으로 TF-IDF 와 코사인 유사도를 계산한다.
  네트워크 호출도, 임베딩 API 도 쓰지 않는다. 같은 코퍼스·같은 질의는 항상 같은 결과다.
* **출처 없는 근거는 사용하지 않는다**: 모든 :class:`~docagent.contracts.RetrievedChunk`
  는 파일명과 조항 표시가 담긴 ``source`` 를 가진다. ``source`` 가 빈 조각은
  색인 단계에서 :class:`ValueError` 로 거부한다.
* **한국어 토큰화는 형태소 분석기 없이** 한다. 어절 분리 + 조사 제거 + 문자 n-gram(2,3)
  혼합 방식이며 근거는 :func:`tokenize` 독스트링에 적었다.

코퍼스 문서 포맷
----------------
``data/corpus/*.md`` 는 다음 규칙을 따른다.

* ``# 제목`` — 문서 제목(H1). 한 파일에 하나.
* ``> ...`` — 고지·출처 메타데이터 줄. **본문에서 제외**되어 검색 대상이 되지 않는다.
  (모든 파일에 반복되는 고지 문구가 검색 결과를 오염시키는 것을 막는다.)
* ``## 소제목`` — 절 제목. ``source`` 표기에 쓰인다.
* 그 외 줄 — 본문 문단. ``[근거: ...]`` 표기가 있으면 ``source`` 뒤에 덧붙는다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from docagent.contracts import RetrievedChunk
from docagent.errors import AdapterUnavailable, DocAgentError

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CORPUS_DIR",
    "KOREAN_PARTICLES",
    "CorpusError",
    "tokenize",
    "strip_particle",
    "chunk_documents",
    "word_tokens",
    "char_ngrams",
    "LocalTfidfRetriever",
    "FaissRetriever",
    "build_index",
    "default_corpus_dir",
]


#: 로드맵이 정한 청크 길이(문자 수).
DEFAULT_CHUNK_SIZE: int = 500
#: 로드맵이 정한 청크 간 겹침 길이(문자 수).
DEFAULT_CHUNK_OVERLAP: int = 50

#: 저장소 기본 코퍼스 디렉터리(``<repo>/data/corpus``).
#:
#: 이 파일 위치가 ``<repo>/src/docagent/agent/rag.py`` 이므로 상위 4단계가 저장소 루트다.
DEFAULT_CORPUS_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "corpus"


class CorpusError(DocAgentError):
    """코퍼스 파일을 읽거나 해석하지 못했을 때 발생한다."""


# --------------------------------------------------------------------------
# 한국어 토큰화
# --------------------------------------------------------------------------

#: 어절 끝에서 제거할 조사·어미 목록(긴 것부터 검사한다).
#:
#: 한국어는 교착어라서 같은 체언이 '동의가 / 동의를 / 동의는 / 동의에' 처럼
#: 조사만 바꿔 가며 나타난다. 형태소 분석기를 쓸 수 없는 환경에서는 어절 끝의
#: 고빈도 조사·어미를 잘라 내는 규칙만으로도 어휘 정규화 효과가 크다.
#: (형태소 분석 없이 색인하는 한국어 검색에서 널리 쓰이는 관행이다.)
#: 다만 '병'(病)처럼 조사와 형태가 겹치는 1음절 명사를 파괴하지 않도록,
#: 자르고 남는 길이가 2자 미만이면 자르지 않는다(:func:`strip_particle` 참조).
KOREAN_PARTICLES: tuple[str, ...] = (
    # 3음절 이상
    "으로써",
    "으로서",
    "이라도",
    "에서는",
    "에게서",
    "이라고",
    # 2음절
    "에서",
    "에게",
    "한테",
    "께서",
    "으로",
    "라도",
    "이나",
    "보다",
    "처럼",
    "같이",
    "까지",
    "부터",
    "조차",
    "마저",
    "밖에",
    "이며",
    "이란",
    "이든",
    "만큼",
    "라는",
    # 1음절
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "와",
    "과",
    "도",
    "만",
    "로",
    "며",
    "나",
    "께",
)

#: 조사 제거 뒤에도 남겨야 하는 최소 한글 길이.
_MIN_STEM_LEN: int = 2

#: 문자 n-gram 크기.
_NGRAM_SIZES: tuple[int, ...] = (2, 3)

#: 어절 추출 정규식(한글·영문·숫자 덩어리).
_WORD_RE = re.compile(r"[가-힣]+|[A-Za-z]+|[0-9]+")

#: 한글 음절만 남기는 정규식(문자 n-gram 생성용).
_HANGUL_RE = re.compile(r"[가-힣]+")


def strip_particle(word: str) -> str:
    """어절 끝의 조사·어미를 한 번 제거한다.

    :param word: 어절 하나(예: ``"동의가"``).
    :returns: 조사를 제거한 어간(예: ``"동의"``).
        한글이 아니거나 제거 후 길이가 2자 미만이 되면 원본을 그대로 돌려준다.

    사용 예::

        >>> strip_particle("동의를")
        '동의'
        >>> strip_particle("병")     # 1음절 명사는 보존한다
        '병'
    """
    if not _HANGUL_RE.fullmatch(word):
        return word
    for particle in KOREAN_PARTICLES:
        if word.endswith(particle) and len(word) - len(particle) >= _MIN_STEM_LEN:
            return word[: -len(particle)]
    return word


def tokenize(text: str) -> list[str]:
    """한국어 텍스트를 검색용 토큰 목록으로 바꾼다.

    형태소 분석기(konlpy·mecab 등)를 쓸 수 없으므로 세 단계를 조합한다.

    1. **어절 분리** — 한글/영문/숫자 덩어리를 뽑는다. 문장부호는 버린다.
    2. **조사 제거** — :func:`strip_particle` 로 어절 끝 조사·어미를 잘라
       ``동의가``/``동의를``/``동의는`` 을 같은 토큰 ``동의`` 로 모은다.
    3. **문자 n-gram(2,3)** — 한글 구간에서 2·3음절 n-gram 을 추가한다.
       복합명사(``개인정보수집``)와 미등록어(``주민번호`` vs ``주민등록번호``)를
       부분 일치로 이어 주는 안전망이다. 어절 토큰만 쓰면 이런 짝을 놓치고,
       n-gram 만 쓰면 정확 일치의 변별력이 떨어지므로 **둘을 함께 쓴다.**

    :param text: 원문 텍스트(한국어 가정, 다른 문자도 허용).
    :returns: 토큰 목록. 중복은 제거하지 않는다(빈도가 TF 에 반영되어야 한다).
        입력이 비면 빈 리스트.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text).lower()
    tokens: list[str] = []
    for word in _WORD_RE.findall(normalized):
        tokens.append(strip_particle(word))
    for run in _HANGUL_RE.findall(normalized):
        for size in _NGRAM_SIZES:
            if len(run) < size:
                continue
            for index in range(len(run) - size + 1):
                tokens.append(run[index : index + size])
    return tokens


def word_tokens(text: str) -> list[str]:
    """어절 단위 토큰만 뽑는다(문자 n-gram 제외).

    :func:`tokenize` 는 검색 재현율을 위해 n-gram 을 섞지만, 어휘 겹침률·근거 커버리지
    계산에는 사람이 읽을 수 있는 어절 단위가 필요하다. 이 함수는 그 용도로 쓴다.

    :param text: 원문 텍스트.
    :returns: 조사를 제거한 어절 토큰 목록(중복 유지). 입력이 비면 빈 리스트.
    """
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text).lower()
    return [strip_particle(word) for word in _WORD_RE.findall(normalized)]


def char_ngrams(word: str, size: int = 2) -> list[str]:
    """단어의 문자 n-gram 목록을 만든다.

    :param word: 대상 단어.
    :param size: n-gram 크기(1 이상).
    :returns: n-gram 목록. 단어가 ``size`` 보다 짧으면 단어 자체를 담은 리스트.
    :raises ValueError: ``size`` 가 1 미만인 경우.
    """
    if size < 1:
        raise ValueError(f"size 는 1 이상이어야 합니다: {size}")
    if len(word) < size:
        return [word] if word else []
    return [word[i : i + size] for i in range(len(word) - size + 1)]


# --------------------------------------------------------------------------
# 코퍼스 로딩·청킹
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Section:
    """코퍼스 문서의 한 절(``##`` 단위) — 청킹 이전의 중간 표현."""

    heading: str
    body: str


def _read_document(path: Path) -> tuple[str, tuple[_Section, ...]]:
    """코퍼스 마크다운 1개를 ``(문서 제목, 절 목록)`` 으로 파싱한다.

    :param path: ``.md`` 파일 경로.
    :returns: ``(H1 제목, (_Section, ...))``.
    :raises CorpusError: 파일을 읽지 못했거나 H1 제목이 없는 경우.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"코퍼스 파일을 읽지 못했습니다: {path}") from exc

    doc_title = ""
    heading = ""
    buffer: list[str] = []
    sections: list[_Section] = []

    def flush() -> None:
        body = " ".join(part.strip() for part in buffer if part.strip()).strip()
        buffer.clear()
        if body:
            sections.append(_Section(heading=heading, body=body))

    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith(">"):
            # 고지·출처 메타데이터 줄. 모든 문서에 반복되므로 본문에서 제외한다.
            continue
        if stripped.startswith("##"):
            flush()
            heading = stripped.lstrip("#").strip()
            continue
        if stripped.startswith("#"):
            flush()
            doc_title = stripped.lstrip("#").strip()
            heading = ""
            continue
        buffer.append(stripped)
    flush()

    if not doc_title:
        raise CorpusError(f"코퍼스 파일에 '# 제목'(H1) 이 없습니다: {path}")
    if not sections:
        raise CorpusError(f"코퍼스 파일에 본문이 없습니다: {path}")
    return doc_title, tuple(sections)


#: ``[근거: ...]`` 표기 추출 정규식.
_BASIS_RE = re.compile(r"\[근거:\s*([^\]]+)\]")


def _split_body(body: str, chunk_size: int, overlap: int) -> list[str]:
    """본문 문자열을 문단 경계 우선으로 잘라 청크 문자열 목록을 만든다.

    문단(= 절 안의 문장 묶음)을 먼저 붙여 나가다가 ``chunk_size`` 를 넘으면 끊는다.
    한 문단이 통째로 ``chunk_size`` 를 넘으면 문장 경계(``. ``/``다. ``)로 다시 쪼개고,
    그래도 넘치면 마지막 수단으로 ``chunk_size`` 만큼 강제 절단한다.
    앞 청크의 꼬리 ``overlap`` 글자를 다음 청크 앞에 붙여 문맥 단절을 줄인다.

    :param body: 절 본문.
    :param chunk_size: 청크 최대 길이(문자 수).
    :param overlap: 청크 간 겹침 길이(문자 수).
    :returns: 청크 문자열 목록(빈 문자열 없음).
    """
    if len(body) <= chunk_size:
        return [body]

    # 문장 경계로 1차 분해한다(한국어 종결어미 '다.' 와 일반 마침표를 함께 본다).
    pieces = [piece.strip() for piece in re.split(r"(?<=다\.)\s+|(?<=\.)\s+", body)]
    pieces = [piece for piece in pieces if piece]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail} {piece}".strip() if tail else piece
        else:
            current = piece
        while len(current) > chunk_size:
            chunks.append(current[:chunk_size])
            current = current[max(0, chunk_size - overlap) :]
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def chunk_documents(
    paths: Iterable[Path | str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RetrievedChunk]:
    """코퍼스 마크다운 파일들을 검색 단위 청크로 나눈다.

    :param paths: ``.md`` 파일 경로들. 정렬은 내부에서 파일명 기준으로 다시 한다
        (색인 결정론을 보장하기 위해서다).
    :param chunk_size: 청크 최대 길이(문자 수). 1 이상.
    :param overlap: 청크 간 겹침 길이(문자 수). 0 이상이며 ``chunk_size`` 미만.
    :returns: :class:`~docagent.contracts.RetrievedChunk` 목록.
        ``chunk_id`` 는 ``"<파일이름(확장자 제외)>#<일련번호>"`` 형식이고
        ``score`` 는 0.0(검색 전)이다. ``source`` 는 항상 채워진다.
    :raises ValueError: ``chunk_size`` 나 ``overlap`` 이 규칙을 어긴 경우.
    :raises CorpusError: 파일을 읽지 못했거나 형식이 어긋난 경우.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size 는 1 이상이어야 합니다: {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap 은 0 이상이어야 합니다: {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap 은 chunk_size 보다 작아야 합니다: overlap={overlap}, chunk_size={chunk_size}"
        )

    ordered = sorted((Path(item) for item in paths), key=lambda p: p.name)
    chunks: list[RetrievedChunk] = []
    for path in ordered:
        doc_title, sections = _read_document(path)
        counter = 0
        for section in sections:
            for text in _split_body(section.body, chunk_size, overlap):
                source = f"{path.name} · {doc_title}"
                if section.heading:
                    source = f"{source} > {section.heading}"
                basis = _BASIS_RE.search(text)
                if basis:
                    source = f"{source} [근거: {basis.group(1).strip()}]"
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{path.stem}#{counter:02d}",
                        text=text,
                        source=source,
                        score=0.0,
                    )
                )
                counter += 1
    return chunks


def default_corpus_dir() -> Path:
    """저장소 기본 코퍼스 디렉터리를 돌려준다.

    :returns: ``<repo>/data/corpus`` 경로(:data:`DEFAULT_CORPUS_DIR`).
    """
    return DEFAULT_CORPUS_DIR


# --------------------------------------------------------------------------
# TF-IDF 검색기
# --------------------------------------------------------------------------


class LocalTfidfRetriever:
    """numpy TF-IDF + 코사인 유사도 기반 로컬 검색기.

    :class:`docagent.interfaces.Retriever` 프로토콜을 만족한다.

    가중치는 다음과 같이 계산한다.

    * ``tf = 1 + log(빈도)`` — 긴 청크가 빈도만으로 유리해지지 않게 로그로 눌러 준다.
    * ``idf = log((N + 1) / (df + 1)) + 1`` — 모든 청크에 나오는 흔한 토큰(예: ``정보``)의
      가중치를 낮춘다. 분모·분자에 1 을 더하는 평활화로 df=0 에서도 정의된다.
    * 문서 벡터와 질의 벡터를 각각 L2 정규화하므로 내적이 곧 코사인 유사도(0.0~1.0)다.

    :param chunks: 색인할 청크 목록. 비어 있으면 :class:`ValueError`.
    :raises ValueError: 청크가 없거나 ``source`` 가 빈 청크가 섞인 경우
        (출처 없는 근거는 사용 금지).
    """

    def __init__(self, chunks: Sequence[RetrievedChunk]) -> None:
        if not chunks:
            raise ValueError("색인할 근거 청크가 없습니다. 코퍼스 디렉터리를 확인하십시오.")
        missing = [chunk.chunk_id for chunk in chunks if not chunk.source.strip()]
        if missing:
            raise ValueError(
                "출처(source)가 비어 있는 근거 청크는 사용할 수 없습니다: "
                + ", ".join(missing)
            )
        self._chunks: tuple[RetrievedChunk, ...] = tuple(chunks)

        tokenized = [tokenize(chunk.text) for chunk in self._chunks]
        vocabulary: dict[str, int] = {}
        for tokens in tokenized:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self._vocabulary = vocabulary

        n_docs = len(self._chunks)
        n_terms = len(vocabulary)
        counts = np.zeros((n_docs, n_terms), dtype=np.float64)
        for row, tokens in enumerate(tokenized):
            for token in tokens:
                counts[row, vocabulary[token]] += 1.0

        document_frequency = (counts > 0).sum(axis=0).astype(np.float64)
        self._idf = np.log((n_docs + 1.0) / (document_frequency + 1.0)) + 1.0

        weights = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
        matrix = weights * self._idf
        self._matrix = _l2_normalize_rows(matrix)

    @property
    def chunks(self) -> tuple[RetrievedChunk, ...]:
        """색인된 청크 전체(입력 순서 유지)."""
        return self._chunks

    def __len__(self) -> int:
        """색인된 청크 개수."""
        return len(self._chunks)

    def _query_vector(self, query: str) -> np.ndarray:
        """질의 문자열을 L2 정규화된 TF-IDF 벡터로 바꾼다.

        :param query: 검색 질의.
        :returns: ``(어휘 수,)`` float64 벡터. 미등록어만 있으면 영벡터.
        """
        vector = np.zeros(len(self._vocabulary), dtype=np.float64)
        for token in tokenize(query):
            index = self._vocabulary.get(token)
            if index is not None:
                vector[index] += 1.0
        vector = np.where(vector > 0, 1.0 + np.log(np.maximum(vector, 1.0)), 0.0)
        vector *= self._idf
        norm = float(np.linalg.norm(vector))
        return vector if norm == 0.0 else vector / norm

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """질의에 적합한 근거 청크를 점수 내림차순으로 반환한다.

        :param query: 검색 질의. **개인정보가 제거된 문자열**이어야 한다.
        :param k: 반환할 최대 청크 수. 1 이상.
        :returns: :class:`~docagent.contracts.RetrievedChunk` 목록(길이 ``k`` 이하).
            ``score`` 는 코사인 유사도(0.0~1.0)로 채워진다. 유사도가 0 인 청크는
            제외하므로, 겹치는 어휘가 하나도 없으면 빈 리스트를 돌려준다.
        :raises ValueError: ``k`` 가 1 미만인 경우.
        """
        if k < 1:
            raise ValueError(f"k 는 1 이상이어야 합니다: {k}")
        vector = self._query_vector(query)
        if not np.any(vector):
            return []
        scores = self._matrix @ vector
        # 동점일 때 색인 순서가 앞선 청크가 이기도록 (-score, index) 로 정렬한다.
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
        results: list[RetrievedChunk] = []
        for index in order[:k]:
            score = float(scores[index])
            if score <= 0.0:
                break
            chunk = self._chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    score=round(score, 6),
                )
            )
        return results

    # ---------------------------------------------------------------- 캐시
    def to_dict(self) -> dict[str, Any]:
        """디스크 캐시용 dict 를 돌려준다(청크 원문만 저장한다).

        가중치 행렬은 저장하지 않는다. 청크에서 항상 같은 행렬이 재계산되므로
        캐시 파일이 작고, 계산식이 바뀌어도 캐시가 낡지 않는다.

        :returns: ``{"version": 1, "chunks": [...]}`` 형태의 dict.
        """
        return {
            "version": 1,
            "chunks": [chunk.to_dict() for chunk in self._chunks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LocalTfidfRetriever":
        """:meth:`to_dict` 결과로부터 검색기를 복원한다.

        :param data: :meth:`to_dict` 가 만든 매핑.
        :returns: :class:`LocalTfidfRetriever`.
        :raises ValueError: ``chunks`` 키가 없거나 비어 있는 경우.
        """
        if "chunks" not in data:
            raise ValueError("색인 캐시에 'chunks' 키가 없습니다.")
        chunks = [RetrievedChunk.from_dict(item) for item in data["chunks"]]
        return cls(chunks)

    @classmethod
    def from_corpus(
        cls,
        corpus_dir: Path | str | None = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> "LocalTfidfRetriever":
        """코퍼스 디렉터리를 읽어 곧바로 검색기를 만든다.

        :param corpus_dir: ``.md`` 파일이 있는 디렉터리. ``None`` 이면
            :data:`DEFAULT_CORPUS_DIR`.
        :param chunk_size: 청크 최대 길이(문자 수).
        :param overlap: 청크 간 겹침 길이(문자 수).
        :returns: :class:`LocalTfidfRetriever`.
        :raises CorpusError: 디렉터리가 없거나 ``.md`` 파일이 하나도 없는 경우.
        """
        directory = Path(corpus_dir) if corpus_dir is not None else DEFAULT_CORPUS_DIR
        if not directory.is_dir():
            raise CorpusError(f"코퍼스 디렉터리를 찾지 못했습니다: {directory}")
        paths = sorted(directory.glob("*.md"), key=lambda p: p.name)
        if not paths:
            raise CorpusError(f"코퍼스 디렉터리에 .md 파일이 없습니다: {directory}")
        return cls(chunk_documents(paths, chunk_size=chunk_size, overlap=overlap))


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """행 단위 L2 정규화. 노름이 0 인 행은 그대로 둔다.

    :param matrix: ``(행, 열)`` float 배열.
    :returns: 정규화된 새 배열.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


def build_index(
    corpus_dir: Path | str | None = None,
    *,
    cache_path: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    refresh: bool = False,
) -> LocalTfidfRetriever:
    """코퍼스로 색인을 만들고, 요청하면 디스크에 JSON 캐시를 남긴다.

    캐시는 결정론적이다. 같은 코퍼스로 만들면 **바이트 단위로 같은 파일**이 나온다
    (키 정렬 + ``ensure_ascii=False`` + 고정 들여쓰기).

    :param corpus_dir: 코퍼스 디렉터리. ``None`` 이면 :data:`DEFAULT_CORPUS_DIR`.
    :param cache_path: 캐시 JSON 경로. ``None`` 이면 캐시를 쓰지도 읽지도 않는다.
    :param chunk_size: 청크 최대 길이(문자 수).
    :param overlap: 청크 간 겹침 길이(문자 수).
    :param refresh: True 면 캐시가 있어도 무시하고 다시 만든 뒤 덮어쓴다.
    :returns: :class:`LocalTfidfRetriever`.
    :raises CorpusError: 코퍼스를 읽지 못했거나 캐시 파일이 깨진 경우.
    """
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.is_file() and not refresh:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorpusError(f"색인 캐시를 읽지 못했습니다: {path}") from exc
        return LocalTfidfRetriever.from_dict(payload)

    retriever = LocalTfidfRetriever.from_corpus(
        corpus_dir, chunk_size=chunk_size, overlap=overlap
    )
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    retriever.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise CorpusError(f"색인 캐시를 쓰지 못했습니다: {path}") from exc
    return retriever


# --------------------------------------------------------------------------
# 선택적 어댑터
# --------------------------------------------------------------------------


class FaissRetriever:
    """FAISS + 임베딩 기반 검색기(선택적 어댑터).

    ``faiss`` 와 임베딩 모델은 이번 범위에서 **설치하지 않는다.** 생성자에서
    지연 import 를 시도하고, 없으면 한국어 설치 안내가 담긴
    :class:`~docagent.errors.AdapterUnavailable` 를 던진다.
    기본 검색기는 :class:`LocalTfidfRetriever` 이며 이 어댑터는 선택이다.

    :param chunks: 색인할 청크 목록.
    :param embedder: ``list[str] -> (N, D) float32 배열`` 형태의 임베딩 함수.
        ``None`` 이면 임베딩 어댑터도 없는 것으로 보고 예외를 던진다.
    :raises docagent.errors.AdapterUnavailable: ``faiss`` 가 없거나 임베더가 없는 경우.
    """

    def __init__(
        self,
        chunks: Sequence[RetrievedChunk],
        embedder: Any | None = None,
    ) -> None:
        try:  # 지연 import — 모듈 최상단에서 절대 import 하지 않는다.
            import faiss  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as exc:
            raise AdapterUnavailable(
                "faiss-cpu",
                feature="FAISS 벡터 색인 기반 근거 검색",
                extra="faiss",
            ) from exc
        if embedder is None:
            raise AdapterUnavailable(
                "sentence-transformers",
                feature="FAISS 검색용 문장 임베딩 생성",
                extra="faiss",
            )
        self._chunks = tuple(chunks)
        self._embedder = embedder

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """질의에 적합한 근거 청크를 반환한다(어댑터 미설치 시 도달하지 않는다).

        :param query: 검색 질의.
        :param k: 반환할 최대 청크 수.
        :returns: :class:`~docagent.contracts.RetrievedChunk` 목록.
        :raises docagent.errors.AdapterUnavailable: 어댑터가 준비되지 않은 경우.
        """
        raise AdapterUnavailable(
            "faiss-cpu",
            feature="FAISS 벡터 색인 기반 근거 검색",
            extra="faiss",
        )

