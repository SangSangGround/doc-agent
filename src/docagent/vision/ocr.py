"""OCR 어댑터와 단어 → 행 → 문단 그룹핑 유틸(Understand 단계 입구).

이 모듈은 :class:`docagent.interfaces.OcrEngine` 프로토콜의 구현 두 가지와,
그 결과를 구조화 단계가 쓰기 좋은 형태로 묶는 순수 함수들을 제공한다.

* :class:`TesseractOcr` — 실제 OCR 어댑터. ``pytesseract`` 를 **함수 내부에서
  지연 import** 한다. 미설치 시 :class:`~docagent.errors.AdapterUnavailable` 로
  한국어 설치 안내를 던진다. 모듈 최상단에 ``pytesseract`` import 는 없다.
* :class:`StubOcr` — 정답 단어 목록을 주입받아 그대로 돌려주는 결정론적 구현.
  합성 문서 테스트와 오프라인 데모의 **기본값**이다. 실제 OCR 과 동일하게
  :class:`~docagent.contracts.OcrWord` 규격(mm 좌표)을 지킨다.

좌표 규약
---------
:class:`OcrWord.box_mm` 는 **정합(deskew·crop)이 끝난 A4 페이지의 mm 좌표**다.
따라서 어댑터는 입력 이미지가 "페이지 전체를 담은 정합된 이미지"라고 가정하고,
``픽셀 → mm`` 환산을 ``페이지_mm / 이미지_px`` 비율로 수행한다. 페이지 크기는
생성자의 ``page_size_mm`` 로 바꿀 수 있다(기본 A4).

읽기 순서
---------
:func:`sort_reading_order` 는 **y 우선, 같은 행 안에서는 x 오름차순**으로 정렬한다.
행 판정 허용오차는 :data:`DEFAULT_LINE_Y_TOL_MM` (mm)이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from docagent.contracts import A4_PAGE_SIZE_MM, BoxMm, OcrWord
from docagent.errors import AdapterUnavailable, VisionError
from docagent.interfaces import ImageArray

__all__ = [
    "DEFAULT_LINE_Y_TOL_MM",
    "DEFAULT_PARAGRAPH_GAP_MM",
    "DEFAULT_OCR_LANG",
    "TextLine",
    "Paragraph",
    "StubOcr",
    "TesseractOcr",
    "sort_reading_order",
    "group_words_to_lines",
    "lines_to_paragraphs",
    "union_box",
]


#: 같은 행으로 볼 y 중심 허용오차(mm). 본문 글자 높이(약 3.6mm)의 70% 수준.
DEFAULT_LINE_Y_TOL_MM: float = 2.5
#: 같은 문단으로 볼 행 간 세로 간격(mm). 이보다 벌어지면 문단을 끊는다.
DEFAULT_PARAGRAPH_GAP_MM: float = 3.0
#: 기본 OCR 언어 조합(한국어 + 영어).
DEFAULT_OCR_LANG: str = "kor+eng"


# --------------------------------------------------------------------------
# 기하 헬퍼
# --------------------------------------------------------------------------


def union_box(boxes: Sequence[BoxMm]) -> BoxMm:
    """여러 mm 사각형을 모두 감싸는 최소 사각형을 반환한다.

    :param boxes: 1개 이상의 :class:`BoxMm`.
    :returns: 합집합 경계 상자.
    :raises ValueError: ``boxes`` 가 비어 있는 경우.
    """
    if not boxes:
        raise ValueError("union_box 에 빈 사각형 목록이 들어왔습니다.")
    left = min(box.x_mm for box in boxes)
    top = min(box.y_mm for box in boxes)
    right = max(box.right_mm for box in boxes)
    bottom = max(box.bottom_mm for box in boxes)
    return BoxMm(left, top, right - left, bottom - top)


def _y_center(box: BoxMm) -> float:
    """사각형의 세로 중심 좌표(mm)."""
    return box.y_mm + box.h_mm / 2.0


# --------------------------------------------------------------------------
# 행 · 문단 (모듈 내부 표현)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextLine:
    """같은 행으로 묶인 단어들.

    계약(:mod:`docagent.contracts`) 에 없는 **Vision 내부 표현**이다.
    모듈 경계를 넘어가는 산출물은 어디까지나 :class:`DocumentStructure` 이며,
    이 타입은 구조화 규칙을 서술하기 위한 중간 자료구조다.

    :param words: 행을 구성하는 단어들(x 오름차순).
    :param text: 단어를 공백 하나로 이어 붙인 행 전체 문자열.
    :param box_mm: 행 전체를 감싸는 mm 경계 상자.
    :param confidence: 행 신뢰도. 구성 단어 신뢰도의 **최솟값**(보수적).
    """

    words: tuple[OcrWord, ...]
    text: str
    box_mm: BoxMm
    confidence: float

    @classmethod
    def from_words(cls, words: Sequence[OcrWord]) -> "TextLine":
        """단어 목록에서 행 하나를 만든다(x 오름차순 정렬 후 결합).

        :param words: 같은 행에 속하는 단어들. 비어 있으면 안 된다.
        :returns: :class:`TextLine`.
        :raises ValueError: ``words`` 가 비어 있는 경우.
        """
        if not words:
            raise ValueError("TextLine.from_words 에 빈 단어 목록이 들어왔습니다.")
        ordered = tuple(sorted(words, key=lambda w: (w.box_mm.x_mm, w.box_mm.y_mm)))
        return cls(
            words=ordered,
            text=" ".join(word.text for word in ordered).strip(),
            box_mm=union_box([word.box_mm for word in ordered]),
            confidence=min(word.confidence for word in ordered),
        )


@dataclass(frozen=True)
class Paragraph:
    """세로로 인접한 행들을 묶은 문단.

    :param lines: 문단을 구성하는 행들(위 → 아래).
    :param text: 행 문자열을 공백 하나로 이어 붙인 문단 전체 문자열.
    :param box_mm: 문단 전체 mm 경계 상자.
    :param confidence: 구성 행 신뢰도의 최솟값.
    """

    lines: tuple[TextLine, ...]
    text: str
    box_mm: BoxMm
    confidence: float

    @classmethod
    def from_lines(cls, lines: Sequence[TextLine]) -> "Paragraph":
        """행 목록에서 문단 하나를 만든다.

        :param lines: 위에서 아래로 정렬된 행들. 비어 있으면 안 된다.
        :returns: :class:`Paragraph`.
        :raises ValueError: ``lines`` 가 비어 있는 경우.
        """
        if not lines:
            raise ValueError("Paragraph.from_lines 에 빈 행 목록이 들어왔습니다.")
        ordered = tuple(sorted(lines, key=lambda line: line.box_mm.y_mm))
        return cls(
            lines=ordered,
            text=" ".join(line.text for line in ordered).strip(),
            box_mm=union_box([line.box_mm for line in ordered]),
            confidence=min(line.confidence for line in ordered),
        )


# --------------------------------------------------------------------------
# 그룹핑 유틸
# --------------------------------------------------------------------------


def sort_reading_order(
    words: Sequence[OcrWord], *, y_tol_mm: float = DEFAULT_LINE_Y_TOL_MM
) -> list[OcrWord]:
    """단어를 읽기 순서(y 우선, 같은 행 안에서는 x 오름차순)로 정렬한다.

    :param words: 정렬할 단어 목록.
    :param y_tol_mm: 같은 행으로 볼 y 중심 허용오차(mm, 0 이상).
    :returns: 새 리스트(입력은 변경하지 않는다).
    :raises ValueError: ``y_tol_mm`` 이 음수인 경우.
    """
    lines = group_words_to_lines(words, y_tol_mm=y_tol_mm)
    ordered: list[OcrWord] = []
    for line in lines:
        ordered.extend(line.words)
    return ordered


def group_words_to_lines(
    words: Sequence[OcrWord], *, y_tol_mm: float = DEFAULT_LINE_Y_TOL_MM
) -> list[TextLine]:
    """단어를 같은 행끼리 묶는다.

    y 중심이 현재 행의 평균 y 중심에서 ``y_tol_mm`` 이내면 같은 행으로 본다.
    공백뿐인 단어는 버린다(빈 문자열은 행 문자열을 오염시키기만 한다).

    :param words: 단어 목록. 순서는 상관없다.
    :param y_tol_mm: 같은 행으로 볼 y 중심 허용오차(mm, 0 이상).
    :returns: 위 → 아래로 정렬된 :class:`TextLine` 목록. 입력이 비면 빈 리스트.
    :raises ValueError: ``y_tol_mm`` 이 음수인 경우.
    """
    if y_tol_mm < 0:
        raise ValueError(f"y_tol_mm 은 0 이상이어야 합니다: {y_tol_mm}")
    usable = [word for word in words if word.text and word.text.strip()]
    if not usable:
        return []

    ordered = sorted(usable, key=lambda w: (_y_center(w.box_mm), w.box_mm.x_mm))
    buckets: list[list[OcrWord]] = []
    centers: list[float] = []
    for word in ordered:
        center = _y_center(word.box_mm)
        if buckets and abs(center - centers[-1]) <= y_tol_mm:
            buckets[-1].append(word)
            # 평균으로 갱신해 행이 조금씩 기울어도 따라간다.
            centers[-1] = sum(_y_center(w.box_mm) for w in buckets[-1]) / len(buckets[-1])
        else:
            buckets.append([word])
            centers.append(center)

    lines = [TextLine.from_words(bucket) for bucket in buckets]
    lines.sort(key=lambda line: (line.box_mm.y_mm, line.box_mm.x_mm))
    return lines


def lines_to_paragraphs(
    lines: Sequence[TextLine], *, gap_mm: float = DEFAULT_PARAGRAPH_GAP_MM
) -> list[Paragraph]:
    """세로로 인접한 행들을 문단으로 묶는다.

    직전 행의 아래 경계와 다음 행의 위 경계 사이 간격이 ``gap_mm`` 이하이고
    가로로 겹치는 부분이 있으면 같은 문단으로 본다.

    :param lines: 행 목록. 순서는 상관없다.
    :param gap_mm: 같은 문단으로 볼 최대 세로 간격(mm, 0 이상).
    :returns: 위 → 아래로 정렬된 :class:`Paragraph` 목록.
    :raises ValueError: ``gap_mm`` 이 음수인 경우.
    """
    if gap_mm < 0:
        raise ValueError(f"gap_mm 은 0 이상이어야 합니다: {gap_mm}")
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (line.box_mm.y_mm, line.box_mm.x_mm))
    buckets: list[list[TextLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        previous = buckets[-1][-1]
        vertical_gap = line.box_mm.y_mm - previous.box_mm.bottom_mm
        overlaps = (
            min(line.box_mm.right_mm, previous.box_mm.right_mm)
            - max(line.box_mm.x_mm, previous.box_mm.x_mm)
        ) > 0.0
        if vertical_gap <= gap_mm and overlaps:
            buckets[-1].append(line)
        else:
            buckets.append([line])
    return [Paragraph.from_lines(bucket) for bucket in buckets]


# --------------------------------------------------------------------------
# 어댑터 구현
# --------------------------------------------------------------------------


def _validate_page_size(page_size_mm: tuple[float, float]) -> tuple[float, float]:
    """페이지 크기 인자를 검증해 ``(가로_mm, 세로_mm)`` 로 정규화한다.

    :param page_size_mm: 길이 2 의 시퀀스.
    :returns: ``(float, float)``.
    :raises ValueError: 길이가 2 가 아니거나 0 이하 값이 있는 경우.
    """
    values = tuple(float(v) for v in page_size_mm)
    if len(values) != 2:
        raise ValueError(f"page_size_mm 은 길이 2 여야 합니다: {page_size_mm!r}")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"page_size_mm 은 0 보다 커야 합니다: {values}")
    return (values[0], values[1])


def _image_shape(image: ImageArray) -> tuple[int, int]:
    """이미지의 ``(height_px, width_px)`` 를 구한다.

    :param image: ``(H, W)`` 또는 ``(H, W, 3)`` 배열.
    :returns: ``(height_px, width_px)``.
    :raises VisionError: 배열이 아니거나 차원이 맞지 않는 경우.
    """
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) not in (2, 3):
        raise VisionError(
            "OCR 입력 이미지는 (H, W) 또는 (H, W, 3) 배열이어야 합니다. "
            f"입력 형태: {shape!r}"
        )
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise VisionError(f"OCR 입력 이미지 크기가 올바르지 않습니다: {shape!r}")
    return (height, width)


class TesseractOcr:
    """``pytesseract`` 기반 OCR 어댑터(:class:`docagent.interfaces.OcrEngine`).

    ``pytesseract`` 는 **선택적 의존 패키지**이므로 모듈 최상단이 아니라
    :meth:`read` 내부에서 지연 import 한다. 미설치이거나 Tesseract 실행 파일이
    없으면 :class:`~docagent.errors.AdapterUnavailable` 를 던진다.

    :param lang: Tesseract 언어 조합. 기본 :data:`DEFAULT_OCR_LANG`.
    :param page_size_mm: 입력 이미지가 담고 있는 페이지 크기 ``(가로_mm, 세로_mm)``.
        픽셀 → mm 환산에 쓰인다. 기본 A4.
    :param min_confidence: 이 값 미만의 단어는 버린다(0.0~1.0).
    :param config: Tesseract 추가 설정 문자열(예: ``"--psm 6"``).
    :raises ValueError: 인자가 허용 범위를 벗어난 경우.
    """

    def __init__(
        self,
        *,
        lang: str = DEFAULT_OCR_LANG,
        page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
        min_confidence: float = 0.0,
        config: str = "",
    ) -> None:
        if not lang:
            raise ValueError("TesseractOcr.lang 은 빈 문자열일 수 없습니다.")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"TesseractOcr.min_confidence 는 0.0~1.0 이어야 합니다: {min_confidence}"
            )
        self.lang = lang
        self.page_size_mm = _validate_page_size(page_size_mm)
        self.min_confidence = float(min_confidence)
        self.config = config

    @staticmethod
    def _import_pytesseract() -> Any:
        """``pytesseract`` 를 지연 import 한다.

        :returns: ``pytesseract`` 모듈.
        :raises docagent.errors.AdapterUnavailable: 패키지가 없는 경우(한국어 안내 포함).
        """
        try:
            import pytesseract  # noqa: PLC0415 — 선택적 어댑터는 지연 import 한다.
        except ImportError as exc:
            raise AdapterUnavailable(
                package="pytesseract",
                feature="종이 문서 OCR 문자 인식",
                extra="ocr",
            ) from exc
        return pytesseract

    def read(self, image: ImageArray) -> list[OcrWord]:
        """이미지에서 단어 단위 인식 결과를 mm 좌표로 반환한다.

        :param image: 정합이 끝난 페이지 이미지. ``(H, W, 3)`` uint8 BGR 또는
            ``(H, W)`` uint8 그레이스케일.
        :returns: 읽기 순서로 정렬된 :class:`OcrWord` 목록. 인식 결과가 없으면 빈 리스트.
        :raises docagent.errors.VisionError: 이미지가 유효하지 않거나 인식이 실패한 경우.
        :raises docagent.errors.AdapterUnavailable: ``pytesseract`` 또는 Tesseract
            실행 파일이 없는 경우.
        """
        height_px, width_px = _image_shape(image)
        pytesseract = self._import_pytesseract()

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:  # Tesseract 미설치는 별도 예외 타입으로 온다.
            not_found = getattr(pytesseract, "TesseractNotFoundError", None)
            if not_found is not None and isinstance(exc, not_found):
                raise AdapterUnavailable(
                    package="tesseract-ocr 실행 파일",
                    feature="종이 문서 OCR 문자 인식",
                    extra="ocr",
                ) from exc
            raise VisionError(f"Tesseract OCR 실행에 실패했습니다: {exc}") from exc

        scale_x = self.page_size_mm[0] / float(width_px)
        scale_y = self.page_size_mm[1] / float(height_px)
        words: list[OcrWord] = []
        count = len(data.get("text", []))
        for index in range(count):
            text = str(data["text"][index]).strip()
            if not text:
                continue
            try:
                raw_conf = float(data["conf"][index])
            except (TypeError, ValueError):
                raw_conf = -1.0
            confidence = 0.0 if raw_conf < 0 else min(1.0, raw_conf / 100.0)
            if confidence < self.min_confidence:
                continue
            words.append(
                OcrWord(
                    text=text,
                    box_mm=BoxMm(
                        x_mm=float(data["left"][index]) * scale_x,
                        y_mm=float(data["top"][index]) * scale_y,
                        w_mm=max(0.0, float(data["width"][index]) * scale_x),
                        h_mm=max(0.0, float(data["height"][index]) * scale_y),
                    ),
                    confidence=confidence,
                )
            )
        return sort_reading_order(words)


class StubOcr:
    """정답 단어 목록을 그대로 돌려주는 결정론적 OCR(:class:`OcrEngine`).

    합성 문서 테스트와 API 키·외부 엔진 없는 오프라인 데모의 기본 구현이다.
    :meth:`read` 는 입력 이미지를 보지 않으며(``None`` 도 허용), 항상 동일한
    결과를 **읽기 순서로** 반환한다.

    :param words: 반환할 :class:`OcrWord` 목록.
    :param page_size_mm: 이 단어들이 놓인 페이지 크기 ``(가로_mm, 세로_mm)``.
    :raises ValueError: ``words`` 에 :class:`OcrWord` 가 아닌 값이 있거나
        신뢰도가 0.0~1.0 을 벗어난 경우.
    """

    def __init__(
        self,
        words: Sequence[OcrWord] = (),
        *,
        page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
    ) -> None:
        validated: list[OcrWord] = []
        for index, word in enumerate(words):
            if not isinstance(word, OcrWord):
                raise ValueError(
                    f"StubOcr.words[{index}] 는 OcrWord 여야 합니다: {type(word).__name__}"
                )
            if not 0.0 <= word.confidence <= 1.0:
                raise ValueError(
                    f"StubOcr.words[{index}].confidence 는 0.0~1.0 이어야 합니다: "
                    f"{word.confidence}"
                )
            validated.append(word)
        self.page_size_mm = _validate_page_size(page_size_mm)
        self._words: tuple[OcrWord, ...] = tuple(sort_reading_order(validated))

    @classmethod
    def from_texts(
        cls,
        entries: Sequence[tuple[str, BoxMm]],
        *,
        confidence: float = 1.0,
        page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
    ) -> "StubOcr":
        """``(문자열, BoxMm)`` 쌍 목록으로부터 손쉽게 생성한다.

        :param entries: ``(text, box_mm)`` 쌍 목록.
        :param confidence: 모든 단어에 부여할 신뢰도(0.0~1.0).
        :param page_size_mm: 페이지 크기 ``(가로_mm, 세로_mm)``.
        :returns: :class:`StubOcr`.
        :raises ValueError: ``confidence`` 가 범위를 벗어난 경우.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence 는 0.0~1.0 이어야 합니다: {confidence}")
        words = [
            OcrWord(text=text, box_mm=box_mm, confidence=confidence)
            for text, box_mm in entries
        ]
        return cls(words, page_size_mm=page_size_mm)

    @property
    def words(self) -> tuple[OcrWord, ...]:
        """주입된 단어들(읽기 순서, 불변 튜플)."""
        return self._words

    def read(self, image: ImageArray = None) -> list[OcrWord]:
        """주입된 단어 목록을 읽기 순서로 반환한다.

        :param image: 무시된다(인터페이스 호환용). ``None`` 허용.
        :returns: :class:`OcrWord` 목록의 **새 리스트**(호출자가 변형해도 안전).
        """
        return list(self._words)
