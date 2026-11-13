"""촬영본 OCR과 등록 원문을 대조한다. 정합 성공만으로 원문을 확정하지 않는다.

한국어 단일 페이지 양식용이다. 공백·장식 문장부호는 허용하되 글자·숫자·
수치 기호가 달라지거나 추가 문구가 읽히면 전체 안내를 보류한다. OCR이 같은
오독을 반복할 가능성까지 제거하는 인증 수단은 아니며 실제 사진 평가가 필요하다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import cv2
import numpy as np

from docagent.vision.ocr import TesseractOcr

_NUMBERS = re.compile(r"[+\-]?(?:\d+(?:[.,]\d+)*|\.\d+)(?:[%‰])?")
_SEMANTIC_SYMBOLS = frozenset("()+<>=≤≥≠±×÷≈~/%‰'\"[]{}")
_TRANSLATE = str.maketrans({"−": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"'})


def _compact(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).translate(_TRANSLATE).split())


def normalized_text(text: str) -> str:
    """띄어쓰기·목록 번호의 마침표·체크칸 모양을 제외한 글자와 의미 기호."""
    return "".join(char for char in _compact(text)
                   if char not in "ㆍᆞ" and (char.isalnum() or char in _SEMANTIC_SYMBOLS))


def numeric_tokens(text: str) -> list[str]:
    """12,000/12.000, 음수, 소수점, 범위, 비율을 문자 대조와 별도로 보존한다."""
    return _NUMBERS.findall(_compact(text))


def compare_content(expected: str, ocr: dict[str, Any]) -> dict[str, Any]:
    if ocr.get("status") != "read" or not isinstance(ocr.get("text"), str) or not ocr["text"].strip():
        return {"status": "unavailable", "method": "korean_line_ocr_exact_text_and_numbers",
                "message": "촬영본의 원문을 읽지 못해 문서 설명과 질문을 보류합니다. OCR 설치와 촬영 상태를 확인해 주세요."}
    actual = ocr["text"]
    words_match = normalized_text(expected) == normalized_text(actual)
    numbers_match = numeric_tokens(expected) == numeric_tokens(actual)
    matched = words_match and numbers_match
    return {"status": "matched" if matched else "mismatch",
            "method": "korean_line_ocr_exact_text_and_numbers",
            "text_matches": words_match, "numbers_match": numbers_match,
            "message": ("촬영본에서 읽은 문구와 수치가 등록 원문과 일치합니다." if matched else
                        "촬영본의 문구나 수치가 등록 원문과 다르거나 정확히 읽히지 않습니다. "
                        "문서 설명과 질문을 보류합니다. 변경된 양식인지 확인하거나 다시 촬영해 주세요.")}


def read_printed_content(image: np.ndarray, manifest: dict[str, Any]) -> dict[str, Any]:
    """촬영본 자체에서 모든 행을 찾는다. 등록된 행만 OCR해 추가 문구를 빠뜨리지 않는다.

    입력은 전체 페이지로 펴진 이미지다. 등록된 체크칸의 표시만 제외한다.
    제목·공통 설명·서명 주변을 포함한 나머지 영역은 모두 검사 대상이다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    # 행 탐색의 픽셀 기준을 일정하게 유지하고 큰 업로드의 처리량을 제한한다.
    if gray.shape[0] != 1800:
        gray = cv2.resize(gray, (round(gray.shape[1] * 1800 / gray.shape[0]), 1800),
                          interpolation=cv2.INTER_CUBIC)
    height, width = gray.shape
    page_width, page_height = manifest["page_size_mm"]
    for group in manifest["groups"]:
        for option in group["options"]:
            x, y, w, h = option["box_mm"]
            left, right = round(x / page_width * width), round((x + w) / page_width * width)
            top, bottom = round(y / page_height * height), round((y + h) / page_height * height)
            gray[max(0, top - 3):min(height, bottom + 4), max(0, left - 3):min(width, right + 4)] = 255
    # 절대 밝기 기준은 옅게 추가된 조건을 배경으로 버릴 수 있다. 주변 종이의
    # 밝기와 비교해 잉크를 찾고, 읽을 수 없는 행도 아래에서 보류 사유로 남긴다.
    ink = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 12) > 0).astype(np.uint8)
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(80, round(width * .15))), np.uint8))
    # 사각형 검출의 1~2px 여유로 남은 페이지 테두리가 모든 행을 하나로
    # 연결하지 않도록 긴 세로선도 제거한다. 글자 영역의 짧은 획은 유지한다.
    rules |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(80, round(height * .15)), 1), np.uint8))
    rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
    gray[rules > 0] = 255
    ink[rules > 0] = 0
    rows = np.flatnonzero(ink.sum(axis=1) > 2)
    bands: list[list[int]] = []
    for row in rows:
        if bands and row - bands[-1][1] <= 5:
            bands[-1][1] = int(row)
        else:
            bands.append([int(row), int(row)])
    if len(bands) > 100:
        return {"status": "unavailable", "text": "", "word_count": 0,
                "message": "문자 행을 안정적으로 구분하지 못했습니다. 그림자와 촬영 각도를 확인해 주세요."}
    texts: list[str] = []
    word_count = 0
    unreadable_lines = 0
    for top, bottom in bands:
        crop = cv2.copyMakeBorder(gray[top:bottom + 1], 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        engine = TesseractOcr(lang="kor", config="--psm 7",
                              page_size_mm=(page_width, crop.shape[0] / height * page_height))
        words = engine.read(crop)
        # 각 crop은 한 행이다. 한글 음절의 세로 중심 차이로 순서가 바뀌지 않게 한다.
        text = " ".join(word.text for word in sorted(words, key=lambda word: word.box_mm.x_mm)).strip()
        if normalized_text(text):
            texts.append(text)
            word_count += len(words)
        else:
            # 추가된 흐린 문구를 OCR이 못 읽었다고 조용히 버리면 등록 원문과
            # 같다는 잘못된 결론에 도달할 수 있다. 잉크 행이 남으면 확인을 보류한다.
            unreadable_lines += 1
    return {"status": "unavailable" if unreadable_lines else "read" if texts else "empty",
            "text": "\n".join(texts), "word_count": word_count, "line_count": len(texts),
            "unreadable_lines": unreadable_lines, "language": "kor", "mode": "line_psm7"}
