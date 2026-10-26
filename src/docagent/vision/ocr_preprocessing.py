"""선택적으로 긴 표 선을 제거하는 OCR 입력 전처리.

원본을 수정하거나 크기를 바꾸지 않는다. 글자와 선이 겹치면 글자 일부도
없어질 수 있으므로 기본 OCR 경로에는 적용하지 않고, 원본 OCR과 함께
검수하는 ``form_lines`` 모드에서만 사용한다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from docagent.errors import VisionError
from docagent.interfaces import ImageArray


def remove_form_lines(
    image: ImageArray, *, page_size_mm: tuple[float, float]
) -> ImageArray:
    """10mm 이상이며 1mm 이하로 얇은 수평·수직 선만 흰색으로 지운다.

    처리 결과는 입력과 크기·채널 수가 같은 새 배열이다. 체크박스나 짧은
    글자 획을 제거 대상으로 삼지 않는다. 물리 크기는 페이지뿐 아니라
    Crop의 실제 크기도 지원하므로 좌표 비율과 선 길이가 일관된다.
    """
    if (getattr(image, "dtype", None) != np.uint8
            or image.ndim not in (2, 3)
            or (image.ndim == 3 and image.shape[2] != 3)
            or min(image.shape[:2]) <= 0):
        raise VisionError("표 선 제거 입력은 비어 있지 않은 uint8 BGR/회색 이미지여야 합니다.")
    if (len(page_size_mm) != 2
            or any(not math.isfinite(v) or v <= 0 for v in page_size_mm)):
        raise ValueError("page_size_mm 은 유한한 양수 두 개여야 합니다.")
    height, width = image.shape[:2]
    sx, sy = width / page_size_mm[0], height / page_size_mm[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    mask = np.zeros((height, width), dtype=np.uint8)
    # 연결 성분을 다시 검사해 큰 검은 영역이 표 선으로 제거되지 않게 한다.
    for horizontal in (True, False):
        # 홀수 커널로 대칭 중심을 맞춰 선 양 끝에 1px 잔상이 남지 않게 한다.
        length = max(3, round((sx if horizontal else sy) * 10.0)) | 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (length, 1) if horizontal else (1, length)
        )
        lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(lines, 8)
        for index in range(1, count):
            x, y, w, h, _ = stats[index]
            long_side, thickness = (w, h) if horizontal else (h, w)
            max_thickness = max(1, round(sy if horizontal else sx))
            if long_side >= length and thickness <= max_thickness and long_side >= thickness * 15:
                region = mask[y:y + h, x:x + w]
                region[labels[y:y + h, x:x + w] == index] = 255
    result = image.copy()
    result[mask != 0] = 255
    return result
