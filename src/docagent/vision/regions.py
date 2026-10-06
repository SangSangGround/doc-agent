"""탐지 주변 OCR 영역. 모든 Crop 은 정합된 페이지에서 잘라낸다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from docagent.contracts import A4_PAGE_SIZE_MM, BoxMm, BoxPx, Detection, OcrWord
from docagent.errors import VisionError


@dataclass(frozen=True)
class OcrRegion:
    """페이지 안의 Crop 위치와 실제 물리 크기(반올림된 픽셀 경계 기준)."""

    box_px: BoxPx
    box_mm: BoxMm

    def restore(self, words: Sequence[OcrWord]) -> list[OcrWord]:
        """Crop 로컬 mm 좌표를 전체 페이지 mm 좌표로 복원한다."""
        return [
            OcrWord(
                text=word.text,
                box_mm=BoxMm(
                    self.box_mm.x_mm + word.box_mm.x_mm,
                    self.box_mm.y_mm + word.box_mm.y_mm,
                    word.box_mm.w_mm,
                    word.box_mm.h_mm,
                ),
                confidence=word.confidence,
            )
            for word in words
        ]


def detection_regions(
    detections: Sequence[Detection],
    image_shape: tuple[int, ...],
    *,
    page_size_mm: tuple[float, float] = A4_PAGE_SIZE_MM,
    padding_mm: tuple[float, float, float, float] = (45.0, 15.0, 45.0, 8.0),
) -> list[OcrRegion]:
    """왼쪽·위·오른쪽·아래 여백을 더하고 경계 보정 후 겹치는 영역을 병합한다.

    mm 여백을 사용하므로 해상도가 바뀌어도 읽는 물리 영역은 같다.
    겹침 병합은 인접 체크박스의 문구가 여러 번 OCR 되는 것을 방지한다.
    탐지가 0건이면 OCR 영역도 0개다. 페이지 밖/퇴화한 탐지는 오류로 처리한다.
    """
    if len(image_shape) not in (2, 3) or min(image_shape[:2]) <= 0:
        raise VisionError("OCR 영역 생성에는 유효한 페이지 이미지가 필요합니다.")
    if len(page_size_mm) != 2 or any(not math.isfinite(v) or v <= 0 for v in page_size_mm):
        raise ValueError("page_size_mm 은 유한한 양수 두 개여야 합니다.")
    if len(padding_mm) != 4 or any(not math.isfinite(v) or v < 0 for v in padding_mm):
        raise ValueError("padding_mm 은 유한한 0 이상 값 네 개여야 합니다.")
    height, width = image_shape[:2]
    sx, sy = page_size_mm[0] / width, page_size_mm[1] / height
    left, top, right, bottom = padding_mm
    rectangles: list[tuple[int, int, int, int]] = []
    for detection in detections:
        box = detection.box_mm
        values = (box.x_mm, box.y_mm, box.w_mm, box.h_mm)
        if (not all(math.isfinite(v) for v in values)
                or box.w_mm <= 0 or box.h_mm <= 0
                or box.right_mm <= 0 or box.bottom_mm <= 0
                or box.x_mm >= page_size_mm[0] or box.y_mm >= page_size_mm[1]):
            raise VisionError("페이지 안에 유효한 크기를 갖지 않는 탐지 영역입니다.")
        rectangles.append((
            max(0, math.floor((box.x_mm - left) / sx)),
            max(0, math.floor((box.y_mm - top) / sy)),
            min(width, math.ceil((box.right_mm + right) / sx)),
            min(height, math.ceil((box.bottom_mm + bottom) / sy)),
        ))

    merged: list[tuple[int, int, int, int]] = []
    for rectangle in sorted(rectangles):
        # 합친 영역이 이전 영역과 새로 겹칠 수 있으므로 고정점까지 반복한다.
        pending = rectangle
        index = 0
        while index < len(merged):
            other = merged[index]
            if (min(pending[2], other[2]) > max(pending[0], other[0])
                    and min(pending[3], other[3]) > max(pending[1], other[1])):
                pending = (
                    min(pending[0], other[0]), min(pending[1], other[1]),
                    max(pending[2], other[2]), max(pending[3], other[3]),
                )
                merged.pop(index)
                index = 0
            else:
                index += 1
        merged.append(pending)
    return [
        OcrRegion(
            BoxPx(x0, y0, x1 - x0, y1 - y0),
            BoxMm(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy),
        )
        for x0, y0, x1, y1 in sorted(merged, key=lambda r: (r[1], r[0]))
    ]
