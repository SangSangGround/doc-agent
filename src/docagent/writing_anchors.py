"""Verify open writable strips against the actual page's printed labels and table lines."""

from __future__ import annotations

import re

import cv2
import numpy as np

from docagent.home.macos_ocr import MacVision, NativeOcrUnavailable
from docagent.writing_geometry import threshold


def printed_words(gray):
    """Use the existing Korean Vision helper locally; a missing language pack withholds open-field guidance."""
    try:
        lines = MacVision().read(gray)
    except NativeOcrUnavailable:
        return []
    result = []
    for line in lines:
        # Korean line confidence is frequently quantized to 0.5 even with a complete literal label.
        if line["confidence"] < 0.45:
            continue
        words = line.get("words") or [line]
        for item in words:
            text = re.sub(r"[^가-힣A-Za-z0-9]", "", item["text"])
            x, y, w, h = item["box"]
            if text and w > 0 and h > 0:
                result.append((text, np.array([x, y, x + w, y + h])))
    return result


def boundary_near(mask, box, margin):
    x1, y1, x2, y2 = np.rint(box).astype(int)
    # Slight perspective can distribute a straight printed rule over several rows.
    h, w = mask.shape
    left, right = max(0, x1), min(w, x2)
    if right - left < 12:
        return False
    for y in (y1, y2):
        crop = mask[max(0, y - margin) : min(h, y + margin + 1), left:right]
        if crop.shape[0] >= 3:
            band = cv2.dilate(crop, np.ones((3, 1), np.uint8))
            if (band > 0).mean(axis=1).max() >= 0.65:
                return True
    return False


def clear_region(mask, area, text_height, obstacles):
    """Return a contiguous low-ink strip, stopping short of any neighboring field or printed text."""
    x1, y1, x2, y2 = area
    pad = max(2.0, text_height * 0.09)
    x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 - pad, y2 - pad
    for other in obstacles:
        ox1, oy1, ox2, oy2 = other
        vertical_overlap = min(oy2, y2) - max(oy1, y1)
        if vertical_overlap < 0.25 * (y2 - y1) or ox2 <= x1 or ox1 >= x2:
            continue
        if ox1 <= x1 + pad:
            x1 = max(x1, ox2 + pad)
        else:
            x2 = min(x2, ox1 - pad)
    if x2 - x1 < max(22, 1.65 * text_height) or y2 - y1 < max(9, 0.45 * text_height):
        return None
    for fraction in (1.0, 0.75, 0.5):
        right = x1 + (x2 - x1) * fraction
        if right - x1 < max(22, 1.65 * text_height):
            break
        bounds = np.array([x1, y1, right, y2])
        left, top, right_pixel, bottom = np.rint(bounds).astype(int)
        crop = mask[top:bottom, left:right_pixel]
        if crop.size and np.mean(crop > 0) < 0.012 and (crop > 0).mean(axis=0).max() < 0.45:
            return bounds
    return None


class WritingAnchors:
    def __init__(self, gray):
        self.words = printed_words(gray)
        self.ink = threshold(gray)

    def region(self, kind, label, source, neighbors):
        target_label = re.sub(r"[^가-힣A-Za-z0-9]", "", label)
        left, top, right, bottom = source
        options = []
        for text, printed in self.words:
            x1, y1, x2, y2 = printed
            height = y2 - y1
            if kind == "signature":
                # '서명' must be on this specific field. Only use space still INSIDE the Gemini proposal.
                if text not in ("서명", "날인", "자필서명", "서명란", "서명또는인"):
                    continue
                if not (
                    left - 0.3 * height <= x1 <= right and top <= (y1 + y2) / 2 <= bottom + 0.2 * height
                ):
                    continue
                if y1 - top < 0.65 * height or y1 - top > 2.5 * height:
                    continue
                area = [max(left, x1), top, right, y1]
            else:
                if len(text) < 2 or text not in target_label:
                    continue
                gap = left - x2
                if not (
                    -0.45 * height <= gap <= 1.5 * height
                    and abs((top + bottom - y1 - y2) / 2) <= 0.85 * height
                ):
                    continue
                area = [
                    max(left, x2),
                    max(top, y1 - 0.12 * height),
                    right,
                    min(bottom, y2 + 0.12 * height),
                ]
            if not boundary_near(self.ink, source, max(5, round(0.55 * height))):
                continue
            obstacles = neighbors + [bounds for _, bounds in self.words]
            patch = clear_region(self.ink, area, height, obstacles)
            if patch is not None:
                options.append(patch)
        # A choice among distinct matching printed labels would be a guess.
        return options[0] if len(options) == 1 else None
