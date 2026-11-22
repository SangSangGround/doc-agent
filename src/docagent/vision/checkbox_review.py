"""빈 양식과 촬영본을 정합한 뒤 선택지별 표시를 보수적으로 판독한다.

이 모듈의 상태는 체크 흔적에 대한 휴리스틱 관측이다. 사용자 의도, 문서 승인,
서명의 진위는 판정하지 않는다. 실제 문서 성능은 별도 정답셋에서 측정해야 한다.
정합 근거가 부족하거나 칸이 가려진 경우에는 미표시로 추정하지 않는다.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from docagent.vision.normalize import detect_document_quad_detailed
from docagent.vision.quality import assess_capture

__all__ = ["verify_checkbox_groups"]


def _validate_groups(groups: list[dict], page_size_mm: tuple[float, float]) -> None:
    if (not isinstance(page_size_mm, (tuple, list)) or len(page_size_mm) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0
                   for x in page_size_mm)):
        raise ValueError("페이지 크기는 양수인 두 mm 값이어야 합니다.")
    if not isinstance(groups, list) or not groups:
        raise ValueError("검증할 선택 항목이 필요합니다.")
    group_ids: set[str] = set()
    boxes: list[list[float]] = []
    for group in groups:
        if (not isinstance(group, dict) or not isinstance(group.get("id"), str)
                or not group["id"].strip() or group["id"] in group_ids):
            raise ValueError("선택 항목 ID가 없거나 중복됩니다.")
        group_ids.add(group["id"])
        options = group.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("선택지가 필요합니다.")
        minimum, maximum = group.get("min_selected", 1), group.get("max_selected", 1)
        if (type(minimum) is not int or type(maximum) is not int
                or not 0 <= minimum <= maximum <= len(options) or maximum < 1):
            raise ValueError("선택 가능한 개수 범위가 잘못되었습니다.")
        option_ids: set[str] = set()
        for option in options:
            if (not isinstance(option, dict) or not isinstance(option.get("id"), str)
                    or not option["id"].strip() or option["id"] in option_ids):
                raise ValueError("선택지 ID가 없거나 중복됩니다.")
            option_ids.add(option["id"])
            box = option.get("box_mm")
            if (not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                raise ValueError("체크박스 좌표는 유한한 x, y, 너비, 높이 mm 값이어야 합니다.")
            x, y, width, height = box
            if (x < 0 or y < 0 or width <= 0 or height <= 0
                    or x + width > page_size_mm[0] or y + height > page_size_mm[1]):
                raise ValueError("체크박스가 페이지 범위를 벗어났습니다.")
            for other in boxes:
                if (min(x + width, other[0] + other[2]) > max(x, other[0])
                        and min(y + height, other[1] + other[3]) > max(y, other[1])):
                    raise ValueError("체크박스 영역이 서로 겹칩니다.")
            boxes.append(list(box))


def _valid_image(image: Any) -> bool:
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.size > 0
            and image.ndim in (2, 3) and min(image.shape[:2]) >= 8
            and image.shape[0] * image.shape[1] <= 64_000_000
            and (image.ndim == 2 or image.shape[2] == 3))


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def _uncertain(groups: list[dict], reason: str, alignment: dict | None = None) -> dict:
    return {"status": "uncertain", "reason": reason, "alignment": alignment or {},
            "groups": [{"id": group["id"], "state": "uncertain", "marked_option_ids": [],
                        "options": [{"id": option["id"], "state": "uncertain", "reason": reason}
                                    for option in group["options"]]} for group in groups]}


def _rect(box: list[float], shape: tuple, page_size: tuple) -> tuple[int, int, int, int]:
    x, y, width, height = box
    h, w = shape[:2]
    return (round(x / page_size[0] * w), round(y / page_size[1] * h),
            round((x + width) / page_size[0] * w), round((y + height) / page_size[1] * h))


def _inner(rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    inset = max(2, round(min(x1 - x0, y1 - y0) * .20))
    return x0 + inset, y0 + inset, x1 - inset, y1 - inset


def _patch(gray: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """칸 바깥 고리로 종이 밝기를 추정해 국소 조명 차이를 완화한다."""
    x0, y0, x1, y1 = rect
    h, w = gray.shape
    pad = max(4, round(min(x1 - x0, y1 - y0) * .5))
    ox0, oy0, ox1, oy1 = max(0, x0-pad), max(0, y0-pad), min(w, x1+pad), min(h, y1+pad)
    outer = gray[oy0:oy1, ox0:ox1]
    ring = np.ones(outer.shape, bool)
    ring[y0-oy0:y1-oy0, x0-ox0:x1-ox0] = False
    paper = float(np.percentile(outer[ring], 85)) if ring.any() else 0
    return gray[y0:y1, x0:x1].astype(np.float32) / max(paper, 1)


def _static_mask(shape: tuple, groups: list[dict], page_size: tuple) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    for group in groups:
        for option in group["options"]:
            x0, y0, x1, y1 = _rect(option["box_mm"], shape, page_size)
            pad = max(5, round(min(x1-x0, y1-y0) * .35))
            mask[max(0, y0-pad):y1+pad, max(0, x0-pad):x1+pad] = 0
    mask[:8] = mask[-8:] = 0
    mask[:, :8] = mask[:, -8:] = 0
    return mask


def _register(reference: np.ndarray, captured: np.ndarray, mask: np.ndarray) -> tuple:
    """인쇄 영역 특징점 정합과 인쇄 획 일치를 모두 요구한다."""
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=10, edgeThreshold=15)
    ref_points, ref_desc = detector.detectAndCompute(reference, mask)
    cap_points, cap_desc = detector.detectAndCompute(captured, None)
    if ref_desc is None or cap_desc is None or min(len(ref_points), len(cap_points)) < 16:
        return None, {"reason": "insufficient_features"}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(cap_desc, ref_desc, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .72 * pair[1].distance]
    # 동일 인쇄 특징점으로 여러 대응점이 몰려 정합이 과대평가되는 것을 막는다.
    unique = {}
    for match in sorted(good, key=lambda item: item.distance):
        unique.setdefault(match.trainIdx, match)
    good = list(unique.values())
    if len(good) < 16:
        return None, {"reason": "insufficient_matches", "matches": len(good)}
    src = np.float32([cap_points[m.queryIdx].pt for m in good])
    dst = np.float32([ref_points[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None, {"reason": "homography_failed"}
    inliers = inliers.ravel().astype(bool)
    count = int(inliers.sum())
    stats = {"matches": len(good), "inliers": count, "inlier_ratio": round(count / len(good), 4)}
    if count < 12 or count / len(good) < .45:
        return None, {**stats, "reason": "weak_registration"}
    h, w = reference.shape
    extent = np.ptp(dst[inliers], axis=0)
    hull = cv2.convexHull(dst[inliers])
    coverage = cv2.contourArea(hull) / (h*w)
    stats["feature_coverage"] = round(coverage, 4)
    if extent[0] < .35*w or extent[1] < .4*h or coverage < .12:
        return None, {**stats, "reason": "localized_registration"}
    corners = np.float32([[[0, 0], [captured.shape[1]-1, 0],
                           [captured.shape[1]-1, captured.shape[0]-1], [0, captured.shape[0]-1]]])
    mapped = cv2.perspectiveTransform(corners, matrix)[0]
    area = cv2.contourArea(mapped, oriented=True) / (h*w)
    if not cv2.isContourConvex(mapped) or not .75 <= area <= 1.3:
        return None, {**stats, "reason": "invalid_page_transform"}
    aligned = cv2.warpPerspective(captured, matrix, (w, h), borderValue=255)
    valid = cv2.warpPerspective(np.full(captured.shape, 255, np.uint8), matrix, (w, h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    stats["page_coverage"] = round(float((valid == 255).mean()), 4)
    if stats["page_coverage"] < .95:
        return None, {**stats, "reason": "page_cutoff"}
    # 일부 특징만 일치하는 다른 페이지를 같은 문서로 취급하지 않는다.
    reference_ink = cv2.adaptiveThreshold(reference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 12) > 0
    captured_ink = cv2.adaptiveThreshold(aligned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 12) > 0
    static = (mask > 0) & (valid == 255)
    reference_ink &= static
    captured_ink &= static
    kernel = np.ones((3, 3), np.uint8)
    ref_near = cv2.dilate(reference_ink.astype(np.uint8), kernel) > 0
    cap_near = cv2.dilate(captured_ink.astype(np.uint8), kernel) > 0
    missing = float((reference_ink & ~cap_near).sum() / max(1, reference_ink.sum()))
    extra = float((captured_ink & ~ref_near).sum() / max(1, captured_ink.sum()))
    stats.update(printed_missing_ratio=round(missing, 4), printed_extra_ratio=round(extra, 4))
    if missing > .06 or extra > .15:
        return None, {**stats, "reason": "printed_content_mismatch"}
    # 전역 비율은 한두 단어가 가려지거나 바뀌어도 작을 수 있다. 겹치는
    # 작은 창으로도 인쇄 내용의 누락을 검사해 다른 안내 원문을 읽지 않게 한다.
    local_missing = 0.0
    for top in range(0,h,64):
        for left in range(0,w,64):
            expected = reference_ink[top:top+128,left:left+128]
            absent = expected & ~cap_near[top:top+128,left:left+128]
            if expected.sum() >= 80 and absent.sum() >= 20:
                local_missing = max(local_missing, float(absent.sum()/expected.sum()))
    stats["max_local_printed_missing_ratio"] = round(local_missing, 4)
    if local_missing > .25:
        return None, {**stats, "reason": "local_printed_content_mismatch"}
    stats.update(reason="registered", method="orb_homography_and_printed_content")
    return (aligned, valid), stats


def _stray_mark(reference: np.ndarray, captured: np.ndarray, rect: tuple, boxes: list[tuple]) -> bool:
    """체크칸 주변에 새로 생긴 획을 놓쳐 정상 완료로 안내하지 않는다."""
    x0, y0, x1, y1 = rect
    pad = max(8, round(min(x1-x0, y1-y0)*.8))
    outer = max(0,x0-pad), max(0,y0-pad), min(reference.shape[1],x1+pad), min(reference.shape[0],y1+pad)
    before, after = _patch(reference, outer), _patch(captured, outer)
    old_near = cv2.dilate((before < .72).astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    fresh = (after < .72) & ~old_near
    for bx0, by0, bx1, by1 in boxes:
        # 다른 정상 선택지 안의 표시는 해당 선택지에서 따로 판독한다.
        left, top = max(0,bx0-2-outer[0]), max(0,by0-2-outer[1])
        right, bottom = min(fresh.shape[1],bx1+2-outer[0]), min(fresh.shape[0],by1+2-outer[1])
        if right > left and bottom > top:
            fresh[top:bottom,left:right] = False
    count, _, stats, _ = cv2.connectedComponentsWithStats(fresh.astype(np.uint8), connectivity=8)
    minimum = max(8, (x1-x0)*(y1-y0)*.015)
    return any(s[cv2.CC_STAT_AREA] >= minimum and
               max(s[cv2.CC_STAT_WIDTH],s[cv2.CC_STAT_HEIGHT]) >= min(x1-x0,y1-y0)*.25
               for s in stats[1:count])


def _observe(reference: np.ndarray, captured: np.ndarray, valid: np.ndarray, rect: tuple,
             boxes: list[tuple]) -> dict:
    x0, y0, x1, y1 = rect
    if min(x1-x0, y1-y0) < 12:
        return {"state": "uncertain", "reason": "checkbox_too_small"}
    if not np.all(valid[max(0, y0-3):y1+3, max(0, x0-3):x1+3] == 255):
        return {"state": "uncertain", "reason": "checkbox_cutoff"}
    if _stray_mark(reference, captured, rect, boxes):
        return {"state": "uncertain", "reason": "mark_outside_checkbox"}
    before, after = _patch(reference, rect), _patch(captured, rect)
    ix0, iy0, ix1, iy1 = _inner(rect)
    inner = (slice(iy0-y0, iy1-y0), slice(ix0-x0, ix1-x0))
    border = np.ones(before.shape, bool)
    border[inner] = False
    printed = (before < .72) & border
    if printed.sum() < min(before.shape):
        return {"state": "uncertain", "reason": "reference_checkbox_border_missing"}
    near_print = cv2.dilate((after < .72).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if (printed & ~near_print).sum() / printed.sum() > .1:
        return {"state": "uncertain", "reason": "checkbox_border_obscured"}
    old_near = cv2.dilate((before < .72).astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    margin_changed = ((before-after) > .035) & ~old_near & border
    # 테두리를 빼는 과정에서 칸 가장자리의 표시까지 사라지면 '미표시'를
    # 잘못 확정할 수 있다. 중앙이 비어 있어도 가장자리의 새 획은 재확인한다.
    margin_mark = int(margin_changed.sum()) >= max(5, round(before.size*.015))
    before, after = before[inner], after[inner]
    old_ink = float((before < .72).mean())
    dark = after < .72
    changed = (before - after) > .035
    dark_ratio = float(dark.mean())
    changed_ratio = float(changed.mean())
    measures = {"reference_ink_ratio": round(old_ink, 5), "ink_ratio": round(dark_ratio, 5),
                "changed_ratio": round(changed_ratio, 5)}
    if old_ink > .015:
        return {"state": "uncertain", "reason": "reference_not_blank", **measures}
    if dark_ratio > .55 or changed_ratio > .65:
        return {"state": "uncertain", "reason": "occluded_or_filled", **measures}
    if dark_ratio <= .01 and changed_ratio <= .02:
        if margin_mark:
            return {"state": "uncertain", "reason": "mark_at_checkbox_margin", **measures}
        return {"state": "unmarked", "reason": "no_visible_mark", **measures}
    # 한두 점, 희미한 획, 임계값 부근은 미표시로 통과시키지 않는다.
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    strokes = [s for s in stats[1:count] if s[cv2.CC_STAT_AREA] >= max(3, dark.size*.025)]
    spans = any(s[cv2.CC_STAT_WIDTH] >= after.shape[1]*.3
                and s[cv2.CC_STAT_HEIGHT] >= after.shape[0]*.3 for s in strokes)
    if dark_ratio >= .04 and spans:
        return {"state": "marked", "reason": "visible_stroke", **measures}
    return {"state": "uncertain", "reason": "faint_or_ambiguous_mark", **measures}


def verify_checkbox_groups(reference: np.ndarray, captured: np.ndarray, groups: list[dict], *,
                           page_size_mm: tuple[float, float] = (210., 297.),
                           capture_mode: str = "photo") -> dict:
    """전체 페이지를 정합해 선택지 표시 상태를 반환한다. 외부 통신은 없다.

    ``reference`` 는 검토된 빈 전체 페이지이며 ``box_mm`` 는 이 페이지 기준이다.
    ``photo`` 는 문서 네 모서리 검출을 요구하고 ``scan`` 은 전체 페이지 스캔용이다.
    모델 확률 대신 관측 상태와 진단값만 제공한다. ``multiple`` 은 복수 표시이며
    그것이 허용되는지는 호출자가 양식의 min/max_selected 규칙으로 확인한다.
    """
    _validate_groups(groups, page_size_mm)
    if capture_mode not in ("photo", "scan"):
        raise ValueError("촬영 모드는 photo 또는 scan이어야 합니다.")
    if not _valid_image(reference) or not _valid_image(captured):
        return _uncertain(groups, "invalid_image")
    if abs(reference.shape[1]/reference.shape[0] / (page_size_mm[0]/page_size_mm[1]) - 1) > .04:
        return _uncertain(groups, "reference_aspect_mismatch")
    reference_quality = assess_capture(reference, mode="scan")
    capture_quality = assess_capture(captured, mode=capture_mode)
    if not reference_quality.can_continue:
        return _uncertain(groups, "reference_quality", {"quality": reference_quality.to_dict()})
    if not capture_quality.can_continue:
        return _uncertain(groups, "capture_quality", {"quality": capture_quality.to_dict()})
    try:
        reference_gray = _gray(reference)
        scale = min(1., 1600/max(reference_gray.shape))
        size = (round(reference_gray.shape[1]*scale), round(reference_gray.shape[0]*scale))
        reference_gray = cv2.resize(reference_gray, size, interpolation=cv2.INTER_AREA)
        if capture_mode == "photo":
            boundary = detect_document_quad_detailed(captured)
            if boundary.fallback or boundary.confidence < .85:
                return _uncertain(groups, "boundary_unknown")
            target = np.float32([[0, 0], [size[0]-1, 0], [size[0]-1, size[1]-1], [0, size[1]-1]])
            transform = cv2.getPerspectiveTransform(boundary.quad.astype(np.float32), target)
            captured_gray = cv2.warpPerspective(_gray(captured), transform, size, borderValue=255)
        else:
            # 픽셀 크기가 같아도 아래 특징점/인쇄 내용 검증을 반드시 수행한다.
            captured_gray = cv2.resize(_gray(captured), size, interpolation=cv2.INTER_AREA)
        static = _static_mask(reference_gray.shape, groups, page_size_mm)
        registration, alignment = _register(reference_gray, captured_gray, static)
        if registration is None:
            return _uncertain(groups, "alignment_failed", alignment)
        aligned, valid = registration
        boxes = [_rect(option["box_mm"], reference_gray.shape, page_size_mm)
                 for group in groups for option in group["options"]]
        output = []
        for group in groups:
            options = [{"id": option["id"], **_observe(reference_gray, aligned, valid,
                         _rect(option["box_mm"], reference_gray.shape, page_size_mm), boxes)}
                       for option in group["options"]]
            marked = [option["id"] for option in options if option["state"] == "marked"]
            state = ("uncertain" if any(option["state"] == "uncertain" for option in options)
                     else "unmarked" if not marked else "single" if len(marked) == 1 else "multiple")
            output.append({"id": group["id"], "state": state, "marked_option_ids": marked, "options": options})
        uncertain = any(group["state"] == "uncertain" for group in output)
        return {"status": "uncertain" if uncertain else "ok",
                "reason": "ambiguous_checkbox" if uncertain else "observed",
                "alignment": alignment, "groups": output}
    except (cv2.error, OverflowError, FloatingPointError):
        return _uncertain(groups, "image_processing_failed")
