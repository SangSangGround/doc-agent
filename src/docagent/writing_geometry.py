"""Team hand-tracking detector geometry, from refine_detections.py and writable_targets.py.

The adapter uses these original pixel matchers on the upright image received
from iOS. Webcam, CLI, manual review files and GUI output are not needed.
"""

import math

import cv2
import numpy as np


def threshold(gray, c=9):
    k = min(31, min(gray.shape[:2]) if min(gray.shape[:2]) % 2 else min(gray.shape[:2]) - 1)
    if k < 3:
        return np.zeros_like(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, c)


def roi(gray, box, padx, pady):
    x1, y1, x2, y2 = box
    x0 = max(0, math.floor(x1 - padx))
    y0 = max(0, math.floor(y1 - pady))
    xe = min(gray.shape[1], math.ceil(x2 + padx))
    ye = min(gray.shape[0], math.ceil(y2 + pady))
    return gray[y0:ye, x0:xe], x0, y0


def center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - x * y
    return x * y / u if u else 0.0


def checkbox_candidates(gray, box):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    p = center(box)
    if min(bw, bh) < 5:
        return []
    g, ox, oy = roi(gray, box, 1.9 * bw, 0.95 * bh)
    if not g.size:
        return []
    out = []
    for c in (9, 5):
        b = threshold(g, c)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b)
        strokes = []
        for j, (x, y, w, h, a) in enumerate(stats[1:], 1):
            if not (0.43 * bh <= h <= 1.35 * bh and 2 <= w <= 0.53 * bh and a >= 5 and h >= 2 * w):
                continue
            z = (lab[y : y + h, x : x + w] == j).astype(np.uint8)
            mid = z[h // 4 : 3 * h // 4]
            k = max(1, round(w * 0.45))
            direction = (int(mid[:, :k].sum()) - int(mid[:, -k:].sum())) / max(1, int(mid.sum()))
            # A bracket has a vertical spine and horizontal caps pointing inward.
            if abs(direction) < 0.28:
                continue
            stroke_coverage = np.count_nonzero(z.sum(axis=1)) / h
            if stroke_coverage < 0.85:
                continue
            side = 1 if direction > 0 else -1
            caps = (z[: max(2, h // 4)], z[-max(2, h // 4) :])
            if any(np.count_nonzero(cap.sum(axis=0)) < max(2, round(w * 0.60)) for cap in caps):
                continue
            strokes.append((x, y, w, h, side, abs(direction)))
        for a in strokes:
            if a[4] != 1:
                continue
            for b2 in strokes:
                if b2[4] != -1:
                    continue
                x, y, w, h, _, da = a
                X, Y, W, H, _, db = b2
                hh = (h + H) / 2
                gap = X - (x + w)
                if not (
                    0.30 * hh <= gap <= 2.2 * hh
                    and abs(h - H) <= 0.30 * hh
                    and abs(y + h / 2 - Y - H / 2) <= 0.24 * hh
                ):
                    continue
                inside = b[max(y, Y) + 2 : min(y + h, Y + H) - 2, x + w : X]
                if inside.size == 0 or np.mean(inside > 0) > 0.11:
                    continue
                bb = np.array([ox + x, oy + min(y, Y), ox + X + W, oy + max(y + h, Y + H)], float)
                delta = center(bb) - p
                if abs(delta[0]) > 1.8 * bw or abs(delta[1]) > 0.85 * bh:
                    continue
                score = (
                    2.4
                    - 0.45 * abs(delta[0]) / bw
                    - 0.60 * abs(delta[1]) / bh
                    - 0.35 * abs(h - H) / hh
                    + 0.15 * (da + db)
                )
                out.append({"box": bb, "score": score})
        # A checked box can keep its outer frame. The iPhone reads its ink state separately.
        contours, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for j, ct in enumerate(contours):
            if hier[0, j, 2] < 0:
                continue
            x, y, w, h = cv2.boundingRect(ct)
            if not (0.5 * bh <= h <= 1.45 * bh and 0.55 <= w / h <= 1.65 and 0.45 * bw <= w <= 1.6 * bw):
                continue
            if cv2.contourArea(ct) / (w * h) < 0.65:
                continue
            poly = cv2.approxPolyDP(ct, 0.045 * cv2.arcLength(ct, True), True)
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            d = max(2, round(min(w, h) * 0.22))
            inside = b[y + d : y + h - d, x + d : x + w - d]
            if not inside.size:
                continue
            bb = np.array([ox + x, oy + y, ox + x + w, oy + y + h], float)
            delta = center(bb) - p
            if abs(delta[0]) > 1.8 * bw or abs(delta[1]) > 0.85 * bh:
                continue
            if np.mean(inside > 0) > 0.16:
                frame = b[y : y + h, x : x + w] > 0
                band = max(2, round(min(w, h) * 0.14))
                core = frame[band:-band, band:-band]
                sides = (
                    frame[:band].any(axis=0),
                    frame[-band:].any(axis=0),
                    frame[:, :band].any(axis=1),
                    frame[:, -band:].any(axis=1),
                )
                if (
                    abs(delta[0]) > 0.45 * bw
                    or abs(delta[1]) > 0.45 * bh
                    or not core.size
                    or core.mean() > 0.85
                    or min(side.mean() for side in sides) < 0.82
                ):
                    continue
            score = 2.25 - 0.45 * abs(delta[0]) / bw - 0.60 * abs(delta[1]) / bh
            out.append({"box": bb, "score": score})
    return out


def cell_candidates(gray, box):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    p = center(box)
    if min(bw, bh) < 8:
        return []
    g, ox, oy = roi(gray, box, 0.24 * bw, 0.72 * bh)
    if not g.size:
        return []
    out = []
    for c in (5, 8):
        b = threshold(g, c)
        b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            continue
        for j, ct in enumerate(contours):
            # Inner contour of a closed printed boundary, not a text blob.
            if hier[0, j, 3] < 0:
                continue
            x, y, w, h = cv2.boundingRect(ct)
            if x <= 1 or y <= 1 or x + w >= g.shape[1] - 1 or y + h >= g.shape[0] - 1:
                continue
            if not (0.65 * bw <= w <= 1.35 * bw and 0.45 * bh <= h <= 1.65 * bh):
                continue
            area = cv2.contourArea(ct)
            if area / (w * h) < 0.68:
                continue
            poly = cv2.approxPolyDP(ct, 0.025 * cv2.arcLength(ct, True), True)
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            # Reject writing/labels inside the candidate. Erode boundary before measuring.
            mask = np.zeros_like(b)
            cv2.fillConvexPoly(mask, poly[:, 0, :], 255)
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
            inside = b[mask > 0]
            if not inside.size or np.mean(inside > 0) > 0.075:
                continue
            bb = np.array([ox + x, oy + y, ox + x + w, oy + y + h], float)
            delta = center(bb) - p
            if abs(delta[0]) > 0.25 * bw or abs(delta[1]) > 0.48 * bh:
                continue
            if iou(bb, box) < 0.46:
                continue
            score = 1.2 + iou(bb, box) - 0.3 * abs(delta[1]) / bh
            out.append({"box": bb, "score": score})
    return out


def choose(candidates):
    distinct = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if not any(iou(c["box"], d["box"]) > 0.6 for d in distinct):
            distinct.append(c)
    if not distinct:
        return None, "no_supported_geometry"
    if distinct[0]["score"] < 1.35:
        return None, "weak_geometry"
    # Different plausible locations: keep the original proposal; no guess.
    if len(distinct) > 1 and distinct[0]["score"] - distinct[1]["score"] < 0.20:
        return None, "ambiguous_candidates"
    return distinct[0], "geometry_matched"


def _edge_peaks(mask, axis, box, ox, oy, tol=0.4):
    """Long straight strokes near original box sides; returns (position, fractional support)."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    if axis == "h":
        lo = max(0, int(round(x1 - ox + bw * 0.06)))
        hi = min(mask.shape[1], int(round(x2 - ox - bw * 0.06)))
        if hi - lo < 10:
            return []
        ys = np.arange(max(0, int(y1 - oy - tol * bh)), min(mask.shape[0], int(y2 - oy + tol * bh) + 1))
        if not ys.size:
            return []
        # 3-pixel tolerance to handle mild perspective and aliased lines
        traces = []
        for y in ys:
            band = mask[max(0, y - 1) : min(mask.shape[0], y + 2), lo:hi]
            traces.append(float(np.any(band > 0, axis=0).mean()) if band.size else 0.0)
        return [(float(y + oy), float(s)) for y, s in zip(ys, traces) if s >= 0.53]
    else:
        lo = max(0, int(round(y1 - oy + bh * 0.08)))
        hi = min(mask.shape[0], int(round(y2 - oy - bh * 0.08)))
        if hi - lo < 8:
            return []
        xs = np.arange(max(0, int(x1 - ox - tol * bw)), min(mask.shape[1], int(x2 - ox + tol * bw) + 1))
        if not xs.size:
            return []
        traces = []
        for x in xs:
            band = mask[lo:hi, max(0, x - 1) : min(mask.shape[1], x + 2)]
            traces.append(float(np.any(band > 0, axis=1).mean()) if band.size else 0.0)
        return [(float(x + ox), float(s)) for x, s in zip(xs, traces) if s >= 0.48]


def _near_peak(peaks, desired, maxdist):
    choices = [(pos, s) for pos, s in peaks if abs(pos - desired) <= maxdist]
    if not choices:
        return None
    return max(choices, key=lambda t: t[1] - 0.55 * abs(t[0] - desired) / max(1, maxdist))


def open_field_candidate(gray, box):
    """Detect partial table borders / underlines; avoid declaring a single line 'verified'.

    Strong: >=2 horizontal + >=1 vertical or >=3 sides.
    Medium: top+bottom only; weak: single baseline, useful solely for manual review.
    """
    x1, y1, x2, y2 = map(float, box)
    bw = x2 - x1
    bh = y2 - y1
    if bw < 30 or bh < 12:
        return None
    g, ox, oy = roi(gray, box, 0.16 * bw, 0.65 * bh)
    if g.size == 0:
        return None
    # Local thresholding only; strong horizontal/vertical morphology suppresses text strokes.
    b = threshold(g, 8)
    kh = max(13, round(bw * 0.23))
    kv = max(9, round(bh * 0.55))
    H = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((1, kh), np.uint8))
    V = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((kv, 1), np.uint8))
    hp = _edge_peaks(H, "h", box, ox, oy, tol=0.45)
    vp = _edge_peaks(V, "v", box, ox, oy, tol=0.40)
    top = _near_peak(hp, y1, 0.32 * bh)
    bottom = _near_peak(hp, y2, 0.32 * bh)
    left = _near_peak(vp, x1, 0.20 * bw)
    right = _near_peak(vp, x2, 0.20 * bw)
    if top and bottom and bottom[0] - top[0] < 0.5 * bh:
        top, bottom = None, None  # do not reuse the same printed line for both sides
    if left and right and right[0] - left[0] < 0.5 * bw:
        left, right = None, None
    sides = {k: v for k, v in [("top", top), ("bottom", bottom), ("left", left), ("right", right)] if v}
    n = len(sides)
    if n == 0:
        return None
    bbox = np.array(
        [
            left[0] if left else x1,
            top[0] if top else y1,
            right[0] if right else x2,
            bottom[0] if bottom else y2,
        ],
        float,
    )
    if bbox[2] - bbox[0] < 0.65 * bw or bbox[3] - bbox[1] < 0.5 * bh or iou(bbox, box) < 0.62:
        return None
    # A line at the bottom and a long line at the top can come from adjacent rows.
    if top and bottom and (abs(top[0] - y1) > 0.2 * bh or abs(bottom[0] - y2) > 0.2 * bh):
        return None
    if n >= 3 and (top or bottom) and (left or right):
        return {"box": bbox}
    return None


def _bbox_clamp(box, w, h):
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(t) for t in (x1, y1, x2, y2)):
        return None
    x1, x2 = min(max(0.0, x1), float(w)), min(max(0.0, x2), float(w))
    y1, y2 = min(max(0.0, y1), float(h)), min(max(0.0, y2), float(h))
    return [x1, y1, x2, y2] if x2 - x1 >= 5 and y2 - y1 >= 5 else None


def _ink_mask(gray):
    # A local adaptive threshold handles uneven illumination, but cannot identify
    # semantics of text, preprinted labels, seals, or handwritten contents.
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    side = min(g.shape[:2])
    blk = min(41, side if side % 2 else side - 1)
    if blk < 3:
        return np.zeros_like(g, dtype=np.uint8)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 12)


def _intersection(a, b):
    if a is None or b is None:
        return None
    c = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return c if c[2] > c[0] and c[3] > c[1] else None


def _expanded(box, pixels, width, height):
    return [
        max(0.0, box[0] - pixels),
        max(0.0, box[1] - pixels),
        min(float(width), box[2] + pixels),
        min(float(height), box[3] + pixels),
    ]


def _mask_box(mask, candidate_box, excluded_box):
    """Remove a protected area from the cropped search region (uint8, value 255)."""
    h, w = mask.shape
    x1, y1, x2, y2 = excluded_box
    ox, oy = candidate_box[:2]
    x0 = max(0, int(math.floor(x1 - ox)))
    x3 = min(w, int(math.ceil(x2 - ox)))
    y0 = max(0, int(math.floor(y1 - oy)))
    y3 = min(h, int(math.ceil(y2 - oy)))
    if x3 > x0 and y3 > y0:
        mask[y0:y3, x0:x3] = 255


def blank_patch(gray, bbox, exclusions=()):
    """Find low-ink INNER region, treating nearby field labels as hard obstacles.

    This only tests contrast/stroke occupancy, not the semantic fact that the
    area may legally be written in. Absence of ink alone is not proof.
    """
    h, w = gray.shape
    b = _bbox_clamp(bbox, w, h)
    if b is None:
        return None
    x1, y1, x2, y2 = b
    bw = x2 - x1
    bh = y2 - y1
    if bw < 27 or bh < 17:
        return None
    mx = max(4, min(22, 0.043 * bw))
    my = max(4, min(16, 0.14 * bh))
    x0 = int(math.ceil(x1 + mx))
    xx = int(math.floor(x2 - mx))
    y0 = int(math.ceil(y1 + my))
    yy = int(math.floor(y2 - my))
    if xx - x0 < 20 or yy - y0 < 12:
        return None
    crop = gray[y0:yy, x0:xx]
    mask = _ink_mask(crop)
    # Inflate printed strokes and table borders so a point is not placed on text.
    dil = min(10, max(3, round(min(crop.shape[:2]) * 0.09)))
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1))
    mask = cv2.dilate(mask, element, iterations=1)
    for box in exclusions:
        if box is None:
            continue
        if _intersection(b, box) is not None:
            _mask_box(mask, [x0, y0, xx, yy], _expanded(box, max(4, dil), w, h))
    H, W = mask.shape
    integr = cv2.integral((mask > 0).astype(np.uint8), sdepth=cv2.CV_32S)
    best = None
    # Try larger regions first. Never trade away all safety margin for one clean pixel.
    for wf in (0.72, 0.58, 0.44, 0.32):
        for hf in (0.58, 0.44, 0.30):
            pw = max(20, round(W * wf))
            ph = max(12, round(H * hf))
            if pw > W or ph > H:
                continue
            sx = max(3, pw // 6)
            sy = max(3, ph // 5)
            xs = sorted(set([*range(0, W - pw + 1, sx), W - pw]))
            ys = sorted(set([*range(0, H - ph + 1, sy), H - ph]))
            for py in ys:
                for px in xs:
                    black = (
                        int(integr[py + ph, px + pw])
                        - int(integr[py, px + pw])
                        - int(integr[py + ph, px])
                        + int(integr[py, px])
                    ) / (pw * ph)
                    if black > 0.018:
                        continue
                    score = 1.25 * (pw * ph) / (W * H) - 3.0 * black - 0.012 * px / max(1, W)
                    if best is None or score > best[0]:
                        best = (score, black, [x0 + px, y0 + py, x0 + px + pw, y0 + py + ph])
    if best is None:
        return None
    return np.asarray(best[2], dtype=float)
