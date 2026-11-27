# -*- coding: utf-8 -*-
r"""STEP 4 v3 — paper-form *ink trace* comparison (NOT handwriting/content validation).

Read-only standalone MVP. Keeps STEP 1–3 Python and JSON untouched.

From hand-tracking project root:
    .\.venv_scan\Scripts\python.exe -u .\step4_verify_writing_mvp.py --demo
    .\.venv_scan\Scripts\python.exe -u .\step4_verify_writing_mvp.py \
        --after .\detection\test_images\document1_after.jpg --type record

The reference is the blank, reviewed original photo (document1.jpg).
Accepted output is deliberately conservative: INK_CANDIDATE, NO_CLEAR_TRACE,
RETAKE_NEEDED, REVIEW_REQUIRED and NOT_CHECKED, not 'completed' or 'correct'.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 840, 1188  # 4 px per physical A4 millimeter
PX_PER_MM = 4
BLOCKED = {'F033'}
TYPE_NAMES = {'record': '기록 신청', 'change': '정정·변경 신청', 'delete': '삭제 신청'}


def imread_unicode(path: Path):
    # Support Korean directory names on Windows/OpenCV
    raw = np.fromfile(str(path), dtype=np.uint8)
    im = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f'사진을 열 수 없습니다: {path}')
    return im


def imwrite_unicode(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f'이미지 저장 실패: {path}')
    path.write_bytes(data.tobytes())


def order_corners(points):
    p = np.asarray(points, dtype=np.float32).reshape((4, 2))
    s = p.sum(axis=1)
    dif = np.diff(p, axis=1).reshape(-1)
    return np.array([p[np.argmin(s)], p[np.argmin(dif)],
                     p[np.argmax(s)], p[np.argmax(dif)]], np.float32)


def detect_page(image):
    """Detect a light, low-saturation *entire paper page* with contrasting background.

    Ambiguous images are rejected. No invented page corners, no fixed crop.
    """
    h, w = image.shape[:2]
    small_scale = min(1.0, 1400.0 / max(h, w))
    reduced = cv2.resize(image, None, fx=small_scale, fy=small_scale) if small_scale < 1 else image
    hsv = cv2.cvtColor(reduced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 58), (179, 75, 255))
    s = max(5, int(11 * small_scale))
    kernel = np.ones((s, s), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(found, key=cv2.contourArea, reverse=True):
        frac = cv2.contourArea(contour) / (reduced.shape[0] * reduced.shape[1])
        if not (0.20 <= frac <= 0.985):
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(cv2.convexHull(contour), 0.025 * perimeter, True)
        if len(approx) != 4:
            continue
        corners = order_corners(approx[:, 0, :] / small_scale)
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        if min(edges) < 0.22 * min(w, h):
            continue
        # Fully visible page requested: corners stuck to multiple image borders often mean cropping.
        border = np.sum((corners[:, 0] < 3) | (corners[:, 0] > w - 4) |
                        (corners[:, 1] < 3) | (corners[:, 1] > h - 4))
        if border > 1:
            continue
        return corners, frac
    raise ValueError('RETAKE_NEEDED: 종이 네 모서리를 안정적으로 찾지 못했습니다. 종이 전체가 배경과 구분되게 다시 촬영해 주세요.')


def detect_reference_page_edges(image):
    """Independent reference-page detector; accept only a broad page-like quadrilateral.

    This fallback only defines the A4 coordinate system for the blank source photo.
    Never guess corners for the user's after photo (use print-feature alignment).
    """
    h, w = image.shape[:2]
    small_scale = min(1., 1400. / max(h, w))
    reduced = cv2.resize(image, None, fx=small_scale, fy=small_scale) if small_scale < 1. else image
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(clahe, (5, 5), 0)
    h0, w0 = gray.shape[:2]
    candidates = []
    for low, high in [(35, 105), (55, 155)]:
        ed = cv2.Canny(gray, low, high)
        k = max(3, int(round(5*small_scale)))
        closed = cv2.morphologyEx(ed, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            area = cv2.contourArea(contour) / (w0*h0)
            if not .50 <= area <= .98:
                continue
            hull = cv2.convexHull(contour)
            per = cv2.arcLength(hull, True)
            for epsilon in [.015, .025, .04]:
                p = cv2.approxPolyDP(hull, per*epsilon, True)
                if len(p) != 4:
                    continue
                ordered = order_corners(p[:, 0, :] / small_scale)
                lengths = np.linalg.norm(np.roll(ordered, -1, axis=0) - ordered, axis=1)
                if np.min(lengths) < .35*min(w,h):
                    continue
                corners_area = cv2.contourArea(ordered)
                frame_ratio = corners_area / float(w*h)
                if not .48 <= frame_ratio <= .995:
                    continue
                inside = (ordered[:,0]>=-3)&(ordered[:,0]<w+3)&(ordered[:,1]>=-3)&(ordered[:,1]<h+3)
                if int(np.sum(inside)) < 3:
                    continue
                candidates.append((frame_ratio, ordered))
    if not candidates:
        raise ValueError('경계선 기반 탐색에서도 문서 전체의 사각형을 찾지 못했습니다')
    # Largest plausible paper quadrilateral
    ratio, corners = max(candidates, key=lambda item:item[0])
    return corners, ratio


def page_warp(image, corners):
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    Hm = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, Hm, (W, H)), Hm


def register_print(before, after):
    """Align detected page against the same pre-printed form; fail when evidence is weak."""
    bg = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    ag = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=10)
    bk, bd = detector.detectAndCompute(bg, None)
    ak, ad = detector.detectAndCompute(ag, None)
    if bd is None or ad is None or len(bk) < 35 or len(ak) < 35:
        raise ValueError('RETAKE_NEEDED: 인쇄된 문서의 특징점을 충분히 찾지 못했습니다.')
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(ad, bd, k=2)
    matches = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(matches) < 35:
        raise ValueError('RETAKE_NEEDED: 작성 전후 사진이 같은 문서인지 확인할 수 없습니다.')
    src = np.float32([ak[m.queryIdx].pt for m in matches]).reshape((-1, 1, 2))
    dst = np.float32([bk[m.trainIdx].pt for m in matches]).reshape((-1, 1, 2))
    mat, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.5)
    if mat is None or inliers is None:
        raise ValueError('RETAKE_NEEDED: 사진 위치 정렬에 실패했습니다.')
    n = int(inliers.sum())
    frac = n / max(1, len(matches))
    if n < 28 or frac < 0.38:
        raise ValueError(f'RETAKE_NEEDED: 문서 정렬 신뢰도가 낮습니다 (일치점 {n}개).')
    # Do not accept a mapping that moves the page wildly.
    page = np.float32([[20, 20], [W - 21, 20], [W - 21, H - 21], [20, H - 21]]).reshape((-1, 1, 2))
    mapped = cv2.perspectiveTransform(page, mat).reshape(-1, 2)
    if not np.isfinite(mapped).all() or np.max(np.abs(mapped - page.reshape(-1, 2))) > 0.17 * H:
        raise ValueError('RETAKE_NEEDED: 두 촬영본의 원근 차이가 너무 큽니다.')
    registered = cv2.warpPerspective(after, mat, (W, H), borderValue=(255, 255, 255))
    return registered, n, round(frac, 3)


def align_direct_by_print(before_page, after_img):
    """Safely register a new photo by printed content when page borders are hard to find.

    Coordinates map photographed after pixels directly into the reviewed A4
    reference. Require many RANSAC inliers spread widely across the document,
    plausible whole-page geometry, and near-complete page coverage.
    """
    gray_ref = cv2.cvtColor(before_page, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
    detector = cv2.ORB_create(nfeatures=7500, fastThreshold=10)
    rk, rd = detector.detectAndCompute(gray_ref, None)
    ak, ad = detector.detectAndCompute(gray_after, None)
    if rd is None or ad is None or len(rk) < 80 or len(ak) < 80:
        raise ValueError('RETAKE_NEEDED: 문서 내부의 글자와 선이 충분히 선명하지 않습니다.')
    candidate_pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(ad, rd, k=2)
    good = [a for pair in candidate_pairs if len(pair) == 2
            for a, b in [pair] if a.distance < 0.78 * b.distance]
    if len(good) < 80:
        raise ValueError('RETAKE_NEEDED: 작성 전후 사진에서 같은 인쇄 양식을 충분히 확인하지 못했습니다.')
    after_points = np.float32([ak[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    reference_points = np.float32([rk[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.findHomography(after_points, reference_points, cv2.RANSAC, 5.0)
    if transform is None or inlier_mask is None:
        raise ValueError('RETAKE_NEEDED: 문서 자동 정렬에 실패했습니다.')
    inliers = inlier_mask.reshape(-1).astype(bool)
    count = int(inliers.sum())
    ratio = count / len(good)
    if count < 90 or ratio < 0.48:
        raise ValueError('RETAKE_NEEDED: 인쇄 내용 일치도가 부족합니다.')
    ref_inliers = reference_points.reshape(-1, 2)[inliers]
    xspread = float(np.ptp(ref_inliers[:, 0]))
    yspread = float(np.ptp(ref_inliers[:, 1]))
    if xspread < W * 0.42 or yspread < H * 0.44:
        raise ValueError('RETAKE_NEEDED: 문서 일부만 확인되어 전체 위치를 안전하게 맞출 수 없습니다.')
    try:
        inverse = np.linalg.inv(transform)
        after_corners = cv2.perspectiveTransform(
            np.float32([[[0, 0]], [[W-1, 0]], [[W-1, H-1]], [[0, H-1]]]),
            inverse).reshape(4, 2)
    except (np.linalg.LinAlgError, cv2.error):
        raise ValueError('RETAKE_NEEDED: 문서 원근 보정이 불안정합니다.')
    ah, aw = after_img.shape[:2]
    if not np.isfinite(after_corners).all():
        raise ValueError('RETAKE_NEEDED: 문서 네 모서리 좌표를 검증할 수 없습니다.')
    # Allow a tiny edge overhang caused by paper curling or camera framing.
    margin_x, margin_y = aw * 0.025, ah * 0.025
    if (np.any(after_corners[:, 0] < -margin_x) or
        np.any(after_corners[:, 0] > aw - 1 + margin_x) or
        np.any(after_corners[:, 1] < -margin_y) or
        np.any(after_corners[:, 1] > ah - 1 + margin_y)):
        raise ValueError('RETAKE_NEEDED: 문서가 사진 밖으로 크게 잘려 전체 필드를 확인할 수 없습니다.')
    quad_area = abs(cv2.contourArea(after_corners.astype(np.float32)))
    area_frac = quad_area / (aw * ah)
    if not 0.20 <= area_frac <= 1.12:
        raise ValueError('RETAKE_NEEDED: 문서 전체 크기와 방향을 신뢰하기 어렵습니다.')
    # Bound local reprojection residual (small line/corner warp is fine).
    aligned_points = cv2.perspectiveTransform(after_points[inliers].reshape(-1,1,2), transform).reshape(-1,2)
    residuals = np.linalg.norm(aligned_points - ref_inliers, axis=1)
    if float(np.median(residuals)) > 3.2:
        raise ValueError('RETAKE_NEEDED: 문서 정렬 오차가 큽니다.')
    aligned = cv2.warpPerspective(after_img, transform, (W, H), borderValue=(255,255,255))
    print(f'[PAGE FALLBACK PASS] 인쇄 글자/표 특징점으로 직접 정렬: '
          f'matched={count} ratio={ratio:.3f} page_area={area_frac:.3f}', flush=True)
    return aligned, count, round(ratio, 3)


def parse_target(raw: str):
    nums = re.findall(r'[-+]?\d+(?:\.\d+)?', raw or '')
    if len(nums) != 2:
        return None
    xy = (float(nums[0]), float(nums[1]))
    if not (0 <= xy[0] <= 210 and 0 <= xy[1] <= 297):
        return None
    return xy


def load_fields(path: Path, application: str):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('STEP3 CSV 파일이 비었습니다.')
    fields = {}
    for row in rows:
        if row['application_type'] != application:
            continue
        fid = row['field_id']
        if fid not in fields:
            fields[fid] = dict(row)
    if not fields:
        raise ValueError(f'신청 유형 {application}에 대응하는 작성란이 없습니다.')
    return list(fields.values())


def roi_size(field):
    kind, name = field['class_name'], field['field_name']
    if kind == 'check_box':
        return (6.0, 6.0)
    if kind == 'signature_field':
        return (19.0, 8.0)
    if any(x in name for x in ('연도', '신청 월', '신청 일')):
        return (9.0, 6.0)
    return (19.0, 6.0)


def make_ink_mask(before, after, print_exclusion_kernel=5):
    bg = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    ag = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    # Equalize smooth illumination differences, retain printed/stroke structure.
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    bg, ag = cl.apply(bg), cl.apply(ag)
    bsoft = cv2.GaussianBlur(bg, (3, 3), 0)
    asoft = cv2.GaussianBlur(ag, (3, 3), 0)
    binbefore = cv2.adaptiveThreshold(bsoft, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 25, 10)
    binafter = cv2.adaptiveThreshold(asoft, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 10)
    existing = cv2.dilate(binbefore, np.ones((print_exclusion_kernel, print_exclusion_kernel), np.uint8))
    newly_dark = (bsoft.astype(np.int16) - asoft.astype(np.int16)) > 28
    mask = ((binafter > 0) & (existing == 0) & newly_dark).astype(np.uint8) * 255
    # Single-camera salt-and-pepper noise is not a handwriting stroke.
    n, comps, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            clean[comps == i] = 255
    return clean


def inspect_field(row, mask, all_centers):
    fid, kind = row['field_id'], row['class_name']
    base = dict(field_id=fid, field_name=row['field_name'], class_name=kind,
                target_mm=row['target_mm'], new_ink_pixels=0, new_ink_ratio=0.0,
                result='NOT_CHECKED', spoken_hint='안내 좌표가 검수되지 않아 작성 여부를 판단하지 않았습니다.')
    if row['safe_to_navigate'].upper() != 'YES' or fid in BLOCKED:
        return base
    xy = parse_target(row['target_mm'])
    if xy is None:
        return base
    dx, dy = roi_size(row)
    cx, cy = [int(v * PX_PER_MM) for v in xy]
    x0 = max(0, int(cx - dx * PX_PER_MM / 2))
    x1 = min(W, int(cx + dx * PX_PER_MM / 2))
    y0 = max(0, int(cy - dy * PX_PER_MM / 2))
    y1 = min(H, int(cy + dy * PX_PER_MM / 2))
    if x1 <= x0 or y1 <= y0:
        return base
    area = mask[y0:y1, x0:x1] > 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    # Prevent wrongly attributing ink of a neighboring field to this field.
    ds = (xx - cx) ** 2 + (yy - cy) ** 2
    for other_id, other_xy in all_centers.items():
        if other_id == fid:
            continue
        ox, oy = (int(other_xy[0] * PX_PER_MM), int(other_xy[1] * PX_PER_MM))
        other_d2 = (xx - ox) ** 2 + (yy - oy) ** 2
        area &= (ds <= other_d2)
    count = int(np.count_nonzero(area))
    valid = max(1, (x1 - x0) * (y1 - y0))
    ratio = count / valid
    # conservative proposal thresholds; NOT clinical/legal validation
    min_pixels = 12 if kind == 'check_box' else 24 if kind == 'signature_field' else 22
    enough = count >= min_pixels and ratio >= (0.015 if kind == 'check_box' else 0.008)
    base['new_ink_pixels'] = count
    base['new_ink_ratio'] = round(ratio, 4)
    base['result'] = 'INK_CANDIDATE' if enough else 'NO_CLEAR_TRACE'
    base['spoken_hint'] = (
        f"{row['field_name']} 항목에 새 필기 흔적으로 보이는 변화가 있습니다. 정확한 작성 내용은 확인하지 못했습니다."
        if enough else
        f"{row['field_name']} 항목에서 뚜렷한 새 필기 흔적을 찾지 못했습니다. 누락으로 단정하지 않으며 필요하면 다시 촬영해 주세요."
    )
    return base


def verify(before_path, after_path, csv_path, application, out):
    print(f'[STEP4] type={application} before={before_path} after={after_path}', flush=True)
    before_img, after_img = imread_unicode(before_path), imread_unicode(after_path)
    print(f'[FILES] before_bytes={before_path.stat().st_size} before_size={before_img.shape[1]}x{before_img.shape[0]} '
          f'after_bytes={after_path.stat().st_size} after_size={after_img.shape[1]}x{after_img.shape[0]}', flush=True)
    try:
        before_corners, _ = detect_page(before_img)
        print('[BEFORE PAGE PASS] 원본 빈 양식의 테두리를 찾았습니다.', flush=True)
    except ValueError:
        print('[BEFORE PAGE FALLBACK] 원본 빈 양식 테두리 1차 검출 실패. 다른 방법으로 다시 찾습니다.', flush=True)
        try:
            before_corners, _ = detect_reference_page_edges(before_img)
            print('[BEFORE PAGE FALLBACK PASS] 원본 빈 양식의 테두리를 찾았습니다.', flush=True)
        except ValueError as exc:
            raise ValueError(f'BEFORE_IMAGE_PROBLEM: 작성 전 원본 사진(document1.jpg)의 테두리를 확인하지 못했습니다. '
                             f'작성 후 사진을 다시 찍을 필요는 없습니다. 원본을 확인해야 합니다. 세부: {exc}') from exc
    before_page, _ = page_warp(before_img, before_corners)
    try:
        after_corners, _ = detect_page(after_img)
    except ValueError as e:
        print('[PAGE FALLBACK] 종이 외곽선 검출 실패. 재촬영 대신 인쇄 특징점으로 자동 정렬을 시도합니다.', flush=True)
        aligned_after, inliers, match_ratio = align_direct_by_print(before_page, after_img)
    else:
        after_page, _ = page_warp(after_img, after_corners)
        try:
            aligned_after, inliers, match_ratio = register_print(before_page, after_page)
        except ValueError:
            print('[PAGE FALLBACK] 외곽선 기반 정렬 실패. 인쇄 특징점 기반 자동 정렬을 시도합니다.', flush=True)
            aligned_after, inliers, match_ratio = align_direct_by_print(before_page, after_img)
    blur = cv2.Laplacian(cv2.cvtColor(aligned_after, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if blur < 35:
        raise ValueError(f'RETAKE_NEEDED: 촬영 이미지가 흐립니다 (선명도 {blur:.0f}).')
    rows = load_fields(csv_path, application)
    centers = {r['field_id']: v for r in rows if (v := parse_target(r['target_mm'])) is not None}
    # Two-scale evidence: reject a candidate if it vanishes after more
    # conservative masking of pre-existing printed text/lines. This avoids
    # calling registration artifacts near signatures and field borders "ink".
    mask = make_ink_mask(before_page, aligned_after, print_exclusion_kernel=5)
    strict_mask = make_ink_mask(before_page, aligned_after, print_exclusion_kernel=15)
    result = [inspect_field(r, mask, centers) for r in rows]
    strict_result = [inspect_field(r, strict_mask, centers) for r in rows]
    for tentative, strict in zip(result, strict_result):
        tentative['strict_new_ink_pixels'] = strict['new_ink_pixels']
        if tentative['result'] == 'INK_CANDIDATE' and strict['result'] != 'INK_CANDIDATE':
            tentative['result'] = 'REVIEW_REQUIRED'
            tentative['spoken_hint'] = (
                f"{tentative['field_name']} 항목은 촬영 차이 또는 인쇄 문구 때문에 "
                '필기 여부를 신뢰성 있게 구분하지 못했습니다. 작성됐다고 판정하지 않습니다.'
            )
    out.mkdir(exist_ok=True, parents=True)
    overlay = aligned_after.copy()
    palette = {'INK_CANDIDATE': (35, 125, 35), 'REVIEW_REQUIRED': (35, 170, 220), 'NO_CLEAR_TRACE': (0, 170, 240)}
    for row, item in zip(rows, result):
        if item['result'] == 'NOT_CHECKED':
            continue
        xy = centers.get(row['field_id'])
        if xy is None:
            continue
        dx, dy = roi_size(row)
        cx, cy = [int(v * PX_PER_MM) for v in xy]
        p0 = (max(0, int(cx - dx * PX_PER_MM / 2)), max(0, int(cy - dy * PX_PER_MM / 2)))
        p1 = (min(W - 1, int(cx + dx * PX_PER_MM / 2)), min(H - 1, int(cy + dy * PX_PER_MM / 2)))
        cv2.rectangle(overlay, p0, p1, palette[item['result']], 2)
        cv2.putText(overlay, row['field_id'], (p0[0], max(15, p0[1] - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, palette[item['result']], 1)
    imwrite_unicode(out / 'document1_after_aligned.jpg', aligned_after)
    imwrite_unicode(out / 'document1_new_ink_mask.png', mask)
    imwrite_unicode(out / 'document1_verification_overlay.jpg', overlay)
    with (out / 'document1_step4_fields.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0].keys()))
        writer.writeheader()
        writer.writerows(result)
    counts = {key: sum(r['result'] == key for r in result)
              for key in ('INK_CANDIDATE', 'REVIEW_REQUIRED', 'NO_CLEAR_TRACE', 'NOT_CHECKED')}
    summary = {'document': 'document1', 'application_type': application,
               'warning': '필기 흔적 후보 검사만 수행. 실제 작성 완료/내용 정확성/서명 진위는 판단 불가.',
               'registration_inliers': inliers, 'registration_match_ratio': match_ratio,
               'after_blur_metric': round(float(blur), 1), 'counts': counts,
               'fields': result}
    (out / 'document1_step4_results.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[ALIGN PASS] matched_print_points={inliers} match_ratio={match_ratio} blur={blur:.0f}', flush=True)
    print(f'[RESULT] {counts}', flush=True)
    print(f'[REPORT] {out / "document1_step4_results.json"}', flush=True)
    print(f'[OVERLAY] {out / "document1_verification_overlay.jpg"}', flush=True)
    print('[IMPORTANT] 후보 검출은 작성 완료/작성 내용 정확성을 의미하지 않습니다.', flush=True)
    return summary


def create_demo(before_path, csv_path, out):
    """Create a clearly marked *synthetic* after picture for smoke testing only."""
    image = imread_unicode(before_path)
    corners, _ = detect_page(image)
    page, Hm = page_warp(image, corners)
    rows = load_fields(csv_path, 'record')
    safe = {r['field_id']: r for r in rows if r['safe_to_navigate'].upper() == 'YES'}
    # Insert obvious fake marks on two of the real, reviewed coordinates.
    for fid in ('F001', 'F010'):
        r = safe.get(fid)
        if r is None:
            raise ValueError(f'데모 대상 검수 항목 누락: {fid}')
        xy = parse_target(r['target_mm'])
        if not xy:
            raise ValueError('검수 위치 없음')
        x, y = (int(xy[0] * PX_PER_MM), int(xy[1] * PX_PER_MM))
        if fid == 'F001':
            pts = np.array([[x - 5, y], [x - 1, y + 5], [x + 8, y - 8]], np.int32)
            cv2.polylines(page, [pts], False, (15, 15, 15), 3, cv2.LINE_AA)
        else:
            pts = np.array([[x - 26, y + 2], [x - 20, y - 6], [x - 12, y + 3],
                            [x - 3, y - 5], [x + 5, y + 4], [x + 17, y - 5]], np.int32)
            cv2.polylines(page, [pts], False, (15, 15, 15), 3, cv2.LINE_AA)
    inv = np.linalg.inv(Hm)
    synth = cv2.warpPerspective(page, inv, (image.shape[1], image.shape[0]))
    # Only replace pixels inside the printed page, preserve the original background.
    page_mask = np.zeros(page.shape[:2], np.uint8) + 255
    original_mask = cv2.warpPerspective(page_mask, inv, (image.shape[1], image.shape[0]))
    mask = original_mask > 240
    image[mask] = synth[mask]
    dst = out / 'DEMO_ONLY_synthetic_after.jpg'
    imwrite_unicode(dst, image)
    print('[DEMO] 합성된 가짜 작성 흔적을 사용합니다. 실제 문서 검증 결과가 아닙니다.', flush=True)
    return dst


def main():
    print(f'[STEP4 V3 VERIFIED RUNNER] {Path(__file__).resolve()}', flush=True)
    p = argparse.ArgumentParser(description='STEP 4: cautious ink-trace comparison against a blank paper form')
    p.add_argument('--before', type=Path, default=Path('detection/test_images/document1.jpg'))
    p.add_argument('--after', type=Path, help='작성 후 같은 인쇄 서식을 찍은 새 사진')
    p.add_argument('--type', choices=tuple(TYPE_NAMES), default='record')
    p.add_argument('--fields', type=Path, default=Path('step3_output/document1_step3_fields.csv'))
    p.add_argument('--out', type=Path, default=Path('step4_output'))
    p.add_argument('--demo', action='store_true', help='명확히 표시된 가상 펜 표시로 시스템 작동 검사')
    p.add_argument('--version-only', action='store_true', help='실행 중인 파일 버전만 확인')
    args = p.parse_args()
    if args.version_only:
        print('[VERSION OK] STEP4 V3 independent file; no photo loaded', flush=True)
        return
    try:
        if not args.before.is_file() or not args.fields.is_file():
            raise FileNotFoundError('작성 전 사진 또는 STEP3 CSV가 없습니다. 경로를 확인해 주세요.')
        if args.demo:
            args.after = create_demo(args.before, args.fields, args.out)
            args.type = 'record'
        if args.after is None:
            p.error('작성 후 사진을 --after로 지정하거나 먼저 --demo를 실행해 주세요.')
        if not args.after.is_file():
            raise FileNotFoundError(f'작성 후 사진을 찾을 수 없습니다: {args.after}')
        verify(args.before, args.after, args.fields, args.type, args.out)
    except (ValueError, FileNotFoundError, cv2.error) as ex:
        print(f'[RETAKE/ERROR] {ex}', file=sys.stderr, flush=True)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
