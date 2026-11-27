# -*- coding: utf-8 -*-
"""Developer-side automatic paper boundary candidate and reviewed-manifest builder.

Does not edit existing project code, approve fields, or infer legal eligibility.
Designed to fail closed when white paper cannot be separated from background.

Preview:
  python auto_paper_corners_safe.py --image detection/test_images/document5.jpg
After independent review:
  python auto_paper_corners_safe.py --image detection/test_images/document5.jpg --build
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def detect_page(image_rgb):
    import cv2
    import numpy as np
    height, width = image_rgb.shape[:2]
    factor = min(1.0, 1100.0 / max(height, width))
    dw, dh = round(width * factor), round(height * factor)
    small = cv2.resize(image_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = cv2.medianBlur(hsv[:, :, 1], 5)
    yy, xx = max(8, dh // 35), max(8, dw // 35)
    border = np.concatenate([sat[:, :xx].ravel(), sat[:, -xx:].ravel(),
                             sat[:yy, :].ravel(), sat[-yy:, :].ravel()])
    central = sat[round(dh * .25):round(dh * .7), round(dw * .25):round(dw * .75)]
    center_s = float(np.median(central))
    background_s = float(np.percentile(border, 70))
    contrast = background_s - center_s
    if contrast < 35:
        raise RuntimeError(
            f'[AUTO CORNERS STOP] 종이/배경 색상 차이가 부족합니다 (S-gap={contrast:.1f}). '
            '배경을 바꾸거나 자동 스캔 재시도가 필요합니다. 임의 좌표는 생성하지 않습니다.')
    threshold = int(np.clip(center_s + 0.42 * contrast, 30, 100))
    binary = np.uint8(sat < threshold) * 255
    ks = max(7, (round(min(dw, dh) * .016) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('[AUTO CORNERS STOP] 종이 경계 후보가 없습니다.')
    contour = max(contours, key=cv2.contourArea)
    page_area_ratio = cv2.contourArea(contour) / float(dw * dh)
    if not .35 < page_area_ratio < .965:
        raise RuntimeError(f'[AUTO CORNERS STOP] 후보 면적 비정상: {page_area_ratio:.3f}')
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for frac in (.008, .01, .012, .016, .02, .025):
        approx = cv2.approxPolyDP(hull, frac * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(float)
            break
    if quad is None:
        raise RuntimeError('[AUTO CORNERS STOP] 종이의 사각형 윤곽을 확신할 수 없습니다.')
    # Work with TL, TR, BR, BL in source-image coordinate frame.
    order = sorted(quad.tolist(), key=lambda p: p[1])
    top = sorted(order[:2], key=lambda p: p[0])
    bottom = sorted(order[2:], key=lambda p: p[0])
    pts = np.asarray([top[0], top[1], bottom[1], bottom[0]], dtype=np.float64)
    poly = pts.astype(np.float32).reshape(-1, 1, 2)
    quad_area = abs(cv2.contourArea(poly)) / float(dw * dh)
    if quad_area < .35 or quad_area > .98:
        raise RuntimeError('[AUTO CORNERS STOP] 사각형 비율이 비정상입니다.')
    # Paper corners close to the image edge may be slightly clipped.
    min_margin = min(float(np.min(pts[:, 0])), float(dw - 1 - np.max(pts[:, 0])),
                     float(np.min(pts[:, 1])), float(dh - 1 - np.max(pts[:, 1])))
    clipped_warning = min_margin < max(3, min(dw, dh) * .008)
    original = pts / factor
    return original, {'saturation_gap': round(contrast, 1), 'area_ratio': round(page_area_ratio, 3),
                      'threshold_s': threshold, 'near_frame_edge': bool(clipped_warning)}, binary


def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image', type=Path, required=True, help='Original untouched EXIF-upright photo')
    ap.add_argument('--build', action='store_true', help='Build manifest only from existing human-approved review')
    ap.add_argument('--targets', type=Path, help='Default: refinement_output/<image-stem>/writable_output/navigation_targets.json')
    ap.add_argument('--review', type=Path, help='Default: same folder/navigation_review.json')
    ap.add_argument('--out', type=Path, help='Default: same folder/navigation_manifest.json')
    args = ap.parse_args()
    project = Path(__file__).resolve().parent
    image = args.image.resolve()
    if not image.is_file():
        raise SystemExit(f'[STOP] Photo missing: {image}')
    with Image.open(image) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    if len(rgb.shape) != 3 or min(rgb.shape[:2]) < 300:
        raise SystemExit('[STOP] Photo size/shape invalid')
    try:
        corners, stats, _ = detect_page(rgb)
    except RuntimeError as ex:
        raise SystemExit(str(ex))
    folder = project / 'detection' / 'refinement_output' / image.stem / 'writable_output'
    folder.mkdir(parents=True, exist_ok=True)
    # Draw at original resolution, with scaled text sizes.
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    coords = corners.round().astype(np.int32)
    cv2.polylines(vis, [coords], isClosed=True, color=(0, 220, 30), thickness=max(3, vis.shape[0] // 800))
    for idx, (x, y) in enumerate(coords.tolist()):
        cv2.circle(vis, (x, y), max(8, vis.shape[0] // 400), (0, 0, 240), -1)
        cv2.putText(vis, ('TL', 'TR', 'BR', 'BL')[idx], (x + 12, max(24, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(.7, vis.shape[0] / 2800), (0, 0, 220), 2)
    preview = folder / 'auto_page_corners_preview.jpg'
    ok, encoded = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise SystemExit('[STOP] Failed to encode preview')
    encoded.tofile(str(preview))
    print('[AUTO CORNERS] image=', str(image), flush=True)
    print('[PAGE CANDIDATE] TL TR BR BL:', coords.tolist(), flush=True)
    print('[METRICS]', stats, flush=True)
    print('[PREVIEW]', preview, flush=True)
    if stats['near_frame_edge']:
        print('[CAUTION] 종이의 일부가 이미지 경계에 매우 가까움. 좌표 정밀도가 떨어질 수 있습니다.', flush=True)
    print('[REVIEW FIRST] 사진과 오버레이를 대조하고 실물로 위치 정확성을 확인해야 합니다.', flush=True)
    if not args.build:
        return
    targets = (args.targets or folder / 'navigation_targets.json').resolve()
    review = (args.review or folder / 'navigation_review.json').resolve()
    out = (args.out or folder / 'navigation_manifest.json').resolve()
    for p in (targets, review):
        if not p.is_file():
            raise SystemExit(f'[BUILD BLOCKED] 검수 파일이 없습니다: {p}')
    sys.path.insert(0, str(project / 'detection'))
    from navigation_adapter import make_manifest, _json_read
    data = _json_read(review)
    count = sum(bool(f.get('approved') and f.get('verified_on_paper')) for f in data.get('fields', []))
    if count == 0:
        raise SystemExit('[BUILD BLOCKED] 검수 승인 항목 0개. field_review_v7_6.py에서 실제 승인 후 실행하세요.')
    manifest = make_manifest(targets, review, image, corners.tolist(), (210.0, 297.0), out)
    print('[MANIFEST] approved=', manifest['approved_fields_count'], 'skipped=', manifest['skipped_approval_counts'])
    print('[SAVED]', out)
    if manifest['approved_fields_count'] == 0:
        print('[BLOCKED] 승인된 손가락 안내 항목이 없습니다. 음성 추적을 시작하지 마세요.')


if __name__ == '__main__':
    main()
