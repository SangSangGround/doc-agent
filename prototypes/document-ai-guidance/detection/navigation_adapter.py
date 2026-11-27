"""Build an explicitly reviewed navigation manifest from writable_targets v2.

This is a developer-only approval/calibration workflow. NO cloud calls.
The source image and live camera frame use DIFFERENT pixel coordinates. Each
must be calibrated to the SAME document plane (millimeters).

Run from project root, e.g.:
  python detection/navigation_adapter.py --targets detection/refinement_output/document1/writable_output/navigation_targets.json --init-review
  python detection/navigation_adapter.py --targets detection/refinement_output/document1/writable_output/navigation_targets.json --build --pick-corners

Review must contain explicit per-field approval, on-paper check and role.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def _json_read(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def _verified_targets_file(path):
    data = _json_read(path)
    if data.get('algorithm_version') != 'writable_targets_v2':
        raise ValueError('Only writable_targets_v2 navigation_targets.json is supported')
    if data.get('preview_rescaled') is not False:
        raise ValueError('Preview or unspecified preview_rescaled flag. Use ORIGINAL photo coordinates only.')
    if not str(data.get('coordinate_system', '')).startswith('EXIF-upright'):
        raise ValueError('Unsupported coordinate system')
    if not isinstance(data.get('targets'), list) or not data['targets']:
        raise ValueError('No targets')
    ids = [t.get('id') for t in data['targets']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate target ids')
    return data


def _read_photo(path: Path):
    from PIL import Image, ImageOps
    import numpy as np
    if not path.is_file():
        raise FileNotFoundError(f'Original source photo not found: {path}')
    with Image.open(path) as im:
        arr = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    return arr


def _resolve_source(data, override, targets_path):
    if override:
        return override.expanduser().resolve()
    p = Path(str(data.get('source_image', '')))
    if p.is_file():
        return p.resolve()
    # The generated JSON typically records an absolute Windows path. If it is
    # unavailable after moving directories, require an explicit --image.
    raise FileNotFoundError('Source photo path in JSON cannot be found. Specify --image PATH.')


def write_review(targets_path: Path, output: Path):
    data = _verified_targets_file(targets_path)
    if output.exists():
        raise FileExistsError(f'Review already exists, will not overwrite: {output}')
    obj = {
        'schema': 'navigation_review_v1',
        'source_targets_sha256': _sha256(targets_path),
        'instructions': 'Approve only after checking the point on paper and who should fill it. For signatures/weak candidates supply manual_target_px and a note.',
        'fields': [
            {'id': t['id'], 'label': t.get('label', ''), 'class_name': t.get('class_name', ''),
             'candidate_status': t.get('candidate_status', ''),
             'suggested_target_px': t.get('target_px'),
             'approved': False, 'verified_on_paper': False,
             'role': 'unknown', 'condition_confirmed': False,
             'manual_target_px': None, 'note': ''}
            for t in data['targets']
        ],
    }
    _save_json(output, obj)
    return output


def _corners_validate(points, width, height):
    import numpy as np
    import cv2
    a = np.asarray(points, dtype=np.float32)
    if a.shape != (4, 2) or not np.isfinite(a).all():
        raise ValueError('Four finite (x,y) paper corners required: TL TR BR BL')
    if (a[:, 0] < 0).any() or (a[:, 0] >= width).any() or (a[:, 1] < 0).any() or (a[:, 1] >= height).any():
        raise ValueError('Paper corners must be inside original image')
    if not cv2.isContourConvex(a) or cv2.contourArea(a) < 0.07 * width * height:
        raise ValueError('Paper corners are nonconvex or cover too little image area. Order: TL TR BR BL')
    return a


def _choose_corners(img):
    import numpy as np
    import cv2
    h, w = img.shape[:2]
    scale = min(1., 750 / h, 750 / w)
    dw, dh = max(1, round(w * scale)), max(1, round(h * scale))
    sx, sy = dw / w, dh / h
    base = cv2.cvtColor(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    selected = []
    title = 'Reference paper: TL TR BR BL | ENTER confirm | R reset | ESC cancel'
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(selected) < 4:
            selected.append([x, y])
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            view = base.copy()
            for i, (x, y) in enumerate(selected):
                cv2.circle(view, (x, y), 6, (0, 180, 255), -1)
                cv2.putText(view, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
            if len(selected) >= 2:
                cv2.polylines(view, [np.asarray(selected, dtype=np.int32)], len(selected) == 4, (0, 220, 0), 2)
            cv2.imshow(title, view)
            key = cv2.waitKey(30) & 0xff
            if key in (10, 13) and len(selected) == 4:
                return [[x / sx, y / sy] for x, y in selected]
            if key in (ord('r'), ord('R')):
                selected.clear()
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError('Reference corner selection cancelled')
    finally:
        cv2.destroyWindow(title)


def _to_xy(value):
    if isinstance(value, dict):
        value = [value.get('x'), value.get('y')]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        out = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(v) for v in out) else None


def _perspective_points(points, H):
    import cv2
    import numpy as np
    a = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, H.astype(np.float64)).reshape(-1, 2)


def make_manifest(targets_path, review_path, image_path, paper_corners, paper_size, output):
    import numpy as np
    import cv2
    data = _verified_targets_file(targets_path)
    review = _json_read(review_path)
    if review.get('schema') != 'navigation_review_v1' or review.get('source_targets_sha256') != _sha256(targets_path):
        raise ValueError('Review source checksum mismatch. Generate a fresh review for this navigation_targets.json.')
    photo = _read_photo(image_path)
    h, w = photo.shape[:2]
    if (w, h) != (int(data['image_width']), int(data['image_height'])):
        raise ValueError(f'Source photo size {w}x{h} differs from targets {data["image_width"]}x{data["image_height"]}')
    corners = _corners_validate(paper_corners, w, h)
    width_mm, height_mm = paper_size
    if min(width_mm, height_mm) <= 0:
        raise ValueError('Paper size must be positive')
    world = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], np.float32)
    H = cv2.getPerspectiveTransform(corners, world)
    check = _perspective_points(corners, H)
    if np.max(np.linalg.norm(check - world, axis=1)) > 1e-2:
        raise ValueError('Invalid projective transform')
    refs = {r.get('id'): r for r in review.get('fields', [])}
    if len(refs) != len(review.get('fields', [])):
        raise ValueError('Duplicate review ids')
    records, skipped = [], Counter()
    for t in data['targets']:
        fid = t['id']
        a = refs.get(fid, {})
        if not a.get('approved'):
            skipped['not_approved'] += 1
            continue
        if not a.get('verified_on_paper'):
            skipped['not_verified_on_paper'] += 1
            continue
        role = a.get('role')
        gate = t.get('role_gate', '')
        if role not in ('applicant', 'conditional') or gate.startswith('excluded_'):
            skipped['role_not_cleared'] += 1
            continue
        if (role == 'conditional' or gate == 'conditional_review') and not a.get('condition_confirmed'):
            skipped['condition_not_confirmed'] += 1
            continue
        manual = _to_xy(a.get('manual_target_px')) if a.get('manual_target_px') is not None else None
        original = _to_xy(t.get('target_px'))
        if a.get('manual_target_px') is not None and manual is None:
            skipped['invalid_manual_target'] += 1
            continue
        # Signature positions and heuristic/weak/missing candidates must be explicitly clicked.
        status = str(t.get('candidate_status', ''))
        extra_review = status != 'candidate_needs_validation' or t.get('class_name') == 'signature_field'
        if extra_review and (manual is None or not str(a.get('note', '')).strip()):
            skipped['manual_point_and_note_required'] += 1
            continue
        pt = manual or original
        if pt is None or not (0 <= pt[0] < w and 0 <= pt[1] < h):
            skipped['missing_or_outside_source_image'] += 1
            continue
        # Reject a manual point outside the source field's broad region unless
        # a deliberate signature exception is documented: a location elsewhere
        # can mean that a different field was mistakenly clicked.
        box = t.get('source_box_xyxy_px')
        if box and len(box) == 4 and t.get('class_name') != 'signature_field':
            pad = 6.0
            if not (box[0] - pad <= pt[0] <= box[2] + pad and box[1] - pad <= pt[1] <= box[3] + pad):
                skipped['point_outside_source_box'] += 1
                continue
        m = _perspective_points([pt], H)[0]
        if not all(math.isfinite(z) for z in m) or not (2 <= m[0] <= width_mm - 2 and 2 <= m[1] <= height_mm - 2):
            skipped['outside_paper_or_edge'] += 1
            continue
        records.append({'id': fid, 'class_name': t['class_name'], 'label': t.get('label', ''),
                        'target_source_px': {'x': round(pt[0], 3), 'y': round(pt[1], 3)},
                        'target_mm': {'x': round(float(m[0]), 3), 'y': round(float(m[1]), 3)},
                        'source_candidate_status': status, 'role': role,
                        'approval_method': 'manually_adjusted' if manual else 'candidate_checked_on_paper',
                        'note': a.get('note', '')})
    manifest = {
        'schema': 'a4_navigation_manifest_v1',
        'coordinate_frame': 'paper plane, x-right y-down, units mm, origin top-left',
        'paper_width_mm': width_mm, 'paper_height_mm': height_mm,
        'source_image': str(image_path), 'source_image_sha256': _sha256(image_path),
        'source_targets_sha256': _sha256(targets_path),
        'reference_corners_px_TL_TR_BR_BL': np.round(corners, 3).tolist(),
        'homography_reference_px_to_mm': H.tolist(),
        'approved_fields_count': len(records), 'skipped_approval_counts': dict(skipped),
        'fields': records,
        'safety_note': ('Developer-reviewed targets only. Homography assumes flat A4 page, fixed camera/paper, '
                        'finger tip on page; parallax and physical error remain unmeasured. Not validated for independent use.')
    }
    _save_json(output, manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', required=True, type=Path, help='writable_output/navigation_targets.json')
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--init-review', action='store_true')
    action.add_argument('--build', action='store_true')
    ap.add_argument('--review', type=Path, help='Default: same folder as targets / navigation_review.json')
    ap.add_argument('--image', type=Path, help='Original source photo; default: source_image from targets JSON')
    ap.add_argument('--out', type=Path, help='Default: same folder / navigation_manifest.json')
    ap.add_argument('--pick-corners', action='store_true', help='Click 4 ORIGINAL PHOTO paper corners TL TR BR BL')
    ap.add_argument('--corners', type=float, nargs=8, metavar=('TLX', 'TLY', 'TRX', 'TRY', 'BRX', 'BRY', 'BLX', 'BLY'))
    ap.add_argument('--paper-mm', type=float, nargs=2, default=(210.0, 297.0), metavar=('WIDTH', 'HEIGHT'))
    args = ap.parse_args()
    targets = args.targets.expanduser().resolve()
    review = (args.review or targets.parent / 'navigation_review.json').expanduser().resolve()
    if args.init_review:
        print('Review template created:', write_review(targets, review))
        print('No items approved. Open JSON, inspect on the REAL PAPER, and explicitly set approved/role/verified_on_paper.')
        return
    data = _verified_targets_file(targets)
    image = _resolve_source(data, args.image, targets)
    photo = _read_photo(image)
    if args.pick_corners and args.corners:
        ap.error('Choose either --pick-corners or --corners')
    if not args.pick_corners and not args.corners:
        ap.error('--build requires --pick-corners or --corners')
    corners = _choose_corners(photo) if args.pick_corners else [args.corners[i:i + 2] for i in range(0, 8, 2)]
    output = (args.out or targets.parent / 'navigation_manifest.json').expanduser().resolve()
    manifest = make_manifest(targets, review, image, corners, args.paper_mm, output)
    print('Approved fields:', manifest['approved_fields_count'])
    print('Skipped reasons:', manifest['skipped_approval_counts'])
    print('Saved:', output)
    if not manifest['fields']:
        print('No approved fields. Tracking v6 will refuse to guide. Approve at least one checked item.')
    else:
        for v in manifest['fields']:
            print(f"  {v['id']} {v['label']} -> ({v['target_mm']['x']:.1f}, {v['target_mm']['y']:.1f}) mm")


if __name__ == '__main__':
    main()
