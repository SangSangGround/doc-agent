"""Original-photo, human-reviewed writable-field picker for navigation v7.5.

Never makes unreviewed Gemini candidates navigable. Pressing A means the OPERATOR
has checked the location and applicant responsibility against the printed paper.
Source coordinates stay in the original EXIF-upright source-image coordinate frame.
No OCR, no API calls, no user form answers collected.

Usage (from project root):
  python detection/field_review_v7_6.py --targets detection/refinement_output/document2/writable_output/navigation_targets.json --image detection/test_images/document2.jpg --list
  python detection/field_review_v7_6.py --targets ... --image ...

Navigation: N/P next/previous, +/- zoom out/in, Z wide view, click actual writing
spot, A approve only after verifying paper/role, C confirm conditional use,
X revoke, S save, Q save and quit. Existing approvals are preserved.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np

from navigation_adapter import (_verified_targets_file, _read_photo, _sha256,
                                _save_json, write_review)

# Never auto-clear the official-field exclusion inherited from writable_targets.
OFFICIAL_PREFIX = 'excluded_'
VALID_CLASSES = {'input_field', 'signature_field', 'check_box'}
WINDOW = 'Navigation Field Review v7.6 | N/P  CLICK  A approve  C conditional  X reject  Z zoom  S save  Q quit'
LEFT = (18, 75, 785, 693)
RIGHT = (825, 76, 345, 693)
CANVAS_W, CANVAS_H = 1190, 800


def _point(value):
    if isinstance(value, dict):
        value = [value.get('x'), value.get('y')]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        xy = np.array(value, dtype=float)
        return xy if np.isfinite(xy).all() else None
    except (TypeError, ValueError):
        return None


def _bbox(target, w, h):
    b = target.get('source_box_xyxy_px')
    if isinstance(b, (list, tuple)) and len(b) == 4:
        try:
            x1, y1, x2, y2 = map(float, b)
            if all(math.isfinite(z) for z in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                return np.array([max(0., x1), max(0., y1), min(float(w), x2), min(float(h), y2)])
        except (TypeError, ValueError):
            pass
    return None


def _unapproved_candidates(data, review, classes):
    by_id = {x['id']: x for x in review['fields']}
    return [(t, by_id[t['id']]) for t in data['targets']
            if t['class_name'] in classes and t['id'] in by_id]


def safe_approve(t, r, click_point, size, conditional=False):
    """Approval gate; position still needs independent physical calibration."""
    w, h = size
    if t.get('class_name') not in VALID_CLASSES:
        raise ValueError('Unknown field class')
    if str(t.get('role_gate', '')).startswith(OFFICIAL_PREFIX):
        raise ValueError('Institutional/official field: upstream role gate forbids automatic guidance')
    p = _point(click_point)
    if p is None or not (0 <= p[0] < w and 0 <= p[1] < h):
        raise ValueError('Click the TRUE writing/checkbox/signature location in the image first')
    b = _bbox(t, w, h)
    if t['class_name'] != 'signature_field' and b is not None:
        if not (b[0] - 6 <= p[0] <= b[2] + 6 and b[1] - 6 <= p[1] <= b[3] + 6):
            raise ValueError('Clicked outside candidate box. This field needs upstream re-labeling; do NOT force it.')
    needs_condition = str(t.get('role_gate', '')) == 'conditional_review'
    if needs_condition and not conditional:
        raise ValueError('Conditional field: press C ONLY after confirming the condition, then A')
    r['approved'] = True
    r['verified_on_paper'] = True  # Only press A after visually verifying actual paper.
    r['role'] = 'conditional' if needs_condition else 'applicant'
    r['condition_confirmed'] = bool(conditional) if needs_condition else False
    r['manual_target_px'] = {'x': round(float(p[0]), 3), 'y': round(float(p[1]), 3)}
    r['note'] = ('Operator clicked original-image target and explicitly attested on-paper '
                 'location, applicant role' + (' and applicability condition' if needs_condition else '')
                 + '; prototype guidance, physical accuracy not independently measured.')
    return r


def revoke(r):
    r['approved'] = False
    r['verified_on_paper'] = False
    r['role'] = 'unknown'
    r['condition_confirmed'] = False
    r['manual_target_px'] = None
    r['note'] = ''


def _print_current(t, r, i, count):
    print('\n' + '-'*70, flush=True)
    print(f'[{i+1}/{count}] {t["id"]} | {t["class_name"]} | {t.get("label", "")}')
    print('role_gate:', t.get('role_gate'), '| candidate_status:', t.get('candidate_status'))
    print('APPROVED:', r.get('approved'), '| manual_point:', r.get('manual_target_px'))
    if str(t.get('role_gate', '')).startswith(OFFICIAL_PREFIX):
        print('BLOCKED: possible official-only field. Will NOT approve here.')
    elif t.get('class_name') == 'signature_field':
        print('SIGNATURE: CLICK TRUE BLANK SIGNING SPACE, NOT THE PRINTED LABEL.')
    print('Confirm the location and role ON REAL PAPER before pressing A.', flush=True)


def _fit(canvas, rgb, rect):
    """Fit RGB image in canvas rectangle; return (x0,y0,scale)."""
    import cv2
    x, y, ww, hh = rect
    ih, iw = rgb.shape[:2]
    s = min(ww / iw, hh / ih)
    dw, dh = max(1, round(iw*s)), max(1, round(ih*s))
    ox = x+(ww-dw)//2
    oy = y+(hh-dh)//2
    canvas[oy:oy+dh, ox:ox+dw] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (dw, dh),
                                           interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_LINEAR)
    return ox, oy, dw/iw, dh/ih, dw, dh


def _roi_rect(target, w, h, wide=False, zoom=1.0):
    b = _bbox(target, w, h)
    if b is None:
        return 0, 0, w, h
    cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
    bw, bh = b[2]-b[0], b[3]-b[1]
    pad = 235 if target['class_name']=='signature_field' else 105
    ext_w = max(290 if not wide else 670, bw+2*pad)
    ext_h = max(240 if not wide else 600, bh+2*pad)
    ext_w *= zoom
    ext_h *= zoom
    x1 = max(0, int(cx-ext_w/2))
    y1 = max(0, int(cy-ext_h/2))
    x2 = min(w, int(cx+ext_w/2))
    y2 = min(h, int(cy+ext_h/2))
    if x2<=x1+5 or y2<=y1+5:
        return 0,0,w,h
    return x1,y1,x2,y2


def run_ui(data, review, source_rgb, out_review, classes):
    import cv2
    h, w = source_rgb.shape[:2]
    # Skip upstream-excluded official fields in the visual GUI.
    # CLI --list reports them for audit.
    items = [(t,r) for t,r in _unapproved_candidates(data, review, classes)
             if not str(t.get('role_gate','')).startswith(OFFICIAL_PREFIX)]
    if not items:
        raise RuntimeError('No candidates of chosen classes')
    index = 0
    zoom = 1.0
    wide = False
    pending_point = None
    pending_conditional = False
    click_map = {}
    notice = ''
    def on_mouse(event, x, y, flags, userdata):
        nonlocal pending_point, notice
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not click_map:
            return
        ox, oy, sx, sy, dw, dh, roi = click_map['map']
        if ox<=x<ox+dw and oy<=y<oy+dh:
            rx1, ry1, rx2, ry2 = roi
            xx = rx1+(x-ox)/sx
            yy = ry1+(y-oy)/sy
            pending_point={'x':round(min(w-1,max(0,xx)),2), 'y':round(min(h-1,max(0,yy)),2)}
            notice='Point chosen. Verify REAL PAPER, then press A to approve.'
            print('Clicked original px:',pending_point, flush=True)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CANVAS_W, CANVAS_H)
    cv2.setMouseCallback(WINDOW, on_mouse)
    _print_current(items[0][0], items[0][1], 0, len(items))
    try:
        while True:
            t,r=items[index]
            canvas=np.full((CANVAS_H,CANVAS_W,3),34,dtype=np.uint8)
            rgb=source_rgb
            cv2.putText(canvas,f'{t["id"]} | {t["class_name"]} | {index+1}/{len(items)}',
                        (22,36),cv2.FONT_HERSHEY_SIMPLEX,.83,(245,245,245),2,cv2.LINE_AA)
            state='APPROVED' if r.get('approved') else 'PENDING'
            color=(60,230,60) if r.get('approved') else (0,180,255)
            cv2.putText(canvas,state,(835,36),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
            rx1,ry1,rx2,ry2=_roi_rect(t,w,h,wide,zoom)
            roi=rgb[ry1:ry2,rx1:rx2].copy()
            ox,oy,sx,sy,dw,dh=_fit(canvas,roi,LEFT)
            click_map['map']=(ox,oy,sx,sy,dw,dh,(rx1,ry1,rx2,ry2))
            bb=_bbox(t,w,h)
            if bb is not None:
                x1=round(ox+(bb[0]-rx1)*sx); y1=round(oy+(bb[1]-ry1)*sy)
                x2=round(ox+(bb[2]-rx1)*sx); y2=round(oy+(bb[3]-ry1)*sy)
                cv2.rectangle(canvas,(x1,y1),(x2,y2),(50,50,250),2)
            p=pending_point if pending_point is not None else _point(r.get('manual_target_px'))
            if p is not None:
                if isinstance(p,dict): p=_point(p)
                px=round(ox+(p[0]-rx1)*sx); py=round(oy+(p[1]-ry1)*sy)
                if ox<=px<ox+dw and oy<=py<oy+dh:
                    cv2.drawMarker(canvas,(px,py),(0,255,120),cv2.MARKER_CROSS,30,2)
            tx,ty,tw,th=RIGHT
            _fit(canvas,rgb,RIGHT)
            # Full-page image coordinate visualization (same fit calculation).
            thumbox,thumboy,thumbsx,thumbsy,_,_=_fit(canvas,rgb,RIGHT)
            if bb is not None:
                cv2.rectangle(canvas,(round(thumbox+bb[0]*thumbsx),round(thumboy+bb[1]*thumbsy)),
                              (round(thumbox+bb[2]*thumbsx),round(thumboy+bb[3]*thumbsy)),(0,60,240),3)
            cv2.putText(canvas,'RED = detected candidate box; GREEN = clicked writing point',
                        (20,774),cv2.FONT_HERSHEY_SIMPLEX,.56,(200,225,235),1,cv2.LINE_AA)
            block = str(t.get('role_gate','')).startswith(OFFICIAL_PREFIX)
            line = 'BLOCKED: OFFICIAL FIELD' if block else ('CONDITIONAL: C then A' if t.get('role_gate')=='conditional_review' else 'Click point, then A (real-paper verified)')
            cv2.putText(canvas,line,(22,60),cv2.FONT_HERSHEY_SIMPLEX,.64,(50,100,245) if block else (90,225,255),2,cv2.LINE_AA)
            cv2.putText(canvas,'N/P next/prev  Z wide view  +/- zoom  A approve  X reject  S save  Q quit',
                        (22,748),cv2.FONT_HERSHEY_SIMPLEX,.61,(230,230,230),1,cv2.LINE_AA)
            if notice:
                cv2.putText(canvas,notice[:90],(22,723),cv2.FONT_HERSHEY_SIMPLEX,.5,(105,220,255),1,cv2.LINE_AA)
            cv2.imshow(WINDOW,canvas)
            key=cv2.waitKey(30) & 255
            if key==255: continue
            if key in (ord('q'),ord('Q'),27):
                break
            if key in (ord('n'),ord('N'),ord('p'),ord('P')):
                index=(index+(1 if key in (ord('n'),ord('N')) else -1))%len(items)
                pending_point=None
                pending_conditional=False
                notice=''
                _print_current(items[index][0],items[index][1],index,len(items))
            elif key in (ord('z'),ord('Z')):
                wide=not wide
                notice=f'WIDE VIEW: {wide}'
            elif key in (ord('+'),ord('=')):
                zoom=max(.55,zoom*.80)
            elif key in (ord('-'),ord('_')):
                zoom=min(3.0,zoom*1.25)
            elif key in (ord('c'),ord('C')):
                if t.get('role_gate')=='conditional_review':
                    pending_conditional=not pending_conditional
                    notice=f'Condition explicitly confirmed: {pending_conditional}. Press A after paper review.'
                else:
                    notice='C applies only to conditional fields.'
            elif key in (ord('x'),ord('X')):
                revoke(r)
                pending_point=None
                _save_json(out_review, review)
                notice='REVOKED. Saved.'
                print('REVOKED:',t['id'])
            elif key in (ord('s'),ord('S')):
                _save_json(out_review, review)
                notice='Review saved.'
                print('Saved:',out_review)
            elif key in (ord('a'),ord('A')):
                # User can retain previously chosen manual point without a new click.
                p=pending_point or r.get('manual_target_px')
                try:
                    safe_approve(t,r,p,(w,h),conditional=pending_conditional)
                except ValueError as exc:
                    notice='APPROVAL REFUSED - see terminal for reason'
                    print('APPROVAL REFUSED:',t['id'],str(exc),flush=True)
                else:
                    _save_json(out_review,review)
                    notice='APPROVED + SAVED.'
                    print('APPROVED:',t['id'],t.get('label'),r.get('manual_target_px'),flush=True)
                    # Stay on field so reviewer can inspect/save; N proceeds.
    finally:
        _save_json(out_review, review)
        cv2.destroyAllWindows()
    print('\nSaved review:',out_review)
    print('Approved:',[(r['id'],r.get('label')) for r in review['fields'] if r.get('approved')])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets',required=True,type=Path)
    ap.add_argument('--image',required=True,type=Path,help='Original untouched EXIF-upright photo used by writable_targets')
    ap.add_argument('--review',type=Path,help='Defaults to targets parent/navigation_review.json')
    ap.add_argument('--include-checkboxes',action='store_true',help='Also display existing checkbox candidates (off by default)')
    ap.add_argument('--list',action='store_true',help='Print candidates without opening GUI or changing review')
    args=ap.parse_args()
    data=_verified_targets_file(args.targets)
    review_path=args.review or args.targets.parent/'navigation_review.json'
    if not review_path.exists():
        if args.list:
            ap.error('Review file missing. Generate it first using navigation_adapter.py --init-review')
        write_review(args.targets,review_path)
        print('Created review:',review_path)
    review=json.loads(review_path.read_text(encoding='utf-8-sig'))
    if review.get('schema')!='navigation_review_v1' or review.get('source_targets_sha256')!=_sha256(args.targets):
        ap.error('Review file/targets mismatch. DO NOT overwrite old approvals; check paths/versions.')
    ids=[r['id'] for r in review.get('fields',[])]
    if len(ids)!=len(set(ids)) or set(ids)!={t['id'] for t in data['targets']}:
        ap.error('Review IDs mismatch source target IDs')
    classes={'input_field','signature_field'}
    if args.include_checkboxes:
        classes.add('check_box')
    items=_unapproved_candidates(data,review,classes)
    if args.list:
        print('REAL ORIGINAL PHOTO:',args.image)
        for t,r in items:
            print(f"{t['id']:4} {t['class_name']:<16} {str(t.get('label','')):<27} "
                  f"{'APPROVED' if r.get('approved') else 'pending':<8} "
                  f"{t.get('role_gate','')} | {t.get('candidate_status','')}")
        return
    source_rgb=_read_photo(args.image)
    if tuple(source_rgb.shape[:2][::-1]) != (data['image_width'],data['image_height']):
        ap.error('Original image size mismatch. DO NOT use a resized/annotated visualization.')
    run_ui(data,review,source_rgb,review_path,classes)


if __name__=='__main__':
    main()
