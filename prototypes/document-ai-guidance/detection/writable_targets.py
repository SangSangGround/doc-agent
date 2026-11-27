r"""Writable-region and navigation-target *candidates* V2 (post-processor).

Does NOT change refine_detections.py, does NOT call Gemini, and never grants
navigation permission. Uses conservative image-ink heuristics (NOT OCR).
Requires: opencv-python, numpy, Pillow (already used by refinement v2).

V2 safety changes: no automatic signature location guessing, glyph-distance-aware
blank-region search, occlusion masks for other fields, and spatial collisions.
All suggestions remain unverified, never navigation-allowed.

Examples (run from hand-tracking project root):
  .\.venv_scan\Scripts\python.exe -u detection\writable_targets.py --image detection\test_images\document1.jpg
  .\.venv_scan\Scripts\python.exe -u detection\writable_targets.py --image detection\test_images\document2.jpg

Inputs: detection/refinement_output/<image stem>/refined_detection_result.json
Outputs: detection/refinement_output/<image stem>/writable_output/*
Coordinates: original EXIF-upright image pixels (NOT paper millimeters).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent


def _bbox_clamp(box, w, h):
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(t) for t in (x1, y1, x2, y2)):
        return None
    x1, x2 = min(max(0., x1), float(w)), min(max(0., x2), float(w))
    y1, y2 = min(max(0., y1), float(h)), min(max(0., y2), float(h))
    return [x1, y1, x2, y2] if x2 - x1 >= 5 and y2 - y1 >= 5 else None


def _center(box):
    return {'x': round((box[0]+box[2])/2,2), 'y': round((box[1]+box[3])/2,2)}


def _point_inside(point, w, h):
    return isinstance(point,dict) and all(k in point for k in ('x','y')) and all(
        isinstance(point[k],(int,float)) and math.isfinite(float(point[k])) for k in ('x','y')
    ) and 0 <= point['x'] < w and 0 <= point['y'] < h


def _relative_status(field):
    """Do NOT infer legal right to fill from a text label."""
    manual = field.get('manual_review') or {}
    if manual.get('status') == 'not_applicable_or_official':
        return 'excluded_manual'
    role = (field.get('role_assessment') or {}).get('role', 'unknown')
    compact = ''.join(str(field.get('label','')).split())
    official_terms = ('접수번호','접수일','관계기관통보','담당공무원','처리기관','처리기간','접수담당')
    if role == 'official_possible' or any(t in compact for t in official_terms):
        return 'excluded_official_possible'
    if role == 'conditional':
        return 'conditional_review'
    return 'role_unverified'


def _ink_mask(gray):
    # A local adaptive threshold handles uneven illumination, but cannot identify
    # semantics of text, preprinted labels, seals, or handwritten contents.
    g = cv2.GaussianBlur(gray, (3,3), 0)
    side = min(g.shape[:2]); blk=min(41, side if side%2 else side-1)
    if blk < 3:
        return np.zeros_like(g, dtype=np.uint8)
    return cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV,blk,12)




def _intersection(a, b):
    if a is None or b is None:
        return None
    c=[max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])]
    return c if c[2]>c[0] and c[3]>c[1] else None


def _area(a):
    return max(0.,a[2]-a[0])*max(0.,a[3]-a[1]) if a is not None else 0.


def _overlap_fraction(a,b):
    z=_intersection(a,b)
    return _area(z)/max(1.,min(_area(a),_area(b))) if z is not None else 0.


def _expanded(box, pixels, width, height):
    return [max(0.,box[0]-pixels), max(0.,box[1]-pixels),
            min(float(width),box[2]+pixels), min(float(height),box[3]+pixels)]


def _mask_box(mask, candidate_box, excluded_box):
    """Remove a protected area from the cropped search region (uint8, value 255)."""
    h,w=mask.shape
    x1,y1,x2,y2=excluded_box
    ox,oy=candidate_box[:2]
    x0=max(0,int(math.floor(x1-ox))); x3=min(w,int(math.ceil(x2-ox)))
    y0=max(0,int(math.floor(y1-oy))); y3=min(h,int(math.ceil(y2-oy)))
    if x3>x0 and y3>y0:mask[y0:y3,x0:x3]=255


def _blank_patch_v2(gray, bbox, exclusions=(), require_refined=True):
    """Find low-ink INNER region, treating nearby field labels as hard obstacles.

    This only tests contrast/stroke occupancy, not the semantic fact that the
    area may legally be written in. Absence of ink alone is not proof.
    """
    h,w=gray.shape
    b=_bbox_clamp(bbox,w,h)
    if b is None:return None
    x1,y1,x2,y2=b;bw=x2-x1;bh=y2-y1
    if bw<27 or bh<17:return None
    mx=max(4,min(22,.043*bw));my=max(4,min(16,.14*bh))
    x0=int(math.ceil(x1+mx));xx=int(math.floor(x2-mx))
    y0=int(math.ceil(y1+my));yy=int(math.floor(y2-my))
    if xx-x0<20 or yy-y0<12:return None
    crop=gray[y0:yy,x0:xx]
    mask=_ink_mask(crop)
    # Inflate printed strokes and table borders so a point is not placed on text.
    dil=min(10,max(3,round(min(crop.shape[:2])*.09)))
    element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1))
    mask=cv2.dilate(mask,element,iterations=1)
    for box in exclusions:
        if box is None:continue
        if _intersection(b,box) is not None:
            _mask_box(mask,[x0,y0,xx,yy],_expanded(box,max(4,dil),w,h))
    H,W=mask.shape
    integr=cv2.integral((mask>0).astype(np.uint8),sdepth=cv2.CV_32S)
    best=None
    # Try larger regions first. Never trade away all safety margin for one clean pixel.
    for wf in (.72,.58,.44,.32):
        for hf in (.58,.44,.30):
            pw=max(20,round(W*wf));ph=max(12,round(H*hf))
            if pw>W or ph>H:continue
            sx=max(3,pw//6);sy=max(3,ph//5)
            xs=sorted(set([*range(0,W-pw+1,sx), W-pw]))
            ys=sorted(set([*range(0,H-ph+1,sy), H-ph]))
            for py in ys:
                for px in xs:
                    black=(int(integr[py+ph,px+pw])-int(integr[py,px+pw])
                           -int(integr[py+ph,px])+int(integr[py,px]))/(pw*ph)
                    if black>.018:continue
                    score=1.25*(pw*ph)/(W*H)-3.0*black-.012*px/max(1,W)
                    if best is None or score>best[0]:
                        best=(score,black,[x0+px,y0+py,x0+px+pw,y0+py+ph])
    if best is None:return None
    _,ink,patch=best
    return {'region':[round(float(x),2) for x in patch],
            'point':_center(patch),'ink_fraction_after_dilation':round(float(ink),5),
            'method':'low_ink_subregion_v2'}


def _protected_boxes(rows, field):
    """Keep a proposed writing patch away from annotated checkboxes and label boxes.

    For signatures, the Gemini box usually encloses the PRINTED '(서명 또는 인)'
    rather than the area for handwriting. It is protected, not used as a target.
    """
    result=[]
    for other in rows:
        if other is field:continue
        if other.get('class_name') not in ('signature_field','check_box'):continue
        b=other.get('refined_bbox_xyxy_px') or other.get('original_bbox_xyxy_px') or other.get('bbox_xyxy_px')
        if b is not None:result.append(b)
    return result


def _new_record(field,w,h):
    refined=_bbox_clamp(field.get('refined_bbox_xyxy_px'),w,h)
    original=_bbox_clamp(field.get('original_bbox_xyxy_px') or field.get('bbox_xyxy_px'),w,h)
    status=(field.get('refinement') or {}).get('status','unknown')
    box=refined or original
    return {'id':field.get('id','?'), 'class_name':field.get('class_name','unknown'),
            'label':field.get('label',''), 'role_gate':_relative_status(field),
            'source_refinement_status':status,
            'region_source':'refined_box' if refined is not None else 'gemini_box_unverified',
            'source_box_xyxy_px':box,'candidate_method':None,
            'writable_region_xyxy_px':None,'target_px':None,
            'candidate_status':'unresolved', 'requires_manual_review':True,
            'navigation_allowed':False,'navigation_verified':False,
            'coordinate_system':'EXIF-upright original image pixels',
            'warning':'Candidate only. NOT OCR-validated, physical-mm measured, or authorized for guidance.'}


def _clear_target(t,reason):
    """If ambiguity remains, do not preserve a misleading point in navigation JSON."""
    t['candidate_status']=reason
    t['conflict_resolution']='withheld_until_review'
    t['target_px']=None
    t['writable_region_xyxy_px']=None
    t['candidate_method']=None


def _resolve_target_conflicts(targets):
    """Check overlap between proposals; never sacrifice a validated *candidate* box.

    Checkbox locations are kept unchanged in normal cases, but ANY competing
    proposed writing patch is withheld to avoid pointing users onto a checkbox.
    A collision of two input fields withholds BOTH (not arbitrarily choosing).
    """
    for i,a in enumerate(targets):
        aa=a.get('writable_region_xyxy_px')
        if aa is None:continue
        for b in targets[i+1:]:
            bb=b.get('writable_region_xyxy_px')
            if bb is None:continue
            if _overlap_fraction(aa,bb)<.12:continue
            acheck=a['class_name']=='check_box'
            bcheck=b['class_name']=='check_box'
            if acheck and not bcheck:
                _clear_target(b,'conflict_with_checkbox');continue
            if bcheck and not acheck:
                _clear_target(a,'conflict_with_checkbox');break
            # Neither candidate may silently be chosen over the other.
            _clear_target(a,'overlap_requires_review')
            _clear_target(b,'overlap_requires_review')
            break
    return targets


def build_targets(image, data, *, permit_preview=False):
    h,w=image.shape[:2]
    if (w,h)!=(int(data['image_width']),int(data['image_height'])):
        raise ValueError(f"Image/JSON size mismatch: image={w}x{h}; json={data['image_width']}x{data['image_height']}")
    if not str(data.get('coordinate_system','')).startswith('EXIF-upright'):
        raise ValueError('Requires EXIF-upright image pixel coordinates')
    if not str(data.get('refinement',{}).get('method','')).startswith('local_geometry_v2'):
        raise ValueError('Requires refine_detections.py v2 JSON input')
    preview=bool(data.get('refinement',{}).get('preview_rescaled',False))
    if preview and not permit_preview:
        raise ValueError('Rescaled preview JSON: use original image or --permit-preview for mock testing')
    rows=data.get('detections',[])
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    out=[]
    for field in rows:
        t=_new_record(field,w,h)
        box=t['source_box_xyxy_px'];cls=t['class_name']
        state=t['source_refinement_status']
        if box is None:
            t['candidate_status']='invalid_box'
        elif t['role_gate'].startswith('excluded_'):
            t['candidate_status']=t['role_gate']
        elif (field.get('manual_review') or {}).get('status')=='developer_clicked' and _point_inside((field.get('manual_review') or {}).get('target_px'),w,h):
            # Developer-picked point remains a review-only suggestion and is
            # independently checked for conflicts later.
            pt=field['manual_review']['target_px']
            t['target_px']={k:round(float(pt[k]),2) for k in ('x','y')}
            t['candidate_method']='manual_image_click'
            t['candidate_status']='manual_point_unverified'
            t['writable_region_xyxy_px']=[pt['x']-1,pt['y']-1,pt['x']+1,pt['y']+1]
        elif cls=='check_box':
            # Preserve v2 refined checkbox coordinates exactly. Never re-detect.
            if state=='matched_candidate' and field.get('refined_bbox_xyxy_px') is not None:
                bbox=_bbox_clamp(field['refined_bbox_xyxy_px'],w,h)
                if bbox:
                    t['target_px']=_center(bbox)
                    t['writable_region_xyxy_px']=[round(float(v),2) for v in bbox]
                    t['candidate_method']='existing_v2_checkbox_match'
                    t['candidate_status']='candidate_needs_validation'
            else:
                t['candidate_status']='unresolved_checkbox'
        elif cls=='input_field':
            protected=_protected_boxes(rows,field)
            guess=_blank_patch_v2(gray,box,protected)
            if guess:
                t['writable_region_xyxy_px']=guess['region']
                t['target_px']=guess['point']
                t['candidate_method']=guess['method']
                t['ink_fraction_after_dilation']=guess['ink_fraction_after_dilation']
                t['candidate_status']=('candidate_needs_validation'
                                       if t['region_source']=='refined_box' and state=='matched_candidate'
                                       else 'weak_proposal_review_only')
            else:
                t['candidate_status']='no_safe_blank_patch'
        elif cls=='signature_field':
            # Do not shift a Gemini label box left: this caused F019 / F040
            # candidates to land on unrelated text or adjacent name fields.
            # A human can review the source box but no signing TARGET is
            # created until handwriting space is identified by OCR/manual check.
            t['candidate_status']='signature_area_manual_review_required'
            t['candidate_method']='no_signature_guess_without_ocr_or_manual_region'
        else:
            t['candidate_status']='unknown_class'
        out.append(t)
    _resolve_target_conflicts(out)
    return {'source':'Gemini + refine_detections.py v2 + offline writable_targets v2',
            'algorithm_version':'writable_targets_v2',
            'image_width':w,'image_height':h,
            'coordinate_system':'EXIF-upright original image pixels',
            'preview_rescaled':preview,'api_calls':0,
            'navigation_allowed':False,'navigation_verified':False,
            'note':('Every candidate is provisional. No OCR text coordinates or physical-mm '
                    'ground truth has been verified. Signatures are never auto-targeted.'),
            'counts':dict(Counter(t['candidate_status'] for t in out)),
            'targets':out}


def visualize(image, meta):
    out=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    h,w=image.shape[:2]
    scale=min(1.,1600/max(h,w))
    if scale<1:
        out=cv2.resize(out,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
    colors={'candidate_needs_validation':(0,160,30),
            'manual_point_unverified':(0,165,255),
            'weak_proposal_review_only':(0,165,255),
            'signature_area_manual_review_required':(0,165,255)}
    for t in meta['targets']:
        b=t['source_box_xyxy_px'];s=t['candidate_status']
        if b is None:continue
        p1=(round(b[0]*scale),round(b[1]*scale));p2=(round(b[2]*scale),round(b[3]*scale))
        cv2.rectangle(out,p1,p2,(190,60,180),1)
        color=colors.get(s,(20,20,210))
        rect=t['writable_region_xyxy_px']
        if rect is not None:
            a=(round(rect[0]*scale),round(rect[1]*scale))
            z=(round(rect[2]*scale),round(rect[3]*scale))
            cv2.rectangle(out,a,z,color,2)
        point=t['target_px']
        if point is not None:
            center=(round(point['x']*scale),round(point['y']*scale))
            cv2.drawMarker(out,center,color,cv2.MARKER_CROSS,10,1)
        cv2.putText(out,t['id']+('?' if s!='candidate_needs_validation' else ''),
                    (p1[0],max(13,p1[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1,cv2.LINE_AA)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',required=True,type=Path)
    ap.add_argument('--refined',type=Path,
        help='default: detection/refinement_output/<image-stem>/refined_detection_result.json')
    ap.add_argument('--out',type=Path,
        help='default: detection/refinement_output/<image-stem>/writable_output/')
    ap.add_argument('--permit-preview',action='store_true',
        help='Debug only. A rescaled preview is NOT suitable for finger guidance.')
    a=ap.parse_args()
    source=a.refined or HERE/'refinement_output'/a.image.stem/'refined_detection_result.json'
    dest=a.out or HERE/'refinement_output'/a.image.stem/'writable_output'
    data=json.loads(source.read_text(encoding='utf-8-sig'))
    with Image.open(a.image) as im:
        img=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    output=build_targets(img,data,permit_preview=a.permit_preview)
    output['source_image']=str(a.image.resolve())
    output['source_refinement_json']=str(source.resolve())
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'navigation_targets.json').write_text(
        json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    cv2.imencode('.jpg',visualize(img,output),
                [cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(dest/'writable_visualization.jpg'))
    with (dest/'writable_review_report.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        cols=['id','class_name','label','role_gate','source_refinement_status',
              'candidate_status','candidate_method','navigation_allowed']
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for t in output['targets']:
            writer.writerow({k:t.get(k) for k in cols})
    print('Total:',len(output['targets']),'| Status:',output['counts'])
    print('Saved:',dest)
    print('API calls: 0 | navigation_allowed=false for ALL targets')
    print('WARNING: Pixel-only review proposals, NOT validated locations.')
    if output['preview_rescaled']:
        print('PREVIEW TEST ONLY. DO NOT use coordinates for navigation.')


if __name__=='__main__':
    main()
