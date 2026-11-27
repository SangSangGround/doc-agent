"""Offline Gemini+OpenCV field refinement v2; preserves the existing checkbox matcher.
Requires opencv-python, numpy, pillow. No API, no OCR, no navigation authorization.
All outputs are in the EXIF-upright supplied image pixel coordinate system.
"""
import argparse
import copy
import json
import math
import csv
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent


def threshold(gray, c=9):
    k = min(31, min(gray.shape[:2]) if min(gray.shape[:2]) % 2 else min(gray.shape[:2])-1)
    if k < 3:
        return np.zeros_like(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, k, c)


def roi(gray, box, padx, pady):
    x1,y1,x2,y2 = box
    x0=max(0,math.floor(x1-padx)); y0=max(0,math.floor(y1-pady))
    xe=min(gray.shape[1],math.ceil(x2+padx)); ye=min(gray.shape[0],math.ceil(y2+pady))
    return gray[y0:ye,x0:xe], x0,y0


def center(box):
    return np.array([(box[0]+box[2])/2,(box[1]+box[3])/2],dtype=float)


def iou(a,b):
    x=max(0,min(a[2],b[2])-max(a[0],b[0])); y=max(0,min(a[3],b[3])-max(a[1],b[1]))
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-x*y
    return x*y/u if u else 0.0


def candidate(box, score, method, polygon=None, target=None):
    return {'box':np.array(box,dtype=float), 'score':float(score), 'method':method,
            'polygon':None if polygon is None else np.array(polygon,dtype=float),
            'target': center(box) if target is None else np.array(target,dtype=float)}


def checkbox_candidates(gray, box):
    x1,y1,x2,y2=box; bw=x2-x1; bh=y2-y1; p=center(box)
    if min(bw,bh)<5: return []
    g,ox,oy=roi(gray,box,1.9*bw,.95*bh)
    if not g.size: return []
    out=[]
    for c in (9,5):
        b=threshold(g,c)
        n,lab,stats,_=cv2.connectedComponentsWithStats(b)
        strokes=[]
        for j,(x,y,w,h,a) in enumerate(stats[1:],1):
            if not (.43*bh<=h<=1.35*bh and 2<=w<=.53*bh and a>=5 and h>=2*w):continue
            z=(lab[y:y+h,x:x+w]==j).astype(np.uint8)
            mid=z[h//4:3*h//4]; k=max(1,round(w*.45))
            direction=(int(mid[:,:k].sum())-int(mid[:,-k:].sum()))/max(1,int(mid.sum()))
            # A bracket has a vertical spine and horizontal caps pointing inward.
            if abs(direction)<.28:continue
            stroke_coverage=np.count_nonzero(z.sum(axis=1))/h
            if stroke_coverage<.85:continue
            side=1 if direction>0 else -1
            caps=(z[:max(2,h//4)],z[-max(2,h//4):])
            if any(np.count_nonzero(cap.sum(axis=0))<max(2,round(w*.60)) for cap in caps):continue
            strokes.append((x,y,w,h,side,abs(direction)))
        for a in strokes:
            if a[4]!=1:continue
            for b2 in strokes:
                if b2[4]!=-1:continue
                x,y,w,h,_,da=a; X,Y,W,H,_,db=b2
                hh=(h+H)/2; gap=X-(x+w)
                if not (.30*hh<=gap<=2.2*hh and abs(h-H)<=.30*hh and abs(y+h/2-Y-H/2)<=.24*hh):continue
                inside=b[max(y,Y)+2:min(y+h,Y+H)-2,x+w:X]
                if inside.size==0 or np.mean(inside>0)>.11:continue
                bb=np.array([ox+x,oy+min(y,Y),ox+X+W,oy+max(y+h,Y+H)],float)
                delta=center(bb)-p
                if abs(delta[0])>1.8*bw or abs(delta[1])>.85*bh:continue
                score=2.4 - .45*abs(delta[0])/bw - .60*abs(delta[1])/bh - .35*abs(h-H)/hh + .15*(da+db)
                out.append(candidate(bb,score,'bracket_pair'))
        # Closed, near-square outlines with a mostly empty center.
        contours,hier=cv2.findContours(b,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:continue
        for j,ct in enumerate(contours):
            if hier[0,j,2]<0:continue
            x,y,w,h=cv2.boundingRect(ct)
            if not (.5*bh<=h<=1.45*bh and .55<=w/h<=1.65 and .45*bw<=w<=1.6*bw):continue
            if cv2.contourArea(ct)/(w*h)<.65:continue
            poly=cv2.approxPolyDP(ct,.045*cv2.arcLength(ct,True),True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):continue
            d=max(2,round(min(w,h)*.22)); inside=b[y+d:y+h-d,x+d:x+w-d]
            if not inside.size or np.mean(inside>0)>.16:continue
            bb=np.array([ox+x,oy+y,ox+x+w,oy+y+h],float); delta=center(bb)-p
            if abs(delta[0])>1.8*bw or abs(delta[1])>.85*bh:continue
            score=2.25-.45*abs(delta[0])/bw-.60*abs(delta[1])/bh
            out.append(candidate(bb,score,'closed_checkbox',poly[:,0,:]+[ox,oy]))
    return out


def cell_candidates(gray,box):
    x1,y1,x2,y2=box; bw=x2-x1; bh=y2-y1; p=center(box)
    if min(bw,bh)<8:return []
    g,ox,oy=roi(gray,box,.24*bw,.72*bh)
    if not g.size:return []
    out=[]
    for c in (5,8):
        b=threshold(g,c)
        b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,hier=cv2.findContours(b,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:continue
        for j,ct in enumerate(contours):
            # Inner contour of a closed printed boundary, not a text blob.
            if hier[0,j,3]<0:continue
            x,y,w,h=cv2.boundingRect(ct)
            if x<=1 or y<=1 or x+w>=g.shape[1]-1 or y+h>=g.shape[0]-1:continue
            if not (.65*bw<=w<=1.35*bw and .45*bh<=h<=1.65*bh):continue
            area=cv2.contourArea(ct)
            if area/(w*h)<.68:continue
            poly=cv2.approxPolyDP(ct,.025*cv2.arcLength(ct,True),True)
            if len(poly)!=4 or not cv2.isContourConvex(poly):continue
            # Reject writing/labels inside the candidate. Erode boundary before measuring.
            mask=np.zeros_like(b);cv2.fillConvexPoly(mask,poly[:,0,:],255)
            mask=cv2.erode(mask,np.ones((5,5),np.uint8)); inside=b[mask>0]
            if not inside.size or np.mean(inside>0)>.075:continue
            bb=np.array([ox+x,oy+y,ox+x+w,oy+y+h],float); delta=center(bb)-p
            if abs(delta[0])>.25*bw or abs(delta[1])>.48*bh:continue
            if iou(bb,box)<.46:continue
            mo=cv2.moments(ct)
            if not mo['m00']:continue
            tgt=[ox+mo['m10']/mo['m00'],oy+mo['m01']/mo['m00']]
            score=1.2+iou(bb,box)-.3*abs(delta[1])/bh
            out.append(candidate(bb,score,'empty_table_cell',poly[:,0,:]+[ox,oy],tgt))
    return out


def choose(candidates):
    distinct=[]
    for c in sorted(candidates,key=lambda x:x['score'],reverse=True):
        if not any(iou(c['box'],d['box'])>.6 for d in distinct):distinct.append(c)
    if not distinct:return None,'no_supported_geometry',0
    if distinct[0]['score']<1.35:return None,'weak_geometry',len(distinct)
    # Different plausible locations: keep the original proposal; no guess.
    if len(distinct)>1 and distinct[0]['score']-distinct[1]['score']<.20:
        return None,'ambiguous_candidates',len(distinct)
    return distinct[0],'geometry_matched',len(distinct)




def _local_ink_hint(gray, box):
    """Diagnostic only. Gray paper, shadows and printed text are NOT authorization cues."""
    x1,y1,x2,y2=[int(round(k)) for k in box]
    x1=max(0,x1);x2=min(gray.shape[1],x2);y1=max(0,y1);y2=min(gray.shape[0],y2)
    if x2-x1<8 or y2-y1<8:return {'shaded_appearance':'unknown'}
    crop=gray[y1:y2,x1:x2]
    i=crop[max(1,crop.shape[0]//5):max(2,4*crop.shape[0]//5),max(1,crop.shape[1]//5):max(2,4*crop.shape[1]//5)]
    if not i.size:return {'shaded_appearance':'unknown'}
    g,ox,oy=roi(gray,box,.15*(x2-x1),.4*(y2-y1))
    if not g.size:return {'shaded_appearance':'unknown'}
    # Trim dark text by looking at the upper 70th percentile.
    field_brightness=float(np.percentile(i,70)); area_brightness=float(np.percentile(g,80))
    return {'shaded_appearance': 'possible' if area_brightness-field_brightness>17 else 'not_obvious',
            'brightness_delta':round(area_brightness-field_brightness,1)}


def _edge_peaks(mask, axis, box, ox, oy, tol=0.4):
    """Long straight strokes near original box sides; returns (position, fractional support)."""
    x1,y1,x2,y2=box;bw=x2-x1;bh=y2-y1
    if axis=='h':
        lo=max(0,int(round(x1-ox+bw*.06)));hi=min(mask.shape[1],int(round(x2-ox-bw*.06)))
        if hi-lo<10:return []
        ys=np.arange(max(0,int(y1-oy-tol*bh)),min(mask.shape[0],int(y2-oy+tol*bh)+1))
        if not ys.size:return []
        # 3-pixel tolerance to handle mild perspective and aliased lines
        traces=[]
        for y in ys:
            band=mask[max(0,y-1):min(mask.shape[0],y+2),lo:hi]
            traces.append(float(np.any(band>0,axis=0).mean()) if band.size else 0.)
        return [(float(y+oy),float(s)) for y,s in zip(ys,traces) if s>=.53]
    else:
        lo=max(0,int(round(y1-oy+bh*.08)));hi=min(mask.shape[0],int(round(y2-oy-bh*.08)))
        if hi-lo<8:return []
        xs=np.arange(max(0,int(x1-ox-tol*bw)),min(mask.shape[1],int(x2-ox+tol*bw)+1))
        if not xs.size:return []
        traces=[]
        for x in xs:
            band=mask[lo:hi,max(0,x-1):min(mask.shape[1],x+2)]
            traces.append(float(np.any(band>0,axis=1).mean()) if band.size else 0.)
        return [(float(x+ox),float(s)) for x,s in zip(xs,traces) if s>=.48]


def _near_peak(peaks, desired, maxdist):
    choices=[(pos,s) for pos,s in peaks if abs(pos-desired)<=maxdist]
    if not choices:return None
    return max(choices,key=lambda t:t[1]-.55*abs(t[0]-desired)/max(1,maxdist))


def open_field_candidate(gray, box):
    """Detect partial table borders / underlines; avoid declaring a single line 'verified'.

    Strong: >=2 horizontal + >=1 vertical or >=3 sides.
    Medium: top+bottom only; weak: single baseline, useful solely for manual review.
    """
    x1,y1,x2,y2=map(float,box);bw=x2-x1;bh=y2-y1
    if bw<30 or bh<12:return None
    g,ox,oy=roi(gray,box,.16*bw,.65*bh)
    if g.size==0:return None
    # Local thresholding only; strong horizontal/vertical morphology suppresses text strokes.
    b=threshold(g,8)
    kh=max(13,round(bw*.23));kv=max(9,round(bh*.55))
    H=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,kh),np.uint8))
    V=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((kv,1),np.uint8))
    hp=_edge_peaks(H,'h',box,ox,oy,tol=.45)
    vp=_edge_peaks(V,'v',box,ox,oy,tol=.40)
    top=_near_peak(hp,y1,.32*bh)
    bottom=_near_peak(hp,y2,.32*bh)
    left=_near_peak(vp,x1,.20*bw)
    right=_near_peak(vp,x2,.20*bw)
    if top and bottom and bottom[0]-top[0]<.5*bh:
        top,bottom=None,None  # do not reuse the same printed line for both sides
    if left and right and right[0]-left[0]<.5*bw:
        left,right=None,None
    sides={k:v for k,v in [('top',top),('bottom',bottom),('left',left),('right',right)] if v}
    n=len(sides)
    if n==0:return None
    bbox=np.array([left[0] if left else x1,top[0] if top else y1,right[0] if right else x2,bottom[0] if bottom else y2],float)
    if bbox[2]-bbox[0]<.65*bw or bbox[3]-bbox[1]<.5*bh or iou(bbox,box)<.62:return None
    # A line at the bottom and a long line at the top can come from adjacent rows.
    if top and bottom and (abs(top[0]-y1)>.2*bh or abs(bottom[0]-y2)>.2*bh):
        return None
    strengths={k:round(v[1],3) for k,v in sides.items()}
    if n>=3 and (top or bottom) and (left or right):
        status='matched_candidate'; method='line_supported_cell'
    elif top and bottom:
        status='needs_review'; method='open_table_row'
    elif bottom:
        status='needs_review'; method='underline_baseline'
    else:
        return None
    # Do not infer that the geometric box is physically clear of printed content.
    return {'box':bbox,'target':center(bbox),'status':status,'method':method,
            'sides':strengths,'side_count':n}


def role_guess(field, gray, box):
    """Conservative heuristic metadata, NOT a legal instruction to fill a field."""
    label=' '.join(str(field.get('label','')).split())
    l=label.replace(' ','')
    official_terms=('접수번호','접수일','담당공무원','관계기관통보','처리기관','처리기간','공무원확인','기관장','접수담당')
    conditional_terms=('정정','변경전','변경후','삭제신청','세대주의','세대원','배우자','해당사항','처분')
    role='unknown';reason='작성 주체를 확정할 문서 규칙 없음'
    if any(t in l for t in official_terms):
        role='official_possible';reason='접수/기관 처리 용어 포함; 신청인 안내에서 제외 권고'
    elif '신청일'==l and box[1]<.3*gray.shape[0]:
        role='official_possible';reason='상단 신청일: 담당자 접수용인지 원문 확인 필요'
    elif any(t in l for t in conditional_terms):
        role='conditional';reason='선택/관계/조건에 따라 작성 여부가 달라질 수 있음'
    elif '서명' in l or '날인' in l or '인'==l:
        role='signature_review';reason='인쇄된 서명 문구와 실제 서명 공간 구분 필요'
    elif any(t in l for t in ('신청인','성명','연락처','주소','생년월일','주민등록번호','외국인등록번호')):
        role='applicant_possible';reason='일반 인적사항 용어, 해당 문서의 작성 주체 확인 필요'
    return {'role':role,'reason':reason,'source':'label_heuristic','review_required':True}


def _serialize_result(field, original, c, method, status, reason, extra=None):
    field['original_bbox_xyxy_px']=rounded(original)
    field['original_center_px']=dict(zip(('x','y'),rounded(center(original))))
    field['navigation_verified']=False
    field['navigation_allowed']=False
    field['review_required']=True
    field['refinement']={'status':status,'reason':reason,
                         'candidate_count':int(extra.get('candidate_count',0)) if extra else (1 if c else 0),
                         'method':method}
    if extra and 'geometry_sides' in extra:field['refinement']['geometry_sides']=extra['geometry_sides']
    if extra and 'shaded_appearance' in extra:field['refinement']['shaded_appearance']=extra['shaded_appearance']
    field['refined_bbox_xyxy_px']=rounded(c['box']) if c is not None and status=='matched_candidate' else None
    field['refined_center_px']=dict(zip(('x','y'),rounded(c['target']))) if c is not None and status=='matched_candidate' else None
    field['refined_polygon_px']=c['polygon'].round(3).tolist() if c is not None and status=='matched_candidate' and c.get('polygon') is not None else None
    field['center_shift_px']=round(float(np.linalg.norm(c['target']-center(original))),3) if c is not None and status=='matched_candidate' else None
    field['review_suggestion_px']=dict(zip(('x','y'),rounded(c['target']))) if c is not None and status=='needs_review' else None
    field['bbox_xyxy_px']=rounded(original)
    field['center_px']=field['original_center_px']


def rounded(a):return [round(float(v),3) for v in a]


def refine(image,result,allow_rescaled=False):
    h,w=image.shape[:2];ow=int(result['image_width']);oh=int(result['image_height'])
    if min(w,h,ow,oh)<=0:raise ValueError('Invalid dimensions')
    resized=(w,h)!=(ow,oh)
    if resized and (not allow_rescaled or abs((w/h)/(ow/oh)-1)>.002):
        raise ValueError(f'Image size {w}x{h} differs from JSON {ow}x{oh}. Use exact original image. --allow-rescaled for PREVIEW ONLY.')
    if not result.get('coordinate_system','').startswith('EXIF-upright'):
        raise ValueError('Expected EXIF-upright source coordinate system')
    scale=min(1.,2048/max(w,h));nw=max(1,round(w*scale));nh=max(1,round(h*scale))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if (nw,nh)!=(w,h):gray=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
    work_scale=np.array([nw/w,nh/h,nw/w,nh/h]);
    input_scale=np.array([w/ow,h/oh,w/ow,h/oh])
    final=copy.deepcopy(result);final['image_width']=w;final['image_height']=h
    final['navigation_verified']=False
    final['refinement']={'method':'local_geometry_v2','working_image_size':[nw,nh],
      'preview_rescaled':resized,'api_calls':0,'navigation_verified':False,
      'note':'candidate matches are NOT ground-truth or finger-navigation validation'}
    rows=[]
    for field in final.get('detections',[]):
        original=np.array(field['bbox_xyxy_px'],float)*input_scale
        if original.shape!=(4,) or not np.isfinite(original).all() or not (0<=original[0]<original[2]<=w and 0<=original[1]<original[3]<=h):
            raise ValueError(f'Invalid bbox: {field.get("id")}')
        box=original*work_scale;cls=field['class_name'];geometry=[]
        if cls=='check_box':geometry=checkbox_candidates(gray,box)
        elif cls in ('input_field','signature_field'):geometry=cell_candidates(gray,box)
        found,reason,n=choose(geometry)
        c=None; status='unresolved';method=None;extra={'candidate_count':n}
        if found:
            c={'box':found['box']/work_scale,'target':found['target']/work_scale[:2],
               'polygon':found['polygon']/work_scale[:2] if found['polygon'] is not None else None}
            status='matched_candidate';method=found['method']
        elif cls=='input_field':
            guess=open_field_candidate(gray,box)
            if guess:
                c={'box':guess['box']/work_scale,'target':guess['target']/work_scale[:2],'polygon':None}
                status=guess['status'];method=guess['method']
                reason='border_evidence' if status=='matched_candidate' else 'partial_line_evidence'
                extra={'candidate_count':1,'geometry_sides':guess['sides']}
        elif cls=='signature_field':
            # An '(서명 또는 인)' printed label is NOT itself a safe signing position.
            reason='signature_area_requires_review';status='needs_review'
        appearance=_local_ink_hint(gray,box)
        extra['shaded_appearance']=appearance
        if cls=='input_field' and 'shaded_appearance' in extra and extra['shaded_appearance'].get('shaded_appearance')=='possible':
            # Do not infer blank-cell clearance from a gray administrative header.
            pass
        _serialize_result(field,original,c,method,status,reason,extra)
        field['role_assessment']=role_guess(field,gray,box)
        if field['role_assessment']['role']=='official_possible':
            field['review_priority']='low_official'
        elif cls=='signature_field' or status!='matched_candidate':field['review_priority']='high'
        else:field['review_priority']='normal'
        rows.append(field)

    # Avoid two fields snapping to the same proposed physical target.
    bad=set()
    for i,a in enumerate(rows):
        ba=a['refined_bbox_xyxy_px']
        if ba is None:continue
        for j in range(i+1,len(rows)):
            b=rows[j];bb=b['refined_bbox_xyxy_px']
            if a['class_name']==b['class_name'] and bb is not None and iou(ba,bb)>.55:bad.update([i,j])
    for i in bad:
        field=rows[i]
        field['refinement'].update(status='needs_review',reason='duplicate_candidate')
        for k in ('refined_bbox_xyxy_px','refined_center_px','refined_polygon_px','center_shift_px'):field[k]=None
    final['refinement']['counts']=dict(Counter(f['refinement']['status'] for f in rows))
    final['refinement']['class_counts']={key:dict(Counter(f['refinement']['status'] for f in rows if f['class_name']==key)) for key in ('check_box','input_field','signature_field')}
    final['refinement']['role_counts']=dict(Counter(f['role_assessment']['role'] for f in rows))
    return final


def overlay(image,rows):
    out=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);h,w=out.shape[:2]
    thick=max(2,round(max(w,h)/1100));fs=max(.45,max(w,h)/3700)
    colors={'matched_candidate':(0,165,10),'needs_review':(0,165,255),'unresolved':(30,40,220)}
    for f in rows:
        a=np.round(f['original_bbox_xyxy_px']).astype(int)
        cv2.rectangle(out,tuple(a[:2]),tuple(a[2:]),(205,50,175),thick)
        status=f['refinement']['status'];color=colors.get(status,(10,10,200));bb=f.get('refined_bbox_xyxy_px')
        q=np.round(bb).astype(int) if bb else a
        if bb:
            polygon=f.get('refined_polygon_px')
            if polygon is not None:cv2.polylines(out,[np.round(polygon).astype(np.int32)],True,color,thick)
            else:cv2.rectangle(out,tuple(q[:2]),tuple(q[2:]),color,thick)
            pt=f['refined_center_px'];cv2.circle(out,(round(pt['x']),round(pt['y'])),max(2,thick),color,-1)
        hint=f.get('review_suggestion_px')
        if hint:
            px=(round(hint['x']),round(hint['y']))
            cv2.drawMarker(out,px,color,cv2.MARKER_TILTED_CROSS,12,thick)
        label=f.get('id','?')+('?' if status!='matched_candidate' else '')
        pt=(int(q[0]),max(12,int(q[1])-5))
        cv2.putText(out,label,pt,cv2.FONT_HERSHEY_SIMPLEX,fs,(255,255,255),thick+2,cv2.LINE_AA)
        cv2.putText(out,label,pt,cv2.FONT_HERSHEY_SIMPLEX,fs,color,thick,cv2.LINE_AA)
    return out


def write_review_csv(path,result):
    with open(path,'w',encoding='utf-8-sig',newline='') as fh:
        cols=['id','class_name','label','status','method','reason','role','review_priority','center_shift_px','navigation_allowed']
        writer=csv.DictWriter(fh,fieldnames=cols);writer.writeheader()
        for f in result['detections']:
            r=f['refinement'];writer.writerow({'id':f['id'],'class_name':f['class_name'],'label':f['label'],
                'status':r['status'],'method':r['method'],'reason':r['reason'],
                'role':f['role_assessment']['role'],'review_priority':f['review_priority'],
                'center_shift_px':f['center_shift_px'],'navigation_allowed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=HERE/'test_document.jpg')
    p.add_argument('--detections',type=Path,help='Gemini detection_result.json (default matches image stem)')
    p.add_argument('--output',type=Path)
    p.add_argument('--allow-rescaled',action='store_true',help='Same-content resized preview ONLY; NOT for navigation')
    a=p.parse_args()
    source=a.detections or HERE/'gemini_output'/a.image.stem/'detection_result.json'
    result=json.loads(source.read_text(encoding='utf-8-sig'))
    with Image.open(a.image) as im:image=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    final=refine(image,result,a.allow_rescaled)
    final['image']=str(a.image.resolve())
    final['refinement']['source_detection_json']=str(source.resolve())
    output=a.output or HERE/'refinement_output'/a.image.stem;output.mkdir(parents=True,exist_ok=True)
    (output/'refined_detection_result.json').write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding='utf-8')
    cv2.imencode('.jpg',overlay(image,final['detections']),[cv2.IMWRITE_JPEG_QUALITY,94])[1].tofile(str(output/'refined_visualization.jpg'))
    write_review_csv(output/'review_report.csv',final)
    for f in final['detections']:
        r=f['refinement'];print(f"{f['id']:5} {f['class_name']:15} {r['status']:18} {str(r['method']):23} {r['reason']:28} | {f['label']}")
    print('Counts:',final['refinement']['counts']);print('Class counts:',final['refinement']['class_counts'])
    print('Saved:',output,'API calls: 0')
    print('CAUTION: Every result requires validation against ground truth; navigation_allowed=false for ALL fields.')
    if final['refinement']['preview_rescaled']:print('WARNING: resized preview coordinates, NOT for physical navigation.')

if __name__=='__main__':main()
