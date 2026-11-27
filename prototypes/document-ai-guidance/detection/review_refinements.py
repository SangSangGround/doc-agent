"""Optional developer-only GUI review for refined_detection_result.json.

Click a candidate writing target to save its image-pixel position. This does NOT verify
finger navigation, physical-mm accuracy, user eligibility, or legal instructions.
Requires a Windows desktop session (not headless). No Gemini API requests.
"""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',type=Path,required=True)
    ap.add_argument('--result',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--all',action='store_true',help='Review all fields, not just unresolved/needs_review')
    opts=ap.parse_args()
    source=opts.result or HERE/'refinement_output'/opts.image.stem/'refined_detection_result.json'
    target=opts.out or source.with_name('reviewed_detection_result.json')
    data=json.loads(source.read_text(encoding='utf-8-sig'))
    with Image.open(opts.image) as im: rgb=np.array(ImageOps.exif_transpose(im).convert('RGB'))
    img=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);h,w=img.shape[:2]
    if (w,h)!=(int(data['image_width']),int(data['image_height'])):
        raise ValueError('Image/JSON size mismatch. Use exactly the same source image.')
    if data.get('refinement',{}).get('preview_rescaled'):
        raise ValueError('Refusing manual annotation of resized preview; use exact original image.')
    items=data.get('detections',[])
    selected=[i for i,f in enumerate(items) if opts.all or f.get('refinement',{}).get('status')!='matched_candidate']
    if not selected:
        print('No unresolved items. Use --all to review matched candidates.');return
    ui={'i':0,'click':None,'frame':None,'crop_meta':None}
    cv2.namedWindow('Review fields (q save, n next, p prev, d disable, r clear)',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Review fields (q save, n next, p prev, d disable, r clear)',1400,900)
    win='Review fields (q save, n next, p prev, d disable, r clear)'

    def on_mouse(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        meta=ui['crop_meta']
        if meta is None:return
        if x<meta['split_x']:
            # overview has its own scale and offsets
            ox,oy,scale=meta['overview']
            if 0<=x-ox<meta['overview_width'] and 0<=y-oy<meta['overview_height']:
                px=(x-ox)/scale;py=(y-oy)/scale
            else:return
        else:
            crop_x,crop_y,crop_w,crop_h=meta['crop_box']
            lx=x-meta['split_x'];ly=y-meta['crop_top']
            if lx<0 or ly<0 or lx>=meta['crop_w_draw'] or ly>=meta['crop_h_draw']:return
            px=crop_x+lx/meta['crop_scale'];py=crop_y+ly/meta['crop_scale']
        ui['click']=[max(0,min(w-1,px)),max(0,min(h-1,py))]

    cv2.setMouseCallback(win,on_mouse)
    def render():
        field=items[selected[ui['i']]]
        # page overview
        overview_w=560; overview_h=800
        s=min(overview_w/w,overview_h/h)
        ow=int(round(w*s));oh=int(round(h*s))
        overview=cv2.resize(img,(ow,oh),interpolation=cv2.INTER_AREA)
        canvas=np.full((900,1400,3),245,np.uint8)
        canvas[:oh,:ow]=overview
        b=field.get('refined_bbox_xyxy_px') or field['bbox_xyxy_px'];x1,y1,x2,y2=map(float,b)
        p1=(int(round(x1*s)),int(round(y1*s)));p2=(int(round(x2*s)),int(round(y2*s)))
        cv2.rectangle(canvas,p1,p2,(0,180,230),2)
        # zoomed crop with generous nearby context
        bw=x2-x1;bh=y2-y1
        padx=max(180,bw*.6);pady=max(150,bh*1.2)
        cx1=max(0,int(x1-padx));cy1=max(0,int(y1-pady));cx2=min(w,int(x2+padx));cy2=min(h,int(y2+pady))
        crop=img[cy1:cy2,cx1:cx2];cw=780;ch=690
        scale=min(cw/max(1,crop.shape[1]),ch/max(1,crop.shape[0]));dw=max(1,round(crop.shape[1]*scale));dh=max(1,round(crop.shape[0]*scale))
        view=cv2.resize(crop,(dw,dh),interpolation=cv2.INTER_LINEAR)
        split=590;top=95
        canvas[top:top+dh,split:split+dw]=view
        def mappt(px,py):return (split+int(round((px-cx1)*scale)),top+int(round((py-cy1)*scale)))
        cv2.rectangle(canvas,mappt(x1,y1),mappt(x2,y2),(0,160,255),2)
        if ui['click'] is not None:
            cv2.drawMarker(canvas,mappt(*ui['click']),(0,150,20),cv2.MARKER_CROSS,21,2)
        saved=field.get('manual_review',{}).get('target_px')
        if saved is not None:cv2.circle(canvas,mappt(saved['x'],saved['y']),7,(200,80,0),3)
        cv2.putText(canvas,f"{ui['i']+1}/{len(selected)} {field['id']} {field['class_name']}",(600,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(canvas,f"status: {field['refinement']['status']}",(600,63),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1,cv2.LINE_AA)
        cv2.putText(canvas,'CLICK target, ENTER save+next | N next P prev | D disable | Q quit',(10,865),cv2.FONT_HERSHEY_SIMPLEX,.57,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(canvas,'R clear current target | No API calls | Physical calibration NOT validated',(10,888),cv2.FONT_HERSHEY_SIMPLEX,.55,(30,30,170),1,cv2.LINE_AA)
        ui['crop_meta']={'split_x':split,'crop_top':top,'crop_box':(cx1,cy1,cx2-cx1,cy2-cy1),
              'crop_scale':scale,'crop_w_draw':dw,'crop_h_draw':dh,
              'overview':(0,0,s),'overview_width':ow,'overview_height':oh}
        cv2.imshow(win,canvas)
        print(f"[{ui['i']+1}/{len(selected)}] {field['id']} {field.get('label')} | status={field['refinement']['status']}")

    def save():
        data['manual_review_metadata']={'source':str(source),'review_tool':'developer_click_v1',
           'notice':'manual image coordinates; user navigation remains NOT validated'}
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Saved',target)

    try:
        while True:
            render()
            key=cv2.waitKey(0)&0xff
            if key in (13,10,32):
                if ui['click'] is not None:
                    f=items[selected[ui['i']]]
                    f['manual_review']={'target_px':{'x':round(ui['click'][0],2),'y':round(ui['click'][1],2)},
                         'status':'developer_clicked','scope':'image_coordinate_only'}
                    f['navigation_allowed']=False;f['navigation_verified']=False
                ui['i']=min(len(selected)-1,ui['i']+1);ui['click']=None
            elif key in (ord('n'),ord('N')):ui['i']=min(len(selected)-1,ui['i']+1);ui['click']=None
            elif key in (ord('p'),ord('P')):ui['i']=max(0,ui['i']-1);ui['click']=None
            elif key in (ord('r'),ord('R')):
                f=items[selected[ui['i']]];f.pop('manual_review',None);ui['click']=None
            elif key in (ord('d'),ord('D')):
                f=items[selected[ui['i']]];f['manual_review']={'status':'not_applicable_or_official','scope':'developer_review'}
                f['navigation_allowed']=False;ui['i']=min(len(selected)-1,ui['i']+1);ui['click']=None
            elif key in (ord('q'),ord('Q'),27):break
    finally:
        save();cv2.destroyAllWindows()

if __name__=='__main__':main()
