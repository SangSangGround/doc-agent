"""v7 - Camo/iPhone-camera automatic document-tracking navigation demo.

Requires: existing multi_field_voice_tracking_v6.py in the same folder,
          document_tracker_v7.py, verified navigation_manifest.json,
          and the EXACT source image used when the manifest was created.

No C / 4-corner clicking on live video. ORB+RANSAC auto-localizes document
against source photo every frame. Low-confidence tracking pauses all guidance.
Optional --allow-contour is a weaker, explicit developer-only fallback.

This runs on Windows, NOT on the iPhone itself. It does not prove physical
writing accuracy, hand-tip contact or safe independent use.

Keys: N/P next/prev approved target, V speak, R reacquire, Q quit.
"""
from __future__ import annotations
import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from multi_field_voice_tracking_v6 import (
    load_manifest, warp_points, movement, Sound, Speaker, create_landmarker, put_line
)
from document_tracker_v7 import PaperTracker


def open_camera(index, width, height):
    if sys.platform.startswith('win'):
        cap=cv2.VideoCapture(index,cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release(); cap=cv2.VideoCapture(index)
    else:
        cap=cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    return cap


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--reference',type=Path,default=None,
                    help='Original image used to build manifest (defaults to manifest.source_image)')
    ap.add_argument('--model',type=Path,default=Path(__file__).resolve().parent.parent/'hand_landmarker.task')
    ap.add_argument('--camera',type=int,default=1,help='Camo iPhone camera index (confirmed index 1)')
    ap.add_argument('--beep',action='store_true')
    ap.add_argument('--tts',action='store_true')
    ap.add_argument('--mirror',action='store_true',help='Use only when Camo is horizontally mirrored')
    ap.add_argument('--allow-contour',action='store_true',
                    help='Unsafe for independent use: allow page contour assuming one upright original document')
    ap.add_argument('--hit-mm',type=float,default=8.)
    ap.add_argument('--frame-width',type=int,default=1280)
    ap.add_argument('--frame-height',type=int,default=720)
    ap.add_argument('--max-feature-size',type=int,default=1080)
    ap.add_argument('--check-setup',action='store_true',help='Validate manifest, reference, and ORB descriptors; no camera')
    args=ap.parse_args()

    manifest=load_manifest(args.manifest)
    if not (3<=args.hit_mm<=25):
        ap.error('--hit-mm must be between 3 and 25')
    reference=args.reference or Path(manifest.get('source_image',''))
    try:
        tracker=PaperTracker(manifest,reference,args.allow_contour,args.max_feature_size)
    except Exception as exc:
        ap.error(f'Cannot initialize reference tracking: {exc}')
    targets=manifest['fields']
    print('v7 PHONE-CAMERA developer PoC | Webcam index:',args.camera)
    print('Approved fields:',len(targets),'| Reference ORB features:',len(tracker.kp_ref))
    print('Reference:',tracker.ref_path)
    print('Automatic document tracking: ORB+RANSAC; uncertainty pauses guidance.')
    print('Contour fallback:', 'ON (experimental; paper identity NOT verified)' if args.allow_contour else 'OFF (strict)')
    print('No live corner clicks. N/P select target; V voice; R reset; Q quit.')
    for t in targets:
        print(f"  {t['id']}: {t['label']} at ({t['target_mm']['x']:.1f},{t['target_mm']['y']:.1f}) mm")
    if args.check_setup:
        print('Setup check PASS (NOT a real-world tracking accuracy test)')
        return
    if not args.model.is_file():
        ap.error(f'MediaPipe .task model missing: {args.model}')
    try:
        mp,landmarker=create_landmarker(args.model)
    except Exception as exc:
        ap.error(f'Cannot load MediaPipe: {exc}')
    cap=open_camera(args.camera,args.frame_width,args.frame_height)
    if not cap.isOpened():
        landmarker.close(); ap.error(f'Cannot open camera {args.camera}; close other apps using Camo.')

    sound=Sound(args.beep)
    speaker=Speaker(args.tts)
    selection=0
    finger_smooth=None
    last_finger=0.
    stable_arrival=0
    hit_announced=False
    next_beep=0.
    voice_direction=None
    last_voice=0.
    last_time=-1
    started=time.monotonic()
    last_tracking_ok=False
    last_alert=0.
    fps_last=time.monotonic()
    fps_value=0.
    window='A4 iPhone Camo Navigation v7 AUTO (N/P select | R reset | Q exit)'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    try:
        with landmarker:
            while True:
                ok,frame=cap.read()
                if not ok:
                    print('Camera frame unavailable; guidance stopped.')
                    break
                if args.mirror:
                    frame=cv2.flip(frame,1)
                h,w=frame.shape[:2]
                now=time.monotonic()
                frame_delta=max(1e-3,now-fps_last)
                fps_value=.88*fps_value+.12*(1./frame_delta) if fps_value else 1./frame_delta
                fps_last=now
                chosen=targets[selection]
                target_mm=(float(chosen['target_mm']['x']),float(chosen['target_mm']['y']))
                view=frame.copy()
                result=tracker.update(frame)
                # No stale H: target/hand guidance only inside a currently valid, stable tracking frame.
                if result is None:
                    if last_tracking_ok:
                        print('[PAUSED] Lost document tracking:',tracker.last_reason)
                        if now-last_alert>2.:
                            speaker.say('문서 위치가 불확실하여 안내를 중단합니다.')
                            last_alert=now
                    last_tracking_ok=False
                    finger_smooth=None
                    stable_arrival=0
                    hit_announced=False
                    next_beep=now+.35
                    voice_direction=None
                    put_line(view,'SEARCHING / PAUSED - NO NAVIGATION',32,(0,70,255),.75)
                    put_line(view,tracker.last_reason[:80],64,(0,170,255),.55)
                    put_line(view,'Keep FULL A4 visible. Hold iPhone steady, rear camera on paper.',95,(230,230,230),.53)
                else:
                    if not last_tracking_ok:
                        print('[TRACKING]',result.method,result.detail)
                        if now-last_alert>1.5:
                            speaker.say('문서 위치를 다시 찾았습니다.')
                            last_alert=now
                    last_tracking_ok=True
                    corners=result.corners
                    cv2.polylines(view,[np.round(corners).astype(np.int32)],True,
                                  (0,205,30) if result.method=='ORB' else (0,180,255),3)
                    for corner in corners:
                        cv2.circle(view,tuple(np.round(corner).astype(int)),5,(0,255,0),-1)
                    target_live=warp_points([target_mm],result.H_mm_to_live)[0]
                    if np.isfinite(target_live).all():
                        tx,ty=map(int,np.rint(target_live))
                        cv2.drawMarker(view,(tx,ty),(0,0,255),cv2.MARKER_CROSS,29,3)
                    put_line(view,f'TRACKING [{result.method}] {result.detail[:55]}',30,(0,255,90),.61)
                    rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                    stamp=max(last_time+1,int((now-started)*1000))
                    last_time=stamp
                    detected=landmarker.detect_for_video(mp_img,stamp)
                    finger_valid=False
                    if detected.hand_landmarks:
                        tip=detected.hand_landmarks[0][8]
                        px,py=float(tip.x*w),float(tip.y*h)
                        if 0<=px<w and 0<=py<h:
                            cv2.circle(view,(round(px),round(py)),8,(255,130,10),-1)
                            if cv2.pointPolygonTest(np.float32(corners),(px,py),False)>=0:
                                finger_mm=warp_points([(px,py)],result.H_live_to_mm)[0]
                                mmw=float(manifest['paper_width_mm']);mmh=float(manifest['paper_height_mm'])
                                if np.isfinite(finger_mm).all() and 0<=finger_mm[0]<=mmw and 0<=finger_mm[1]<=mmh:
                                    finger_valid=True
                                    if finger_smooth is None or now-last_finger>.45:
                                        finger_smooth=finger_mm
                                    else:
                                        finger_smooth=.48*finger_mm+.52*finger_smooth
                                    last_finger=now
                                    dist,direction,primary=movement(finger_smooth,target_mm)
                                    put_line(view,f'FINGER ({finger_smooth[0]:.1f},{finger_smooth[1]:.1f}) mm DIST={dist:.1f} mm {direction}',61,(255,255,255),.61)
                                    if dist<=args.hit_mm:
                                        stable_arrival+=1
                                        if stable_arrival>=7:
                                            put_line(view,'NEAR TARGET (NOT verified physical contact)',93,(0,255,90),.61)
                                            if not hit_announced:
                                                sound.play(1200,150)
                                                speaker.say('목표 위치 부근입니다. 실제 종이 위치를 확인해 주세요.')
                                                hit_announced=True
                                    else:
                                        stable_arrival=0
                                        hit_announced=False
                                        if now>=next_beep:
                                            interval=float(np.interp(np.clip(dist,8,150),[8,150],[.18,1.2]))
                                            next_beep=now+interval
                                            sound.play(800,45)
                                        if args.tts and direction!=voice_direction and now-last_voice>4.5:
                                            korean={'right':'오른쪽','left':'왼쪽','up':'위쪽','down':'아래쪽'}[direction]
                                            speaker.say(f'{korean}으로 약 {primary/10:.1f}센티미터 움직이세요.')
                                            voice_direction=direction
                                            last_voice=now
                    if not finger_valid:
                        finger_smooth=None
                        stable_arrival=0
                        hit_announced=False
                        next_beep=now+.25
                        put_line(view,'Finger not on paper / no guidance',61,(0,190,255),.58)
                put_line(view,f'FIELD {selection+1}/{len(targets)}: {chosen["id"]} | CAMERA {args.camera} | {fps_value:.1f} FPS',h-39,(245,245,245),.58)
                put_line(view,'N/P target | V voice | R reset document tracking | Q exit',h-13,(210,210,210),.53)
                cv2.imshow(window,view)
                key=cv2.waitKey(1)&0xff
                if key in (27,ord('q'),ord('Q')):
                    break
                if key in (ord('r'),ord('R')):
                    tracker.last_good=None
                    tracker.pending=None
                    tracker.pending_count=0
                    tracker.fail_frames=0
                    finger_smooth=None
                    last_tracking_ok=False
                    print('Document tracking reset. Automatic reacquisition in progress.')
                elif key in (ord('n'),ord('N'),ord('p'),ord('P')):
                    selection=(selection+(1 if chr(key).lower()=='n' else -1))%len(targets)
                    finger_smooth=None
                    stable_arrival=0
                    hit_announced=False
                    voice_direction=None
                    next_beep=now+.4
                    print('Selected:',targets[selection]['id'],targets[selection]['label'])
                    speaker.say(f"{targets[selection]['label']} 항목을 안내합니다.")
                elif key in (ord('v'),ord('V')):
                    speaker.say(f"{chosen['label']} 위치를 안내 중입니다. 카메라가 흔들리면 안내가 멈출 수 있습니다.")
    finally:
        speaker.stop.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
