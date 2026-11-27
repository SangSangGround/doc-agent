"""v7.3.1: close paper navigation with voice prompt decoupled from tracking.

Runs on WINDOWS, using an iPhone solely as a camera through Camo.
A reviewed manifest is mandatory: nonapproved fields are NEVER navigated.
Guide JSON is human-authored for this particular form; NOT live Gemini/OCR.
Experimental close-range ORB matching can localize a *partial* known page,
provided enough unique print is visible. If localization or target visibility
fails, guidance pauses. No physical contact guarantee.

Keys: 1/2/3 choose a form option, N/P navigate approved fields,
O overview, I description, V location, B choose again, Space pause,
R clear tracking, Q quit. Korean speech can do the same when --stt enabled.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
import time
import cv2
import numpy as np

from multi_field_voice_tracking_v6 import (
    load_manifest, warp_points, movement, Sound, create_landmarker, put_line
)
from document_tracker_v7 import PaperTracker
from document_tracker_v7_3 import CloseRangePaperTracker
from narration_v7_1 import (
    Narrator, load_guide, single_choice_group,
    field_spoken_description, overview_text, voice_diagnostic
)
from voice_commands_v7_3 import SpeechCommands


def open_camera(index, width, height, backend='msmf'):
    backends = {
        'msmf': cv2.CAP_MSMF,
        'dshow': cv2.CAP_DSHOW,
        'auto': cv2.CAP_ANY,
    }
    api = backends[backend] if sys.platform == 'win32' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, api)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _in_image(point, shape, pad=10):
    h, w = shape[:2]
    x, y = map(float, point)
    return math.isfinite(x) and math.isfinite(y) and pad <= x < w-pad and pad <= y < h-pad


def interval_for_distance(dist_mm):
    if dist_mm > 100:
        return None     # Far: direction speech only.
    if dist_mm > 50:
        return .80
    if dist_mm > 30:
        return .50
    if dist_mm > 15:
        return .25
    if dist_mm > 8:
        return .14
    return .11


def korean_direction(direction, primary):
    name={'right':'오른쪽','left':'왼쪽','up':'위쪽','down':'아래쪽'}[direction]
    if primary < 20:
        return f'조금 더 {name}으로 움직이세요.'
    cm=round(primary/10, 1)
    return f'{name}으로 약 {cm}센티미터 이동하세요.'


def announce_overview(fields, guide, w, h, speech_enabled):
    txt=overview_text(fields,guide,w,h)
    if speech_enabled:
        txt=txt.replace('키보드에서 번호를 누르세요.',
                        '신규, 변경, 해지 중 원하는 항목을 말해 주세요. 또는 키보드에서 번호를 누르세요.')
    group=single_choice_group(guide, fields)
    if group:
        grouped=set(group['field_ids'])
        others=[f for f in fields if f['id'] not in grouped]
        if others:
            txt+=' 선택한 뒤에는 검증된 다른 작성란을 위에서부터 차례대로 안내합니다. 다음이라고 말씀하세요.'
        else:
            txt+=' 나머지 작성란과 서명란은 위치 확인이 끝나지 않아 아직 자동 안내하지 않습니다.'
    return txt


def make_sequence(fields, guide, selected_choice_id):
    """One chosen checkbox plus all OTHER approved fields, top to bottom.

    Remaining choices in the same mutually exclusive group are excluded.
    Conditions must already have been verified in navigation_adapter.
    """
    group=single_choice_group(guide, fields)
    if group:
        all_choice_ids=set(group['field_ids'])
        if selected_choice_id not in all_choice_ids:
            return []
        base=[next(t for t in fields if t['id']==selected_choice_id)]
        selected_label=str(base[0].get('label','')).replace(' ','')
        choice_labels={str(t.get('label','')).replace(' ','')
                       for t in fields if t['id'] in all_choice_ids}
        rest=[]
        for t in fields:
            if t['id'] in all_choice_ids:
                continue
            # This form can repeat 신규/변경/해지 farther down the paper.
            # Never guide a contradictory repeated choice when another is picked.
            label=str(t.get('label','')).replace(' ','')
            if t.get('class_name')=='check_box' and label in choice_labels and label!=selected_label:
                continue
            rest.append(t)
        return base+rest
    return list(fields)


def paper_present(frame, result):
    # The H is only trusted for the current frame, never stale.
    if result is None:
        return False
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--guide',type=Path)
    ap.add_argument('--model',type=Path,
                    default=Path(__file__).resolve().parent.parent/'hand_landmarker.task')
    ap.add_argument('--camera',type=int,default=1)
    ap.add_argument('--camera-backend',choices=['msmf','dshow','auto'],default='msmf')
    ap.add_argument('--frame-width',type=int,default=1280)
    ap.add_argument('--frame-height',type=int,default=720)
    ap.add_argument('--beep',action='store_true')
    ap.add_argument('--tts',action='store_true')
    ap.add_argument('--tts-rate',type=int,default=4)
    ap.add_argument('--tts-backend',choices=['powershell','pyttsx3'],default='powershell')
    ap.add_argument('--stt',action='store_true',help='Recognize Korean commands via microphone + Google (online)')
    ap.add_argument('--mic-index',type=int,default=None)
    ap.add_argument('--list-mics',action='store_true')
    ap.add_argument('--test-voice',action='store_true')
    ap.add_argument('--strict-full-page',action='store_true',
                    help='Use only old strict full-page ORB; close-range partial mode otherwise')
    ap.add_argument('--max-feature-size',type=int,default=1080)
    ap.add_argument('--hit-mm',type=float,default=8.,
                    help='Unverified proximity band, not proof of fingertip contact')
    ap.add_argument('--hold-sec',type=float,default=.65)
    ap.add_argument('--mirror',action='store_true')
    ap.add_argument('--check-setup',action='store_true')
    args=ap.parse_args()
    if args.list_mics:
        SpeechCommands.list_microphones()
        return
    if args.test_voice:
        raise SystemExit(voice_diagnostic(args.tts_backend,args.tts_rate))
    if not (3 <= args.hit_mm <= 20) or not (.35 <= args.hold_sec <= 2):
        ap.error('hit-mm must be 3..20, hold-sec must be 0.35..2')
    if args.manifest is None or args.reference is None:
        ap.error('--manifest and --reference are required unless --list-mics or --test-voice')
    manifest=load_manifest(args.manifest)
    targets=sorted(manifest['fields'], key=lambda f:(float(f['target_mm']['y']),float(f['target_mm']['x'])))
    guide=load_guide(args.guide,targets)
    group=single_choice_group(guide,targets)
    group_ids=list(group['field_ids']) if group else []
    group_fields=[next(f for f in targets if f['id']==fid) for fid in group_ids]
    try:
        if args.strict_full_page:
            tracker=PaperTracker(manifest,args.reference,max_dim=args.max_feature_size)
        else:
            tracker=CloseRangePaperTracker(manifest,args.reference,max_dim=args.max_feature_size)
    except Exception as exc:
        ap.error(f'Cannot initialize paper tracker: {exc}')
    print('v7.3.1 | Camera:', args.camera, args.camera_backend.upper(),
          '| close-range mode:', not args.strict_full_page)
    print('Reference:', tracker.ref_path, '| approved fields:', len(targets),
          '| reference ORB keypoints:', len(tracker.kp_ref))
    print('STT:', 'ON (online Google Korean, not iOS native)' if args.stt else 'OFF',
          '| beep:', args.beep, '| TTS:', args.tts)
    if group_ids:
        print('Mutually exclusive:',group_ids)
    print('Approved navigation targets:')
    for f in targets:
        p=f['target_mm']
        print(' ',f['id'],f.get('label',''),f.get('class_name',''),f'({p["x"]:.1f},{p["y"]:.1f})mm')
    if args.check_setup:
        print('Setup check PASS. Actual phone positioning accuracy NOT verified.')
        return
    if not args.model.is_file():
        ap.error('MediaPipe model not found: '+str(args.model))
    try:
        mp, landmark=create_landmarker(args.model)
    except Exception as exc:
        ap.error('MediaPipe error: '+str(exc))
    cap=open_camera(args.camera,args.frame_width,args.frame_height,args.camera_backend)
    if not cap.isOpened():
        landmark.close()
        ap.error('Camera cannot be opened. Check Camo virtual camera and MSMF.')
    sound=Sound(args.beep)
    voice=Narrator(args.tts,args.tts_backend,args.tts_rate)
    speech=SpeechCommands(args.stt,voice,args.mic_index)
    speech.start()
    # If no choice group, navigate from the first reviewed field.
    choice_id=None if group else ''
    sequence=make_sequence(targets,guide,choice_id)
    current=0
    paused=False
    overview_announced=False
    last_overview_lock=None
    last_tracked=False
    last_tracking_alert=0.
    tracking_lost_since=None
    loss_alerted=False
    last_debug_print=0.
    last_target_offscreen=False
    last_offscreen_alert=0.
    next_beep=0.
    last_direction_voice=0.
    prev_voice_direction=None
    arrived_since=None
    arrived_latched=False
    finger_smooth=None
    last_finger_seen=0.
    last_mp_ms=-1
    started=time.monotonic()
    fps=0.; fps_last=started
    window='iPhone Camo v7.3.1 | close paper | speech commands | Q quit'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    # Announce the reviewed items even if the paper temporarily cannot be tracked.
    # Do not announce successful navigation until geometry is trustworthy.
    if args.tts:
        voice.say(announce_overview(targets, guide,
                  float(manifest['paper_width_mm']),
                  float(manifest['paper_height_mm']), args.stt))
        overview_announced=True

    def speak_field():
        if sequence:
            field=sequence[current]
            voice.say(field_spoken_description(field,guide,
                      float(manifest['paper_width_mm']),float(manifest['paper_height_mm'])))

    def reset_target_audio():
        nonlocal arrived_since, arrived_latched, next_beep, last_direction_voice
        nonlocal prev_voice_direction, finger_smooth
        arrived_since=None
        arrived_latched=False
        next_beep=time.monotonic()+.45
        last_direction_voice=0.
        prev_voice_direction=None
        finger_smooth=None

    def choose(index):
        nonlocal choice_id,sequence,current,paused
        if not (0 <= index < len(group_fields)):
            return
        choice_id=group_fields[index]['id']
        sequence=make_sequence(targets,guide,choice_id)
        current=0
        paused=False
        reset_target_audio()
        print('[CHOICE]',choice_id,'| sequence:',[f['id'] for f in sequence])
        voice.say(f'{group_fields[index]["label"]} 항목을 선택했습니다. '+
                  field_spoken_description(sequence[0],guide))
        if len(sequence)==1:
            voice.say('이 문서의 다른 작성란은 아직 개발자 확인을 거치지 않아 자동 안내하지 않습니다.')

    def change_field(step):
        nonlocal current
        if not sequence:
            voice.say('먼저 신규, 변경, 해지 중 하나를 선택해 주세요.')
            return
        current=(current+step)%len(sequence)
        reset_target_audio()
        print('[SELECT]',sequence[current]['id'],sequence[current].get('label',''))
        speak_field()

    def act(action):
        nonlocal paused,choice_id,sequence,current
        op,arg=action
        if op=='choice':
            choose(arg)
        elif op=='next':
            change_field(1)
        elif op=='prev':
            change_field(-1)
        elif op=='overview':
            voice.say(announce_overview(targets,guide,float(manifest['paper_width_mm']),
                                         float(manifest['paper_height_mm']),args.stt))
        elif op=='detail':
            speak_field()
        elif op=='position':
            if sequence:
                speak_field()
            else:
                voice.say('안내받을 항목을 먼저 선택해 주세요.')
        elif op=='pause':
            paused=True
            reset_target_audio()
            voice.say('안내를 일시 중지합니다. 계속이라고 말하면 다시 시작합니다.')
        elif op=='resume':
            paused=False
            reset_target_audio()
            voice.say('안내를 재개합니다.')
        elif op=='choices':
            if not group_ids:
                voice.say('선택형 체크박스가 없습니다.')
                return
            choice_id=None
            sequence=[]
            current=0
            reset_target_audio()
            voice.say('신규, 변경, 해지 중 무엇을 선택하시겠습니까?')
        elif op=='done':
            # A self-reported completion is NOT a verified mark on paper.
            voice.say('작성했다고 말씀하신 것으로 기록하지는 않습니다. 다음 안내 항목으로 넘어갑니다.')
            change_field(1)

    try:
        with landmark:
            while True:
                ok,frame=cap.read()
                if not ok or frame is None:
                    print('Camera frame lost. Guidance stopped.')
                    break
                if args.mirror:
                    frame=cv2.flip(frame,1)
                h,w=frame.shape[:2]
                now=time.monotonic()
                fps_now=1/max(.001,now-fps_last)
                fps=fps*.86+fps_now*.14 if fps else fps_now
                fps_last=now
                # STT events are emitted on a separate thread.
                cmd=speech.get()
                if cmd:
                    act(cmd)
                vis=frame.copy()
                result=tracker.update(frame)
                if result is None:
                    # Loss of homography pauses guidance immediately, but speech
                    # is rate-limited to avoid noisy acquired/lost ping-pong.
                    if tracking_lost_since is None:
                        tracking_lost_since=now
                    if now-last_debug_print>2.0:
                        print('[TRACK LOST]', tracker.last_reason, flush=True)
                        last_debug_print=now
                    if (not loss_alerted and now-tracking_lost_since>3.0
                            and now-last_tracking_alert>15.0):
                        voice.say('문서 추적이 잠시 불안정합니다. 화면에서 종이의 인쇄된 글씨가 선명하게 보이도록 해 주세요.')
                        last_tracking_alert=now
                        loss_alerted=True
                    if last_tracked:
                        reset_target_audio()
                    last_tracked=False
                    last_overview_lock=None
                    last_target_offscreen=False
                    finger_smooth=None
                    arrived_since=None
                    put_line(vis,'SEARCHING / PAUSED - TRACKING NOT CONFIDENT',29,(0,60,255),.66)
                    put_line(vis,tracker.last_reason[:85],60,(0,200,255),.53)
                    put_line(vis,'Show a textured area of SAME paper; avoid blur/occlusion.',90,(230,230,230),.48)
                else:
                    if tracking_lost_since is not None and now-last_debug_print>2.0:
                        print('[TRACK REACQUIRED]', result.method, result.detail, flush=True)
                        last_debug_print=now
                    tracking_lost_since=None
                    loss_alerted=False
                    last_tracked=True
                    if last_overview_lock is None:
                        last_overview_lock=now
                    if not overview_announced and now-last_overview_lock>.8:
                        overview_announced=True
                        voice.say(announce_overview(targets,guide,
                                  float(manifest['paper_width_mm']),float(manifest['paper_height_mm']),args.stt))
                    # Drawing a quadrilateral is valid even when its corners
                    # lie offscreen; OpenCV clips lines to image boundaries.
                    q=result.corners
                    if np.max(np.abs(q))<30000:
                        cv2.polylines(vis,[np.round(q).astype(np.int32)],True,(0,210,35),2)
                    put_line(vis,f'TRACKING [{result.method}] {result.detail[:45]}',29,(0,240,80),.59)
                    if paused:
                        put_line(vis,'PAUSED BY USER',61,(0,200,255),.67)
                    elif not sequence:
                        put_line(vis,'SAY: NEW / CHANGE / CANCEL (or press 1/2/3)',61,(0,210,255),.59)
                    else:
                        field=sequence[current]
                        target=field['target_mm']
                        target_mm=(float(target['x']),float(target['y']))
                        loc=warp_points([target_mm],result.H_mm_to_live)[0]
                        if not _in_image(loc,frame.shape,pad=12):
                            finger_smooth=None
                            arrived_since=None
                            if not last_target_offscreen and now-last_offscreen_alert>2:
                                voice.say('목표 작성란이 현재 카메라 화면 밖에 있습니다. 휴대폰 위치를 조금 이동해 주세요.')
                                last_offscreen_alert=now
                            last_target_offscreen=True
                            put_line(vis,'TARGET OUTSIDE CAMERA VIEW - NO GUIDANCE',62,(0,160,255),.58)
                        else:
                            last_target_offscreen=False
                            cv2.drawMarker(vis,tuple(np.rint(loc).astype(int)),(0,0,255),
                                           cv2.MARKER_CROSS,27,3)
                            rgb=np.ascontiguousarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                            mp_img=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                            stamp=max(last_mp_ms+1,int((now-started)*1000))
                            last_mp_ms=stamp
                            detected=landmark.detect_for_video(mp_img,stamp)
                            finger_valid=False
                            if detected.hand_landmarks:
                                tip=detected.hand_landmarks[0][8]
                                px,py=float(tip.x*w),float(tip.y*h)
                                if 0<=px<w and 0<=py<h:
                                    cv2.circle(vis,(round(px),round(py)),8,(255,120,0),-1)
                                    world=warp_points([(px,py)],result.H_live_to_mm)[0]
                                    pw=float(manifest['paper_width_mm'])
                                    ph=float(manifest['paper_height_mm'])
                                    if np.isfinite(world).all() and 0<=world[0]<=pw and 0<=world[1]<=ph:
                                        finger_valid=True
                                        if finger_smooth is None or now-last_finger_seen>.5:
                                            finger_smooth=world
                                        else:
                                            finger_smooth=.55*world+.45*finger_smooth
                                        last_finger_seen=now
                                        dist,direction,primary=movement(finger_smooth,target_mm)
                                        put_line(vis,f'TIP ({finger_smooth[0]:.0f},{finger_smooth[1]:.0f})mm  DIST {dist:.1f}mm  {direction}',
                                                 61,(245,245,245),.56)
                                        if dist <= args.hit_mm:
                                            if arrived_since is None:
                                                arrived_since=now
                                            if now-arrived_since>=args.hold_sec:
                                                put_line(vis,'AT TARGET AREA (CONTACT NOT VERIFIED)',91,(0,255,100),.58)
                                                if not arrived_latched:
                                                    arrived_latched=True
                                                    sound.play(1280,800)  # long victory tone from older design
                                                    voice.say('여기입니다. 목표 위치 부근입니다. 실제 종이에 닿은 위치를 확인해 주세요.')
                                        else:
                                            arrived_since=None
                                            if arrived_latched and dist>args.hit_mm*1.65:
                                                arrived_latched=False
                                            if not arrived_latched:
                                                interval=interval_for_distance(dist)
                                                if interval is not None and now>=next_beep and not voice.busy:
                                                    sound.play(920,55)
                                                    next_beep=now+interval
                                                if args.tts and not voice.busy:
                                                    need=now-last_direction_voice > (3.7 if dist>100 else 5.2)
                                                    if need and (dist>100 or dist>25 or direction!=prev_voice_direction):
                                                        voice.say(korean_direction(direction,primary))
                                                        last_direction_voice=now
                                                        prev_voice_direction=direction
                            if not finger_valid:
                                finger_smooth=None
                                arrived_since=None
                                put_line(vis,'NO VALID FINGER ON PAPER - GUIDANCE PAUSED',61,(0,180,255),.54)
                if sequence:
                    curr=sequence[current]
                    caption=f'{curr["id"]}: {curr.get("label","")}  {current+1}/{len(sequence)}'
                else:
                    caption='CHOICE REQUIRED'
                put_line(vis,caption[:65],h-37,(240,240,240),.54)
                put_line(vis,f'CAM {args.camera} MSMF | {fps:.0f}fps | STT {"ON" if speech.ready else "OFF"}',h-12,(240,240,240),.47)
                cv2.imshow(window,vis)
                key=cv2.waitKey(1)&255
                if key in (ord('q'),ord('Q'),27):
                    break
                elif key in (ord('1'),ord('2'),ord('3')) and group_ids:
                    act(('choice',key-ord('1')))
                elif key in (ord('n'),ord('N')):
                    act(('next',None))
                elif key in (ord('p'),ord('P')):
                    act(('prev',None))
                elif key in (ord('o'),ord('O')):
                    act(('overview',None))
                elif key in (ord('i'),ord('I')):
                    act(('detail',None))
                elif key in (ord('v'),ord('V')):
                    act(('position',None))
                elif key in (ord('b'),ord('B')):
                    act(('choices',None))
                elif key==32:
                    act(('resume',None) if paused else ('pause',None))
                elif key in (ord('r'),ord('R')):
                    tracker.last_good=None
                    tracker.pending=None
                    tracker.pending_count=0
                    tracker.fail_frames=0
                    finger_smooth=None
                    arrived_since=None
                    print('Tracker reset, re-acquire current page.')
    finally:
        speech.stop.set()
        voice.stop.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__=='__main__':
    main()
