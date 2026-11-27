import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
from pathlib import Path
from collections import deque
from dataclasses import dataclass

W, H = 210, 297
WINDOW = 'Multi Field Voice Navigation'
DEST = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
MODEL = Path(__file__).resolve().parent / 'hand_landmarker.task'

@dataclass(frozen=True)
class Field:
    kind: str
    number: int
    box: tuple  # x1, y1, x2, y2 in paper mm (not camera pixels)

    @property
    def name(self):
        return f'{self.kind} {self.number}번'

    @property
    def target(self):
        x1, y1, x2, y2 = self.box
        if self.kind == '체크박스':
            return ((x1+x2)/2, (y1+y2)/2)
        return (x1+3, (y1+y2)/2) # demo signature anchor, not verified writing point

    @property
    def tolerance(self):
        return 4.0 if self.kind == '체크박스' else 10.0

# DEMO ONLY: replace with verified detections in rectified paper coordinates.
FIELDS = [Field('체크박스', 1, (20, 75, 29, 84)),
          Field('체크박스', 2, (20, 105, 29, 114)),
          Field('체크박스', 3, (20, 155, 29, 164)),
          Field('서명란', 1, (145, 220, 190, 240)),
          Field('서명란', 2, (145, 255, 190, 275))]

def location(x, y):
    vertical = '위쪽' if y < H/3 else '가운데' if y < 2*H/3 else '아래쪽'
    horizontal = '왼쪽' if x < W/3 else '중앙' if x < 2*W/3 else '오른쪽'
    return f'{vertical} {horizontal}'

def introduction():
    counts = '체크박스 세 개와 서명란 두 개' if len(FIELDS)==5 and sum(f.kind=='체크박스' for f in FIELDS)==3 else ', '.join(f'{kind} {sum(f.kind==kind for f in FIELDS)}개' for kind in ('체크박스','서명란'))
    return '실험용 문서입니다. ' + counts + '가 있습니다. ' + '. '.join(f'{i+1}번 선택 항목, {f.name}, {location(*f.target)}' for i,f in enumerate(FIELDS)) + '. 안내받을 항목을 선택해 주세요.'

# Adaptive guidance: voice only beyond 100 mm; voice + beeps within 100 mm.
BEEP_START_MM = 100.0

def beep_interval(distance):
    """Seconds between beep starts; None means voice-only mode."""
    if distance > BEEP_START_MM:
        return None
    return float(np.interp(distance, [0, 15, 30, 50, 100], [.12, .12, .25, .50, .80]))

def speech_intervals(distance):
    # Reduce repeated speech near the target so the beeps dominate.
    return (2.5, 6.0) if distance <= 15 else (1.5, 4.0)


def direction_for(dx, dy):
    if abs(dx) >= abs(dy):
        return ('RIGHT', '오른쪽으로') if dx > 0 else ('LEFT', '왼쪽으로')
    return ('DOWN','아래쪽으로') if dy > 0 else ('UP','위쪽으로')

class Audio:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.generation = 0
        self.pending = None
        self.state = ('waiting', '', None, time.monotonic())
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def switch(self, message=None):
        with self.lock:
            self.generation += 1
            self.pending = message
            self.state = ('waiting', '', None, time.monotonic())

    def update(self, status, direction='', distance=None):
        with self.lock:
            self.state = (status, direction, distance, time.monotonic())

    def run(self):
        # SAPI is initialized and used only in this audio thread.
        import comtypes
        import comtypes.client
        comtypes.CoInitialize()
        voice = None
        try:
            voice = comtypes.client.CreateObject('SAPI.SpVoice', dynamic=True)
            for token in voice.GetVoices():
                desc = token.GetDescription().lower()
                if any(k in desc for k in ('korean', 'heami', '한국')):
                    voice.Voice = token
                    break
            voice.Rate = 0
            voice.Volume = 100
            print('[TTS] Ready:', voice.Voice.GetDescription(), flush=True)
        except Exception as e:
            print('[TTS] Initialization failed:', repr(e), flush=True)
            voice = None

        def say(message):
            print('[VOICE]', message, flush=True)
            if voice is not None:
                try:
                    voice.Speak(message, 3)  # async + purge previous speech
                    return True
                except Exception as e:
                    print('[TTS] Speak failed:', repr(e), flush=True)
            return False

        def cancel():
            if voice is not None:
                try:
                    voice.Speak('', 3)
                except Exception as e:
                    print('[TTS] Cancel failed:', repr(e), flush=True)

        def beep(freq, duration):
            try:
                winsound.Beep(freq, duration)
            except RuntimeError as e:
                print('[BEEP]', e, flush=True)
                self.stop.wait(duration/1000)

        generation = -1
        last_voice = last_beep = -100.0
        last_phrase = ''
        announced_arrival = False
        speaking = False
        speech_kind = ''
        try:
            while not self.stop.is_set():
                with self.lock:
                    g = self.generation
                    pending = self.pending
                    self.pending = None
                    status, direction, distance, updated = self.state
                now = time.monotonic()
                if now-updated > 1.0:
                    status = 'waiting'
                if g != generation:
                    cancel()
                    generation = g
                    speaking = False
                    last_voice = last_beep = -100.0
                    last_phrase = ''
                    announced_arrival = False
                if pending:
                    speaking = say(pending)
                    speech_kind = 'intro'
                    last_voice = now
                if speaking:
                    # Cancel obsolete guidance, but allow document introduction to finish.
                    if speech_kind == 'direction' and status in ('lost', 'outside', 'arrived', 'waiting', 'holding'):
                        cancel()
                        speaking = False
                    else:
                        try:
                            speaking = not bool(voice.WaitUntilDone(0))
                        except Exception as e:
                            print('[TTS] Playback check failed:', repr(e), flush=True)
                            cancel()
                            speaking = False
                    if speaking:
                        self.stop.wait(.02)
                        continue
                    last_voice = time.monotonic()
                if status == 'arrived':
                    if not announced_arrival:
                        beep(1400, 850)
                        with self.lock:
                            valid = self.generation == g and self.state[0] == 'arrived' and time.monotonic()-self.state[3] <= 1.0
                        if valid and not self.stop.is_set():
                            speaking = say('목표 위치에 도착했습니다. 다음 항목을 선택해 주세요.')
                            speech_kind = 'arrival'
                            announced_arrival = True
                elif status in ('moving', 'holding') and distance is not None:
                    phrase = direction + (' 조금 더 이동하세요.' if distance < 30 else ' 이동하세요.')
                    interval = beep_interval(distance)
                    change_delay, repeat_delay = speech_intervals(distance)
                    if status == 'moving' and ((phrase != last_phrase and now-last_voice >= change_delay) or now-last_voice >= repeat_delay):
                        speaking = say(phrase)
                        speech_kind = 'direction'
                        last_phrase = phrase
                        last_voice = now
                    elif interval is not None and now-last_beep >= interval:
                        last_beep = now
                        beep(1050, 65)
                elif status in ('lost', 'outside'):
                    phrase = {'lost':'검지가 인식되지 않습니다. 손을 카메라 안에 보여주세요.', 'outside':'검지를 문서 위에 올려주세요.'}[status]
                    if (phrase != last_phrase and now-last_voice >= 1.5) or now-last_voice >= 5.0:
                        speaking = say(phrase)
                        speech_kind = 'status'
                        last_phrase = phrase
                        last_voice = now
                self.stop.wait(.015)
        finally:
            cancel()
            voice = None
            comtypes.CoUninitialize()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=4)

def main():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_buffer=MODEL.read_bytes()),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=.3, min_hand_presence_confidence=.3,
        min_tracking_confidence=.5)
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('카메라를 열 수 없습니다.')
    audio = None
    corners = []
    locked = None
    forward = inverse = None
    history = deque(maxlen=5)
    selected = None
    reached = False
    arrival_start = None
    missing_since = None
    failed = 0
    last_ts = -1
    start = time.monotonic()

    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and locked is None and len(corners)<4:
            corners.append((x,y))
            print('Corner', len(corners), (x,y))

    def pixel(x, y):
        p = cv2.perspectiveTransform(np.array([[[x,y]]],np.float32), inverse)[0,0]
        return tuple(np.rint(p).astype(int))

    def choose(index, speak=True):
        nonlocal selected, reached, arrival_start, missing_since
        selected = index
        reached = False
        arrival_start = None
        missing_since = None
        history.clear()
        f = FIELDS[index]
        audio.switch(f'{f.name}으로 안내합니다. 검지를 문서 위에 올려주세요.' if speak else None)
        print('Selected:', index+1, f.name)

    def text(img, value, y, color=(0,255,255)):
        cv2.putText(img, value, (12,y), cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)

    try:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, click)
        audio = Audio()
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failed += 1
                    history.clear(); arrival_start = None
                    audio.update('waiting')
                    print(f'Frame retry {failed}/10')
                    if failed >= 10 or (cv2.waitKey(1)&255)==ord('q'):
                        break
                    time.sleep(.1)
                    continue
                failed = 0
                now = time.monotonic()
                last_ts = max(last_ts+1, int((now-start)*1000))
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),last_ts)
                display = frame.copy() # no mirror; fixed document coordinates
                if locked is None:
                    for i,p in enumerate(corners):
                        cv2.circle(display,p,6,(255,0,0),-1)
                        cv2.putText(display,str(i+1),p,cv2.FONT_HERSHEY_SIMPLEX,.6,(255,0,0),2)
                    text(display,f'Corners {len(corners)}/4: TL TR BR BL; SPACE',25)
                else:
                    cv2.polylines(display,[locked.astype(np.int32)],True,(0,255,0),2)
                    for i,f in enumerate(FIELDS):
                        x1,y1,x2,y2 = f.box
                        poly = np.array([pixel(x1,y1),pixel(x2,y1),pixel(x2,y2),pixel(x1,y2)],np.int32)
                        color = (0,220,0) if f.kind=='체크박스' else (255,0,0)
                        if i==selected: color = (0,255,255)
                        cv2.polylines(display,[poly],True,color,3 if i==selected else 2)
                        cv2.putText(display,f'{i+1}: '+('C' if f.kind=='체크박스' else 'S')+str(f.number),pixel(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
                    text(display,'LOCKED | 1-5 select, N/P next/prev, I intro',25)
                if result.hand_landmarks:
                    h,w=frame.shape[:2]
                    tip=result.hand_landmarks[0][8]
                    x,y=tip.x*w,tip.y*h
                    cv2.circle(display,(round(x),round(y)),8,(0,0,255),-1)
                    missing_since=None
                    if forward is not None and selected is not None:
                        f=FIELDS[selected]
                        tx,ty=f.target
                        target_px=pixel(tx,ty)
                        cv2.circle(display,target_px,6,(255,0,0),-1)
                        pos=cv2.perspectiveTransform(np.array([[[x,y]]],np.float32),forward)[0,0]
                        if np.isfinite(pos).all():
                            history.append(pos)
                            mx,my=np.mean(history,axis=0)
                            dx,dy=tx-mx,ty-my
                            dist=float(np.hypot(dx,dy))
                            raw_dist=float(np.hypot(tx-pos[0],ty-pos[1]))
                            inside=0<=mx<=W and 0<=my<=H and 0<=pos[0]<=W and 0<=pos[1]<=H
                            text(display,f'A4 ({mx:.1f}, {my:.1f}) mm | distance {dist:.1f} mm',53)
                            if reached:
                                text(display,'ARRIVED - select another field',81,(0,255,0))
                                audio.update('arrived')
                            elif not inside:
                                arrival_start=None
                                text(display,'Finger outside paper',81)
                                audio.update('outside')
                            elif dist<=f.tolerance and raw_dist<=f.tolerance:
                                if arrival_start is None: arrival_start=now
                                if now-arrival_start>=.7:
                                    reached=True
                                    audio.update('arrived')
                                    text(display,'TARGET REACHED',81,(0,255,0))
                                else:
                                    audio.update('holding', distance=dist)
                                    text(display,'Hold still',81)
                            else:
                                arrival_start=None
                                en,ko=direction_for(dx,dy)
                                audio.update('moving',ko,dist)
                                text(display,'Move '+en,81)
                                text(display,'VOICE ONLY' if dist>BEEP_START_MM else 'VOICE + BEEP',109)
                                cv2.arrowedLine(display,(round(x),round(y)),target_px,(0,255,255),2,tipLength=.15)
                        else:
                            history.clear(); arrival_start=None; audio.update('waiting')
                else:
                    history.clear(); arrival_start=None
                    if missing_since is None: missing_since=now
                    if selected is not None:
                        if reached: audio.update('arrived')
                        elif now-missing_since>=.5: audio.update('lost')
                        else: audio.update('waiting')
                        text(display,'Finger not detected',53,(0,0,255))
                cv2.imshow(WINDOW,display)
                key=cv2.waitKey(1)&255
                if key==ord('q'): break
                if key==32 and locked is None and len(corners)==4:
                    candidate=np.array(corners,np.float32)
                    con=candidate.reshape(4,1,2)
                    if cv2.isContourConvex(con) and cv2.contourArea(con)>1000:
                        locked=candidate.copy()
                        forward=cv2.getPerspectiveTransform(locked,DEST)
                        inverse=cv2.getPerspectiveTransform(DEST,locked)
                        audio.switch(introduction())
                        print('LOCKED. Select 1-5; N/P next/previous; I introduction; R reset; Q quit')
                    else: print('Invalid corners: press R and retry.')
                elif key==ord('r'):
                    corners.clear(); locked=forward=inverse=None
                    selected=None; reached=False; arrival_start=None; missing_since=None
                    history.clear(); audio.switch()
                elif locked is not None:
                    if ord('1')<=key<=ord('9') and key-ord('1')<len(FIELDS): choose(key-ord('1'))
                    elif key==ord('n'): choose(0 if selected is None else (selected+1)%len(FIELDS))
                    elif key==ord('p'): choose(len(FIELDS)-1 if selected is None else (selected-1)%len(FIELDS))
                    elif key==ord('i'):
                        selected=None; reached=False; arrival_start=None; missing_since=None
                        history.clear(); audio.switch(introduction())
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if audio is not None: audio.close()

if __name__=='__main__':
    main()
