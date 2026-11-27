import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
import re
import queue
import speech_recognition as sr
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
    description: str = ""  # verified OCR/template text, or explicit demo text

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
FIELDS = [Field('체크박스', 1, (20, 75, 29, 84), '개인정보 수집 동의 여부'),
          Field('체크박스', 2, (20, 105, 29, 114), '개인정보 제3자 제공 동의 여부'),
          Field('체크박스', 3, (20, 155, 29, 164), '안내 사항 확인 여부'),
          Field('서명란', 1, (145, 220, 190, 240), '신청인 본인 서명란'),
          Field('서명란', 2, (145, 255, 190, 275), '확인자 서명란')]

ORDINALS = ('첫 번째', '두 번째', '세 번째', '네 번째', '다섯 번째',
            '여섯 번째', '일곱 번째', '여덟 번째', '아홉 번째', '열 번째')
COUNTS = ('영', '한', '두', '세', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열')

def location(x, y):
    vertical = '상단' if y < H/3 else '중앙부' if y < 2*H/3 else '하단부'
    horizontal = '왼쪽' if x < W/3 else '중앙' if x < 2*W/3 else '오른쪽'
    return f'{vertical} {horizontal}'

def sino_number(number):
    """Render a nonnegative integer as Korean Sino-Korean numerals."""
    number = int(number)
    if number < 0:
        return '마이너스 ' + sino_number(-number)
    if number == 0:
        return '영'
    digits = '영일이삼사오육칠팔구'
    large_units = ('', '만', '억', '조', '경', '해', '자', '양', '구', '간', '정', '재', '극')
    groups = []
    index = 0
    while number:
        number, group = divmod(number, 10000)
        if index >= len(large_units):
            # Guard against identifiers beyond conventional Korean large units.
            raise ValueError('Field number exceeds supported Korean number units')
        if group:
            parts = []
            for value, unit in ((1000, '천'), (100, '백'), (10, '십'), (1, '')):
                digit, group = divmod(group, value)
                if digit:
                    parts.append(('' if digit == 1 and unit else digits[digit]) + unit)
            groups.append(''.join(parts) + large_units[index])
        index += 1
    return ''.join(reversed(groups))


def field_label(f):
    return f'{sino_number(f.number)}번 {f.kind}'


def introduction():
    counts = []
    for kind in ('체크박스', '서명란', '작성란'):
        n = sum(f.kind == kind for f in FIELDS)
        if n:
            counts.append(f'{kind} {COUNTS[n] if n < len(COUNTS) else sino_number(n)} 개')
    if not counts:
        return '실험용 문서입니다. 안내할 항목이 없습니다.'
    summary = ('와 '.join(counts) if len(counts) == 2 else ', '.join(counts))
    lines = []
    for i, f in enumerate(FIELDS):
        ordinal = ORDINALS[i] if i < len(ORDINALS) else f'{sino_number(i+1)} 번째'
        description = f.description.strip().rstrip('.!?。')
        prefix = f'{ordinal} 선택 항목, {field_label(f)}.'
        position = location(*f.target)
        if description:
            lines.append(f'{prefix} {description}이며 {position}에 있습니다.')
        else:
            lines.append(f'{prefix} 항목 설명을 확인하지 못했습니다. {position}에 있습니다.')
    return '실험용 문서입니다. ' + summary + '가 있습니다. ' + ' '.join(lines) + ' 안내받을 항목을 선택해 주세요.'

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

def parse_command(text):
    """Return (action, argument); do not guess ambiguous field selections."""
    t = re.sub(r"[^가-힣0-9]", "", text.lower())
    t = t.replace('싸인란', '서명란').replace('사인란', '서명란').replace('사인', '서명란').replace('체크박쓰', '체크박스')
    if any(k in t for k in ('중단','멈춰','취소','그만')): return ('stop', None)
    if '종료' in t: return ('quit', None)
    if any(k in t for k in ('문서정보','문서설명','다시알려','다시설명','항목목록')): return ('intro', None)
    kinds = [k for k in ('체크박스','서명란') if k in t]
    if len(kinds)>1: return ('unknown', None)
    if '다음' in t: return ('next', kinds[0] if kinds else None)
    if '이전' in t: return ('prev', kinds[0] if kinds else None)
    if any(k in t for k in ('아니','말고','제외')): return ('unknown', None)
    # Explicit menu number takes precedence only when the remaining label agrees.
    menu = re.search(r'(첫번째|두번째|세번째|네번째|다섯번째|[0-9]+(?:번째|번))선택항목', t)
    if menu:
        token = menu.group(1)
        ordinal_map = {'첫번째':1, '두번째':2, '세번째':3, '네번째':4, '다섯번째':5}
        n = ordinal_map[token] if token in ordinal_map else int(re.match(r'[0-9]+', token).group())
        if not 1 <= n <= len(FIELDS): return ('unknown', None)
        f = FIELDS[n-1]
        if kinds and f.kind != kinds[0]: return ('unknown', None)
        rest = t[:menu.start()] + t[menu.end():]
        if kinds:
            local_numbers = re.findall(r'([0-9]+)(?:번째|번)', rest)
            if any(int(v) != f.number for v in local_numbers): return ('unknown', None)
        return ('select', n-1)
    numbers = set()
    for n, words in [(1,('첫번째','첫째','한번째','일번')), (2,('두번째','둘째','이번째')), (3,('세번째','셋째','삼번')), (4,('네번째','넷째','사번')), (5,('다섯번째','다섯째','오번'))]:
        if any(w in t for w in words): numbers.add(n)
    numbers.update(int(n) for n in re.findall(r'([0-9]+)(?:번째|번)', t))
    if len(numbers)!=1: return ('unknown', None)
    n = numbers.pop()
    if kinds:
        matches = [i for i,f in enumerate(FIELDS) if f.kind==kinds[0] and f.number==n]
        return ('select',matches[0]) if len(matches)==1 else ('unknown',None)
    return ('select',n-1) if 1<=n<=len(FIELDS) else ('unknown',None)


class Audio:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.generation = 0
        self.pending = None
        self.listen_requested = False
        self.commands = queue.Queue()
        self.state = ('waiting', '', None, time.monotonic())
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def switch(self, message=None, listen=False):
        with self.lock:
            self.generation += 1
            self.pending = message
            self.listen_requested = listen
            self.state = ('waiting', '', None, time.monotonic())

    def request_listen(self):
        with self.lock:
            self.generation += 1
            self.pending = '음성 명령을 말씀해 주세요.'
            self.listen_requested = True

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

        recognizer = sr.Recognizer()
        recognizer.operation_timeout = 5
        recognizer.pause_threshold = .7

        def listen_once(g):
            # Half-duplex: all TTS and beeps are silent in this audio thread.
            self.stop.wait(.35)
            if self.stop.is_set(): return
            with self.lock:
                if self.generation != g: return
            print('[STT] Listening...', flush=True)
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=.35)
                    audio = recognizer.listen(source, timeout=4, phrase_time_limit=5)
                phrase = recognizer.recognize_google(audio, language='ko-KR')
                print('[STT]', phrase, flush=True)
                command = parse_command(phrase)
            except sr.WaitTimeoutError:
                command = ('timeout', None)
            except sr.UnknownValueError:
                command = ('unknown', None)
            except (sr.RequestError, OSError) as exc:
                print('[STT] Error:', exc, flush=True)
                command = ('stt_error', None)
            with self.lock:
                if self.generation == g and not self.stop.is_set():
                    self.commands.put((g, command))

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
                with self.lock:
                    listen = self.generation == g and self.listen_requested
                    if listen: self.listen_requested = False
                if listen:
                    listen_once(g)
                    last_voice = time.monotonic()
                    continue
                if status == 'arrived':
                    if not announced_arrival:
                        beep(1400, 850)
                        with self.lock:
                            valid = self.generation == g and self.state[0] == 'arrived' and time.monotonic()-self.state[3] <= 1.0
                        if valid and not self.stop.is_set():
                            speaking = say('목표 위치에 도착했습니다. 다음 항목을 선택해 주세요.')
                            speech_kind = 'arrival'
                            announced_arrival = True
                            with self.lock:
                                if self.generation == g: self.listen_requested = True
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
        audio.switch(f'{field_label(f)}. {f.description or "항목 설명을 확인하지 못했습니다"}. 해당 위치로 안내합니다. 검지를 문서 위에 올려주세요.' if speak else None)
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
                    text(display,'LOCKED | V voice, 1-5 select, I intro',25)
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
                # Execute microphone commands on the main thread, never in the audio thread.
                try:
                    gen, (action, arg) = audio.commands.get_nowait()
                except queue.Empty:
                    action = None
                else:
                    with audio.lock:
                        if gen != audio.generation: action = None
                if action == 'quit': break
                if locked is not None and action is not None:
                    if action == 'select': choose(arg)
                    elif action in ('next','prev'):
                        indices = [i for i,f in enumerate(FIELDS) if arg is None or f.kind == arg]
                        if action == 'next':
                            index = next((i for i in indices if selected is None or i>selected), indices[0])
                        else:
                            index = next((i for i in reversed(indices) if selected is None or i<selected), indices[-1])
                        choose(index)
                    elif action in ('stop','intro'):
                        selected=None; reached=False; arrival_start=None; missing_since=None
                        history.clear()
                        audio.switch(introduction() if action=='intro' else '안내를 중단했습니다. 다시 안내받으려면 음성 명령을 사용해 주세요.', listen=action=='intro')
                    elif action == 'unknown':
                        audio.switch('명령을 확인하지 못했습니다. 다시 말씀해 주세요.', listen=True)
                    elif action == 'timeout':
                        if selected is None or reached:
                            audio.switch('음성 명령을 기다리고 있습니다.', listen=True)
                        else:
                            audio.switch('안내를 계속합니다.')
                    elif action == 'stt_error':
                        audio.switch('음성 인식에 문제가 있습니다. 브이 키로 다시 시도해 주세요.')

                if key==32 and locked is None and len(corners)==4:
                    candidate=np.array(corners,np.float32)
                    con=candidate.reshape(4,1,2)
                    if cv2.isContourConvex(con) and cv2.contourArea(con)>1000:
                        locked=candidate.copy()
                        forward=cv2.getPerspectiveTransform(locked,DEST)
                        inverse=cv2.getPerspectiveTransform(DEST,locked)
                        audio.switch(introduction(), listen=True)
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
                    elif key==ord('v'): audio.request_listen()
                    elif key==ord('i'):
                        selected=None; reached=False; arrival_start=None; missing_since=None
                        history.clear(); audio.switch(introduction(), listen=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if audio is not None: audio.close()

if __name__=='__main__':
    main()
