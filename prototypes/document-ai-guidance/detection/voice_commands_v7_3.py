"""Korean speech commands for the Windows developer PoC.

Uses SpeechRecognition's online Google recognizer in this prototype.
Not the Apple Vision OCR, not a local/offline STT model, and not an iOS app.
Microphone capture runs in a daemon thread; OpenCV UI is never blocked.
Pauses recognition during TTS as a half-duplex safeguard.
"""
from __future__ import annotations
import queue
import re
import threading
import time


def normalize(text: str) -> str:
    return re.sub(r'[^0-9가-힣]', '', str(text).lower())


def parse_command(text):
    """Return (action, optional index) or None. Only short, explicit commands."""
    s = normalize(text)
    if not s or len(s) > 20:
        return None
    if any(t in s for t in ('일시중지', '중단', '멈춰', '멈춰줘', '정지')):
        return ('pause', None)
    if any(t in s for t in ('다시시작', '이어하기', '재개', '계속해', '계속')):
        return ('resume', None)
    if any(t in s for t in ('다음항목', '다음으로', '다음', '넘어가')):
        return ('next', None)
    if any(t in s for t in ('이전항목', '이전으로', '이전', '뒤로가')):
        return ('prev', None)
    if any(t in s for t in ('전체설명', '처음부터', '다시읽어', '문서설명', '문서내용', '전체읽어')):
        return ('overview', None)
    if any(t in s for t in ('위치안내', '어디있', '어디야', '위치알려', '좌표')):
        return ('position', None)
    if any(t in s for t in ('항목설명', '설명해', '자세히', '뭐하는')):
        return ('detail', None)
    if any(t in s for t in ('선택변경', '선택바꿔', '다른선택')):
        return ('choices', None)
    if any(t in s for t in ('완료했', '다했어', '작성완료', '체크했')):
        return ('done', None)
    # Only choice utterances, not arbitrary text containing '변경'.
    vals = [
        (0, ('신규', '새로신청', '일번', '1번', '첫번째', '첫째')),
        (1, ('변경', '이번', '2번', '두번째', '둘째')),
        (2, ('해지', '삼번', '3번', '세번째', '셋째')),
    ]
    for i, aliases in vals:
        if any(s == alias or (len(s) < 12 and s in (alias+'선택', alias+'할게', alias+'으로해줘',alias+'으로할게')) for alias in aliases):
            return ('choice', i)
    return None


class SpeechCommands:
    def __init__(self, enabled, speaker, mic_index=None):
        self.enabled = bool(enabled)
        self.speaker = speaker
        self.mic_index = mic_index
        self.events = queue.Queue(maxsize=12)
        self.stop = threading.Event()
        self.error = None
        self.ready = False
        self.last_heard = ''
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            r.dynamic_energy_threshold = True
            r.pause_threshold = .65
            r.non_speaking_duration = .25
            with sr.Microphone(device_index=self.mic_index) as microphone:
                print('[STT] Measuring ambient noise for 0.7 seconds...')
                r.adjust_for_ambient_noise(microphone, duration=.7)
                self.ready = True
                last_busy = time.monotonic()
                print('[STT] Listening for: 신규/변경/해지, 다음, 이전, 위치, 설명, 중단, 계속.')
                while not self.stop.is_set():
                    if self.speaker.busy:
                        last_busy = time.monotonic()
                        time.sleep(.10)
                        continue
                    if time.monotonic() - last_busy < .8:
                        time.sleep(.1)
                        continue
                    try:
                        # Short listen window; thread runs independently from video.
                        audio = r.listen(microphone, timeout=.8, phrase_time_limit=3.2)
                    except sr.WaitTimeoutError:
                        continue
                    if self.speaker.busy:
                        last_busy = time.monotonic()
                        continue
                    try:
                        transcript = r.recognize_google(audio, language='ko-KR')
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as exc:
                        self.error = f'STT network error: {exc}'
                        print('[STT]', self.error)
                        time.sleep(2)
                        continue
                    command = parse_command(transcript)
                    self.last_heard = transcript
                    if command is not None and not self.speaker.busy:
                        print('[STT]', transcript, '=>', command, flush=True)
                        try:
                            self.events.put_nowait(command)
                        except queue.Full:
                            pass
                    else:
                        print('[STT ignored]', transcript, flush=True)
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            self.error = str(exc)
            print('[STT INIT ERROR]', exc, flush=True)
            print('Install with: .\\.venv_scan\\Scripts\\python.exe -m pip install SpeechRecognition pyaudio', flush=True)
        finally:
            self.ready = False

    @staticmethod
    def list_microphones():
        import speech_recognition as sr
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f'{i}: {name}')

    def get(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None
