"""v8 state-aware Korean voice intents + half-duplex STT listener.

The STT thread ONLY captures text. Meaning is decided on the main thread by
DialogueController using the CURRENT dialogue state, because the same word
('예', '변경', '다음') means different things in different states.

Echo guard: audio captured while (or right after) our own TTS was speaking is
discarded, so the recognizer cannot hear its own prompt and change state.

v7.5 modules (voice_commands_v7_5, narration_v7_1) are imported, not modified.
STT engine is still SpeechRecognition + online Google (ko-KR), as in v7.5.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass

from narration_v7_1 import Narrator, speak_sync


def normalize(text) -> str:
    """Same rule as voice_commands_v7_5.normalize: keep digits and Hangul only."""
    return re.sub(r'[^0-9가-힣]', '', str(text).lower())


@dataclass(frozen=True)
class Intent:
    kind: str            # yes no done next prev skip repeat overview position pause resume change_type choice
    value: int | None = None
    raw: str = ''


# ---------------------------------------------------------------- lexicon
# Keyword (substring) commands, checked in this order. More specific first.
KEYWORD_INTENTS = (
    ('pause', ('일시중지', '일시정지', '중단', '멈춰', '정지', '그만', '잠깐')),
    ('resume', ('다시시작', '이어서', '이어하기', '재개', '계속')),
    ('change_type', ('다른신청', '선택변경', '선택바꿔', '다른선택', '신청유형', '유형변경',
                     '유형바꿔', '처음으로', '처음부터')),
    ('overview', ('문서설명', '전체설명', '문서내용', '무슨문서', '전체읽어', '문서다시')),
    ('position', ('위치', '어디있', '어디야', '어디예요', '어디에요', '어디죠')),
    ('skip', ('건너뛰', '건너뛸', '건너띄', '패스', '스킵', '생략', '넘겨')),
    ('done', ('완료', '다썼', '다했', '썼어', '썼습니다', '썼어요', '적었', '작성했', '체크했',
              '표시했', '서명했', '찍었', '끝났', '다됐')),
    ('next', ('다음', '넘어가')),
    ('prev', ('이전', '뒤로', '앞항목', '전항목', '앞으로돌아', '돌아가')),
    ('repeat', ('다시설명', '다시말해', '다시읽어', '다시한번', '한번더', '뭐라고', '설명해',
                '무슨뜻', '무슨의미', '못들었', '다시')),
)

# '어' is deliberately NOT accepted: STT often returns it for hesitation/noise.
YES_HEADS = ('예', '네', '넵', '넹', '내', '응', '그래', '그래요', '좋아', '좋아요', '맞아', '맞아요',
             '맞습니다', '알겠어', '알겠어요', '알겠습니다', '오케이', '시작', '시작해', '이동',
             '안내해', '부탁해', '부탁해요', '부탁합니다', '해줘', '해주세요', '그렇게해')
YES_TAILS = ('', '요', '예', '네', '해줘', '해주세요', '좋아요', '좋아', '부탁해요', '부탁합니다',
             '안내해주세요', '안내해줘', '이동해주세요', '이동해줘', '시작해주세요', '시작해줘',
             '그렇게해주세요', '알겠습니다', '할게요', '할게', '주세요', '줘')
NO_HEADS = ('아니', '아니요', '아니오', '아뇨', '아니야', '아니에요', '아니예요', '싫어', '싫어요',
            '안해', '안할래', '안할게', '안할게요', '됐어', '됐어요', '괜찮아', '괜찮아요',
            '괜찮습니다', '필요없어', '필요없어요', '필요없습니다', '노', '취소')
NO_TAILS = ('', '요', '괜찮아요', '됐어요', '안해요', '싫어요')

ORDINALS = (
    ('첫번째', '첫째', '첫번', '일번', '1번', '하나', '1'),
    ('두번째', '둘째', '두번', '이번', '2번', '둘', '2'),
    ('세번째', '셋째', '세번', '삼번', '3번', '셋', '3'),
    ('네번째', '넷째', '네번', '사번', '4번', '넷', '4'),
    ('다섯번째', '다섯째', '오번', '5번', '다섯', '5'),
    ('여섯번째', '여섯째', '육번', '6번', '여섯', '6'),
    ('일곱번째', '일곱째', '칠번', '7번', '일곱', '7'),
    ('여덟번째', '여덟째', '팔번', '8번', '여덟', '8'),
    ('아홉번째', '아홉째', '구번', '9번', '아홉', '9'),
)
CHOICE_TAILS = ('', '요', '이요', '입니다', '이에요', '예요', '선택', '선택할게', '선택할게요',
                '선택해줘', '선택해주세요', '할게', '할게요', '하겠습니다', '해줘', '해주세요',
                '으로', '로', '으로해줘', '으로해주세요', '로해줘', '로해주세요', '으로할게',
                '으로할게요', '로할게', '로할게요', '을선택', '를선택', '으로선택', '로선택',
                '신청', '신청이요', '신청할게', '신청할게요', '신청하겠습니다', '신청하고싶어요',
                '신청해줘', '신청해주세요', '하고싶어요', '이요요', '것', '거', '거요', '항목')


def _exact(s, heads, tails):
    return any(s == h + t for h in heads for t in tails)


def _match_option(s, options, fuzzy):
    """options: list of alias lists (index = option number). Return index or None."""
    if not options:
        return None
    # 1) ordinal ('첫 번째', '2번', ...)
    for i, tokens in enumerate(ORDINALS[:len(options)]):
        if _exact(s, tokens, CHOICE_TAILS):
            return i
    # 2) exact alias + polite tail
    hits = {i for i, aliases in enumerate(options)
            if any(a and _exact(s, (a,), CHOICE_TAILS) for a in aliases)}
    if len(hits) == 1:
        return hits.pop()
    if not fuzzy or len(s) < 2:
        return None
    # 3) unique containment ('배우자' alone is ambiguous on most forms -> None)
    hits = set()
    for i, aliases in enumerate(options):
        for a in aliases:
            if len(a) >= 2 and (a in s or (len(s) >= 3 and s in a)):
                hits.add(i)
    return hits.pop() if len(hits) == 1 else None


def parse_intent(text, options=None):
    """Return Intent or None.

    options: optional list of alias lists for the CURRENT choice context
    (application types or one choice group). Exact option matches win over
    keyword commands so that e.g. '변경' selects an application type.
    """
    s = normalize(text)
    if not s or len(s) > 40:
        return None
    raw = str(text)
    idx = _match_option(s, options, fuzzy=False)
    if idx is not None:
        return Intent('choice', idx, raw)
    for kind, keys in KEYWORD_INTENTS:
        if any(k in s for k in keys):
            return Intent(kind, None, raw)
    idx = _match_option(s, options, fuzzy=True)
    if idx is not None:
        return Intent('choice', idx, raw)
    if _exact(s, NO_HEADS, NO_TAILS):
        return Intent('no', None, raw)
    if _exact(s, YES_HEADS, YES_TAILS):
        return Intent('yes', None, raw)
    return None


# ---------------------------------------------------------------- TTS echo guard
class EchoGuardNarrator(Narrator):
    """Narrator (v7.1) that timestamps TTS activity for the STT echo guard."""

    def __init__(self, enabled, backend='powershell', rate=4):
        self.last_activity = -1e9
        super().__init__(enabled, backend, rate)

    def mark_activity(self):
        self.last_activity = time.monotonic()

    def say(self, message):
        if str(message).strip():
            self.mark_activity()
        super().say(message)

    def active_since(self, t):
        """True if TTS was queued/speaking at any time since monotonic t."""
        return self.busy or self.last_activity >= t

    def _worker(self):  # same as Narrator._worker plus activity timestamps
        while not self.stop.is_set():
            try:
                message = self.queue.get(timeout=.12)
            except queue.Empty:
                continue
            self._speaking.set()
            self.mark_activity()
            try:
                rc, output, error = speak_sync(message, self.backend, self.rate)
                if rc:
                    print(f'[TTS ERROR] {error or output or "Unknown TTS failure"}', flush=True)
                    self.failed = True
                    self.enabled = False
            except Exception as exc:
                print('[TTS ERROR]', repr(exc), flush=True)
                self.failed = True
                self.enabled = False
            finally:
                self.mark_activity()
                self._speaking.clear()
                self.queue.task_done()


def should_drop_audio(speaker, listen_started, margin=.35):
    """Drop a captured utterance if our own TTS overlapped its capture window."""
    active_since = getattr(speaker, 'active_since', None)
    if active_since is not None:
        return bool(active_since(listen_started - margin))
    return bool(getattr(speaker, 'busy', False))


class SpeechListener:
    """Microphone thread. Emits ('heard', text) or ('unclear', None).

    Retry prompts are decided by the caller, which knows whether beeps are
    playing (guidance) and applies a cooldown. Silence never emits anything.
    """

    def __init__(self, enabled, speaker, mic_index=None, quiet_after_tts=.8):
        self.enabled = bool(enabled)
        self.speaker = speaker
        self.mic_index = mic_index
        self.quiet_after_tts = float(quiet_after_tts)
        self.events = queue.Queue(maxsize=12)
        self.stop = threading.Event()
        self.ready = False
        self.error = None
        self.last_heard = ''
        self.dropped_echo = 0
        self.thread = None

    def start(self):
        if not self.enabled:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, event):
        try:
            self.events.put_nowait(event)
        except queue.Full:
            pass

    def _run(self):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            r.dynamic_energy_threshold = True
            r.pause_threshold = .65
            r.non_speaking_duration = .25
            with sr.Microphone(device_index=self.mic_index) as microphone:
                print('[STT] Measuring ambient noise for 0.7 seconds...', flush=True)
                r.adjust_for_ambient_noise(microphone, duration=.7)
                self.ready = True
                print('[STT] v8 listening (state-aware commands).', flush=True)
                while not self.stop.is_set():
                    # Half duplex: wait until TTS has been quiet for a moment.
                    if should_drop_audio(self.speaker, time.monotonic() - self.quiet_after_tts + .35):
                        time.sleep(.1)
                        continue
                    started = time.monotonic()
                    try:
                        audio = r.listen(microphone, timeout=.8, phrase_time_limit=3.5)
                    except sr.WaitTimeoutError:
                        continue
                    if should_drop_audio(self.speaker, started):
                        self.dropped_echo += 1
                        print('[STT echo-guard] dropped audio overlapping TTS', flush=True)
                        continue
                    try:
                        transcript = r.recognize_google(audio, language='ko-KR')
                    except sr.UnknownValueError:
                        self._put(('unclear', None))
                        continue
                    except sr.RequestError as exc:
                        self.error = f'STT network error: {exc}'
                        print('[STT]', self.error, flush=True)
                        time.sleep(2)
                        continue
                    if should_drop_audio(self.speaker, started):
                        self.dropped_echo += 1
                        print('[STT echo-guard] dropped transcript', repr(transcript), flush=True)
                        continue
                    self.last_heard = transcript
                    print('[STT heard]', repr(transcript), flush=True)
                    self._put(('heard', transcript))
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            self.error = str(exc)
            print('[STT INIT ERROR]', exc, flush=True)
            print('Install with: .\\.venv_scan\\Scripts\\python.exe -m pip install SpeechRecognition pyaudio', flush=True)
        finally:
            self.ready = False

    def get(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None
