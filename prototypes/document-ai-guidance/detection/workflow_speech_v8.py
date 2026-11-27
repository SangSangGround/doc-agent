"""v8 speech recognition via Deepgram Nova-3 (Korean).

Drop-in replacement for detection/workflow_speech_v8.py.
Existing multi_field_voice_tracking_v8.py uses WorkflowSpeechCommands.
- Captures audio using SpeechRecognition / PyAudio (Camo microphone)
- Sends short WAV clips to Deepgram /v1/listen (pre-recorded REST)
- Waits for TTS to finish to avoid capturing the assistant
- Provides microphone fallback and the same event queue interface
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
import traceback


def normalize(text: str) -> str:
    return re.sub(r"[^0-9가-힣]", "", str(text or "").lower())


def device_name_key(name: str) -> str:
    return re.sub(r"[^a-z0-9가-힣]", "", str(name or "").lower())


def candidate_indices(names: list[str], selected: int | None) -> list[int | None]:
    if selected is None:
        return [None]
    if not 0 <= selected < len(names):
        raise ValueError(f"Invalid mic index {selected}. Valid range: 0..{len(names)-1}")
    key = device_name_key(names[selected])
    return [selected] + [
        i for i, name in enumerate(names)
        if i != selected and device_name_key(name) == key
    ]


def parse_deepgram_transcript(result: dict) -> str:
    """Extract the single-channel best transcript; fail closed on missing data."""
    channels = result.get("results", {}).get("channels", [])
    if not channels:
        return ""
    alternatives = channels[0].get("alternatives", [])
    if not alternatives:
        return ""
    return str(alternatives[0].get("transcript") or "").strip()


class WorkflowSpeechCommands:
    def __init__(self, enabled, speaker, mic_index=None):
        self.enabled = bool(enabled)
        self.speaker = speaker
        self.mic_index = mic_index
        self.events = queue.Queue(maxsize=12)
        self.stop = threading.Event()
        self.error = None
        self.ready = False
        self.last_heard = ""
        self.thread = None
        self.last_retry_at = -1e9
        self.retry_cooldown_sec = 6.0
        self.active_mic_index = None
        self.attempt_count = 0

    def _busy(self) -> bool:
        return bool(getattr(self.speaker, "busy", False))

    def _wait_speaker_idle(self, quiet_for=1.0) -> bool:
        quiet_since = None
        last_warning = time.monotonic()
        while not self.stop.is_set():
            if self._busy():
                quiet_since = None
                if time.monotonic() - last_warning >= 15.0:
                    print("[STT WAIT WARNING] TTS is still busy", flush=True)
                    last_warning = time.monotonic()
            elif quiet_since is None:
                quiet_since = time.monotonic()
            elif time.monotonic() - quiet_since >= quiet_for:
                return True
            self.stop.wait(0.10)
        return False

    def _queue_retry(self, reason: str) -> bool:
        now = time.monotonic()
        if (self._busy() or self.stop.is_set()
                or now - self.last_retry_at < self.retry_cooldown_sec):
            return False
        self.last_retry_at = now
        try:
            self.events.put_nowait(("retry", None))
            print(f"[STT RETRY] {reason}: 다시 말씀해 주세요.", flush=True)
            return True
        except queue.Full:
            return False

    def start(self):
        if not self.enabled:
            print("[STT OFF] --stt not enabled", flush=True)
            return
        if self.thread is not None and self.thread.is_alive():
            return
        if self.stop.is_set():
            self.stop.clear()
        self.thread = threading.Thread(
            target=self._run, name="v8-deepgram-nova3-stt", daemon=True
        )
        self.thread.start()

    def _transcribe(self, http_client, wav_bytes: bytes) -> str:
        # WAV file uploads documented at https://api.deepgram.com/v1/listen.
        # language=ko avoids using a multilingual billing mode.
        params = {
            "model": os.getenv("DEEPGRAM_STT_MODEL", "nova-3"),
            "language": "ko",
            "smart_format": "true",
            "punctuate": "true",
        }
        response = http_client.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            content=wav_bytes,
            headers={"Content-Type": "audio/wav"},
        )
        response.raise_for_status()
        return parse_deepgram_transcript(response.json())

    def _listen(self, sr, index: int | None):
        import httpx

        model = os.getenv("DEEPGRAM_STT_MODEL", "nova-3")
        key = os.environ["DEEPGRAM_API_KEY"]
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.95
        recognizer.non_speaking_duration = 0.45
        # Do not log the API key or request authorization header.
        with httpx.Client(
            timeout=httpx.Timeout(20.0, connect=8.0),
            headers={"Authorization": f"Token {key}"},
        ) as http_client:
            with sr.Microphone(device_index=index) as microphone:
                if getattr(microphone, "stream", None) is None:
                    raise RuntimeError("Microphone opened without an audio stream")

                print(f"[STT CALIBRATE] Microphone {index} (0.7s)", flush=True)
                recognizer.adjust_for_ambient_noise(microphone, duration=0.7)
                self.ready = True
                self.error = None
                self.active_mic_index = index
                print(
                    f"[STT READY] Microphone {index} | "
                    f"threshold={recognizer.energy_threshold:.0f} | "
                    f"Deepgram {model} (ko)",
                    flush=True,
                )
                last_busy = time.monotonic()
                while not self.stop.is_set():
                    if self._busy():
                        last_busy = time.monotonic()
                        self.stop.wait(0.1)
                        continue
                    if time.monotonic() - last_busy < 0.8:
                        self.stop.wait(0.1)
                        continue
                    try:
                        audio = recognizer.listen(
                            microphone, timeout=1.2, phrase_time_limit=5.0
                        )
                    except sr.WaitTimeoutError:
                        continue

                    if self.stop.is_set():
                        break
                    if self._busy():
                        last_busy = time.monotonic()
                        print("[STT DROPPED] Speech overlapped with TTS", flush=True)
                        continue

                    duration = (
                        len(audio.frame_data)
                        / max(1, audio.sample_rate * audio.sample_width)
                    )
                    print(
                        f"[STT CAPTURED] audio={duration:.2f}s -> Deepgram {model}",
                        flush=True,
                    )
                    try:
                        wav_bytes = audio.get_wav_data(
                            convert_rate=16000, convert_width=2
                        )
                        transcript = self._transcribe(http_client, wav_bytes)
                    except Exception as exc:
                        code = getattr(getattr(exc, "response", None), "status_code", None)
                        self.error = f"{type(exc).__name__} (HTTP {code}): {exc}"
                        print(f"[STT DEEPGRAM ERROR] {self.error}", flush=True)
                        # 401 invalid key, 402 no credits, 429 throttling.
                        self.stop.wait(10.0 if code in (401, 402, 429) else 2.0)
                        continue

                    if not transcript:
                        print(
                            f"[STT UNRECOGNIZED] No transcript for {duration:.2f}s",
                            flush=True,
                        )
                        self._queue_retry("speech not understood")
                        continue

                    self.last_heard = transcript
                    if normalize(transcript) and not self._busy() and not self.stop.is_set():
                        event = ("utterance", transcript)
                        print(f"[STT HEARD] {transcript!r} => {event}", flush=True)
                        try:
                            self.events.put_nowait(event)
                        except queue.Full:
                            print("[STT QUEUE FULL] Dropping speech", flush=True)
                    else:
                        print(f"[STT IGNORED] During TTS: {transcript!r}", flush=True)
        self.ready = False

    def _run(self):
        if not os.getenv("DEEPGRAM_API_KEY"):
            self.error = "DEEPGRAM_API_KEY is missing"
            print(f"[STT CONFIG ERROR] {self.error}", flush=True)
            return
        try:
            import httpx  # noqa: F401
            import speech_recognition as sr
            names = sr.Microphone.list_microphone_names()
            indices = candidate_indices(names, self.mic_index)
            print(
                "[STT DEVICES]",
                [(i, names[i] if i is not None else "(default mic)") for i in indices],
                flush=True,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[STT CONFIG ERROR] {self.error}", flush=True)
            traceback.print_exc()
            return

        cycle = 0
        while not self.stop.is_set():
            cycle += 1
            print("[STT WAIT] Waiting for TTS to finish", flush=True)
            if not self._wait_speaker_idle(quiet_for=1.0):
                break
            for index in indices:
                if self.stop.is_set():
                    break
                self.attempt_count += 1
                print(
                    f"[STT OPEN] Trying microphone {index}, "
                    f"attempt {self.attempt_count}",
                    flush=True,
                )
                try:
                    self._listen(sr, index)
                    if self.stop.is_set():
                        break
                    print(f"[STT SESSION ENDED] Microphone {index}", flush=True)
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                    print(f"[STT MIC ERROR] Microphone {index}: {self.error}", flush=True)
                    if cycle == 1:
                        traceback.print_exc()
                finally:
                    self.ready = False
                    self.active_mic_index = None
                if not self.stop.is_set():
                    self.stop.wait(0.4)
            if not self.stop.is_set():
                delay = min(8, cycle * 2)
                print(f"[STT RETRY OPEN] Retrying in {delay}s", flush=True)
                self.stop.wait(delay)
        self.ready = False
        print("[STT STOPPED]", flush=True)

    @staticmethod
    def list_microphones():
        import speech_recognition as sr
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{index}: {name}")

    def get(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None
