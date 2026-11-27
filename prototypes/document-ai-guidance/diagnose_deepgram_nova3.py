r"""Diagnose Deepgram Nova-3 STT vs Groq using exactly the same mic recording.

Run from hand-tracking root:
  .\.venv_scan\Scripts\python.exe diagnose_deepgram_nova3.py --mic-index 20
Requires DEEPGRAM_API_KEY; optional GROQ_API_KEY for comparison.
This program does NOT save your audio and never prints API keys.
"""
from __future__ import annotations

import argparse
from array import array
import math
import os
import sys
import traceback


def audio_stats(audio):
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    samples = array('h')
    samples.frombytes(raw)
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples:
        return 0, 0, 0.0
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    duration = len(samples) / 16000
    return peak, rms, duration


def deepgram_transcribe(wav: bytes):
    import httpx
    key = os.getenv('DEEPGRAM_API_KEY')
    if not key:
        print('[DEEPGRAM] No DEEPGRAM_API_KEY in current terminal')
        return
    try:
        response = httpx.post(
            'https://api.deepgram.com/v1/listen',
            params={'model': 'nova-3', 'language': 'ko', 'smart_format': 'true', 'punctuate': 'true'},
            headers={'Authorization': f'Token {key}', 'Content-Type': 'audio/wav'},
            content=wav,
            timeout=20.0,
        )
        print(f'[DEEPGRAM HTTP] status={response.status_code}')
        if response.status_code != 200:
            # Error payload only: no authorization header/key is logged.
            print('[DEEPGRAM ERROR BODY]', response.text[:500])
            return
        payload = response.json()
        metadata = payload.get('metadata', {})
        channels = payload.get('results', {}).get('channels', [])
        alternatives = (channels[0].get('alternatives', []) if channels else [])
        best = alternatives[0] if alternatives else {}
        print('[DEEPGRAM TRANSCRIPT]', repr(best.get('transcript', '')))
        print('[DEEPGRAM CONFIDENCE]', best.get('confidence', 'N/A'))
        print('[DEEPGRAM WORDS]', len(best.get('words', [])))
        print('[DEEPGRAM REQUEST ID]', metadata.get('request_id', 'N/A'))
        print('[DEEPGRAM AUDIO DURATION]', metadata.get('duration', 'N/A'))
    except Exception as exc:
        print('[DEEPGRAM REQUEST ERROR]', type(exc).__name__, str(exc)[:400])


def groq_transcribe(wav: bytes):
    if not os.getenv('GROQ_API_KEY'):
        print('[GROQ] No GROQ_API_KEY; comparison skipped')
        return
    try:
        from groq import Groq
        result = Groq(timeout=20.0, max_retries=0).audio.transcriptions.create(
            file=('speech.wav', wav, 'audio/wav'),
            model='whisper-large-v3',
            language='ko',
            response_format='json',
            temperature=0.0,
        )
        print('[GROQ TRANSCRIPT]', repr(result.text))
    except Exception as exc:
        print('[GROQ ERROR]', type(exc).__name__, str(exc)[:400])


def main():
    parser = argparse.ArgumentParser(description='Compare Deepgram and Groq on the same recorded Korean voice')
    parser.add_argument('--mic-index', type=int, default=20)
    parser.add_argument('--samples', type=int, default=2, choices=(1, 2, 3))
    args = parser.parse_args()
    if not os.getenv('DEEPGRAM_API_KEY'):
        print('[STOP] Set DEEPGRAM_API_KEY first in this PowerShell window.')
        return 2

    import speech_recognition as sr
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    recognizer.non_speaking_duration = 0.45
if __name__ == '__main__':
      raise SystemExit(main())
  