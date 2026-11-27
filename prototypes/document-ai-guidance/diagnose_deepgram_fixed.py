r"""Standalone Deepgram Nova-3 diagnostic for hand-tracking.
Run from project root:
    .\.venv_scan\Scripts\python.exe -u .\diagnose_deepgram_fixed.py --mic-index 20
Requires DEEPGRAM_API_KEY in this terminal. Does not persist audio or print keys.
"""
import argparse
from array import array
import math
import os
import sys
import traceback


def main():
    print('[START] Deepgram Nova-3 microphone/API diagnostic v2', flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic-index', type=int, default=20)
    parser.add_argument('--samples', type=int, default=1)
    args = parser.parse_args()
    if not os.environ.get('DEEPGRAM_API_KEY'):
        print('[STOP] DEEPGRAM_API_KEY is missing. Set it in this PowerShell window.')
        return 2
    try:
        import speech_recognition as sr
        import httpx
    except ImportError as e:
        print(f'[IMPORT ERROR] {e} | try: .\\.venv_scan\\Scripts\\python.exe -m pip install SpeechRecognition PyAudio httpx')
        return 1

    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.0
    recognizer.non_speaking_duration = 0.45
    try:
        names = sr.Microphone.list_microphone_names()
        if not 0 <= args.mic_index < len(names):
            print(f'[MIC ERROR] Index {args.mic_index} is outside 0..{len(names)-1}')
            for i, name in enumerate(names):
                print(f'{i}: {name}')
            return 2
        print(f'[MIC] index={args.mic_index} name={names[args.mic_index]!r}')
        with sr.Microphone(device_index=args.mic_index) as source:
            print('[CALIBRATE] Keep quiet for 0.8 seconds', flush=True)
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print(f'[CALIBRATE] threshold={recognizer.energy_threshold:.0f}')
            for index in range(args.samples):
                print(f'[LISTEN {index+1}/{args.samples}] Say: 네 진행해주세요', flush=True)
                try:
                    audio = recognizer.listen(source, timeout=6.0, phrase_time_limit=5.0)
                except sr.WaitTimeoutError:
                    print('[NO AUDIO] No voice detected. Check Camo mic index/volume.')
                    continue
                wav = audio.get_wav_data(convert_rate=16000, convert_width=2)
                raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                values = array('h')
                values.frombytes(raw)
                if sys.byteorder != 'little':
                    values.byteswap()
                peak = max((abs(v) for v in values), default=0)
                rms = math.sqrt(sum(v*v for v in values)/len(values)) if values else 0
                print(f'[AUDIO] seconds={len(values)/16000:.2f} peak={peak} rms={rms:.0f} wav_bytes={len(wav)}', flush=True)
                if rms < 100:
                    print('[AUDIO WARNING] Recording very quiet; wrong mic or low input volume is possible')
                print('[DEEPGRAM SEND] Posting WAV to Nova-3, language=ko', flush=True)
                try:
                    response = httpx.post(
                        'https://api.deepgram.com/v1/listen',
                        params={'model': 'nova-3', 'language': 'ko', 'punctuate': 'true', 'smart_format': 'true'},
                        headers={'Authorization': 'Token ' + os.environ['DEEPGRAM_API_KEY'], 'Content-Type': 'audio/wav'},
                        content=wav,
                        timeout=25.0,
                    )
                    print(f'[DEEPGRAM HTTP] status={response.status_code}', flush=True)
                    if response.status_code != 200:
                        print('[DEEPGRAM ERROR]', response.text[:600])
                        continue
                    data = response.json()
                    channels = data.get('results', {}).get('channels', [])
                    alternatives = channels[0].get('alternatives', []) if channels else []
                    best = alternatives[0] if alternatives else {}
                    print('[DEEPGRAM TRANSCRIPT]', repr(best.get('transcript', '')), flush=True)
                    print('[DEEPGRAM CONFIDENCE]', best.get('confidence', 'N/A'))
                    print('[DEEPGRAM WORDS]', len(best.get('words', [])))
                except Exception as e:
                    print('[DEEPGRAM ERROR]', type(e).__name__, str(e)[:500])
    except Exception as e:
        print('[MIC ERROR]', type(e).__name__, str(e)[:500])
        traceback.print_exc(limit=2)
        return 1
    print('[DONE] Diagnostic finished; no audio file saved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
