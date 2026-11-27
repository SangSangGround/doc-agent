r"""Diagnose microphone audio and Google Korean STT independently of Camo/v8.

Run from the project root with the SAME Python interpreter used for v8:
    .\.venv_scan\Scripts\python.exe detection\diagnose_stt_v8.py --list
    .\.venv_scan\Scripts\python.exe detection\diagnose_stt_v8.py --mic-index 2
No project files are changed. Requires SpeechRecognition and PyAudio.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List microphone input devices')
    parser.add_argument('--mic-index', type=int, default=None,
                        help='Input device number from --list; default Windows input')
    parser.add_argument('--language', default='ko-KR')
    args = parser.parse_args()

    try:
        import speech_recognition as sr
    except ImportError as exc:
        print('[ERROR] SpeechRecognition import failed:', exc)
        print(r'Install: .\.venv_scan\Scripts\python.exe -m pip install SpeechRecognition pyaudio')
        return 2

    try:
        names = sr.Microphone.list_microphone_names()
        print('[MICROPHONES]')
        for index, name in enumerate(names):
            print(f'  {index}: {name}')
    except Exception as exc:
        print('[ERROR] Cannot enumerate microphones:', repr(exc))
        print('Check that PyAudio is installed and Windows microphone permissions are enabled.')
        return 2

    if args.list:
        return 0
    print('[SELECTED]', 'Windows default input' if args.mic_index is None else
          f'{args.mic_index}: {names[args.mic_index] if 0 <= args.mic_index < len(names) else "UNKNOWN INDEX"}')
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.65
    try:
        with sr.Microphone(device_index=args.mic_index) as source:
            print('[CALIBRATE] Be quiet for 0.8 seconds...', flush=True)
            recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print('[CALIBRATE] energy_threshold =', round(recognizer.energy_threshold, 1), flush=True)
            print('[LISTENING] Say "기록" or "삭제" clearly now (5-second timeout)...', flush=True)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
            except sr.WaitTimeoutError:
                print('[NO AUDIO] No speech crossed the microphone energy threshold.')
                print('Check Windows Settings > System > Sound > Input and choose the active microphone.')
                return 3
    except (OSError, ValueError, AttributeError) as exc:
        print('[MIC ERROR]', repr(exc))
        return 2

    print('[AUDIO CAPTURED] duration=', round(len(audio.frame_data) / (audio.sample_rate * audio.sample_width), 2),
          'seconds, bytes=', len(audio.frame_data), flush=True)
    print('[GOOGLE STT] Recognizing...', flush=True)
    try:
        result = recognizer.recognize_google(audio, language=args.language)
    except sr.UnknownValueError:
        print('[UNRECOGNIZED] Mic recorded audio but speech was not understood.')
        return 4
    except sr.RequestError as exc:
        print('[NETWORK ERROR]', repr(exc))
        return 5
    print('[HEARD]', repr(result))
    print('[OK] The mic and Google STT work independently of the v8 runtime.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
