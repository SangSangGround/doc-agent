"""Optional Windows Korean TTS + developer-reviewed form overview for v7.1.

No OCR inference: Only the reviewed manifest and optional authored guide can be read.
TTS uses Windows PowerShell System.Speech by default, not pyttsx3 threads.
"""
from __future__ import annotations
import base64
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _run_ps(code: str, *, timeout=20):
    if sys.platform != 'win32':
        raise RuntimeError('Windows PowerShell TTS is available only on Windows.')
    powershell = shutil.which('powershell.exe') or shutil.which('powershell')
    if not powershell:
        raise RuntimeError('powershell.exe not found.')
    encoded = base64.b64encode(code.encode('utf-16-le')).decode('ascii')
    result = subprocess.run([powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand', encoded],
                            capture_output=True, encoding='utf-8', errors='replace', timeout=timeout)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def installed_voices():
    script = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    foreach ($v in $s.GetInstalledVoices()) {
        if ($v.Enabled) { Write-Output ($v.VoiceInfo.Culture.Name + " | " + $v.VoiceInfo.Name) }
    }
} finally { $s.Dispose() }
'''
    return _run_ps(script, timeout=15)


def _speak_powershell(text: str, culture='ko-KR', rate=4):
    message = base64.b64encode(text.encode('utf-8')).decode('ascii')
    # SpeechSynthesizer can only use installed desktop System.Speech voices.
    # Absence of ko-KR is an explicit failure, not silent English fallback.
    script = r'''
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $voice = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq 'CULTURE_REPLACE' } | Select-Object -First 1
    if ($null -eq $voice) { Write-Error 'Korean System.Speech TTS voice not installed. Check Windows speech language settings.'; exit 12 }
    $s.SelectVoice($voice.VoiceInfo.Name)
    $s.Rate = RATE_REPLACE
    $s.Volume = 100
    $message = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('B64_REPLACE'))
    $s.Speak($message)
} finally { $s.Dispose() }
'''.replace('CULTURE_REPLACE', culture).replace('B64_REPLACE', message).replace('RATE_REPLACE', str(max(-10, min(10, int(rate)))))
    return _run_ps(script, timeout=max(20, min(110, len(text)//6 + 20)))


def _speak_pyttsx3(text: str, lang='ko', rate=4):
    try:
        import pyttsx3
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pythoncom = None
        engine = pyttsx3.init()
        korean = []
        for v in engine.getProperty('voices'):
            desc = (str(v.id) + str(v.name) + str(getattr(v, 'languages', ''))).lower()
            if 'korean' in desc or 'ko-kr' in desc or '한국' in desc or 'heami' in desc:
                korean.append(v)
        if not korean:
            raise RuntimeError('No Korean SAPI TTS voice found by pyttsx3.')
        engine.setProperty('voice', korean[0].id)
        engine.setProperty('rate', max(120, min(270, 170 + int(rate)*12)))
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        if pythoncom is not None:
            pythoncom.CoUninitialize()
        return 0, '', ''
    except Exception as exc:
        return 1, '', str(exc)


def speak_sync(text, backend='powershell', rate=4):
    if backend == 'pyttsx3':
        return _speak_pyttsx3(text, rate=rate)
    return _speak_powershell(text, rate=rate)


class Narrator:
    def __init__(self, enabled: bool, backend='powershell', rate=4):
        self.enabled = bool(enabled)
        self.backend = backend
        self.rate = max(-10, min(10, int(rate)))
        self.queue = queue.Queue(maxsize=16)
        self.stop = threading.Event()
        self._speaking = threading.Event()
        self.failed = False
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    @property
    def busy(self):
        return self._speaking.is_set() or not self.queue.empty()

    def say(self, message):
        message = str(message).strip()
        if not message:
            return
        print('[VOICE]', message, flush=True)
        if not self.enabled or self.failed:
            return
        # Never stall OpenCV camera loop for narration.
        try:
            self.queue.put_nowait(message)
        except queue.Full:
            print('[VOICE] Too many queued prompts; dropping latest.', flush=True)

    def _worker(self):
        while not self.stop.is_set():
            try:
                message = self.queue.get(timeout=.12)
            except queue.Empty:
                continue
            self._speaking.set()
            try:
                rc, output, error = speak_sync(message, self.backend, self.rate)
                if rc:
                    print(f'[TTS ERROR] {error or output or "Unknown TTS failure"}', flush=True)
                    print('Try: python detection\\narration_v7_1.py --test-voice', flush=True)
                    self.failed = True
                    self.enabled = False
                    # Do not claim successful spoken prompts after engine failure.
            except Exception as exc:
                print('[TTS ERROR]', repr(exc), flush=True)
                self.failed = True
                self.enabled = False
            finally:
                self._speaking.clear()
                self.queue.task_done()


def load_guide(path, fields):
    if path is None:
        return {'schema':'navigation_guide_v1','document_title':'',
                'document_summary':'', 'field_descriptions':{},'selection_groups':[]}
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if raw.get('schema') != 'navigation_guide_v1':
        raise ValueError('guide schema must be navigation_guide_v1')
    ids = {f['id'] for f in fields}
    for g in raw.get('selection_groups',[]):
        if g.get('type') != 'single_choice':
            raise ValueError('Only explicitly reviewed single_choice groups supported')
        if not g.get('field_ids') or not set(g['field_ids']).issubset(ids):
            raise ValueError('Guide group must list only approved manifest field IDs')
    return raw


def single_choice_group(guide, fields):
    for g in guide.get('selection_groups',[]):
        if g['type'] == 'single_choice':
            return g
    return None


def field_spoken_description(field, guide, w=210., h=297.):
    label = str(field.get('label') or field['id'])
    cls = field.get('class_name','')
    kind = {'check_box':'체크박스', 'input_field':'작성란', 'signature_field':'서명란'}.get(cls,'입력 항목')
    p=field['target_mm']
    x,y=float(p['x']),float(p['y'])
    horizontal = '왼쪽' if x < .34*w else ('오른쪽' if x > .68*w else '가운데')
    vertical = '상단' if y < .33*h else ('하단' if y > .68*h else '중간')
    explanation=guide.get('field_descriptions',{}).get(field['id'], '')
    return f'{label} {kind}. 종이 {vertical} {horizontal}, 위에서 약 {y/10:.1f}센티미터 위치입니다. {explanation}'


def overview_text(fields, guide, w=210., h=297.):
    """Short opening announcement; detailed field descriptions are read on selection/I/V.

    This reads an authored, form-specific guide plus approved manifest fields, NOT live OCR.
    Never narrate '0개' (which Korean TTS may pronounce ambiguously).
    """
    chunks = ['문서를 인식했습니다.']
    if guide.get('document_title'):
        chunks.append(f"문서 이름은 {guide['document_title']}입니다.")
    if guide.get('document_summary'):
        chunks.append(guide['document_summary'])
    kind_names={'check_box':'체크박스','input_field':'작성란','signature_field':'서명란'}
    nums={1:'한',2:'두',3:'세',4:'네',5:'다섯',6:'여섯',7:'일곱',8:'여덟',9:'아홉',10:'열'}
    group = single_choice_group(guide, fields)
    if group:
        ids=set(group['field_ids'])
        selected=[f for f in fields if f['id'] in ids]
        names=' '.join(f['label'] for f in selected)
        cnt=nums.get(len(selected),str(len(selected)))
        chunks.append(f'현재 안내 가능한 선택지는 {names}, {cnt} 가지입니다.')
        chunks.append(group.get('question','어느 항목을 선택하시겠습니까?'))
        choices=' '.join(f"{i+1}번 {next(t['label'] for t in fields if t['id']==ident)}." for i,ident in enumerate(group['field_ids']))
        chunks.append(f'키보드에서 번호를 누르세요. {choices}')
    else:
        active=[]
        for cls, name in kind_names.items():
            count=sum(f.get('class_name')==cls for f in fields)
            if count:
                cnt=nums.get(count,str(count))
                active.append(f'{name} {cnt} 개')
        if active:
            chunks.append('현재 안내 가능한 항목은 '+', '.join(active)+'입니다.')
            chunks.append('문서 위쪽 항목부터 순서대로 안내하겠습니다. 엔 키로 다음 항목으로 이동할 수 있습니다.')
        else:
            chunks.append('현재 검증된 작성 목표가 없습니다. 위치 안내를 시작할 수 없습니다.')
    return ' '.join(chunks)


def voice_diagnostic(backend='powershell', rate=4):
    print('VOICE TEST: backend=', backend)
    if backend == 'powershell':
        try:
            rc, stdout, stderr = installed_voices()
            print('System.Speech voices:', stdout or '(none)')
            if rc:
                print('VOICE CHECK FAILED:', stderr or 'voice query failed')
                return 1
            if 'ko-KR' not in stdout:
                print('VOICE CHECK FAILED: no ko-KR System.Speech voice available.')
                return 2
        except Exception as exc:
            print('VOICE CHECK FAILED:', exc)
            return 3
    try:
        rc, stdout, stderr = speak_sync('안녕하세요. 문서 작성 안내 음성 테스트입니다.', backend, rate)
        if rc:
            print('VOICE CHECK FAILED:', stderr or stdout)
            return 4
        print('VOICE COMMAND COMPLETED. Check whether sound was audible.')
        return 0
    except Exception as exc:
        print('VOICE CHECK FAILED:', exc)
        return 5

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--test-voice',action='store_true')
    parser.add_argument('--backend',choices=['powershell','pyttsx3'],default='powershell')
    parser.add_argument('--rate',type=int,default=4,help='Windows TTS speech rate, -10 to +10; default +4')
    args=parser.parse_args()
    if not (-10 <= args.rate <= 10):
        parser.error('--rate must be between -10 and +10')
    if args.test_voice:
        raise SystemExit(voice_diagnostic(args.backend, args.rate))
    parser.print_help()
