"""외부 요청 없이 설치·언어 데이터·로컬 음성·키 설정 여부를 점검한다."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence

from docagent.detector.roboflow_detector import _load_dotenv

PACKAGES = {'numpy':'numpy', 'opencv-python-headless':'cv2', 'Pillow':'PIL',
            'pytesseract':'pytesseract', 'inference-sdk':'inference_sdk', 'python-dotenv':'dotenv'}
PROFILES = {'core':{'python','numpy','opencv-python-headless','Pillow'},
            'ocr':{'pytesseract','tesseract','kor','eng'},
            'roboflow':{'inference-sdk','api_key'}, 'tts':{'local_tts','voice'}}


def probe(command: list[str]) -> str | None:
    """실패 시 명령 출력·환경변수·예외 원문을 노출하지 않는다."""
    try:
        result=subprocess.run(command, capture_output=True, text=True, encoding='utf-8',errors='replace', timeout=5,check=False)
        return result.stdout if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def inspect_environment(*, profile: str = 'core', voice: str = 'Yuna') -> dict[str, Any]:
    if profile not in (*PROFILES,'all'): raise ValueError('지원하지 않는 점검 프로필입니다.')
    checks={'python':dict(available=sys.version_info >= (3,10),version=platform.python_version())}
    for distribution,module in PACKAGES.items():
        try:
            importlib.import_module(module)
            checks[distribution]=dict(available=True,version=importlib.metadata.version(distribution))
        except (ImportError,OSError,RuntimeError,importlib.metadata.PackageNotFoundError):
            checks[distribution]=dict(available=False,version=None)
    executable=shutil.which('tesseract')
    languages=probe([executable,'--list-langs']) if executable else None
    installed=set(languages.splitlines()) if languages is not None else set()
    checks['tesseract']=dict(available=languages is not None)
    for language in ('kor','eng'): checks[language]=dict(available=language in installed)
    checks['api_key']=dict(available=bool(os.environ.get('ROBOFLOW_API_KEY','').strip()))
    say=shutil.which('say') if sys.platform=='darwin' else None
    voices=probe([say,'-v','?']) if say else None
    checks['local_tts']=dict(available=voices is not None)
    checks['voice']=dict(available=bool(voices and any(line.startswith(voice+' ') for line in voices.splitlines())))
    required=set(PROFILES['core'])
    for name,items in PROFILES.items():
        if name==profile or profile=='all': required |= items
    missing=sorted(name for name in required if not checks[name]['available'])
    return dict(schema_version=1,profile=profile,status='ready' if not missing else 'missing_dependencies',
                missing=missing,checks=checks,external_requests=0,
                note='설치 점검만 수행합니다. API 인증·실제 음성 출력·실제 문서 성능 검증은 포함하지 않습니다.')


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description='설치 상태와 키 설정 여부만 확인합니다. 키 값은 출력하지 않습니다.')
    parser.add_argument('--profile',choices=(*PROFILES,'all'),default='core')
    parser.add_argument('--voice',default='Yuna')
    args=parser.parse_args(argv)
    _load_dotenv()
    result=inspect_environment(profile=args.profile,voice=args.voice)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='ready' else 1


if __name__=='__main__':
    raise SystemExit(main())
