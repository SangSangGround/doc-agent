"""외부 통신 없는 촬영 품질 검사 CLI."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from docagent.vision.quality import assess_capture


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description="외부 전송 전에 사용하는 로컬 촬영 품질 검사")
    parser.add_argument('image',type=Path)
    parser.add_argument('--mode',choices=('photo','scan'),default='photo')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:
        image=cv2.imdecode(np.frombuffer(args.image.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        result=assess_capture(image,mode=args.mode)
        text=json.dumps(result.to_dict(),ensure_ascii=False,indent=2)+'\n'
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(text,encoding='utf-8')
        else: print(text,end='')
        return 0 if result.can_continue else 2
    except (ValueError,OSError,cv2.error):
        print('[ERROR] 촬영 검사 입력을 읽거나 결과를 저장할 수 없습니다.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
