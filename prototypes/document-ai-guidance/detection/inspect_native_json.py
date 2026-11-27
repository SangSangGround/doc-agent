"""Only shows schema shapes: DO NOT infer the legacy manifest contract from guesses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def shape(value, depth=0):
    if depth >= 3:
        return type(value).__name__
    if isinstance(value, dict):
        return {key: shape(item, depth + 1) for key, item in list(value.items())[:25]}
    if isinstance(value, list):
        return {'list_length': len(value), 'first_item_shape': shape(value[0], depth + 1) if value else None}
    return type(value).__name__


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='기존 JSON 구조 확인(값 및 개인 정보는 출력하지 않음)')
    p.add_argument('files', nargs='+', type=Path)
    args = p.parse_args()
    for f in args.files:
        print('===', f, '===')
        if not f.exists():
            print('MISSING')
            continue
        try:
            data = json.loads(f.read_text(encoding='utf-8-sig'))
            print(json.dumps(shape(data), ensure_ascii=False, indent=2))
        except (ValueError, OSError) as exc:
            print('ERROR:', exc)
