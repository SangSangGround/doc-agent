"""승인 안내문 검색: python -m docagent.guide --catalog ... --observation ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from docagent.guidance.rules import Observation, RuleCatalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="확인된 문서와 OCR 관측값으로 승인 문구만 검색합니다.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--demo", action="store_true", help="가상 데이터 사용 허용; 실제 문서 안내 금지")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        catalog = RuleCatalog.load(args.catalog)
        observation = Observation.from_dict(json.loads(args.observation.read_text(encoding="utf-8")))
        result = catalog.match(observation, demo=args.demo)
        encoded = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        return 0 if result.status == "matched" else 2
    except (OSError, ValueError, TypeError, KeyError):
        # JSON parser errors can quote original OCR/PII; never echo exception text.
        print("[ERROR] 안내 데이터 또는 관측값을 읽을 수 없습니다. 스키마와 승인 기록을 확인하세요.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
