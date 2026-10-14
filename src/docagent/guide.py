"""승인 안내문 검색: python -m docagent.guide --catalog ... --observation ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from docagent.guidance.analysis import observation_from_analysis
from docagent.errors import DocAgentError
from docagent.io.playback import GuidancePlayer, SayBackend
from docagent.guidance.rules import Observation, RuleCatalog


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="확인된 문서와 OCR 관측값으로 승인 문구만 검색합니다.")
    parser.add_argument("--catalog", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--observation", type=Path)
    source.add_argument("--analysis", type=Path, help="로컬 원문을 포함한 Step 2 JSON")
    parser.add_argument("--document-context", type=Path, help="별도로 확인한 문서 ID·버전·페이지")
    parser.add_argument("--field-id", help="분석 결과의 대상 항목 ID")
    parser.add_argument("--demo", action="store_true", help="가상 데이터 사용 허용; 실제 문서 안내 금지")
    parser.add_argument("--output", type=Path)
    audio = parser.add_mutually_exclusive_group()
    audio.add_argument("--speak", action="store_true", help="macOS 로컬 TTS로 그대로 낭독")
    audio.add_argument("--audio-file", type=Path, help="스피커 재생 대신 .aiff 저장")
    parser.add_argument("--voice", default="Yuna")
    parser.add_argument("--interactive", action="store_true", help="s: 정지, r: 다시 듣기, q: 종료")
    args = parser.parse_args(argv)
    if args.interactive and not args.speak:
        parser.error("--interactive 는 --speak 와 함께 사용합니다.")
    if args.analysis and not (args.document_context and args.field_id):
        parser.error("--analysis 에는 --document-context 와 --field-id 가 필요합니다.")
    if args.observation and (args.document_context or args.field_id):
        parser.error("--document-context 와 --field-id 는 --analysis 와 함께 사용합니다.")
    player = None
    try:
        catalog = RuleCatalog.load(args.catalog)
        if args.analysis:
            observation = observation_from_analysis(
                json.loads(args.analysis.read_text(encoding="utf-8")),
                json.loads(args.document_context.read_text(encoding="utf-8")), args.field_id,
            )
        else:
            observation = Observation.from_dict(json.loads(args.observation.read_text(encoding="utf-8")))
        result = catalog.match(observation, demo=args.demo)
        if args.speak or args.audio_file:
            player = GuidancePlayer(catalog, SayBackend(voice=args.voice, output_file=args.audio_file))
            result = player.play(observation, demo=args.demo)
        encoded = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded, end="")
        if player:
            if args.interactive:
                print("s: 정지 / r: 다시 듣기 / q: 종료", file=sys.stderr)
                while True:
                    try:
                        command = input().strip().lower()
                    except EOFError:
                        break
                    if command == "q": break
                    if command == "s": player.stop()
                    elif command == "r": player.replay()
                    # Detect asynchronous engine failure without changing the approved sentence.
                    if player.backend.state == "failed": player.wait()
            else:
                player.wait()
        return 0 if result.status == "matched" else 2
    except KeyboardInterrupt:
        return 130
    except DocAgentError:
        print("[ERROR] 로컬 음성 재생에 실패했습니다. 음성 설치와 출력 장치를 확인하세요.", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, KeyError):
        # JSON parser errors can quote original OCR/PII; never echo exception text.
        print("[ERROR] 안내 데이터 또는 관측값을 읽을 수 없습니다. 스키마와 승인 기록을 확인하세요.", file=sys.stderr)
        return 1
    finally:
        if player is not None:
            player.stop()


if __name__ == "__main__":
    raise SystemExit(main())
