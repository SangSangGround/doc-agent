"""정합된 검토 이미지와 미승인 마스킹 계획을 로컬에 준비한다."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from docagent.errors import DocAgentError
from docagent.pii.image_mask import draft_plan, MaskPlan
from docagent.vision.normalize import normalize_document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="로컬 마스킹 검토 준비. 네트워크 호출 없음.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--plan", type=Path, help="검토 완료 계획으로 마스킹 미리보기 생성")
    args = parser.parse_args(argv)
    try:
        if not 72 <= args.dpi <= 600:
            raise ValueError("dpi")
        image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        normalized = normalize_document(image, dpi=args.dpi).image
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.plan:
            result = MaskPlan.load(args.plan).apply(normalized)
            (args.output_dir / "masked.png").write_bytes(cv2.imencode('.png', result)[1].tobytes())
        else:
            # Avoid silently overwriting a review in progress.
            with (args.output_dir / "mask-plan.json").open('x', encoding='utf-8') as f:
                json.dump(draft_plan(normalized), f, ensure_ascii=False, indent=2)
            (args.output_dir / "normalized.png").write_bytes(cv2.imencode('.png', normalized)[1].tobytes())
        print("로컬 검토 파일을 저장했습니다. 개인정보가 모두 가려졌는지 확인하세요.")
    except (DocAgentError, ValueError, OSError, cv2.error):
        print("[ERROR] 마스킹 검토 파일을 준비하지 못했습니다. 입력·출력 경로와 계획을 확인하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
