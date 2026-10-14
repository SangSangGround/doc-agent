"""품질·전송 검토·OCR·승인 안내를 잇는 공식 실행 경로."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.analyze import build_report
from docagent.config import DocAgentConfig
from docagent.detector.roboflow_detector import _load_dotenv
from docagent.errors import DocAgentError
from docagent.guidance.rules import RuleCatalog
from docagent.guidance.session import GuidanceSession, validate_context
from docagent.io.playback import SayBackend
from docagent.pii.image_mask import BlankDocumentDetector, MaskedDetector, MaskPlan
from docagent.pipeline import analyze_document
from docagent.vision.detector_factory import build_detector
from docagent.vision.ocr import TesseractOcr
from docagent.vision.normalize import normalize_document
from docagent.vision.quality import assess_capture


def execute(*, catalog: RuleCatalog, context: dict[str, Any], report: dict[str, Any] | None = None,
            image: np.ndarray | None = None, capture_mode: str = "photo",
            config: DocAgentConfig | None = None, plan: MaskPlan | None = None,
            allow_unmasked_blank: bool = False, demo: bool = False,
            detector_factory: Any = None, ocr_factory: Any = None) -> tuple[dict, GuidanceSession | None]:
    """단계별 상태를 반환한다. 오류 상세·원본 OCR은 반환하지 않는다."""
    events: list[dict] = []
    stage = "document_review"
    try:
        validate_context(context)
        events.append(dict(stage=stage, status="passed"))
        if (image is None) == (report is None):
            raise ValueError("이미지 또는 분석 결과 하나가 필요합니다.")
        if image is not None:
            stage = "capture_quality"
            quality = assess_capture(image, mode=capture_mode)
            events.append(dict(stage=stage, status=quality.status, reasons=list(quality.reasons)))
            if not quality.can_continue:
                return dict(status="blocked", reason=stage, events=events), None
            cfg = config or DocAgentConfig(ocr_mode="regions")
            stage = "image_privacy"
            if plan is not None and allow_unmasked_blank is not False:
                raise ValueError("전송 정책을 하나만 선택하세요.")
            if cfg.detector_prefer == "roboflow" and plan is None and allow_unmasked_blank is not True:
                return dict(status="blocked", reason="mask_required", events=events + [
                    dict(stage=stage, status="blocked")]), None
            if plan is not None:
                plan.apply(normalize_document(image, dpi=cfg.dpi).image)
            events.append(dict(stage=stage, status="passed"))
            # OCR 준비 실패가 외부 추론 요청보다 먼저 발생하도록 한다.
            stage = "ocr_preflight"
            reader = (ocr_factory or TesseractOcr)(lang=cfg.ocr_lang, page_size_mm=cfg.page_size_mm,
                                                  config="--psm 11" if cfg.ocr_mode == "regions" else "")
            reader.check_available()
            events.append(dict(stage=stage, status="passed"))
            stage = "analysis"
            detector = (detector_factory or build_detector)(cfg.detector_weights, cfg.detector_prefer)
            if plan is not None:
                detector = MaskedDetector(detector, plan)
            elif allow_unmasked_blank is True:
                detector = BlankDocumentDetector(detector)
            analysis = analyze_document(image, detector=detector, ocr=reader, config=cfg,
                                        document_id=context.get("analysis_document_id", "session_0001"))
            report = build_report(analysis, include_text=True)
            report["capture_quality"] = quality.to_dict()
        events.append(dict(stage="analysis", status="completed"))
        stage = "guidance"
        session = GuidanceSession(catalog, report, context, demo=demo)
        if not session.ready:
            return dict(status="blocked", reason=session.reason, events=events + [
                dict(stage=stage, status="blocked")]), None
        return dict(status="ready", events=events + [dict(stage=stage, status="passed")],
                    current=session.current()), session
    except (DocAgentError, ValueError, KeyError, TypeError, OSError, cv2.error):
        return dict(status="blocked", reason=stage + "_invalid", events=events + [
            dict(stage=stage, status="failed")]), None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="검토된 문서의 승인 안내만 실행합니다. LLM·하드웨어 호출 없음.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--analysis", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--document-context", type=Path, required=True)
    parser.add_argument("--capture-mode", choices=("photo", "scan"), default="photo")
    parser.add_argument("--detector", choices=("heuristic", "roboflow"), default="heuristic")
    parser.add_argument("--ocr-mode", choices=("page", "regions"), default="regions")
    parser.add_argument("--dpi", type=int, default=300)
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument("--mask-plan", type=Path)
    privacy.add_argument("--allow-unmasked-blank", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    session = None
    try:
        _load_dotenv()
        result, session = execute(
            catalog=RuleCatalog.load(args.catalog),
            context=json.loads(args.document_context.read_text(encoding="utf-8")),
            report=json.loads(args.analysis.read_text(encoding="utf-8")) if args.analysis else None,
            image=cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_COLOR) if args.image else None,
            capture_mode=args.capture_mode, config=DocAgentConfig(dpi=args.dpi, detector_prefer=args.detector,
                                                                 ocr_mode=args.ocr_mode),
            plan=MaskPlan.load(args.mask_plan) if args.mask_plan else None,
            allow_unmasked_blank=args.allow_unmasked_blank, demo=args.demo,
        )
        if session and args.speak:
            session.backend = SayBackend()
            session.command("r")
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
        if session and args.interactive:
            print("n: 다음 / p: 이전 / r: 다시 듣기 / s: 정지 / q: 종료", file=sys.stderr)
            while True:
                try:
                    command = input().strip().lower()
                except EOFError:
                    break
                if command not in ("n", "p", "r", "s", "q"):
                    continue
                if session.backend and session.backend.state == "failed":
                    session.backend.wait()
                print(json.dumps(session.command(command), ensure_ascii=False))
                if command == "q":
                    break
        elif session and session.backend:
            session.backend.wait()
        return 0 if session else 2
    except KeyboardInterrupt:
        return 130
    except (DocAgentError, ValueError, TypeError, KeyError, OSError, cv2.error):
        print('{"status":"error","reason":"input_or_playback_failed"}', file=sys.stderr)
        return 1
    finally:
        if session and session.backend:
            session.backend.stop()


if __name__ == "__main__":
    raise SystemExit(main())
