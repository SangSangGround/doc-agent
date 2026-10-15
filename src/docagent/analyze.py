"""빈 양식의 탐지 → 로컬 OCR → 구조화 실행: python -m docagent.analyze."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from docagent.config import DocAgentConfig
from docagent.contracts import FieldRole, FieldType, VISION_TRUST_THRESHOLD
from docagent.detector.roboflow_detector import _load_dotenv
from docagent.errors import DocAgentError, VisionError
from docagent.pii.image_mask import MaskPlan, MaskedDetector, BlankDocumentDetector, image_digest
from docagent.pipeline import DocumentSession, VisionAnalysis, analyze_document
from docagent.vision.detector_factory import build_detector
from docagent.vision.ocr import TesseractOcr
from docagent.vision.quality import assess_capture


def build_report(session: DocumentSession | VisionAnalysis, *, include_text: bool = False) -> dict[str, Any]:
    """탐지와 OCR 신뢰도를 각각 보존하는 로컬 검증용 JSON."""
    fields = []
    for field in session.structure.fields:
        item = field.to_dict()
        item["needs_review"] = (
            field.confidence < VISION_TRUST_THRESHOLD
            or not field.title
            or any(not option.label for option in field.options)
            or (field.type is FieldType.SIGNATURE and field.role is FieldRole.UNKNOWN)
        )
        fields.append(item)
    report = {
        "document_id": session.structure.document_id,
        "image_sha256": image_digest(session.normalized.image),
        "normalized_image_shape": list(session.normalized.image.shape),
        "source_to_normalized": session.normalized.homography.tolist(),
        "coordinate_system": "normalized_a4_mm",
        "ocr_mode": session.config.ocr_mode,
        "needs_review": (
            not fields or any(field["needs_review"] for field in fields)
            or any(not stage.ok for stage in session.stages)
            or session.config.ocr_mode == "regions"
        ),
        "fields": fields,
        "detections": [item.to_dict() for item in session.detections],
        "ocr_words": [word.to_dict() for word in session.words],
        "warnings": list(session.structure.warnings),
        "stages": [stage.to_dict() for stage in session.stages],
    }
    report["text_included"] = include_text
    if not include_text:
        # Pattern masking cannot guarantee removal of names/address/OCR errors.
        # Remove all free text in shareable output, preserving numeric evidence.
        report["document_id"] = "local-analysis"
        report["warnings_count"] = len(report.pop("warnings"))
        for index, item in enumerate(report["fields"], 1):
            item["id"] = f"field_{index}"
            item["title"] = ""
            item["clause_text"] = ""
            for option in item["options"]:
                option["label"] = ""
        for detection in report["detections"]:
            detection["source"] = "detector"
        for word in report["ocr_words"]:
            word["text"] = ""
        for stage in report["stages"]:
            stage["detail"] = ""
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="빈 A4 양식의 작성 영역과 주변 텍스트를 분석합니다. "
        "roboflow 선택 시 이미지를 외부 서버로 보내므로 개인정보 없는 양식만 사용하십시오."
    )
    parser.add_argument("image", type=Path, help="빈 문서 이미지 경로")
    parser.add_argument("--detector", choices=("auto", "heuristic", "yolo", "roboflow"))
    parser.add_argument("--ocr-mode", choices=("page", "regions"),
                        help="page: 전체 문서, regions: 탐지 주변만 OCR (기본 regions)")
    parser.add_argument("--lang", help="OCR 언어 (기본 kor+eng)")
    parser.add_argument("--conf", type=float, help="탐지 신뢰도 임계값")
    parser.add_argument("--dpi", type=int, help="정합 해상도 (기본 300)")
    parser.add_argument("--output", type=Path, help="검증 결과 JSON 저장 경로")
    privacy = parser.add_mutually_exclusive_group()
    privacy.add_argument("--mask-plan", type=Path, help="검토된 정합 이미지 마스킹 계획")
    privacy.add_argument("--allow-unmasked-blank", action="store_true",
                         help="개인정보 없는 빈 양식임을 직접 확인한 경우에만 원본 전송 허용")
    parser.add_argument("--include-ocr-text", action="store_true",
                        help="로컬 검토용 OCR 원문 포함. 개인정보가 포함될 수 있으므로 공유 금지")
    parser.add_argument("--capture-mode", choices=("photo", "scan"),
                        help="전송 전 품질 검사. photo: 촬영본, scan: 이미 잘라낸 원본. 미통과 시 중단")
    args = parser.parse_args(argv)

    try:
        if not args.image.is_file():
            raise VisionError("입력 이미지 파일이 없습니다.")
        image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise VisionError("이미지를 읽을 수 없습니다. PNG 또는 JPEG 파일을 사용하십시오.")
        if args.conf is not None and not 0 <= args.conf <= 1:
            raise ValueError("--conf 는 0.0~1.0 이어야 합니다.")
        quality = assess_capture(image, mode=args.capture_mode) if args.capture_mode else None
        if quality is not None and not quality.can_continue:
            raise VisionError(quality.instruction)
        _load_dotenv()
        cfg = DocAgentConfig.from_env(base=DocAgentConfig(ocr_mode="regions"))
        cfg = cfg.with_(
            detector_prefer=args.detector or cfg.detector_prefer,
            ocr_kind="tesseract", ocr_mode=args.ocr_mode or cfg.ocr_mode,
            ocr_lang=args.lang or cfg.ocr_lang,
            dpi=args.dpi if args.dpi is not None else cfg.dpi,
            llm_kind="offline", serial_port=None, audit_path=None,
        )
        reader = TesseractOcr(
            lang=cfg.ocr_lang, page_size_mm=cfg.page_size_mm,
            config="--psm 11" if cfg.ocr_mode == "regions" else "",
        )
        reader.check_available()
        if cfg.detector_prefer == "roboflow" and not (args.mask_plan or args.allow_unmasked_blank):
            raise VisionError("Roboflow에는 --mask-plan 또는 빈 양식 확인 --allow-unmasked-blank 가 필요합니다.")
        plan = MaskPlan.load(args.mask_plan) if args.mask_plan else None
        detector = build_detector(cfg.detector_weights, cfg.detector_prefer, conf=args.conf)
        if plan is not None:
            detector = MaskedDetector(detector, plan)
        elif args.allow_unmasked_blank:
            detector = BlankDocumentDetector(detector)
        session = analyze_document(
            image, detector=detector, ocr=reader, config=cfg,
            document_id=args.image.stem,
        )
        payload = build_report(session, include_text=args.include_ocr_text)
        payload["source_image_sha256"] = image_digest(image)
        payload["capture_quality"] = quality.to_dict() if quality is not None else {"status": "not_checked"}
        report = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report + "\n", encoding="utf-8")
            print(f"분석 결과 저장: {args.output}")
        else:
            print(report)
    except (DocAgentError, ValueError, OSError, cv2.error) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
