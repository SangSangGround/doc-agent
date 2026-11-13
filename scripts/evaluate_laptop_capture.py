"""고정 양식에 합성 촬영 오류를 가해 준비 판정과 문서 정합을 별도로 기록한다.

실제 카메라·사용자·광학 반사 성능을 측정하는 도구가 아니다. OCR/LLM을
끄고 이미지 처리 경로만 평가한다. 임계값 조정용 자료와 실측 평가를 구분한다.

    python scripts/evaluate_laptop_capture.py \
        --template outputs/laptop-demo-20261113/template.json --demo \
        --output outputs/laptop-capture-evaluation
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from docagent.home.capture import CaptureCoach
from docagent.home.server import HomeApplication, encode_image
from docagent.home.templates import HomeTemplate, load_template

FRAME_TIMESTAMPS = (0., .35, .7)
DEFAULT_SEED = 20261113


def make_scenarios(reference: np.ndarray, seed: int = DEFAULT_SEED) -> list[dict]:
    """같은 기준 페이지·배경에서 다섯 조건을 만들고 알려진 변형을 정답으로 쓴다."""
    rng = np.random.default_rng(seed)
    width, height, page_width = 1920, 1440, 800
    page_height = round(page_width * reference.shape[0] / reference.shape[1])
    if page_height > height - 180:
        page_height = height - 180
        page_width = round(page_height * reference.shape[1] / reference.shape[0])
    page = cv2.resize(reference, (page_width, page_height), interpolation=cv2.INTER_AREA)
    background = np.clip(rng.normal(65, 2, (height, width, 1)), 0, 255).astype(np.uint8)
    background = np.repeat(background, 3, axis=2)
    x0, y0 = (width - page_width) // 2, (height - page_height) // 2
    normal = background.copy()
    normal[y0:y0 + page_height, x0:x0 + page_width] = page

    # 문서만 회전한다. 20도에서는 이 합성 A4 문서의 모서리가 프레임 안에 남는다.
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 20, 1)
    tilted = cv2.warpAffine(normal, matrix, (width, height), borderValue=(65, 65, 65))

    # 왼쪽 문서 면적 15%를 실제로 제거하되 출력 프레임 크기는 동일하게 유지한다.
    clipped = background.copy()
    removed = round(page_width * .15)
    clipped[y0:y0 + page_height, :page_width - removed] = page[:, removed:]
    blurred = cv2.GaussianBlur(normal, (51, 51), 14)

    # 실제 빛 반사의 물리적 모델이 아니다. 본문 위의 흰 덮임으로 정보 손실만 시험한다.
    obscured = normal.copy()
    white_box = [x0 + round(page_width * .08), y0 + round(page_height * .32),
                 x0 + round(page_width * .92), y0 + round(page_height * .39)]
    bx0, by0, bx1, by1 = white_box
    obscured[by0:by1, bx0:bx1] = 255

    return [
        {"id": "normal", "condition": "정상", "image": normal, "expected_usable": True,
         "transformation": {"page_width_px": page_width, "page_height_px": page_height,
                            "margin_xy_px": [x0, y0]}},
        {"id": "tilted", "condition": "기울어짐", "image": tilted, "expected_usable": False,
         "transformation": {"rotation_degrees": 20}},
        {"id": "clipped", "condition": "잘림", "image": clipped, "expected_usable": False,
         "transformation": {"removed_page_width_fraction": .15, "edge": "left"}},
        {"id": "blurred", "condition": "흐림", "image": blurred, "expected_usable": False,
         "transformation": {"gaussian_kernel": [51, 51], "sigma": 14}},
        {"id": "white_occlusion", "condition": "흰 덮임 — 빛 반사 자체를 재현한 것은 아님",
         "image": obscured, "expected_usable": False,
         "transformation": {"white_box_xyxy_px": white_box, "physical_glare_simulation": False}},
    ]


def rate(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def classification_metrics(cases: list[dict], key: str) -> dict:
    """전부 거절했을 때도 정상 통과율과 정확도를 함께 드러낸다."""
    good = [case for case in cases if case["expected_usable"]]
    bad = [case for case in cases if not case["expected_usable"]]
    return {
        "accuracy": rate(sum(bool(case[key]) == case["expected_usable"] for case in cases), len(cases)),
        "bad_image_pass_rate": rate(sum(bool(case[key]) for case in bad), len(bad)),
        "normal_image_pass_rate": rate(sum(bool(case[key]) for case in good), len(good)),
    }


def evaluate(template: HomeTemplate, output: Path, *, seed: int = DEFAULT_SEED) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("평가 결과는 비어 있는 폴더에 저장하세요.")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for scenario in make_scenarios(template.reference, seed):
        image = scenario.pop("image")
        filename = scenario["id"] + ".png"
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("합성 촬영 이미지를 저장하지 못했습니다.")
        (output / filename).write_bytes(png.tobytes())
        coach = CaptureCoach(mode="photo")
        frames, elapsed_ms = [], []
        for stamp in FRAME_TIMESTAMPS:
            start = time.perf_counter()
            frames.append(coach.update(image, timestamp=stamp))
            elapsed_ms.append(round((time.perf_counter() - start) * 1000, 3))
        # 준비 판정이 거절한 사진도 강제로 캡처해 뒤 단계의 방어를 독립적으로 측정한다.
        app = HomeApplication(template, read_ocr=False)
        payload = {"image": encode_image(image), "mode": "photo"}
        start = time.perf_counter()
        captured = app.handle("capture", payload)
        capture_ms = round((time.perf_counter() - start) * 1000, 3)
        observation = captured["observation"]
        results.append({
            **scenario, "image_file": filename, "image_sha256": hashlib.sha256(png).hexdigest(),
            "frame_shape": list(image.shape), "frames": frames,
            "ready_to_capture": frames[-1]["ready_to_capture"],
            "registration_passed": observation.get("alignment", {}).get("reason") == "registered",
            "document_ready": captured["document_ready"],
            "capture_observation": observation,
            "timing_ms": {"frame_processing": elapsed_ms,
                          "capture_processing_including_image_decode": capture_ms},
        })

    report = {
        "schema_version": 1, "dataset_kind": "synthetic_development_smoke", "seed": seed,
        "template_digest": template.manifest_sha256,
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                        "system": platform.system(), "machine": platform.machine()},
        "protocol": {
            "capture_mode": "photo", "frame_timestamps_seconds": list(FRAME_TIMESTAMPS),
            "same_still_image_repeated": True, "ocr_enabled": False, "llm_enabled": False,
            "force_capture_even_if_not_ready": True, "physical_glare_simulation": False,
            "timing_scope": "Python 이미지 처리만 측정. 브라우저 전송·카메라·음성·OCR·LLM 시간 제외.",
        },
        "metrics": {
            "capture_readiness": classification_metrics(results, "ready_to_capture"),
            "image_registration": classification_metrics(results, "registration_passed"),
            "document_ready_with_ocr_disabled": classification_metrics(results, "document_ready"),
            "information_extraction_accuracy": None,
            "unsupported_answer_count": None,
            "user_task_completion_seconds": None,
            "user_recapture_count": None,
        },
        "limitations": [
            "같은 양식의 합성 5건이며 실제 사용자·카메라 정확도 또는 독립 평가 성능이 아닙니다.",
            "흰 덮임은 인쇄 내용 손실 검사이며 실제 반사·번짐을 재현하거나 검증하지 않습니다.",
            "동일 이미지를 반복하므로 실제 움직임과 프레임 지연은 평가하지 않습니다.",
            "OCR을 끈 이미지 경로 평가이며 금액·조건 문자 대조와 질문 근거 정확도는 별도 평가가 필요합니다.",
            "과제 완료 시간·재촬영 횟수는 사용자 실측이 필요하므로 null입니다.",
        ],
        "cases": results,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = evaluate(load_template(args.template, demo=args.demo), args.output, seed=args.seed)
    except (OSError, ValueError) as exc:
        print(f"평가 실패: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"상세 결과: {args.output / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
