"""Replay saved Gemini boxes on the teammate's four source photos without network access.

PYTHONPATH=src venv/bin/python scripts/evaluate_writing_fields.py /path/to/hand-tracking
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
from unittest.mock import patch

from PIL import Image, ImageOps

from docagent import writing_fields
from docagent.cloud_providers import json_bytes


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepare_jpeg(source):
    with Image.open(source) as photo:
        image = ImageOps.exif_transpose(photo).convert("RGB")
    width, height = image.size
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=90)
    data = encoded.getvalue()
    return (
        json_bytes(
            {
                "allowImageProcessing": True,
                "imageJPEG": base64.b64encode(data).decode("ascii"),
                "imageSHA256": hashlib.sha256(data).hexdigest(),
            }
        ),
        width,
        height,
    )


def replay(detection):
    pages = []
    for stem in ("document1", "document2", "document3", "test_document"):
        photo = (detection if stem == "test_document" else detection / "test_images") / (stem + ".jpg")
        saved = detection / "gemini_output" / stem
        captures = read_json(saved / "detection_result.json")
        proposals = read_json(saved / "gemini_raw.json")
        body, w, h = prepare_jpeg(photo)
        if (captures["image_width"], captures["image_height"]) != (w, h):
            raise ValueError(f"Upright dimensions differ from the stored Gemini analysis for {stem}")
        detector = writing_fields.WritingFieldDetector("offline-fixture", "recorded")
        with patch.object(writing_fields, "gemini_boxes", return_value=proposals) as provider:
            before = time.perf_counter()
            result = detector.fields(body)
            cold_ms = (time.perf_counter() - before) * 1000
            before = time.perf_counter()
            assert result == detector.fields(body)
            warm_ms = (time.perf_counter() - before) * 1000
            assert provider.call_count == 1
        row = {
            "page": stem,
            "historical_gemini_seconds": captures["latency_seconds"],
            "gemini_proposals": len(proposals["boxes"]),
            "retained_regions": len(result["fields"]),
            "cold_local_replay_ms": round(cold_ms, 1),
            "warm_exact_replay_ms": round(warm_ms, 1),
        }
        if stem == "document2":
            truth = read_json(
                detection / "refinement_output/document2/writable_output/navigation_manifest.json"
            )
            review = read_json(
                detection / "refinement_output/document2/writable_output/navigation_review.json"
            )
            if truth["source_image_sha256"] != hashlib.sha256(photo.read_bytes()).hexdigest():
                raise ValueError("Reviewed reference points belong to a different page")
            verified = {
                item["id"] for item in review["fields"] if item["approved"] and item["verified_on_paper"]
            }
            predicted = {f["id"]: f for f in result["fields"]}
            alignment = {}
            for item in truth["fields"]:
                if item["id"] not in verified:
                    continue
                counts = alignment.setdefault(
                    item["class_name"],
                    {"reviewed_points": 0, "region_returned": 0, "point_in_region": 0},
                )
                counts["reviewed_points"] += 1
                if item["id"] in predicted:
                    counts["region_returned"] += 1
                    x, y, width, height = predicted[item["id"]]["bounds"]
                    p = item["target_source_px"]
                    counts["point_in_region"] += int(
                        x <= p["x"] / w <= x + width and y <= p["y"] / h <= y + height
                    )
            row["teammate_reviewed_points"] = alignment
        pages.append(row)
    return {
        "condition": "EXIF-upright source photos, longest side <= 2048, JPEG quality 90; saved Gemini boxes, local Mac timings; identical bytes for second pass",
        "pages": pages,
        "median_local_cold_ms": statistics.median(p["cold_local_replay_ms"] for p in pages),
        "median_local_warm_ms": statistics.median(p["warm_exact_replay_ms"] for p in pages),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.team_folder / "detection"), indent=2, ensure_ascii=False))
