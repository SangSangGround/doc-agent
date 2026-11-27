
import os
import cv2
import json
import time
import numpy as np

from pathlib import Path
from collections import Counter
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)


# =====================================
# 1. 설정
# =====================================

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = Path(__file__).resolve().parent / "scan_output" / "binary_document.png"
API_KEY = os.getenv("ROBOFLOW_API_KEY")

WORKSPACE_NAME = "-0qtd3"
WORKFLOW_ID = "my-first-project-lkpc5"

CONFIDENCE_THRESHOLD = 0.3

COLORS = {
    "check_box": (0, 255, 0),
    "signature_field": (255, 0, 255),
    "input_field": (255, 100, 0)
}

if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY를 설정해주세요.")


# =====================================
# 2. 이미지 로드
# =====================================

image = cv2.imdecode(
    np.fromfile(str(IMAGE_PATH), dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if image is None:
    raise RuntimeError("이미지를 불러올 수 없습니다.")

height, width = image.shape[:2]
display = image.copy()

print(f"Image size: {width} x {height}")


# =====================================
# 3. Roboflow API 연결
# =====================================

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)


# =====================================
# 4. RF-DETR 추론
# =====================================

print("\nRunning RF-DETR inference...")

start = time.perf_counter()

result = client.run_workflow(
    workspace_name=WORKSPACE_NAME,
    workflow_id=WORKFLOW_ID,
    images={
        "image": str(IMAGE_PATH)
    },
    use_cache=False
)

elapsed = time.perf_counter() - start

print(f"API elapsed time: {elapsed:.2f}s")


# =====================================
# 5. API 원본 결과 저장
# =====================================

with open(
    BASE_DIR / "raw_result.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(result, f, ensure_ascii=False, indent=4)


# =====================================
# 6. 객체 탐지 결과 추출
# =====================================

output = result[0]

if "predictions" not in output:
    raise RuntimeError(
        f"predictions 출력이 없습니다: {list(output.keys())}"
    )

prediction_data = output["predictions"]

if isinstance(prediction_data, dict):
    predictions = prediction_data["predictions"]
else:
    predictions = prediction_data

detections = []

for pred in predictions:

    confidence = float(pred["confidence"])

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    class_name = pred["class"]

    x = float(pred["x"])
    y = float(pred["y"])
    w = float(pred["width"])
    h = float(pred["height"])

    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(width - 1, int(x + w / 2))
    y2 = min(height - 1, int(y + h / 2))

    detection = {
        "id": len(detections) + 1,
        "class": class_name,
        "confidence": confidence,
        "bbox": [x1, y1, x2, y2],
        "center": [x, y],
        "width": w,
        "height": h
    }

    detections.append(detection)


    # =================================
    # 7. Bounding Box 시각화
    # =================================

    color = COLORS.get(class_name, (0, 165, 255))

    cv2.rectangle(
        display,
        (x1, y1),
        (x2, y2),
        color,
        3
    )

    cv2.circle(
        display,
        (int(x), int(y)),
        5,
        color,
        -1
    )

    label = (
        f"{detection['id']}. "
        f"{class_name} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        display,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

    print(label)


# =====================================
# 8. 구조화된 JSON 저장
# =====================================

class_counts = Counter(
    item["class"] for item in detections
)

structured_result = {
    "image_width": width,
    "image_height": height,
    "coordinate_unit": "pixel",
    "api_elapsed_seconds": elapsed,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "total_detections": len(detections),
    "class_counts": dict(class_counts),
    "detections": detections
}

with open(
    BASE_DIR / "detection_result.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        structured_result,
        f,
        ensure_ascii=False,
        indent=4
    )


# =====================================
# 9. 시각화 이미지 저장
# =====================================

success, encoded = cv2.imencode(".jpg", display)

if not success:
    raise RuntimeError("시각화 이미지 인코딩 실패")

encoded.tofile(
    str(BASE_DIR / "detection_visualization.jpg")
)


# =====================================
# 10. 결과 출력
# =====================================

print("\n========== DETECTION RESULT ==========")

print(f"Total: {len(detections)}")

for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

print("\nRF-DETR INFERENCE COMPLETE!")
print("Raw JSON, structured JSON and image saved.")
