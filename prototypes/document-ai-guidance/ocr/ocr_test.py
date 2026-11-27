
import cv2
import numpy as np
import pytesseract
import json

from pathlib import Path
from pytesseract import Output


# =====================================
# 1. 경로 설정
# =====================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATH = BASE_DIR.parent / "detection" / "test_document.jpg"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =====================================
# 2. 이미지 불러오기
# =====================================

# Windows 한글 경로 문제 방지
image = cv2.imdecode(
    np.fromfile(str(IMAGE_PATH), dtype=np.uint8),
    cv2.IMREAD_COLOR
)

if image is None:
    raise RuntimeError("문서 이미지를 불러올 수 없습니다.")

height, width = image.shape[:2]

display = image.copy()


# =====================================
# 3. 이미지 전처리
# =====================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

processed = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
).apply(gray)


# =====================================
# 4. OCR 실행
# =====================================

data = pytesseract.image_to_data(
    processed,
    lang="kor+eng",
    config="--oem 3 --psm 11",
    output_type=Output.DICT
)

results = []


# =====================================
# 5. 텍스트 및 좌표 추출
# =====================================

for i in range(len(data["text"])):

    text = data["text"][i].strip()
    confidence = float(data["conf"][i])

    if not text or confidence < 0:
        continue

    x = int(data["left"][i])
    y = int(data["top"][i])
    w = int(data["width"][i])
    h = int(data["height"][i])

    item = {
        "id": len(results) + 1,
        "text": text,
        "confidence": confidence,
        "bbox": [x, y, x + w, y + h],
        "center": [x + w / 2, y + h / 2],
        "block_id": int(data["block_num"][i]),
        "paragraph_id": int(data["par_num"][i]),
        "line_id": int(data["line_num"][i])
    }

    results.append(item)

    # OCR 탐지 영역 표시
    color = (0, 255, 0) if confidence >= 50 else (0, 165, 255)

    cv2.rectangle(
        display,
        (x, y),
        (x + w, y + h),
        color,
        2
    )

    cv2.putText(
        display,
        str(item["id"]),
        (x, max(15, y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1
    )

    print(
        f"[{item['id']}] "
        f"{text} / confidence={confidence:.1f}"
    )


# =====================================
# 6. JSON 결과 저장
# =====================================

output = {
    "image_width": width,
    "image_height": height,
    "coordinate_unit": "pixel",
    "total_texts": len(results),
    "texts": results
}

with open(
    BASE_DIR / "ocr_result.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=4
    )


# =====================================
# 7. 시각화 이미지 저장
# =====================================

success, encoded = cv2.imencode(".jpg", display)

if success:
    encoded.tofile(
        str(BASE_DIR / "ocr_visualization.jpg")
    )

print("\nOCR COMPLETE!")
print(f"Detected texts: {len(results)}")
print("JSON and visualization saved.")
