"""카메라용 일반 문서 경계 검출. A4 정합·OCR 품질 검사와 분리한다."""
from __future__ import annotations

import cv2
import numpy as np

from docagent.vision.geometry import order_quad


def document_boundary(image: np.ndarray) -> dict:
    h, w = image.shape[:2]
    scale = min(1., 800 / max(h, w))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.resize(gray, (round(w * scale), round(h * scale)))
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.morphologyEx(cv2.Canny(smooth, 30, 100), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, bright = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, -4)
    candidates = []
    for binary in (edges, bright, adaptive):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:18]:
            area = cv2.contourArea(contour) / gray.size
            if not .10 <= area <= .99:
                continue
            perimeter = cv2.arcLength(contour, True)
            for epsilon in (.015, .025, .04):
                poly = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                if len(poly) != 4 or not cv2.isContourConvex(poly):
                    continue
                q = order_quad(poly.reshape(4, 2)).astype(np.float32)
                sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
                if min(sides) < 35 or max(sides) / min(sides) > 6:
                    continue
                # 심하게 찌그러진 사각형과 화면 경계를 대체한 사각형을 제외한다.
                angles = []
                for i in range(4):
                    a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
                    angles.append(abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))))
                if max(angles) > .8:
                    continue
                mask = np.zeros_like(gray)
                cv2.fillConvexPoly(mask, q.astype(np.int32), 255)
                inner = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
                outer = (cv2.dilate(mask, np.ones((13, 13), np.uint8)) > 0) & (mask == 0)
                contrast = float(np.mean(gray[inner]) - np.mean(gray[outer])) if outer.any() else 0.
                # 표 테두리 등 종이 내부의 사각형을 문서 경계로 삼지 않는다.
                if contrast < 7:
                    continue
                margin = max(2, min(gray.shape) * .008)
                clipped = bool(np.any(q[:, 0] <= margin) or np.any(q[:, 1] <= margin)
                               or np.any(q[:, 0] >= gray.shape[1] - 1 - margin)
                               or np.any(q[:, 1] >= gray.shape[0] - 1 - margin))
                candidates.append((area, q / scale, clipped, contrast))
                break
    if not candidates:
        return {"status": "searching", "quad": None, "area_ratio": 0.,
                "instruction": "종이의 테두리를 찾고 있어요. 네 모서리에 여백을 주고, 종이와 색이 다른 배경에 놓아 주세요."}
    area, quad, clipped, contrast = max(candidates, key=lambda item: item[0])
    return {"status": "clipped" if clipped else "framed", "quad": quad.tolist(),
            "area_ratio": round(area, 3), "edge_contrast": round(contrast, 1),
            "instruction": "문서가 화면 밖으로 나갔어요. 카메라와 문서를 조금 멀리 떨어뜨려 주세요."
            if clipped else "문서가 들어왔어요. 잠시 그대로 유지해 주세요."}


def crop_document(image: np.ndarray, boundary: dict) -> np.ndarray:
    """검출한 네 꼭짓점을 원래 종횡비로 펴며, 미검출 때 임의로 자르지 않는다."""
    if boundary["status"] != "framed":
        return image.copy()
    q = np.asarray(boundary["quad"], np.float32)
    sides = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
    w, h = round(max(sides[0], sides[2])), round(max(sides[1], sides[3]))
    target = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(q, target), (w, h))
