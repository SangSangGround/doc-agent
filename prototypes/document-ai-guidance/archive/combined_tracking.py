
import os

# OpenCV를 불러오기 전에 설정
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path


# ==========================================
# 1. MediaPipe 초기화
# ==========================================

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# Python으로 모델 파일을 직접 읽기
model_bytes = MODEL_PATH.read_bytes()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_buffer=model_bytes
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)


# ==========================================
# 2. 종이 모서리 정렬
# ==========================================

def sort_corners(points):
    points = np.asarray(points, dtype=np.float32)

    # 합과 차이를 이용해 모서리 정렬
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)

    return np.array([
        points[np.argmin(s)],  # 왼쪽 위
        points[np.argmin(d)],  # 오른쪽 위
        points[np.argmax(s)],  # 오른쪽 아래
        points[np.argmax(d)]   # 왼쪽 아래
    ], dtype=np.float32)


# ==========================================
# 3. OpenCV 종이 탐지 함수
# ==========================================

def detect_paper(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    # 경계선 검출
    mask = cv2.Canny(
        blurred, 30, 100
    )

    # 끊어진 선 연결
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    frame_area = frame.shape[0] * frame.shape[1]
    paper = None

    # 큰 윤곽선부터 검사
    for contour in sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    ):
        area = cv2.contourArea(contour)

        if area < frame_area * 0.08:
            continue

        perimeter = cv2.arcLength(
            contour, True
        )

        for epsilon in [0.02, 0.03, 0.04]:
            corners = cv2.approxPolyDP(
                contour,
                epsilon * perimeter,
                True
            )

            if (
                len(corners) == 4
                and cv2.isContourConvex(corners)
            ):
                paper = sort_corners(
                    corners.reshape(4, 2)
                )
                break

        if paper is not None:
            break

    return paper, mask


# ==========================================
# 4. 카메라 초기화
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError(
        "카메라를 열 수 없습니다."
    )

start_time = time.monotonic()
last_timestamp = -1
last_paper = None


# ==========================================
# 5. 실시간 처리
# ==========================================

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        while cap.isOpened():

            success, frame = cap.read()

            if not success:
                print("카메라 프레임을 가져오지 못했습니다.")
                break

            # 기존 손가락 추적 코드와 동일하게 좌우 반전
            frame = cv2.flip(frame, 1)

            # --------------------------
            # A. 종이 탐지
            # --------------------------

            paper, mask = detect_paper(frame)
            
            if paper is not None:
                last_paper = paper.copy()
            elif last_paper is not None:
                paper = last_paper.copy()
                
            transform_matrix = None

            if paper is not None:

                points = paper.astype(int)

                # 종이 외곽선 표시
                cv2.polylines(
                    frame,
                    [points],
                    True,
                    (0, 255, 0),
                    3
                )

                for px, py in points:
                    cv2.circle(
                        frame,
                        (px, py),
                        7,
                        (255, 0, 0),
                        -1
                    )

                cv2.putText(
                    frame,
                    "Paper detected",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # A4 크기: 210 x 297 mm
                destination = np.array([
                    [0, 0],
                    [210, 0],
                    [210, 297],
                    [0, 297]
                ], dtype=np.float32)

                transform_matrix = cv2.getPerspectiveTransform(
                    paper,
                    destination
                )

            else:
                cv2.putText(
                    frame,
                    "Paper not detected",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # --------------------------
            # B. 손가락 탐지
            # --------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp + 1
            )

            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]
                finger = hand[8]

                height, width = frame.shape[:2]

                x = int(finger.x * width)
                y = int(finger.y * height)

                # 손가락 위치 표시
                cv2.circle(
                    frame,
                    (x, y),
                    10,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"Index Finger: ({x}, {y})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # --------------------------
                # C. 픽셀 → mm 좌표 변환
                # --------------------------

                if transform_matrix is not None:

                    finger_point = np.array(
                        [[[x, y]]],
                        dtype=np.float32
                    )

                    paper_point = cv2.perspectiveTransform(
                        finger_point,
                        transform_matrix
                    )

                    x_mm, y_mm = paper_point[0][0]

                    # 종이 영역 안에 있을 때만 표시
                    cv2.putText(
                        frame,
                        f"A4 Position: ({x_mm:.1f}, {y_mm:.1f}) mm",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2
                    )

                    print(f"Paper Position: x={x_mm:.1f}, y={y_mm:.1f}")

            # --------------------------
            # D. 결과 출력
            # --------------------------

            cv2.imshow(
                "Combined Tracking",
                frame
            )

            cv2.imshow(
                "Paper Mask",
                mask
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
