
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from collections import deque


# ==========================================
# 1. 기본 설정
# ==========================================

A4_WIDTH = 210
A4_HEIGHT = 297

TARGET_X = 145
TARGET_Y = 240

# 목표점 도착 허용 거리(mm)
TOLERANCE = 10

WINDOW_NAME = "Signature Navigation"

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

model_bytes = MODEL_PATH.read_bytes()

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_buffer=model_bytes
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.5
)


# ==========================================
# 2. 상태 변수
# ==========================================

clicked_points = []
locked_paper = None

transform_matrix = None
inverse_matrix = None

position_history = deque(maxlen=5)

destination = np.array([
    [0, 0],
    [210, 0],
    [210, 297],
    [0, 297]
], dtype=np.float32)


# ==========================================
# 3. 마우스 클릭으로 종이 모서리 등록
# ==========================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if locked_paper is None and len(clicked_points) < 4:

            clicked_points.append([x, y])

            print(f"Corner {len(clicked_points)}: ({x}, {y})")


# ==========================================
# 4. 목표점까지 이동 방향 계산
# ==========================================

def get_direction(x_mm, y_mm):

    dx = TARGET_X - x_mm
    dy = TARGET_Y - y_mm

    distance = np.sqrt(dx**2 + dy**2)

    if distance <= TOLERANCE:
        return "TARGET REACHED!", distance

    # 더 많이 이동해야 하는 방향을 먼저 안내
    if abs(dx) >= abs(dy):

        direction = "RIGHT" if dx > 0 else "LEFT"
        amount = abs(dx)

    else:

        direction = "DOWN" if dy > 0 else "UP"
        amount = abs(dy)

    return f"Move {direction} {amount / 10:.1f} cm", distance


# ==========================================
# 5. mm 좌표를 화면 픽셀로 역변환
# ==========================================

def mm_to_pixel(x, y):

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    converted = cv2.perspectiveTransform(
        point,
        inverse_matrix
    )

    px, py = converted[0][0]

    return int(px), int(py)


# ==========================================
# 6. 카메라 초기화
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

start_time = time.monotonic()
last_timestamp = -1

failed_frames = 0


# ==========================================
# 7. 실시간 처리
# ==========================================

try:

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            if not success or frame is None:

                failed_frames += 1
                position_history.clear()

                print(f"Frame retry: {failed_frames}/10")

                if failed_frames >= 10:
                    break

                time.sleep(0.1)
                continue

            failed_frames = 0
            display = frame.copy()

            # ----------------------------------
            # A. 손가락 탐지
            # ----------------------------------

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms, last_timestamp + 1
            )

            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # ----------------------------------
            # B. 종이 좌표 등록 및 표시
            # ----------------------------------

            if locked_paper is None:

                for i, point in enumerate(clicked_points):

                    px, py = point

                    cv2.circle(
                        display, (px, py),
                        7, (255, 0, 0), -1
                    )

                    cv2.putText(
                        display, str(i + 1),
                        (px + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2
                    )

                cv2.putText(
                    display,
                    f"Select corners: {len(clicked_points)}/4",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2
                )

            else:

                cv2.polylines(
                    display,
                    [locked_paper.astype(np.int32)],
                    True, (0, 255, 0), 3
                )

                cv2.putText(
                    display, "Paper LOCKED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2
                )

                # ----------------------------------
                # C. 가상의 서명란 표시
                # ----------------------------------

                signature_box = np.array([
                    mm_to_pixel(145, 240),
                    mm_to_pixel(190, 240),
                    mm_to_pixel(190, 265),
                    mm_to_pixel(145, 265)
                ], dtype=np.int32)

                cv2.polylines(
                    display,
                    [signature_box],
                    True, (255, 0, 0), 2
                )

                target_pixel = mm_to_pixel(
                    TARGET_X, TARGET_Y
                )

                cv2.circle(
                    display,
                    target_pixel,
                    7, (255, 0, 0), -1
                )

                cv2.putText(
                    display, "SIGN HERE",
                    (target_pixel[0], target_pixel[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 2
                )

            # ----------------------------------
            # D. 손가락 좌표 계산
            # ----------------------------------

            if result.hand_landmarks:

                finger = result.hand_landmarks[0][8]

                height, width = frame.shape[:2]

                x = int(finger.x * width)
                y = int(finger.y * height)

                cv2.circle(
                    display,
                    (x, y),
                    9, (0, 0, 255), -1
                )

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

                    if np.isfinite([x_mm, y_mm]).all():

                        position_history.append([x_mm, y_mm])

                        smooth_x, smooth_y = np.mean(
                            position_history,
                            axis=0
                        )

                        # 좌표 표시
                        cv2.putText(
                            display,
                            f"A4: ({smooth_x:.1f}, {smooth_y:.1f}) mm",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2
                        )

                        # ----------------------------------
                        # E. 거리 및 방향 계산
                        # ----------------------------------

                        direction, distance = get_direction(
                            smooth_x, smooth_y
                        )

                        arrived = distance <= TOLERANCE

                        inside = (
                            0 <= smooth_x <= A4_WIDTH
                            and 0 <= smooth_y <= A4_HEIGHT
                        )

                        if arrived and inside:
                            color = (0, 255, 0)

                            cv2.circle(
                                display,
                                target_pixel,
                                14, color, 3
                            )

                        else:
                            color = (0, 255, 255)

                            if inside:
                                cv2.arrowedLine(
                                    display,
                                    (x, y),
                                    target_pixel,
                                    color, 2,
                                    tipLength=0.15
                                )

                        if not inside:
                            direction = "Finger outside paper"

                        cv2.putText(
                            display,
                            f"Distance: {distance:.1f} mm",
                            (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, color, 2
                        )

                        cv2.putText(
                            display,
                            direction,
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, color, 2
                        )

                    else:
                        position_history.clear()

            else:

                position_history.clear()

                cv2.putText(
                    display, "Finger not detected",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2
                )

            # ----------------------------------
            # F. 화면 출력
            # ----------------------------------

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            # SPACE: 종이 좌표 고정
            if key == 32:

                if len(clicked_points) == 4:

                    candidate = np.array(
                        clicked_points,
                        dtype=np.float32
                    )

                    contour = candidate.reshape(4, 1, 2)

                    if (
                        cv2.isContourConvex(contour)
                        and cv2.contourArea(contour) > 1000
                    ):

                        locked_paper = candidate.copy()

                        transform_matrix = cv2.getPerspectiveTransform(
                            locked_paper,
                            destination
                        )

                        inverse_matrix = cv2.getPerspectiveTransform(
                            destination,
                            locked_paper
                        )

                        position_history.clear()

                        print("Paper LOCKED!")

                    else:
                        print("Invalid corners. Press R.")

            # R: 초기화
            elif key == ord("r"):

                clicked_points.clear()
                position_history.clear()

                locked_paper = None
                transform_matrix = None
                inverse_matrix = None

                print("Paper RESET!")

            # Q: 종료
            elif key == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()
