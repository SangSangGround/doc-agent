
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

WINDOW_NAME = "Locked Paper Tracking"

MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"

# 한글 경로 문제 방지
model_bytes = MODEL_PATH.read_bytes()

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_buffer=model_bytes
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1
)


# ==========================================
# 2. 상태 변수
# ==========================================

clicked_points = []
locked_paper = None
transform_matrix = None

# 최근 5개 좌표를 저장하여 흔들림 완화
position_history = deque(maxlen=5)

destination = np.array([
    [0, 0],
    [210, 0],
    [210, 297],
    [0, 297]
], dtype=np.float32)


# ==========================================
# 3. 마우스로 종이 모서리 등록
# ==========================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if locked_paper is None and len(clicked_points) < 4:

            clicked_points.append([x, y])

            print(
                f"Corner {len(clicked_points)}: ({x}, {y})"
            )


# ==========================================
# 4. 카메라 초기화
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
# 5. 실시간 추적
# ==========================================

try:

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:

            success, frame = cap.read()

            # 일시적인 프레임 실패 시 재시도
            if not success or frame is None:

                failed_frames += 1

                print(f"Frame retry: {failed_frames}/10")

                if failed_frames >= 10:
                    print("카메라 프레임 수신 실패")
                    break

                time.sleep(0.1)
                continue

            failed_frames = 0

            # 실제 좌표 방향을 위해 좌우 반전하지 않음
            display = frame.copy()

            # ==================================
            # A. MediaPipe 손가락 추적
            # ==================================

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

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

            # ==================================
            # B. 종이 모서리 표시
            # ==================================

            if locked_paper is None:

                for i, point in enumerate(clicked_points):

                    px, py = point

                    cv2.circle(
                        display,
                        (px, py),
                        7,
                        (255, 0, 0),
                        -1
                    )

                    cv2.putText(
                        display,
                        str(i + 1),
                        (px + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    display,
                    f"Select corners: {len(clicked_points)}/4",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.polylines(
                    display,
                    [locked_paper.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    display,
                    "Paper LOCKED",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ==================================
            # C. 손가락 좌표 계산
            # ==================================

            if result.hand_landmarks:

                finger = result.hand_landmarks[0][8]

                height, width = frame.shape[:2]

                x = int(finger.x * width)
                y = int(finger.y * height)

                cv2.circle(
                    display,
                    (x, y),
                    9,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    f"Finger: ({x}, {y}) px",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # ==============================
                # D. 픽셀 좌표 -> mm 좌표
                # ==============================

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

                        # 이동평균으로 좌표 흔들림 완화
                        position_history.append(
                            [x_mm, y_mm]
                        )

                        smooth_x, smooth_y = np.mean(
                            position_history,
                            axis=0
                        )

                        cv2.putText(
                            display,
                            f"A4: ({smooth_x:.1f}, {smooth_y:.1f}) mm",
                            (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

                        if (
                            0 <= smooth_x <= A4_WIDTH
                            and 0 <= smooth_y <= A4_HEIGHT
                        ):
                            status = "Finger inside paper"
                        else:
                            status = "Finger outside paper"

                        cv2.putText(
                            display,
                            status,
                            (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 255),
                            2
                        )

            else:

                position_history.clear()

            # ==================================
            # E. 화면 출력 및 키보드 제어
            # ==================================

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

                        position_history.clear()

                        print("Paper LOCKED!")

                    else:
                        print("Invalid corners. Press R and retry.")

                else:
                    print("Select all four corners first.")

            # R: 종이 좌표 초기화
            elif key == ord("r"):

                clicked_points.clear()
                locked_paper = None
                transform_matrix = None
                position_history.clear()

                print("Paper RESET!")

            # Q: 프로그램 종료
            elif key == ord("q"):
                break

finally:

    cap.release()
    cv2.destroyAllWindows()
