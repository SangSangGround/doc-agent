
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import pyttsx3
import winsound

from pathlib import Path
from collections import deque


# ==========================================
# 1. 기본 설정
# ==========================================

A4_WIDTH = 210
A4_HEIGHT = 297

TARGET_X = 145
TARGET_Y = 240

TOLERANCE = 10

WINDOW_NAME = "Voice Signature Navigation"

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

# 음성 안내용 공유 상태
audio_lock = threading.Lock()
stop_event = threading.Event()

audio_state = {
    "status": "waiting",
    "message": "",
    "distance": None,
    "updated": time.monotonic()
}


def update_audio(status, message="", distance=None):

    with audio_lock:
        audio_state.update({
            "status": status,
            "message": message,
            "distance": distance,
            "updated": time.monotonic()
        })


# ==========================================
# 3. 음성 및 비프음 처리
# ==========================================

def audio_worker():

    engine = None

    try:
        engine = pyttsx3.init()

        for voice in engine.getProperty("voices"):
            if (
                "korean" in voice.name.lower()
                or "heami" in voice.name.lower()
            ):
                engine.setProperty("voice", voice.id)
                break

        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)

    except Exception as e:
        print("TTS initialization error:", e)
        engine = None

    last_speech = -100
    last_beep = -100
    previous_status = None

    while not stop_event.is_set():

        with audio_lock:
            state = audio_state.copy()

        now = time.monotonic()

        status = state["status"]
        message = state["message"]
        distance = state["distance"]

        # 오래된 좌표로 안내하지 않음
        if now - state["updated"] > 1.0:
            status = "waiting"

        if status == "waiting":

            previous_status = status
            stop_event.wait(0.05)
            continue

        # 목표점 도착
        if status == "arrived":

            if previous_status != "arrived":

                winsound.Beep(1200, 400)

                if engine is not None:
                    engine.say("목표 위치에 도착했습니다.")
                    engine.runAndWait()

                last_speech = time.monotonic()

        else:

            # 음성은 최소 2.5초 간격
            if message and now - last_speech >= 2.5:

                if engine is not None:
                    engine.say(message)
                    engine.runAndWait()

                last_speech = time.monotonic()

            # 이동 중 거리별 비프음
            elif status == "moving" and distance is not None:

                if distance >= 100:
                    interval = 1.5
                elif distance >= 50:
                    interval = 1.0
                elif distance >= 20:
                    interval = 0.6
                else:
                    interval = 0.3

                if now - last_beep >= interval:

                    winsound.Beep(850, 70)
                    last_beep = time.monotonic()

        previous_status = status

        stop_event.wait(0.05)

    if engine is not None:
        engine.stop()


# ==========================================
# 4. 종이 모서리 등록
# ==========================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        if locked_paper is None and len(clicked_points) < 4:

            clicked_points.append([x, y])

            print(
                f"Corner {len(clicked_points)}: ({x}, {y})"
            )


# ==========================================
# 5. 목표점 거리 및 방향 계산
# ==========================================

def get_direction(x_mm, y_mm):

    dx = TARGET_X - x_mm
    dy = TARGET_Y - y_mm

    distance = np.hypot(dx, dy)

    if distance <= TOLERANCE:
        return "TARGET REACHED!", "", distance

    if abs(dx) >= abs(dy):

        direction = "RIGHT" if dx > 0 else "LEFT"
        korean = "오른쪽" if dx > 0 else "왼쪽"

        amount = abs(dx)

    else:

        direction = "DOWN" if dy > 0 else "UP"
        korean = "아래쪽" if dy > 0 else "위쪽"

        amount = abs(dy)

    cm = amount / 10

    text = f"Move {direction} {cm:.1f} cm"
    speech = f"{korean}으로 {cm:.1f}센티미터 이동하세요."

    return text, speech, distance


# ==========================================
# 6. mm 좌표를 픽셀 좌표로 역변환
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
# 7. 카메라 초기화
# ==========================================

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

start_time = time.monotonic()
last_timestamp = -1
failed_frames = 0

# 목표 위치에 일정 시간 머무르는지 확인
arrival_start = None

# 음성 안내 스레드 실행
audio_thread = threading.Thread(
    target=audio_worker,
    daemon=True
)

audio_thread.start()


# ==========================================
# 8. 실시간 추적
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
                arrival_start = None

                update_audio("waiting")

                print(f"Frame retry: {failed_frames}/10")

                if failed_frames >= 10:
                    break

                time.sleep(0.1)
                continue

            failed_frames = 0

            display = frame.copy()

            # ----------------------------------
            # A. MediaPipe 손가락 추적
            # ----------------------------------

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

            # ----------------------------------
            # B. 종이 모서리 등록 및 표시
            # ----------------------------------

            if locked_paper is None:

                update_audio("waiting")

                for i, point in enumerate(clicked_points):

                    px, py = point

                    cv2.circle(
                        display, (px, py),
                        7, (255, 0, 0), -1
                    )

                    cv2.putText(
                        display,
                        str(i + 1),
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
                    display,
                    "Paper LOCKED",
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
                    display,
                    "SIGN HERE",
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

                    if not np.isfinite([x_mm, y_mm]).all():

                        position_history.clear()
                        arrival_start = None

                        update_audio("waiting")
                        continue

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
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2
                    )

                    # ----------------------------------
                    # E. 방향 및 거리 계산
                    # ----------------------------------

                    direction, speech, distance = get_direction(
                        smooth_x,
                        smooth_y
                    )

                    inside = (
                        0 <= smooth_x <= A4_WIDTH
                        and 0 <= smooth_y <= A4_HEIGHT
                    )

                    color = (0, 255, 255)

                    # 종이 밖에 있는 경우
                    if not inside:

                        arrival_start = None
                        direction = "Finger outside paper"

                        update_audio(
                            "outside",
                            "손가락을 종이 위에 올려주세요."
                        )

                    # 목표점에 도착한 경우
                    elif distance <= TOLERANCE:

                        if arrival_start is None:
                            arrival_start = time.monotonic()

                        # 0.7초 이상 머물러야 도착 판정
                        if time.monotonic() - arrival_start >= 0.7:

                            direction = "TARGET REACHED!"
                            color = (0, 255, 0)

                            update_audio("arrived")

                            cv2.circle(
                                display,
                                target_pixel,
                                14, color, 3
                            )

                        else:

                            direction = "Hold your finger"

                            update_audio(
                                "holding",
                                "손가락을 잠시 유지하세요."
                            )

                    # 목표점으로 이동 중
                    else:

                        arrival_start = None

                        update_audio(
                            "moving",
                            speech,
                            distance
                        )

                        cv2.arrowedLine(
                            display,
                            (x, y),
                            target_pixel,
                            color,
                            2,
                            tipLength=0.15
                        )

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
                arrival_start = None

                if locked_paper is not None:

                    update_audio(
                        "lost",
                        "손가락이 인식되지 않습니다."
                    )

                cv2.putText(
                    display,
                    "Finger not detected",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2
                )

            # ----------------------------------
            # F. 화면 출력 및 키보드 제어
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
                        arrival_start = None

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

                arrival_start = None

                update_audio("waiting")

                print("Paper RESET!")

            # Q: 종료
            elif key == ord("q"):
                break

finally:

    update_audio("waiting")
    stop_event.set()

    cap.release()
    cv2.destroyAllWindows()

    audio_thread.join(timeout=3)
