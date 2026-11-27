
import cv2
import mediapipe as mp
import time

# MediaPipe 설정
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

# 웹캠 열기
cap = cv2.VideoCapture(0)
start_time = time.monotonic()
last_timestamp = -1

with HandLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 카메라 이미지 변환
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((time.monotonic() - start_time) * 1000)
        timestamp_ms = max(timestamp_ms, last_timestamp + 1)
        last_timestamp = timestamp_ms

        # 손 탐지
        result = landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            # 8번 = 검지 끝
            finger = hand[8]

            height, width = frame.shape[:2]

            x = int(finger.x * width)
            y = int(finger.y * height)

            # 검지 끝에 빨간 점 표시
            cv2.circle(
                frame, (x, y), 10, (0, 0, 255), -1
            )

            # 좌표 표시
            cv2.putText(
                frame,
                f"Index Finger: ({x}, {y})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Hand Tracking", frame)

        # q를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
