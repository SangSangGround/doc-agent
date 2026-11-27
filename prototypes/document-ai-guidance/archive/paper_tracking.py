import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

def sort_corners(points):
    # 네 모서리를 왼쪽 위 → 오른쪽 위 → 오른쪽 아래 → 왼쪽 아래 순으로 정렬
    points = np.asarray(points, dtype=np.float32)
    by_y = points[np.argsort(points[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array(
        [top[0], top[1], bottom[1], bottom[0]],
        dtype=np.float32
    )

while True:
    success, frame = cap.read()
    if not success:
        break

    # 1. 이미지 전처리
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 밝은 종이를 흰색 영역으로 분리
    _, edges = cv2.threshold(
        blurred, 170, 255, cv2.THRESH_BINARY
    )

    # 3. 끊어진 영역 연결
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE, kernel
    )

    # 4. 윤곽선 탐지
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    paper = None
    frame_area = frame.shape[0] * frame.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < frame_area * 0.15:
            continue

        perimeter = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(corners) == 4 and cv2.isContourConvex(corners):
            paper = sort_corners(corners.reshape(4, 2))
            break

    # 5. 종이 탐지 결과 표시
    if paper is not None:
        points = paper.astype(int)

        cv2.polylines(
            frame, [points], True, (0, 255, 0), 3
        )

        labels = ["TL", "TR", "BR", "BL"]

        for i, (x, y) in enumerate(points):
            cv2.circle(
                frame, (x, y), 7, (0, 0, 255), -1
            )
            cv2.putText(
                frame,
                labels[i],
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            "A4 detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "Paper not detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # 6. 결과 출력
    cv2.imshow("A4 Paper Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()