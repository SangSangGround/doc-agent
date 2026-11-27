import cv2

for backend_name, backend in [
    ("MSMF", cv2.CAP_MSMF),
    ("DSHOW", cv2.CAP_DSHOW)
]:
    for index in range(3):
        cap = cv2.VideoCapture(index, backend)
        opened = cap.isOpened()
        ret = False

        if opened:
            ret, frame = cap.read()

        print(
            f"{backend_name} / Camera {index}: "
            f"opened={opened}, frame={ret}"
        )

        cap.release()