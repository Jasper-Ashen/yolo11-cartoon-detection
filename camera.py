
import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/runs/detect/nai_v1/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("无法打开摄像头")


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        device=0,
        conf=0.35,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Nai Detect", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
