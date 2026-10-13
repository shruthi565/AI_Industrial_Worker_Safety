
from pathlib import Path
from ultralytics import YOLO
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fall_model.pt"

if not MODEL_PATH.exists():
    print("ERROR: Fall model not found:", MODEL_PATH)
    raise SystemExit(1)

model = YOLO(str(MODEL_PATH))

print("Fall model classes:", model.names)

video_path = input("Enter the full path of your test video: ").strip().strip('"')

if not Path(video_path).is_file():
    print("ERROR: Video file not found.")
    raise SystemExit(1)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    raise SystemExit(1)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(
        frame,
        conf=0.60,
        imgsz=640,
        verbose=False
    )

    result = results[0]
    counts = {}

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        name = result.names[class_id]
        counts[name] = counts.get(name, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 120),
            2
        )

    print("Current frame:", counts)

    cv2.imshow("Fall Model Test - Press Q to Exit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
