from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO(
    r"C:\Users\shrut\Desktop\AI_Industrial_Worker_Safety\models\ppe_model.pt"
)

# Open video
video_path = r"C:\Users\shrut\Downloads\archive\source_files\source_files\hardhat.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Video could not be opened")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # AI detection
    results = model(frame, conf=0.25)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("AI Industrial Worker Safety", annotated_frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()