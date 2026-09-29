from ultralytics import YOLO
import cv2


MODEL_PATH = r"C:\Users\shrut\Desktop\AI_Industrial_Worker_Safety\models\ppe_model.pt"

VIDEO_PATH = r"C:\Users\shrut\Downloads\archive\source_files\source_files\hardhat.mp4"


model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Video could not be opened.")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        break


    results = model(
        frame,
        conf=0.20,
        iou=0.50,
        verbose=False
    )

    result = results[0]

    output = result.plot()

    fall_count = 0
    person_count = 0


    for box, cls, conf in zip(
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf
    ):

        class_id = int(cls)

        class_name = model.names[class_id]

        if class_name != "Person":
            continue


        person_count += 1


        x1 = float(box[0])
        y1 = float(box[1])
        x2 = float(box[2])
        y2 = float(box[3])


        box_width = x2 - x1
        box_height = y2 - y1


        # ----------------------------------------
        # FALL DETECTION
        # ----------------------------------------

        if box_width > box_height * 0.90:

            fall_count += 1

            cv2.rectangle(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                4
            )

            cv2.putText(
                output,
                "POSSIBLE FALL",
                (
                    int(x1),
                    max(30, int(y1) - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )


    # ----------------------------------------
    # STATUS
    # ----------------------------------------

    if fall_count > 0:

        status = "FALL ALERT"

        color = (0, 0, 255)

    else:

        status = "POSSIBLE FALL / LOW POSTURE"

        color = (0, 200, 0)


    # ----------------------------------------
    # PANEL
    # ----------------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (430, 95),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        output,
        "FALL DETECTION",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        f"Workers Detected: {person_count}",
        (25, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        status,
        (245, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2
    )


    cv2.imshow(
        "Fall Detection",
        output
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()