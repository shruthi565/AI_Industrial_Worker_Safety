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


    hardhat = 0
    no_hardhat = 0
    vest = 0
    no_vest = 0
    persons = 0


    for cls, conf in zip(
        result.boxes.cls,
        result.boxes.conf
    ):

        class_id = int(cls)

        class_name = model.names[class_id]

        if class_name == "Person":
            persons += 1

        elif class_name == "Hardhat":
            hardhat += 1

        elif class_name == "NO-Hardhat":
            no_hardhat += 1

        elif class_name == "Safety Vest":
            vest += 1

        elif class_name == "NO-Safety Vest":
            no_vest += 1


    # ---------------------------------------------
    # SAFETY STATUS
    # ---------------------------------------------

    if no_hardhat > 0 or no_vest > 0:

        status = "PPE ALERT"

    else:

        status = "PPE SAFE"


    # ---------------------------------------------
    # DISPLAY
    # ---------------------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (400, 155),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        output,
        "SAFETY MONITOR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        f"Workers: {persons}",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        f"Hardhat: {hardhat}  No Hardhat: {no_hardhat}",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        f"Vest: {vest}  No Vest: {no_vest}",
        (25, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    color = (0, 0, 255) if "ALERT" in status else (0, 200, 0)

    cv2.putText(
        output,
        status,
        (25, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


    cv2.imshow(
        "Safety / PPE Detection",
        output
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()