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
        conf=0.30,
        iou=0.50,
        verbose=False
    )


    result = results[0]

    output = result.plot()


    height, width = output.shape[:2]


    # =====================================================
    # DANGER ZONE
    # =====================================================
    #
    # This region represents the operating area
    # of the industrial machine/robot.
    #
    # Adjust these coordinates according to your camera.
    #

    zone_x1 = int(width * 0.10)

    zone_y1 = int(height * 0.15)

    zone_x2 = int(width * 0.55)

    zone_y2 = int(height * 0.75)


    # Draw danger-zone boundary

    cv2.rectangle(
        output,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 165, 255),
        3
    )


    cv2.putText(
        output,
        "MACHINE DANGER ZONE",
        (zone_x1 + 10, zone_y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )


    danger_workers = 0

    total_workers = 0


    # =====================================================
    # CHECK PERSON POSITION
    # =====================================================

    for box, cls in zip(
        result.boxes.xyxy,
        result.boxes.cls
    ):

        class_id = int(cls)

        class_name = model.names[class_id]


        if class_name != "Person":
            continue


        total_workers += 1


        bx1 = float(box[0])
        by1 = float(box[1])
        bx2 = float(box[2])
        by2 = float(box[3])


        # Person center

        center_x = int(
            (bx1 + bx2) / 2
        )

        center_y = int(
            (by1 + by2) / 2
        )


        # =================================================
        # PERSON INSIDE DANGER ZONE?
        # =================================================

        inside_zone = (
            zone_x1 < center_x < zone_x2
            and
            zone_y1 < center_y < zone_y2
        )


        if inside_zone:

            danger_workers += 1


            # Red point

            cv2.circle(
                output,
                (center_x, center_y),
                8,
                (0, 0, 255),
                -1
            )


            # Red warning

            cv2.putText(
                output,
                "DANGER",
                (
                    int(bx1),
                    max(30, int(by1) - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3
            )


    # =====================================================
    # STATUS
    # =====================================================

    if danger_workers > 0:

        status = "DANGER ZONE ALERT"

        color = (0, 0, 255)

    else:

        status = "AREA CLEAR"

        color = (0, 200, 0)


    # =====================================================
    # INFORMATION PANEL
    # =====================================================

    cv2.rectangle(
        output,
        (10, 10),
        (410, 95),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        output,
        "DANGER ZONE MONITOR",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        f"Workers: {total_workers}",
        (25, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        status,
        (190, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


    cv2.imshow(
        "Danger Zone Detection",
        output
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()

cv2.destroyAllWindows()