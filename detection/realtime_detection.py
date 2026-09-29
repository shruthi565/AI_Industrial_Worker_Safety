from ultralytics import YOLO
import cv2
import time

# =========================================================
# LOAD TRAINED AI MODEL
# =========================================================

model = YOLO(
    r"C:\Users\shrut\Desktop\AI_Industrial_Worker_Safety\models\ppe_model.pt"
)

# =========================================================
# INPUT VIDEO
# =========================================================

video_path = (
    r"C:\Users\shrut\Downloads\archive\source_files\source_files\hardhat.mp4"
)

cap = cv2.VideoCapture(video_path)
# =========================================================
# OUTPUT VIDEO
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

fps_video = cap.get(cv2.CAP_PROP_FPS)

if fps_video <= 0:
    fps_video = 25

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = (
    r"C:\Users\shrut\Desktop\AI_Industrial_Worker_Safety"
    r"\results\industrial_safety_demo.mp4"
)

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps_video,
    (frame_width, frame_height)
)

if not cap.isOpened():
    print("Video could not be opened")
    exit()

prev_time = time.time()

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # =====================================================
    # AI DETECTION
    # =====================================================

    results = model(frame, conf=0.25)

    output = results[0].plot()

    height, width = output.shape[:2]

    # =====================================================
    # DANGER ZONE
    # =====================================================

    x1 = int(width * 0.35)
    y1 = int(height * 0.25)

    x2 = int(width * 0.75)
    y2 = int(height * 0.90)

    # Transparent danger zone
    overlay = output.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 165, 255),
        -1
    )

    output = cv2.addWeighted(
        overlay,
        0.12,
        output,
        0.88,
        0
    )

    # Danger zone border
    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 165, 255),
        3
    )

    # Danger zone label
    cv2.putText(
        output,
        "DANGER ZONE",
        (x1 + 10, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    # =====================================================
    # COUNTERS
    # =====================================================

    worker_count = 0
    danger_workers = 0
    fall_detected = 0

    hardhat_count = 0
    no_hardhat_count = 0

    vest_count = 0
    no_vest_count = 0

    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for box, cls in zip(
        results[0].boxes.xyxy,
        results[0].boxes.cls
    ):

        class_id = int(cls)

        class_name = model.names[class_id]

        # =================================================
        # PERSON DETECTION
        # =================================================

        if class_name == "Person":

            worker_count += 1

            # ---------------------------------------------
            # Get person dimensions
            # ---------------------------------------------

            person_width = float(box[2] - box[0])
            person_height = float(box[3] - box[1])

            # ---------------------------------------------
            # Simple fall indication
            # ---------------------------------------------

            if person_width > person_height * 1.2:

                fall_detected += 1

                # Draw warning around possible fallen worker
                cv2.rectangle(
                    output,
                    (
                        int(box[0]),
                        int(box[1])
                    ),
                    (
                        int(box[2]),
                        int(box[3])
                    ),
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    output,
                    "POSSIBLE FALL",
                    (
                        int(box[0]),
                        max(int(box[1]) - 10, 80)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            # ---------------------------------------------
            # Person center
            # ---------------------------------------------

            x_center = int(
                (float(box[0]) + float(box[2])) / 2
            )

            y_center = int(
                (float(box[1]) + float(box[3])) / 2
            )

            # ---------------------------------------------
            # Danger zone check
            # ---------------------------------------------

            if (
                x1 < x_center < x2
                and
                y1 < y_center < y2
            ):

                danger_workers += 1

                # Mark worker inside danger zone
                cv2.circle(
                    output,
                    (x_center, y_center),
                    7,
                    (0, 0, 255),
                    -1
                )

        # =================================================
        # PPE DETECTION
        # =================================================

        elif class_name == "Hardhat":

            hardhat_count += 1

        elif class_name == "NO-Hardhat":

            no_hardhat_count += 1

        elif class_name == "Safety Vest":

            vest_count += 1

        elif class_name == "NO-Safety Vest":

            no_vest_count += 1

    # =====================================================
    # SAFETY STATUS
    # =====================================================

    if fall_detected > 0:

        status = "FALL ALERT"
        status_color = (0, 0, 255)

    elif danger_workers > 0:

        status = "DANGER"
        status_color = (0, 0, 255)

    elif no_hardhat_count > 0 or no_vest_count > 0:

        status = "PPE ALERT"
        status_color = (0, 165, 255)

    else:

        status = "SAFE"
        status_color = (0, 200, 0)

    # =====================================================
    # TOP HEADER
    # =====================================================

    cv2.rectangle(
        output,
        (0, 0),
        (width, 75),
        (25, 25, 25),
        -1
    )

    # Main title
    cv2.putText(
        output,
        "AI INDUSTRIAL WORKER SAFETY",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Subtitle
    cv2.putText(
        output,
        "REAL-TIME AI MONITORING SYSTEM",
        (20, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1
    )

    # =====================================================
    # STATUS PANEL
    # =====================================================

    panel_x = width - 280

    cv2.rectangle(
        output,
        (panel_x, 10),
        (width - 10, 65),
        (45, 45, 45),
        -1
    )

    cv2.putText(
        output,
        f"STATUS: {status}",
        (panel_x + 15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_color,
        2
    )

    # =====================================================
    # SAFETY MONITOR PANEL
    # =====================================================

    panel_width = 310
    panel_height = 200

    panel_x = 15
    panel_y = height - panel_height - 15

    cv2.rectangle(
        output,
        (panel_x, panel_y),
        (panel_x + panel_width, height - 15),
        (30, 30, 30),
        -1
    )

    # Panel title
    cv2.putText(
        output,
        "SAFETY MONITOR",
        (panel_x + 15, panel_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Workers
    cv2.putText(
        output,
        f"Workers detected : {worker_count}",
        (panel_x + 15, panel_y + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    # Hardhats
    cv2.putText(
        output,
        f"Hardhats         : {hardhat_count}",
        (panel_x + 15, panel_y + 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 0),
        1
    )

    # No hardhat
    cv2.putText(
        output,
        f"No Hardhat       : {no_hardhat_count}",
        (panel_x + 15, panel_y + 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255) if no_hardhat_count > 0 else (180, 180, 180),
        1
    )

    # Safety vest
    cv2.putText(
        output,
        f"Safety Vests     : {vest_count}",
        (panel_x + 15, panel_y + 124),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 220, 0),
        1
    )

    # No safety vest
    cv2.putText(
        output,
        f"No Safety Vest   : {no_vest_count}",
        (panel_x + 15, panel_y + 147),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255) if no_vest_count > 0 else (180, 180, 180),
        1
    )

    # Fall detection
    cv2.putText(
        output,
        f"Possible Falls    : {fall_detected}",
        (panel_x + 15, panel_y + 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 255) if fall_detected > 0 else (180, 180, 180),
        1
    )

    # =====================================================
    # DANGER ZONE STATUS
    # =====================================================

    cv2.putText(
        output,
        f"DANGER ZONE : {danger_workers}",
        (width - 250, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2
    )

    # =====================================================
    # FPS
    # =====================================================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    cv2.putText(
        output,
        f"FPS: {fps:.1f}",
        (width - 100, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1
    )

    # =====================================================
    # DISPLAY
    # =====================================================
    
    cv2.imshow(
        "AI Industrial Worker Safety",
        output
    )
    out.write(output)
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# RELEASE RESOURCES
# =========================================================
out.release()
cap.release()
cv2.destroyAllWindows()

print("Demo video saved successfully!")
print(output_path)