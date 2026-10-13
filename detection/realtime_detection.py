
from pathlib import Path
import time
import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = (
    Path.home()
    / "OneDrive"
    / "Documents"
    / "AI_Industrial_Worker_Safety"
)

PPE_MODEL_PATH = BASE_DIR / "models" / "ppe_model.pt"
FALL_MODEL_PATH = BASE_DIR / "models" / "fall_model.pt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================

PPE_CONFIDENCE = 0.20
FALL_CONFIDENCE = 0.15
FALL_CONFIRM_FRAMES = 3
IMG_SIZE = 640


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def choose_video():
    """Allow the user to select a video each time."""
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askopenfilename(
        title="Select Video for Industrial Safety Detection",
        filetypes=[
            (
                "Video files",
                "*.mp4 *.avi *.mov *.mkv *.wmv *.mpeg *.mpg",
            ),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not selected:
        return None

    return Path(selected)


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. CHECK MODEL FILES
    # -----------------------------------------------------

    if not PPE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"PPE model not found:\n{PPE_MODEL_PATH}"
        )

    if not FALL_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Fall model not found:\n{FALL_MODEL_PATH}"
        )

    # -----------------------------------------------------
    # 2. SELECT ANY SUPPORTED VIDEO
    # -----------------------------------------------------

    video_path = choose_video()

    if video_path is None:
        print("No video selected. Exiting.")
        return

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Selected video does not exist:\n{video_path}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    output_path = (
        RESULTS_DIR
        / f"{video_path.stem}_safety_result_{timestamp}.mp4"
    )

    print("Selected video:", video_path)

    # -----------------------------------------------------
    # 3. LOAD TRAINED MODELS
    # -----------------------------------------------------

    print("Loading PPE model...")
    ppe_model = YOLO(str(PPE_MODEL_PATH))

    print("Loading fall model...")
    fall_model = YOLO(str(FALL_MODEL_PATH))

    print("PPE classes:", ppe_model.names)
    print("Fall classes:", fall_model.names)

    # -----------------------------------------------------
    # 4. OPEN VIDEO
    # -----------------------------------------------------

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open video:\n{video_path}\n"
            "Please try another supported video file."
        )

    fps_video = cap.get(cv2.CAP_PROP_FPS)

    if not (0 < fps_video <= 120):
        fps_video = 25.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0 or frame_height <= 0:
        cap.release()
        raise RuntimeError(
            "The selected video has invalid dimensions."
        )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps_video,
        (frame_width, frame_height),
    )

    if not out.isOpened():
        cap.release()
        out.release()
        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # -----------------------------------------------------
    # 5. INITIALIZE DETECTION STATE
    # -----------------------------------------------------

    fall_streak = 0
    previous_time = time.time()
    frame_number = 0

    total_fall_alert_frames = 0
    total_ppe_alert_frames = 0
    total_pre_fall_frames = 0

    # Danger-zone coordinates
    zone_x1 = int(frame_width * 0.35)
    zone_y1 = int(frame_height * 0.25)
    zone_x2 = int(frame_width * 0.75)
    zone_y2 = int(frame_height * 0.90)

    # -----------------------------------------------------
    # 6. PROCESS VIDEO FRAMES
    # -----------------------------------------------------

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1
            output = frame.copy()

            # Reset counters for this frame
            workers = 0
            hardhats = 0
            no_hardhats = 0
            safety_vests = 0
            no_vests = 0
            masks = 0
            no_masks = 0

            standing = 0
            sitting = 0
            bending = 0
            assisting = 0
            pre_fall_risks = 0
            danger_workers = 0

            fallen_this_frame = False
            best_fall = None

            # =================================================
            # 6A. PPE DETECTION
            # =================================================

            ppe_results = ppe_model.predict(
                source=frame,
                conf=PPE_CONFIDENCE,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            for result in ppe_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    raw_name = ppe_model.names[class_id]
                    name = normalize(raw_name)

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    color = (180, 180, 180)
                    label = str(raw_name).upper()

                    if name == "person":

                        workers += 1
                        color = (255, 200, 0)
                        label = "PERSON"

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if (
                            zone_x1 < center_x < zone_x2
                            and zone_y1 < center_y < zone_y2
                        ):
                            danger_workers += 1

                    elif name == "hardhat":

                        hardhats += 1
                        color = (0, 220, 0)

                    elif name == "no hardhat":

                        no_hardhats += 1
                        color = (0, 0, 255)

                    elif name == "safety vest":

                        safety_vests += 1
                        color = (0, 220, 0)

                    elif name in ("no safety vest", "no vest"):

                        no_vests += 1
                        color = (0, 0, 255)

                    elif name == "mask":

                        masks += 1
                        color = (0, 220, 0)

                    elif name == "no mask":

                        no_masks += 1
                        color = (0, 0, 255)

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        output,
                        f"{label} {confidence:.0%}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # =================================================
            # 6B. FALL AND POSTURE DETECTION
            # =================================================

            fall_results = fall_model.predict(
                source=frame,
                conf=FALL_CONFIDENCE,
                imgsz=IMG_SIZE,
                verbose=False,
            )

            for result in fall_results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    raw_name = fall_model.names[class_id]
                    name = normalize(raw_name)

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    color = (255, 180, 0)
                    label = str(raw_name).upper()

                    if name in ("fallen", "falling"):

                        fallen_this_frame = True

                        if (
                            best_fall is None
                            or confidence > best_fall[0]
                        ):
                            best_fall = (
                                confidence,
                                (x1, y1, x2, y2),
                            )

                        color = (0, 0, 255)
                        label = "FALLEN"

                    elif "pre" in name and "fall" in name:

                        pre_fall_risks += 1
                        color = (0, 165, 255)
                        label = "PRE-FALL RISK"

                    elif name == "standing":

                        standing += 1
                        color = (0, 220, 0)
                        label = "STANDING"

                    elif name == "sitting":

                        sitting += 1
                        label = "SITTING"

                    elif name == "bending":

                        bending += 1
                        label = "BENDING"

                    elif name == "assisting":

                        assisting += 1
                        color = (255, 0, 255)
                        label = "ASSISTING"

                    else:
                        continue

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    cv2.putText(
                        output,
                        f"{label} {confidence:.0%}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            # =================================================
            # 6C. CONFIRM FALL ACROSS CONSECUTIVE FRAMES
            # =================================================

            if fallen_this_frame:
                fall_streak += 1
            else:
                fall_streak = 0

            fall_alert = (
                fall_streak >= FALL_CONFIRM_FRAMES
            )

            if fall_alert:

                total_fall_alert_frames += 1

                if best_fall is not None:

                    confidence, (x1, y1, x2, y2) = best_fall

                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        4,
                    )

                    cv2.putText(
                        output,
                        f"FALL ALERT {confidence:.0%}",
                        (x1, max(35, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3,
                    )

            if pre_fall_risks:
                total_pre_fall_frames += 1

            if no_hardhats or no_vests or no_masks:
                total_ppe_alert_frames += 1

            # =================================================
            # 6D. DRAW DANGER ZONE
            # =================================================

            overlay = output.copy()

            cv2.rectangle(
                overlay,
                (zone_x1, zone_y1),
                (zone_x2, zone_y2),
                (0, 165, 255),
                -1,
            )

            output = cv2.addWeighted(
                overlay, 0.10, output, 0.90, 0
            )

            cv2.rectangle(
                output,
                (zone_x1, zone_y1),
                (zone_x2, zone_y2),
                (0, 165, 255),
                2,
            )

            cv2.putText(
                output,
                "DANGER ZONE",
                (zone_x1 + 8, zone_y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
            )

            # =================================================
            # 6E. SAFETY STATUS
            # =================================================

            if fall_alert:

                status = "FALL ALERT"
                status_color = (0, 0, 255)

            elif pre_fall_risks:

                status = "PRE-FALL WARNING"
                status_color = (0, 165, 255)

            elif danger_workers:

                status = "DANGER ZONE OCCUPIED"
                status_color = (0, 0, 255)

            elif no_hardhats or no_vests or no_masks:

                status = "PPE ALERT"
                status_color = (0, 165, 255)

            else:

                status = "MONITORING"
                status_color = (0, 180, 0)

            cv2.rectangle(
                output,
                (0, 0),
                (frame_width, 45),
                (25, 25, 25),
                -1,
            )

            cv2.putText(
                output,
                f"AI INDUSTRIAL SAFETY | {status}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            # =================================================
            # 6F. DETECTION SUMMARY
            # =================================================

            lines = [
                f"Workers: {workers}",
                f"Hardhats: {hardhats}",
                f"No Hardhat: {no_hardhats}",
                f"Safety Vests: {safety_vests}",
                f"No Vest: {no_vests}",
                f"Masks: {masks}",
                f"No Mask: {no_masks}",
                f"Standing: {standing}",
                f"Sitting: {sitting}",
                f"Bending: {bending}",
                f"Assisting: {assisting}",
                f"Pre-Fall Risks: {pre_fall_risks}",
                f"Fall Alert: {'YES' if fall_alert else 'NO'}",
                f"Danger Zone: {danger_workers}",
            ]

            panel_x = 12
            panel_y = 55
            line_height = 22
            panel_width = 235
            panel_height = len(lines) * line_height + 18

            cv2.rectangle(
                output,
                (panel_x, panel_y),
                (
                    min(
                        frame_width - 1,
                        panel_x + panel_width,
                    ),
                    min(
                        frame_height - 1,
                        panel_y + panel_height,
                    ),
                ),
                (25, 25, 25),
                -1,
            )

            for i, line in enumerate(lines):

                y = panel_y + 25 + i * line_height

                if y >= frame_height - 5:
                    break

                cv2.putText(
                    output,
                    line,
                    (panel_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # =================================================
            # 6G. FPS AND FRAME NUMBER
            # =================================================

            current_time = time.time()

            fps = 1.0 / max(
                current_time - previous_time,
                0.001,
            )

            previous_time = current_time

            cv2.putText(
                output,
                f"FPS: {fps:.1f} | Frame: {frame_number}",
                (max(10, frame_width - 250), frame_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # =================================================
            # 6H. DISPLAY AND SAVE FRAME
            # =================================================

            cv2.imshow(
                "AI Industrial Worker Safety - Press Q to stop",
                output,
            )

            out.write(output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Processing stopped by user.")
                break

    finally:

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # -----------------------------------------------------
    # 7. FINAL REPORT
    # -----------------------------------------------------

    print("\nVideo processing finished.")
    print("Frames processed:", frame_number)
    print(
        "Frames with confirmed fall alerts:",
        total_fall_alert_frames,
    )
    print(
        "Frames with PPE alerts:",
        total_ppe_alert_frames,
    )
    print(
        "Frames with pre-fall detections:",
        total_pre_fall_frames,
    )
    print("Output video saved to:", output_path)


if __name__ == "__main__":
    main()
