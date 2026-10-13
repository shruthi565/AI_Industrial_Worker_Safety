from ultralytics import YOLO
import cv2
import math


# YOLO Pose model
model = YOLO("yolov8n-pose.pt")


VIDEO_PATH = r"C:\Users\shrut\Downloads\archive\source_files\source_files\hardhat.mp4"


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
        verbose=False
    )


    output = results[0].plot()


    fall_count = 0


    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy


        for person in keypoints:

            # Need enough keypoints
            if len(person) < 17:
                continue


            # COCO keypoints
            nose = person[0]

            left_shoulder = person[5]
            right_shoulder = person[6]

            left_hip = person[11]
            right_hip = person[12]

            left_knee = person[13]
            right_knee = person[14]

            left_ankle = person[15]
            right_ankle = person[16]


            # -----------------------------------------
            # SHOULDER CENTER
            # -----------------------------------------

            shoulder_x = (
                left_shoulder[0] +
                right_shoulder[0]
            ) / 2

            shoulder_y = (
                left_shoulder[1] +
                right_shoulder[1]
            ) / 2


            # -----------------------------------------
            # HIP CENTER
            # -----------------------------------------

            hip_x = (
                left_hip[0] +
                right_hip[0]
            ) / 2

            hip_y = (
                left_hip[1] +
                right_hip[1]
            ) / 2


            # -----------------------------------------
            # BODY ANGLE
            # -----------------------------------------

            dx = hip_x - shoulder_x
            dy = hip_y - shoulder_y


            angle = abs(
                math.degrees(
                    math.atan2(dy, dx)
                )
            )


            # Convert angle to 0-90 range

            if angle > 90:
                angle = 180 - angle


            # -----------------------------------------
            # BODY LENGTH
            # -----------------------------------------

            body_height = abs(
                hip_y - shoulder_y
            )


            body_width = abs(
                shoulder_x - hip_x
            )


            # -----------------------------------------
            # FALL CONDITION
            # -----------------------------------------
            #
            # If the upper body becomes nearly horizontal
            # and body width is larger than body height,
            # consider it a possible fall.
            #

            if (
                angle < 35
                and
                body_width > body_height * 1.2
            ):

                fall_count += 1


                cv2.putText(
                    output,
                    "POSSIBLE FALL",
                    (
                        int(shoulder_x),
                        int(shoulder_y) - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    3
                )


    # -----------------------------------------
    # STATUS
    # -----------------------------------------

    if fall_count > 0:

        status = "FALL ALERT"

        color = (0, 0, 255)

    else:

        status = "NO FALL DETECTED"

        color = (0, 200, 0)


    # -----------------------------------------
    # PANEL
    # -----------------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (380, 80),
        (30, 30, 30),
        -1
    )


    cv2.putText(
        output,
        "FALL MONITOR",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        output,
        status,
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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