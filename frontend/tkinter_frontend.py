import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import time


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    r"C:\Users\shrut\Desktop\AI_Industrial_Worker_Safety"
    r"\models\ppe_model.pt"
)


# ============================================c================
# ADMIN LOGIN
# ============================================================

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"


# ============================================================
# COLORS
# ============================================================

BG = "#101820"
HEADER = "#17212B"
PANEL = "#1B2733"

WHITE = "#FFFFFF"
GREY = "#AAB7C4"

GREEN = "#20C997"
RED = "#FF4D4D"
ORANGE = "#FFB020"
BLUE = "#3498DB"
PURPLE = "#8E44AD"
DARK_GREY = "#444444"


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = YOLO(MODEL_PATH)

except Exception as e:

    print("Model loading error:", e)

    messagebox.showerror(
        "Model Error",
        f"Could not load the YOLO model.\n\n{e}"
    )

    raise


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title(
    "AI Industrial Worker Safety System"
)

root.geometry(
    "1250x780"
)

root.configure(
    bg=BG
)

root.resizable(
    True,
    True
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

cap = None

running = False

selected_mode = "Safety / PPE Detection"

last_time = time.time()

fps = 0


# ============================================================
# LOGIN PAGE
# ============================================================

def show_login_page():

    global login_frame

    # --------------------------------------------------------
    # Clear window
    # --------------------------------------------------------

    for widget in root.winfo_children():

        widget.destroy()


    # --------------------------------------------------------
    # Main login background
    # --------------------------------------------------------

    login_frame = tk.Frame(
        root,
        bg=BG
    )

    login_frame.pack(
        fill="both",
        expand=True
    )


    # --------------------------------------------------------
    # Login card
    # --------------------------------------------------------

    card = tk.Frame(
        login_frame,
        bg=PANEL,
        width=450,
        height=480
    )

    card.place(
        relx=0.5,
        rely=0.5,
        anchor="center"
    )

    card.pack_propagate(False)


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    tk.Label(
        card,
        text="ADMIN LOGIN",
        font=("Arial", 25, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        pady=(45, 8)
    )


    tk.Label(
        card,
        text="AI Industrial Worker Safety System",
        font=("Arial", 11),
        fg=GREY,
        bg=PANEL
    ).pack(
        pady=(0, 30)
    )


    # --------------------------------------------------------
    # Username
    # --------------------------------------------------------

    tk.Label(
        card,
        text="USERNAME",
        font=("Arial", 10, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        anchor="w",
        padx=55
    )


    username_entry = tk.Entry(
        card,
        font=("Arial", 12),
        bg="#243442",
        fg=WHITE,
        insertbackground=WHITE,
        relief="flat"
    )

    username_entry.pack(
        fill="x",
        padx=55,
        ipady=10,
        pady=(5, 18)
    )


    # --------------------------------------------------------
    # Password
    # --------------------------------------------------------

    tk.Label(
        card,
        text="PASSWORD",
        font=("Arial", 10, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        anchor="w",
        padx=55
    )


    password_frame = tk.Frame(
        card,
        bg=PANEL
    )

    password_frame.pack(
        fill="x",
        padx=55,
        pady=(5, 8)
    )


    password_entry = tk.Entry(
        password_frame,
        font=("Arial", 12),
        bg="#243442",
        fg=WHITE,
        insertbackground=WHITE,
        relief="flat",
        show="*"
    )

    password_entry.pack(
        side="left",
        fill="x",
        expand=True,
        ipady=10
    )


    # --------------------------------------------------------
    # Show password
    # --------------------------------------------------------

    show_password_var = tk.BooleanVar(
        value=False
    )


    def toggle_password():

        if show_password_var.get():

            password_entry.config(
                show=""
            )

        else:

            password_entry.config(
                show="*"
            )


    tk.Checkbutton(
        password_frame,
        text="Show",
        variable=show_password_var,
        command=toggle_password,
        bg=PANEL,
        fg=GREY,
        selectcolor=PANEL,
        activebackground=PANEL,
        activeforeground=WHITE
    ).pack(
        side="right",
        padx=(8, 0)
    )


    # --------------------------------------------------------
    # Login status
    # --------------------------------------------------------

    login_status = tk.Label(
        card,
        text="",
        font=("Arial", 10),
        fg=RED,
        bg=PANEL
    )

    login_status.pack(
        pady=8
    )


    # --------------------------------------------------------
    # LOGIN FUNCTION
    # --------------------------------------------------------

    def login():

        username = username_entry.get().strip()

        password = password_entry.get()


        if (
            username == ADMIN_USERNAME
            and
            password == ADMIN_PASSWORD
        ):

            login_status.config(
                text="Login successful",
                fg=GREEN
            )

            root.after(
                400,
                show_dashboard
            )

        else:

            login_status.config(
                text="Invalid username or password",
                fg=RED
            )

            password_entry.delete(
                0,
                tk.END
            )


    # --------------------------------------------------------
    # Login button
    # --------------------------------------------------------

    tk.Button(
        card,
        text="LOGIN",
        command=login,
        font=("Arial", 12, "bold"),
        bg=BLUE,
        fg=WHITE,
        activebackground=BLUE,
        activeforeground=WHITE,
        relief="flat",
        padx=50,
        pady=10,
        cursor="hand2"
    ).pack(
        pady=10
    )


    # --------------------------------------------------------
    # Demo credentials
    # --------------------------------------------------------

    tk.Label(
        card,
        text="Demo Login: admin / admin123",
        font=("Arial", 9),
        fg=GREY,
        bg=PANEL
    ).pack(
        pady=(15, 0)
    )


    # --------------------------------------------------------
    # Enter key
    # --------------------------------------------------------

    password_entry.bind(
        "<Return>",
        lambda event: login()
    )

    username_entry.bind(
        "<Return>",
        lambda event: password_entry.focus()
    )


    username_entry.focus()


# ============================================================
# DASHBOARD
# ============================================================

def show_dashboard():

    global cap
    global running

    # --------------------------------------------------------
    # Stop previous video
    # --------------------------------------------------------

    running = False

    if cap is not None:

        cap.release()

        cap = None


    # --------------------------------------------------------
    # Clear login page
    # --------------------------------------------------------

    for widget in root.winfo_children():

        widget.destroy()


    # ========================================================
    # HEADER
    # ========================================================

    header = tk.Frame(
        root,
        bg=HEADER,
        height=80
    )

    header.pack(
        fill="x"
    )


    title_frame = tk.Frame(
        header,
        bg=HEADER
    )

    title_frame.pack(
        side="left",
        padx=25
    )


    tk.Label(
        title_frame,
        text="AI INDUSTRIAL WORKER SAFETY",
        font=("Arial", 21, "bold"),
        fg=WHITE,
        bg=HEADER
    ).pack(
        anchor="w",
        pady=(10, 0)
    )


    tk.Label(
        title_frame,
        text="Real-Time Safety Monitoring and Anomaly Detection",
        font=("Arial", 10),
        fg=GREY,
        bg=HEADER
    ).pack(
        anchor="w"
    )


    # --------------------------------------------------------
    # Logout
    # --------------------------------------------------------

    tk.Button(
        header,
        text="LOGOUT",
        command=logout,
        font=("Arial", 10, "bold"),
        bg=RED,
        fg=WHITE,
        activebackground=RED,
        activeforeground=WHITE,
        relief="flat",
        padx=18,
        pady=7,
        cursor="hand2"
    ).pack(
        side="right",
        padx=25
    )


    # ========================================================
    # DETECTION CONTROL
    # ========================================================

    control_frame = tk.Frame(
        root,
        bg=PANEL
    )

    control_frame.pack(
        fill="x",
        padx=15,
        pady=7
    )


    tk.Label(
        control_frame,
        text="DETECTION TYPE",
        font=("Arial", 10, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        side="left",
        padx=(15, 8),
        pady=10
    )


    mode_var = tk.StringVar()

    mode_var.set(
        "Safety / PPE Detection"
    )


    mode_menu = tk.OptionMenu(
        control_frame,
        mode_var,
        "Safety / PPE Detection",
        "Fall Detection",
        "Danger Zone Detection"
    )


    mode_menu.config(
        font=("Arial", 10),
        bg=BLUE,
        fg=WHITE,
        activebackground=BLUE,
        activeforeground=WHITE,
        width=23,
        relief="flat"
    )


    mode_menu["menu"].config(
        font=("Arial", 10)
    )


    mode_menu.pack(
        side="left",
        padx=8
    )


    # ========================================================
    # INPUT TYPE
    # ========================================================

    tk.Label(
        control_frame,
        text="INPUT",
        font=("Arial", 10, "bold"),
        fg=WHITE,
        bg=PANEL
    ).pack(
        side="left",
        padx=(25, 8)
    )


    input_var = tk.StringVar()

    input_var.set(
        "Video"
    )


    input_menu = tk.OptionMenu(
        control_frame,
        input_var,
        "Video",
        "Image"
    )


    input_menu.config(
        font=("Arial", 10),
        bg=PURPLE,
        fg=WHITE,
        activebackground=PURPLE,
        activeforeground=WHITE,
        width=10,
        relief="flat"
    )


    input_menu["menu"].config(
        font=("Arial", 10)
    )


    input_menu.pack(
        side="left",
        padx=8
    )


    # ========================================================
    # STATUS
    # ========================================================

    status_frame = tk.Frame(
        root,
        bg=PANEL
    )

    status_frame.pack(
        fill="x",
        padx=15,
        pady=4
    )


    status_label = tk.Label(
        status_frame,
        text="SYSTEM READY",
        font=("Arial", 14, "bold"),
        fg=GREEN,
        bg=PANEL
    )

    status_label.pack(
        pady=6
    )


    # ========================================================
    # VIDEO DISPLAY
    # ========================================================

    video_frame = tk.Frame(
        root,
        bg="black",
        width=850,
        height=330
    )

    video_frame.pack(
        padx=15,
        pady=4
    )

    video_frame.pack_propagate(
        False
    )


    video_label = tk.Label(
        video_frame,
        text="Upload an image or video",
        font=("Arial", 15),
        fg=GREY,
        bg="black"
    )

    video_label.pack(
        expand=True
    )


    # ========================================================
    # DASHBOARD
    # ========================================================

    dashboard = tk.Frame(
        root,
        bg=BG
    )

    dashboard.pack(
        fill="x",
        padx=15,
        pady=5
    )


    def create_card(
        parent,
        title,
        value,
        color
    ):

        frame = tk.Frame(
            parent,
            bg=PANEL,
            height=62
        )

        frame.pack(
            side="left",
            padx=2,
            expand=True,
            fill="both"
        )

        frame.pack_propagate(
            False
        )


        tk.Label(
            frame,
            text=title,
            font=("Arial", 8),
            fg=GREY,
            bg=PANEL
        ).pack(
            pady=(4, 0)
        )


        value_label = tk.Label(
            frame,
            text=value,
            font=("Arial", 17, "bold"),
            fg=color,
            bg=PANEL
        )

        value_label.pack()


        return value_label


    worker_value = create_card(
        dashboard,
        "WORKERS",
        "0",
        WHITE
    )


    hardhat_value = create_card(
        dashboard,
        "HARDHATS",
        "0",
        GREEN
    )


    no_hardhat_value = create_card(
        dashboard,
        "NO HARDHAT",
        "0",
        RED
    )


    vest_value = create_card(
        dashboard,
        "SAFETY VEST",
        "0",
        GREEN
    )


    no_vest_value = create_card(
        dashboard,
        "NO VEST",
        "0",
        RED
    )


    danger_value = create_card(
        dashboard,
        "DANGER ZONE",
        "0",
        ORANGE
    )


    fall_value = create_card(
        dashboard,
        "FALL",
        "0",
        RED
    )


    # ========================================================
    # RESET DASHBOARD
    # ========================================================

    def reset_dashboard():

        worker_value.config(text="0")

        hardhat_value.config(text="0")

        no_hardhat_value.config(text="0")

        vest_value.config(text="0")

        no_vest_value.config(text="0")

        danger_value.config(text="0")

        fall_value.config(text="0")


    # ========================================================
    # DISPLAY OUTPUT
    # ========================================================

    def show_output(output):

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )


        image = Image.fromarray(
            output_rgb
        )


        image.thumbnail(
            (850, 320)
        )


        photo = ImageTk.PhotoImage(
            image=image
        )


        video_label.config(
            image=photo,
            text=""
        )


        video_label.image = photo


    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def detect_frame(frame):

        selected_mode = mode_var.get()


        results = model(
            frame,
            conf=0.50,
            iou=0.50,
            verbose=False
        )


        result = results[0]


        output = result.plot()


        height, width = output.shape[:2]


        workers = 0

        hardhats = 0

        no_hardhats = 0

        vests = 0

        no_vests = 0

        danger_workers = 0

        fall_count = 0


        # ====================================================
        # DANGER ZONE
        # ====================================================

        zone_x1 = int(width * 0.10)

        zone_y1 = int(height * 0.15)

        zone_x2 = int(width * 0.55)

        zone_y2 = int(height * 0.75)


        if selected_mode == "Danger Zone Detection":

            overlay = output.copy()


            cv2.rectangle(
                overlay,
                (zone_x1, zone_y1),
                (zone_x2, zone_y2),
                (0, 165, 255),
                -1
            )


            output = cv2.addWeighted(
                overlay,
                0.10,
                output,
                0.90,
                0
            )


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


        # ====================================================
        # DETECTION LOOP
        # ====================================================

        for box, cls in zip(
            result.boxes.xyxy,
            result.boxes.cls
        ):

            class_id = int(cls)

            class_name = model.names[class_id]


            bx1 = float(box[0])

            by1 = float(box[1])

            bx2 = float(box[2])

            by2 = float(box[3])


            box_width = bx2 - bx1

            box_height = by2 - by1


            center_x = int(
                (bx1 + bx2) / 2
            )

            center_y = int(
                (by1 + by2) / 2
            )


            # =================================================
            # PERSON
            # =================================================

            if class_name == "Person":

                workers += 1


                # ---------------------------------------------
                # DANGER ZONE
                # ---------------------------------------------

                if selected_mode == "Danger Zone Detection":

                    inside_zone = (
                        zone_x1 < center_x < zone_x2
                        and
                        zone_y1 < center_y < zone_y2
                    )


                    if inside_zone:

                        danger_workers += 1


                        cv2.circle(
                            output,
                            (center_x, center_y),
                            8,
                            (0, 0, 255),
                            -1
                        )


                        cv2.putText(
                            output,
                            "DANGER",
                            (
                                int(bx1),
                                max(
                                    30,
                                    int(by1) - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3
                        )


                # ---------------------------------------------
                # FALL DETECTION
                # ---------------------------------------------

                if selected_mode == "Fall Detection":

                    if box_width > box_height * 0.90:

                        fall_count += 1


                        cv2.rectangle(
                            output,
                            (
                                int(bx1),
                                int(by1)
                            ),
                            (
                                int(bx2),
                                int(by2)
                            ),
                            (0, 0, 255),
                            4
                        )


                        cv2.putText(
                            output,
                            "POSSIBLE FALL",
                            (
                                int(bx1),
                                max(
                                    30,
                                    int(by1) - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            3
                        )


            # =================================================
            # PPE
            # =================================================

            elif class_name == "Hardhat":

                hardhats += 1


            elif class_name == "NO-Hardhat":

                no_hardhats += 1


            elif class_name == "Safety Vest":

                vests += 1


            elif class_name == "NO-Safety Vest":

                no_vests += 1


        # ====================================================
        # STATUS
        # ====================================================

        if selected_mode == "Safety / PPE Detection":

            if (
                no_hardhats > 0
                or
                no_vests > 0
            ):

                status = "PPE ALERT"

                status_color = RED

            else:

                status = "PPE SAFE"

                status_color = GREEN


        elif selected_mode == "Fall Detection":

            if fall_count > 0:

                status = "POSSIBLE FALL DETECTED"

                status_color = RED

            else:

                status = "NO FALL DETECTED"

                status_color = GREEN


        else:

            if danger_workers > 0:

                status = "DANGER ZONE ALERT"

                status_color = RED

            else:

                status = "AREA CLEAR"

                status_color = GREEN


        # ====================================================
        # UPDATE STATUS
        # ====================================================

        status_label.config(
            text=status,
            fg=status_color
        )


        # ====================================================
        # UPDATE DASHBOARD
        # ====================================================

        worker_value.config(
            text=str(workers)
        )


        hardhat_value.config(
            text=str(hardhats)
        )


        no_hardhat_value.config(
            text=str(no_hardhats)
        )


        vest_value.config(
            text=str(vests)
        )


        no_vest_value.config(
            text=str(no_vests)
        )


        danger_value.config(
            text=str(danger_workers)
        )


        fall_value.config(
            text=str(fall_count)
        )


        return output


    # ========================================================
    # UPLOAD IMAGE
    # ========================================================

    def upload_image():

        global running
        global cap


        running = False


        if cap is not None:

            cap.release()

            cap = None


        file_path = filedialog.askopenfilename(
            title="Select Industrial Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                (
                    "All Files",
                    "*.*"
                )
            ]
        )


        if not file_path:

            return


        frame = cv2.imread(
            file_path
        )


        if frame is None:

            messagebox.showerror(
                "Error",
                "Could not open the selected image."
            )

            return


        reset_dashboard()


        status_label.config(
            text="PROCESSING IMAGE...",
            fg=GREEN
        )


        try:

            output = detect_frame(
                frame
            )


            show_output(
                output
            )

        except Exception as e:

            messagebox.showerror(
                "Detection Error",
                str(e)
            )


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    def process_video():

        global cap
        global running
        global fps
        global last_time


        if (
            not running
            or
            cap is None
        ):

            return


        ret, frame = cap.read()


        if not ret:

            cap.release()

            cap = None

            running = False


            status_label.config(
                text="VIDEO COMPLETED",
                fg=GREEN
            )

            return


        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time
            -
            last_time
        )


        if elapsed > 0:

            fps = 1 / elapsed


        last_time = current_time


        try:

            output = detect_frame(
                frame
            )

        except Exception as e:

            running = False

            if cap is not None:

                cap.release()

                cap = None


            messagebox.showerror(
                "Detection Error",
                str(e)
            )

            return


        # ----------------------------------------------------
        # FPS text
        # ----------------------------------------------------

        height = output.shape[0]


        cv2.putText(
            output,
            f"FPS: {fps:.1f}",
            (
                20,
                height - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        show_output(
            output
        )


        if running:

            root.after(
                30,
                process_video
            )


    # ========================================================
    # UPLOAD VIDEO
    # ========================================================

    def upload_video():

        global cap
        global running
        global last_time


        # Stop previous video

        running = False


        if cap is not None:

            cap.release()

            cap = None


        file_path = filedialog.askopenfilename(
            title="Select Industrial Video",
            filetypes=[
                (
                    "Video Files",
                    "*.mp4 *.avi *.mov *.mkv"
                ),
                (
                    "All Files",
                    "*.*"
                )
            ]
        )


        if not file_path:

            return


        cap = cv2.VideoCapture(
            file_path
        )


        if not cap.isOpened():

            messagebox.showerror(
                "Error",
                "Could not open the selected video."
            )

            cap = None

            return


        reset_dashboard()


        running = True

        last_time = time.time()


        status_label.config(
            text=(
                mode_var.get().upper()
                +
                " ACTIVE"
            ),
            fg=GREEN
        )


        process_video()


    # ========================================================
    # STOP
    # ========================================================

    def stop_video():

        global running
        global cap


        running = False


        if cap is not None:

            cap.release()

            cap = None


        status_label.config(
            text="SYSTEM STOPPED",
            fg=RED
        )


        video_label.config(
            image="",
            text="Video stopped"
        )


        video_label.image = None


    # ========================================================
    # BUTTONS
    # ========================================================

    button_frame = tk.Frame(
        root,
        bg=BG
    )

    button_frame.pack(
        pady=6
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    tk.Button(
        button_frame,
        text="UPLOAD IMAGE",
        command=upload_image,
        font=("Arial", 10, "bold"),
        bg=PURPLE,
        fg=WHITE,
        activebackground=PURPLE,
        activeforeground=WHITE,
        relief="flat",
        padx=20,
        pady=8,
        cursor="hand2"
    ).pack(
        side="left",
        padx=6
    )


    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    tk.Button(
        button_frame,
        text="UPLOAD VIDEO",
        command=upload_video,
        font=("Arial", 10, "bold"),
        bg=BLUE,
        fg=WHITE,
        activebackground=BLUE,
        activeforeground=WHITE,
        relief="flat",
        padx=20,
        pady=8,
        cursor="hand2"
    ).pack(
        side="left",
        padx=6
    )


    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    tk.Button(
        button_frame,
        text="STOP",
        command=stop_video,
        font=("Arial", 10, "bold"),
        bg=DARK_GREY,
        fg=WHITE,
        activebackground=DARK_GREY,
        activeforeground=WHITE,
        relief="flat",
        padx=25,
        pady=8,
        cursor="hand2"
    ).pack(
        side="left",
        padx=6
    )


    # ========================================================
    # FOOTER
    # ========================================================

    tk.Label(
        root,
        text=(
            "AI-Based Real-Time Industrial Worker Safety "
            "and Anomaly Detection System"
        ),
        font=("Arial", 8),
        fg=GREY,
        bg=BG
    ).pack(
        pady=3
    )


# ============================================================
# LOGOUT
# ============================================================

def logout():

    global cap
    global running


    running = False


    if cap is not None:

        cap.release()

        cap = None


    show_login_page()


# ============================================================
# START APPLICATION
# ============================================================

show_login_page()


root.mainloop()