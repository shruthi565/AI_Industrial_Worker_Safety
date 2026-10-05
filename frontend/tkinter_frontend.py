
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path


# ---------------- CONFIGURATION ----------------


BASE_DIR = Path(
    r"C:\Users\shrut\OneDrive\Documents\AI_Industrial_Worker_Safety"
)

PPE_MODEL_PATH = BASE_DIR / "models" / "ppe_model.pt"
FALL_MODEL_PATH = BASE_DIR / "models" / "fall_model.pt"

PPE_CONFIDENCE = 0.25
FALL_CONFIDENCE = 0.25

# ------------------------------------------------


class IndustrialSafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Industrial Worker Safety Monitoring")
        self.root.geometry("1200x760")
        self.root.minsize(950, 650)
        self.root.configure(bg="#eef2f7")

        self.cap = None
        self.video_running = False
        self.current_image = None
        self.logged_in = False

        self.ppe_model = None
        self.fall_model = None

        self.mode_var = tk.StringVar(value="PPE Detection")
        self.status_var = tk.StringVar(value="Please log in")
        self.file_var = tk.StringVar(value="No file selected")

        self.stats = {
            "Workers": tk.IntVar(value=0),
            "Hardhats": tk.IntVar(value=0),
            "No Hardhat": tk.IntVar(value=0),
            "Safety Vests": tk.IntVar(value=0),
            "No Vest": tk.IntVar(value=0),
            "Falls": tk.IntVar(value=0),
            "Pre-Fall Risks": tk.IntVar(value=0),
            "Sitting": tk.IntVar(value=0),
            "Bending": tk.IntVar(value=0),
            "Assisting": tk.IntVar(value=0),
        }

        self.build_login()

    # ---------------- LOGIN ----------------

    def build_login(self):
        self.login_frame = tk.Frame(self.root, bg="#eef2f7")
        self.login_frame.pack(fill="both", expand=True)

        tk.Label(
            self.login_frame,
            text="AI INDUSTRIAL SAFETY",
            font=("Arial", 25, "bold"),
            fg="#16324f",
            bg="#eef2f7"
        ).pack(pady=(100, 10))

        tk.Label(
            self.login_frame,
            text="Worker Safety Monitoring System",
            font=("Arial", 13),
            bg="#eef2f7",
            fg="#475569"
        ).pack(pady=5)

        card = tk.Frame(
            self.login_frame,
            bg="white",
            padx=35,
            pady=30,
            highlightbackground="#d5dce5",
            highlightthickness=1
        )
        card.pack(pady=25)

        tk.Label(
            card, text="Username",
            bg="white", font=("Arial", 11)
        ).grid(row=0, column=0, sticky="w", pady=8)

        self.username_entry = ttk.Entry(card, width=28)
        self.username_entry.grid(row=0, column=1, pady=8, padx=10)

        tk.Label(
            card, text="Password",
            bg="white", font=("Arial", 11)
        ).grid(row=1, column=0, sticky="w", pady=8)

        self.password_entry = ttk.Entry(card, width=28, show="*")
        self.password_entry.grid(row=1, column=1, pady=8, padx=10)
        self.password_entry.bind("<Return>", lambda event: self.login())

        ttk.Button(
            card, text="Login", command=self.login
        ).grid(row=2, column=0, columnspan=2, pady=20, ipadx=30)

        tk.Label(
            self.login_frame,
            text="Demo login: admin / admin123",
            bg="#eef2f7",
            fg="#64748b",
            font=("Arial", 10)
        ).pack()

    def login(self):
        username = self.username_entry.get().strip()
        password = self.password_entry.get()

        if username == "admin" and password == "admin123":
            self.logged_in = True
            self.login_frame.destroy()

            try:
                self.load_models()
                self.build_dashboard()
            except Exception as error:
                messagebox.showerror(
                    "Model Loading Error",
                    f"Could not load a model.\n\n{error}"
                )
                self.root.destroy()
        else:
            messagebox.showerror(
                "Login Failed",
                "Incorrect username or password."
            )

    # ---------------- LOAD MODELS ----------------

    def load_models(self):
        if not PPE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"PPE model not found:\n{PPE_MODEL_PATH}"
            )

        if not FALL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Fall model not found:\n{FALL_MODEL_PATH}"
            )

        self.ppe_model = YOLO(str(PPE_MODEL_PATH))
        self.fall_model = YOLO(str(FALL_MODEL_PATH))

        print("PPE model classes:", self.ppe_model.names)
        print("Fall model classes:", self.fall_model.names)

    # ---------------- DASHBOARD ----------------

    def build_dashboard(self):
        header = tk.Frame(self.root, bg="#16324f", height=75)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="AI INDUSTRIAL WORKER SAFETY",
            font=("Arial", 20, "bold"),
            bg="#16324f",
            fg="white"
        ).pack(side="left", padx=25, pady=18)

        ttk.Button(
            header, text="Logout", command=self.logout
        ).pack(side="right", padx=20, pady=18)

        controls = tk.Frame(self.root, bg="#eef2f7")
        controls.pack(fill="x", padx=20, pady=12)

        tk.Label(
            controls,
            text="Detection Mode:",
            font=("Arial", 11, "bold"),
            bg="#eef2f7"
        ).pack(side="left", padx=(0, 8))

        self.mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["PPE Detection", "Fall Detection"],
            state="readonly",
            width=18
        )
        self.mode_box.pack(side="left", padx=5)
        self.mode_box.bind(
            "<<ComboboxSelected>>",
            lambda event: self.reset_stats()
        )

        ttk.Button(
            controls, text="Open Image",
            command=self.open_image
        ).pack(side="left", padx=6)

        ttk.Button(
            controls, text="Open Video",
            command=self.open_video
        ).pack(side="left", padx=6)

        ttk.Button(
            controls, text="Stop Video",
            command=self.stop_video
        ).pack(side="left", padx=6)

        tk.Label(
            self.root,
            textvariable=self.file_var,
            bg="#eef2f7",
            fg="#475569",
            anchor="w"
        ).pack(fill="x", padx=22)

        content = tk.Frame(self.root, bg="#eef2f7")
        content.pack(fill="both", expand=True, padx=20, pady=10)

        video_panel = tk.Frame(
            content, bg="#111827", bd=0
        )
        video_panel.pack(
            side="left", fill="both", expand=True, padx=(0, 12)
        )

        self.video_label = tk.Label(
            video_panel,
            text="Open an image or video to begin",
            bg="#111827",
            fg="white",
            font=("Arial", 15),
            compound="center"
        )
        self.video_label.pack(
            fill="both", expand=True, padx=5, pady=5
        )

        side_panel = tk.Frame(
            content,
            bg="white",
            width=250,
            padx=15,
            pady=15
        )
        side_panel.pack(side="right", fill="y")
        side_panel.pack_propagate(False)

        tk.Label(
            side_panel,
            text="DETECTION SUMMARY",
            font=("Arial", 13, "bold"),
            bg="white",
            fg="#16324f"
        ).pack(anchor="w", pady=(0, 12))

        self.stat_labels = {}

        for name, variable in self.stats.items():
            row = tk.Frame(side_panel, bg="white")
            row.pack(fill="x", pady=5)

            tk.Label(
                row, text=name,
                bg="white", fg="#334155",
                font=("Arial", 10)
            ).pack(side="left")

            value_label = tk.Label(
                row, textvariable=variable,
                bg="white", fg="#16324f",
                font=("Arial", 11, "bold")
            )
            value_label.pack(side="right")
            self.stat_labels[name] = row

        ttk.Separator(side_panel).pack(fill="x", pady=12)

        tk.Label(
            side_panel,
            text="SYSTEM STATUS",
            font=("Arial", 11, "bold"),
            bg="white",
            fg="#16324f"
        ).pack(anchor="w")

        self.status_label = tk.Label(
            side_panel,
            textvariable=self.status_var,
            bg="#e2e8f0",
            fg="#334155",
            font=("Arial", 10, "bold"),
            wraplength=210,
            justify="left",
            padx=10,
            pady=12
        )
        self.status_label.pack(fill="x", pady=10)

        tk.Label(
            side_panel,
            text="Alerts are AI predictions. Verify detections before taking action.",
            bg="white",
            fg="#64748b",
            font=("Arial", 9),
            wraplength=215,
            justify="left"
        ).pack(anchor="w", pady=(10, 0))

        self.update_mode_display()

    # ---------------- COUNTERS ----------------

    def reset_stats(self):
        for variable in self.stats.values():
            variable.set(0)

        self.status_var.set("Ready for detection")
        self.update_mode_display()

    def update_mode_display(self):
        mode = self.mode_var.get()

        if mode == "Fall Detection":
            visible = {
                "Workers", "Falls", "Pre-Fall Risks",
                "Sitting", "Bending", "Assisting"
            }
        else:
            visible = {
                "Workers", "Hardhats", "No Hardhat",
                "Safety Vests", "No Vest"
            }

        for name, row in self.stat_labels.items():
            if name in visible:
                row.pack(fill="x", pady=5)
            else:
                row.pack_forget()

    # ---------------- DETECTION ----------------

    def detect_frame(self, frame):
        counts = {name: 0 for name in self.stats}
        mode = self.mode_var.get()

        if mode == "Fall Detection":
            model = self.fall_model
            confidence = FALL_CONFIDENCE
        else:
            model = self.ppe_model
            confidence = PPE_CONFIDENCE

        results = model.predict(
            source=frame,
            conf=confidence,
            iou=0.50,
            verbose=False
        )

        annotated = frame.copy()
        names = model.names

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                class_name = str(names[class_id]).strip().lower()

                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                if mode == "Fall Detection":
                    color = (0, 200, 0)
                    label = class_name.replace("_", " ").upper()

                    if class_name == "fallen":
                        counts["Falls"] += 1
                        color = (0, 0, 255)
                        label = "FALL DETECTED"

                    elif class_name in ("pre_falling", "pre-falling"):
                        counts["Pre-Fall Risks"] += 1
                        color = (0, 165, 255)
                        label = "POSSIBLE FALL RISK"

                    elif class_name == "standing":
                        counts["Workers"] += 1

                    elif class_name == "assisting":
                        counts["Assisting"] += 1
                        counts["Workers"] += 1

                    elif class_name == "sitting":
                        counts["Sitting"] += 1
                        color = (255, 180, 0)

                    elif class_name == "bending":
                        counts["Bending"] += 1
                        color = (255, 180, 0)

                    else:
                        # Unknown class: still show its prediction.
                        color = (180, 180, 180)

                    # Draw bounding box and label.
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2),
                        color, 2
                    )

                    text = f"{label} {score:.0%}"
                    text_y = max(25, y1 - 8)

                    cv2.putText(
                        annotated, text,
                        (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, color, 2,
                        cv2.LINE_AA
                    )

                else:
                    color = (0, 200, 0)
                    label = class_name.upper()

                    if class_name == "person":
                        counts["Workers"] += 1

                    elif class_name == "hardhat":
                        counts["Hardhats"] += 1
                        color = (0, 200, 0)

                    elif class_name in ("no-hardhat", "no_hardhat"):
                        counts["No Hardhat"] += 1
                        color = (0, 0, 255)
                        label = "NO HARDHAT"

                    elif class_name in ("safety vest", "safety_vest"):
                        counts["Safety Vests"] += 1
                        color = (0, 200, 0)

                    elif class_name in (
                        "no-safety vest", "no_safety_vest"
                    ):
                        counts["No Vest"] += 1
                        color = (0, 0, 255)
                        label = "NO SAFETY VEST"

                    elif class_name == "mask":
                        label = "MASK"

                    elif class_name in ("no-mask", "no_mask"):
                        label = "NO MASK"
                        color = (0, 0, 255)

                    elif class_name in (
                        "safety cone", "safety_cone",
                        "machinery", "vehicle"
                    ):
                        # Keep these predictions visible, but do not
                        # count them as worker/PPE statistics.
                        label = class_name.upper()

                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2),
                        color, 2
                    )

                    cv2.putText(
                        annotated,
                        f"{label} {score:.0%}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 2,
                        cv2.LINE_AA
                    )

        # Display an alert banner.
        if mode == "Fall Detection":
            if counts["Falls"] > 0:
                status = "ALERT: POSSIBLE FALL DETECTED"
                banner_color = (0, 0, 220)
            elif counts["Pre-Fall Risks"] > 0:
                status = "WARNING: POSSIBLE FALL RISK"
                banner_color = (0, 140, 255)
            else:
                status = "No fall detected in this frame"
                banner_color = (0, 125, 0)
        else:
            if counts["No Hardhat"] > 0 or counts["No Vest"] > 0:
                status = "WARNING: PPE ISSUE DETECTED"
                banner_color = (0, 0, 220)
            else:
                status = "PPE monitoring active"
                banner_color = (0, 125, 0)

        cv2.rectangle(
            annotated, (0, 0), (annotated.shape[1], 38),
            banner_color, -1
        )

        cv2.putText(
            annotated, status,
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2,
            cv2.LINE_AA
        )

        return annotated, counts, status

    def update_counters(self, counts, status):
        for name, variable in self.stats.items():
            variable.set(counts.get(name, 0))

        self.status_var.set(status)

        if "ALERT" in status or "WARNING" in status:
            self.status_label.configure(
                bg="#fee2e2", fg="#991b1b"
            )
        else:
            self.status_label.configure(
                bg="#dcfce7", fg="#166534"
            )

    # ---------------- IMAGE ----------------

    def open_image(self):
        self.stop_video()

        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        frame = cv2.imread(path)

        if frame is None:
            messagebox.showerror(
                "Image Error", "Could not read the selected image."
            )
            return

        try:
            annotated, counts, status = self.detect_frame(frame)
            self.file_var.set(f"Image: {Path(path).name}")
            self.update_counters(counts, status)
            self.show_frame(annotated)
        except Exception as error:
            messagebox.showerror(
                "Detection Error", str(error)
            )

    # ---------------- VIDEO ----------------

    def open_video(self):
        self.stop_video()

        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            cap.release()
            messagebox.showerror(
                "Video Error",
                "Could not open this video. Try another video file."
            )
            return

        self.cap = cap
        self.video_running = True
        self.file_var.set(f"Video: {Path(path).name}")
        self.process_video_frame()

    def process_video_frame(self):
        if not self.video_running or self.cap is None:
            return

        success, frame = self.cap.read()

        if not success:
            self.stop_video()
            self.status_var.set("Video finished")
            return

        try:
            annotated, counts, status = self.detect_frame(frame)
            self.update_counters(counts, status)
            self.show_frame(annotated)
        except Exception as error:
            self.stop_video()
            messagebox.showerror(
                "Video Detection Error", str(error)
            )
            return

        # Schedule the next frame without blocking the interface.
        self.root.after(20, self.process_video_frame)

    def stop_video(self):
        self.video_running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------------- DISPLAY ----------------

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        panel_width = max(self.video_label.winfo_width(), 640)
        panel_height = max(self.video_label.winfo_height(), 420)

        height, width = rgb.shape[:2]

        scale = min(
            panel_width / width,
            panel_height / height
        )

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(
            rgb, (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        image = Image.fromarray(resized)
        self.current_image = ImageTk.PhotoImage(image)

        self.video_label.configure(
            image=self.current_image,
            text=""
        )

    # ---------------- LOGOUT / CLOSE ----------------

    def logout(self):
        self.stop_video()
        self.root.destroy()

    def on_close(self):
        self.stop_video()
        self.root.destroy()


# ---------------- RUN APPLICATION ----------------

if __name__ == "__main__":
    root = tk.Tk()
    app = IndustrialSafetyApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
