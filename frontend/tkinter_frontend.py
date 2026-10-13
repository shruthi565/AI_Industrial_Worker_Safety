
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None


# =========================================================
# PATHS AND SETTINGS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PPE_MODEL_PATH = BASE_DIR / "models" / "ppe_model.pt"
FALL_MODEL_PATH = BASE_DIR / "models" / "fall_model.pt"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

APP_BG = "#0B1220"
SIDEBAR = "#101A2D"
PANEL = "#162237"
PANEL_ALT = "#1C2B43"
ACCENT = "#20C997"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
DANGER = "#F05252"
WARNING = "#F4B740"
BORDER = "#293951"

ALERT_COOLDOWN = 3.0


class WorkerSafetyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SafeVision | Industrial AI")
        self.root.geometry("1280x800")
        self.root.minsize(1000, 650)
        self.root.configure(bg=APP_BG)

        self.ppe_model = None
        self.fall_model = None
        self.models_loaded = False
        self.model_loading = False
        self.closing = False

        self.cap = None
        self.video_job = None
        self.paused = False
        self.current_image = None
        self.photo = None

        self.mode = tk.StringVar(value="PPE Detection")
        self.ppe_conf = tk.DoubleVar(value=0.25)
        self.fall_conf = tk.DoubleVar(value=0.30)
        self.sound_enabled = tk.BooleanVar(value=True)

        self.alert_history = []
        self.last_alert_time = {}

        # Persons = current-frame count.
        # Other detection metrics = cumulative box detections.
        self.stats = {
            "Frames Processed": 0,
            "Persons": 0,
            "Hardhats": 0,
            "No Hardhats": 0,
            "Masks": 0,
            "No Masks": 0,
            "Safety Vests": 0,
            "No Safety Vests": 0,
            "Falls": 0,
            "Pre-Falls": 0,
            "Standing": 0,
            "Sitting": 0,
            "Bending": 0,
            "Assisting": 0,
        }

        self.page = None
        self.nav_buttons = {}
        self.stat_labels = {}

        self.status_text = tk.StringVar(value="SYSTEM READY")
        self.alert_text = tk.StringVar(value="No active alerts")
        self.login_user = tk.StringVar()
        self.login_password = tk.StringVar()
        self.login_error = tk.StringVar()

        self.configure_styles()
        self.show_login()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # =====================================================
    # STYLES
    # =====================================================
    def configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background=PANEL,
            foreground=TEXT,
            fieldbackground=PANEL,
            rowheight=30,
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background=PANEL_ALT,
            foreground=TEXT,
            font=("Segoe UI", 10, "bold"),
            relief="flat"
        )
        style.map(
            "Treeview",
            background=[("selected", "#245C68")],
            foreground=[("selected", "white")]
        )
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=PANEL_ALT,
            background=ACCENT
        )

    def make_button(self, parent, text, command, primary=False):
        bg = ACCENT if primary else PANEL_ALT
        fg = "#07141D" if primary else TEXT

        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground="#36D9AC" if primary else BORDER,
            activeforeground=fg,
            relief="flat",
            bd=0,
            padx=15,
            pady=9,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2"
        )

    def panel(self, parent, padding=16):
        return tk.Frame(
            parent,
            bg=PANEL,
            padx=padding,
            pady=padding,
            highlightbackground=BORDER,
            highlightthickness=1
        )

    def heading(self, parent, text, size=15):
        return tk.Label(
            parent,
            text=text,
            bg=parent.cget("bg"),
            fg=TEXT,
            font=("Segoe UI", size, "bold")
        )

    # =====================================================
    # LOGIN PAGE
    # =====================================================
    def show_login(self):
        self.stop_video()

        if self.page is not None:
            self.page.destroy()

        self.page = tk.Frame(self.root, bg=APP_BG)
        self.page.pack(fill="both", expand=True)

        left = tk.Frame(self.page, bg=SIDEBAR, width=560)
        left.pack(side="left", fill="both", expand=True)
        left.pack_propagate(False)

        tk.Label(
            left,
            text="SAFETY  /  INTELLIGENCE",
            bg=SIDEBAR,
            fg=ACCENT,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w", padx=55, pady=(90, 20))

        tk.Label(
            left,
            text="Protect people.\nPrevent incidents.",
            bg=SIDEBAR,
            fg=TEXT,
            font=("Segoe UI", 34, "bold"),
            justify="left"
        ).pack(anchor="w", padx=55)

        tk.Label(
            left,
            text=(
                "AI-powered industrial worker safety monitoring\n"
                "with real-time PPE and fall detection."
            ),
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 12),
            justify="left"
        ).pack(anchor="w", padx=55, pady=(20, 35))

        for title, subtitle in [
            ("01   PPE COMPLIANCE", "Helmet • Mask • Safety vest"),
            ("02   FALL MONITORING", "Posture classification"),
            ("03   EVENT MANAGEMENT", "Alerts • Reports • Statistics"),
        ]:
            row = tk.Frame(left, bg=PANEL, padx=16, pady=12)
            row.pack(fill="x", padx=55, pady=6)

            tk.Label(
                row,
                text=title,
                bg=PANEL,
                fg=ACCENT,
                font=("Segoe UI", 10, "bold")
            ).pack(anchor="w")

            tk.Label(
                row,
                text=subtitle,
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 9)
            ).pack(anchor="w", pady=(4, 0))

        tk.Label(
            left,
            text="SAFEVISION  •  INDUSTRIAL AI",
            bg=SIDEBAR,
            fg="#64748B",
            font=("Segoe UI", 9, "bold")
        ).pack(side="bottom", anchor="w", padx=55, pady=30)

        right = tk.Frame(self.page, bg=APP_BG)
        right.pack(side="right", fill="both", expand=True)

        card = tk.Frame(right, bg=APP_BG, width=370)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            card,
            text="WELCOME BACK",
            bg=APP_BG,
            fg=ACCENT,
            font=("Segoe UI", 10, "bold")
        ).pack(anchor="w")

        tk.Label(
            card,
            text="Sign in to continue",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI", 25, "bold")
        ).pack(anchor="w", pady=(8, 8))

        tk.Label(
            card,
            text="Access your industrial safety dashboard.",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 30))

        tk.Label(
            card,
            text="USERNAME",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(0, 7))

        user_entry = tk.Entry(
            card,
            textvariable=self.login_user,
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 12),
            bd=0
        )
        user_entry.pack(fill="x", ipady=12, pady=(0, 20))

        tk.Label(
            card,
            text="PASSWORD",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 9, "bold")
        ).pack(anchor="w", pady=(0, 7))

        password_entry = tk.Entry(
            card,
            textvariable=self.login_password,
            show="*",
            bg=PANEL,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 12),
            bd=0
        )
        password_entry.pack(fill="x", ipady=12)

        tk.Label(
            card,
            textvariable=self.login_error,
            bg=APP_BG,
            fg=DANGER,
            font=("Segoe UI", 9),
            wraplength=350
        ).pack(anchor="w", pady=(10, 0))

        self.make_button(
            card, "SIGN IN  →", self.login, primary=True
        ).pack(fill="x", pady=(20, 10))

        tk.Label(
            card,
            text="Demo credentials: admin / admin123",
            bg=APP_BG,
            fg="#64748B",
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        password_entry.bind("<Return>", lambda _e: self.login())
        user_entry.bind(
            "<Return>", lambda _e: password_entry.focus_set()
        )
        user_entry.focus_set()

    def login(self):
        if (
            self.login_user.get().strip() == "admin"
            and self.login_password.get() == "admin123"
        ):
            self.login_error.set("")
            self.show_dashboard()
        else:
            self.login_error.set(
                "Invalid username or password. Please try again."
            )

    # =====================================================
    # DASHBOARD LAYOUT
    # =====================================================
    def show_dashboard(self):
        if self.page is not None:
            self.page.destroy()

        self.page = tk.Frame(self.root, bg=APP_BG)
        self.page.pack(fill="both", expand=True)

        sidebar = tk.Frame(self.page, bg=SIDEBAR, width=225)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar,
            text="◆  SAFEVISION",
            bg=SIDEBAR,
            fg=ACCENT,
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w", padx=22, pady=(25, 3))

        tk.Label(
            sidebar,
            text="INDUSTRIAL AI PLATFORM",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=23, pady=(0, 25))

        nav_items = [
            ("Dashboard", self.show_dashboard_home),
            ("Live Monitoring", self.show_monitoring),
            ("Alerts", self.show_alerts),
            ("Reports", self.show_reports),
            ("Statistics", self.show_statistics),
            ("Settings", self.show_settings),
            ("About", self.show_about),
        ]

        for label, action in nav_items:
            btn = tk.Button(
                sidebar,
                text="   " + label,
                anchor="w",
                command=action,
                bg=SIDEBAR,
                fg=TEXT,
                activebackground=PANEL_ALT,
                activeforeground=ACCENT,
                relief="flat",
                bd=0,
                padx=15,
                pady=12,
                font=("Segoe UI", 10),
                cursor="hand2"
            )
            btn.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[label] = btn

        tk.Frame(
            sidebar, bg=BORDER, height=1
        ).pack(fill="x", padx=18, pady=(18, 12))

        tk.Label(
            sidebar,
            text="SYSTEM STATUS",
            bg=SIDEBAR,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", padx=22)

        self.sidebar_status = tk.Label(
            sidebar,
            text="● INITIALIZING",
            bg=SIDEBAR,
            fg=WARNING,
            font=("Segoe UI", 9, "bold")
        )
        self.sidebar_status.pack(anchor="w", padx=22, pady=8)

        self.make_button(
            sidebar, "LOG OUT", self.logout
        ).pack(side="bottom", fill="x", padx=15, pady=18)

        workspace = tk.Frame(self.page, bg=APP_BG)
        workspace.pack(side="right", fill="both", expand=True)

        topbar = tk.Frame(workspace, bg=APP_BG, padx=25, pady=17)
        topbar.pack(fill="x")

        top_left = tk.Frame(topbar, bg=APP_BG)
        top_left.pack(side="left")

        self.page_title = tk.Label(
            top_left,
            text="Dashboard",
            bg=APP_BG,
            fg=TEXT,
            font=("Segoe UI", 21, "bold")
        )
        self.page_title.pack(anchor="w")

        tk.Label(
            top_left,
            text="REAL-TIME INDUSTRIAL SAFETY CONTROL",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", pady=(4, 0))

        tk.Label(
            topbar,
            text="● SYSTEM ONLINE",
            bg=APP_BG,
            fg=ACCENT,
            font=("Segoe UI", 10, "bold")
        ).pack(side="right", padx=5)

        tk.Frame(workspace, bg=BORDER, height=1).pack(fill="x")

        self.content = tk.Frame(
            workspace, bg=APP_BG, padx=25, pady=20
        )
        self.content.pack(fill="both", expand=True)

        self.show_dashboard_home()

        if not self.models_loaded:
            self.load_models_async()
        else:
            self.sidebar_status.configure(
                text="● MODELS LOADED", fg=ACCENT
            )

    def set_active_nav(self, label):
        for name, button in self.nav_buttons.items():
            button.configure(
                bg=PANEL_ALT if name == label else SIDEBAR,
                fg=ACCENT if name == label else TEXT
            )

    def clear_content(self, title):
        for widget in self.content.winfo_children():
            widget.destroy()
        self.page_title.configure(text=title)
        self.stat_labels = {}

    # =====================================================
    # MODEL LOADING
    # =====================================================
    def load_models_async(self):
        if self.models_loaded or self.model_loading:
            return

        self.model_loading = True
        self.sidebar_status.configure(
            text="● LOADING MODELS", fg=WARNING
        )

        def worker():
            try:
                if not PPE_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"PPE model missing:\n{PPE_MODEL_PATH}"
                    )

                if not FALL_MODEL_PATH.exists():
                    raise FileNotFoundError(
                        f"Fall model missing:\n{FALL_MODEL_PATH}"
                    )

                ppe = YOLO(str(PPE_MODEL_PATH))
                fall = YOLO(str(FALL_MODEL_PATH))

                def success():
                    self.ppe_model = ppe
                    self.fall_model = fall
                    self.models_loaded = True
                    self.model_loading = False

                    if hasattr(self, "sidebar_status"):
                        self.sidebar_status.configure(
                            text="● MODELS LOADED", fg=ACCENT
                        )

                if not self.closing:
                    self.root.after(0, success)

            except Exception as error:
                def failure(err=error):
                    self.model_loading = False

                    if not self.closing:
                        self.sidebar_status.configure(
                            text="● MODEL ERROR", fg=DANGER
                        )
                        messagebox.showerror(
                            "Model Loading Error", str(err)
                        )

                if not self.closing:
                    self.root.after(0, failure)

        threading.Thread(target=worker, daemon=True).start()

    # =====================================================
    # DASHBOARD HOME
    # =====================================================
    def show_dashboard_home(self):
        self.set_active_nav("Dashboard")
        self.clear_content("Dashboard")

        tk.Label(
            self.content,
            text="Safety overview and monitoring summary",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 15))

        cards = tk.Frame(self.content, bg=APP_BG)
        cards.pack(fill="x")

        metrics = [
            ("FRAMES PROCESSED", "Frames Processed", ACCENT),
            ("PERSONS IN CURRENT FRAME", "Persons", "#62A8FF"),
            ("PPE VIOLATIONS", "No Hardhats", DANGER),
            ("FALL DETECTIONS", "Falls", WARNING),
        ]

        for title, key, color in metrics:
            card = self.panel(cards, 15)
            card.pack(
                side="left", fill="both", expand=True, padx=(0, 10)
            )

            tk.Label(
                card,
                text=title,
                bg=PANEL,
                fg=MUTED,
                font=("Segoe UI", 8, "bold")
            ).pack(anchor="w")

            label = tk.Label(
                card,
                text=str(self.stats[key]),
                bg=PANEL,
                fg=color,
                font=("Segoe UI", 25, "bold")
            )
            label.pack(anchor="w", pady=(8, 2))
            self.stat_labels[key] = label

        lower = tk.Frame(self.content, bg=APP_BG)
        lower.pack(fill="both", expand=True, pady=(18, 0))

        left = self.panel(lower)
        left.pack(
            side="left", fill="both", expand=True, padx=(0, 10)
        )

        self.heading(left, "Live Monitoring Preview").pack(anchor="w")

        tk.Label(
            left,
            text="Choose a video or image to begin AI detection.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(8, 16))

        controls = tk.Frame(left, bg=PANEL)
        controls.pack(anchor="w")

        self.make_button(
            controls, "Open Video", self.open_video, True
        ).pack(side="left", padx=(0, 8))

        self.make_button(
            controls, "Open Image", self.open_image
        ).pack(side="left")

        self.heading(left, "Current Alert").pack(
            anchor="w", pady=(25, 8)
        )

        tk.Label(
            left,
            textvariable=self.alert_text,
            bg=PANEL,
            fg=WARNING,
            font=("Segoe UI", 10, "bold"),
            wraplength=420,
            justify="left"
        ).pack(anchor="w")

        right = self.panel(lower)
        right.pack(side="right", fill="y", padx=(10, 0))

        self.heading(right, "Quick Actions").pack(anchor="w")

        for text, action in [
            ("Live Monitoring", self.show_monitoring),
            ("View Alert History", self.show_alerts),
            ("Export Safety Report", self.export_report),
            ("View Statistics", self.show_statistics),
        ]:
            self.make_button(right, text, action).pack(
                fill="x", pady=(12, 0)
            )

        self.refresh_home_stats()

    def refresh_home_stats(self):
        for key, label in list(self.stat_labels.items()):
            if label.winfo_exists():
                label.configure(text=str(self.stats.get(key, 0)))

    # =====================================================
    # LIVE MONITORING
    # =====================================================
    def show_monitoring(self):
        self.set_active_nav("Live Monitoring")
        self.clear_content("Live Monitoring")

        toolbar = tk.Frame(self.content, bg=APP_BG)
        toolbar.pack(fill="x", pady=(0, 12))

        tk.Label(
            toolbar,
            text="Detection mode",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(side="left")

        mode_box = ttk.Combobox(
            toolbar,
            textvariable=self.mode,
            values=["PPE Detection", "Fall Detection"],
            state="readonly",
            width=18
        )
        mode_box.pack(side="left", padx=10)
        mode_box.bind("<<ComboboxSelected>>", self.change_mode)

        self.make_button(
            toolbar, "Open Video", self.open_video, True
        ).pack(side="left", padx=3)

        self.make_button(
            toolbar, "Open Image", self.open_image
        ).pack(side="left", padx=3)

        self.make_button(
            toolbar, "Pause", self.pause_video
        ).pack(side="left", padx=3)

        self.make_button(
            toolbar, "Continue", self.continue_video
        ).pack(side="left", padx=3)

        self.make_button(
            toolbar, "Stop", self.stop_video
        ).pack(side="left", padx=3)

        self.monitor_panel = self.panel(self.content, 5)
        self.monitor_panel.pack(fill="both", expand=True)

        self.video_canvas = tk.Canvas(
            self.monitor_panel,
            bg="#070B12",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True)
        self.video_canvas.bind(
            "<Configure>", self.redraw_current_frame
        )

        bottom = tk.Frame(self.content, bg=APP_BG)
        bottom.pack(fill="x", pady=(10, 0))

        tk.Label(
            bottom,
            textvariable=self.status_text,
            bg=APP_BG,
            fg=ACCENT,
            font=("Segoe UI", 9, "bold")
        ).pack(side="left")

        tk.Label(
            bottom,
            textvariable=self.alert_text,
            bg=APP_BG,
            fg=WARNING,
            font=("Segoe UI", 9),
            wraplength=600
        ).pack(side="right")

    # =====================================================
    # IMAGE AND VIDEO SELECTION
    # =====================================================
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_video()
        frame = cv2.imread(path)

        if frame is None:
            messagebox.showerror(
                "Image Error", "Unable to open the selected image."
            )
            return

        self.ensure_monitoring_page()
        self.process_frame(frame)
        self.status_text.set(f"Image loaded: {Path(path).name}")

    def open_video(self):
        path = filedialog.askopenfilename(
            title="Select a video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        self.stop_video()
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            messagebox.showerror(
                "Video Error",
                "Unable to open the selected video."
            )
            return

        self.ensure_monitoring_page()
        self.paused = False
        self.status_text.set(f"Playing: {Path(path).name}")
        self.read_video_frame()

    def ensure_monitoring_page(self):
        if not hasattr(self, "video_canvas"):
            self.show_monitoring()

    def read_video_frame(self):
        if self.closing or self.cap is None:
            return

        if self.paused:
            self.video_job = self.root.after(
                100, self.read_video_frame
            )
            return

        ok, frame = self.cap.read()

        if not ok:
            self.stop_video()
            self.status_text.set("Video finished")
            return

        self.process_frame(frame)

        if self.cap is not None:
            self.video_job = self.root.after(
                20, self.read_video_frame
            )

    def pause_video(self):
        if self.cap is not None:
            self.paused = True
            self.status_text.set("Video paused")

    def continue_video(self):
        if self.cap is not None:
            self.paused = False
            self.status_text.set("Video playing")

    def stop_video(self):
        if self.video_job is not None:
            try:
                self.root.after_cancel(self.video_job)
            except tk.TclError:
                pass
            self.video_job = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.paused = False

    def change_mode(self, _event=None):
        self.stop_video()
        self.stats["Persons"] = 0
        self.refresh_home_stats()
        self.status_text.set(f"Mode changed to {self.mode.get()}")

    # =====================================================
    # FRAME PROCESSING
    # =====================================================
    def process_frame(self, frame):
        if not self.models_loaded:
            self.status_text.set("AI models are still loading...")
            return

        try:
            if self.mode.get() == "PPE Detection":
                output = self.detect_ppe(frame)
            else:
                output = self.detect_fall(frame)

            self.stats["Frames Processed"] += 1
            self.current_image = output
            self.display_frame(output)
            self.refresh_home_stats()
            self.status_text.set("AI detection running")

        except Exception as error:
            self.status_text.set("Detection error")
            messagebox.showerror("Detection Error", str(error))

    @staticmethod
    def normalize_name(name):
        return (
            name.lower()
            .replace("-", " ")
            .replace("_", " ")
            .strip()
        )

    # =====================================================
    # PPE DETECTION
    # =====================================================
    def detect_ppe(self, frame):
        result = self.ppe_model.predict(
            source=frame,
            conf=float(self.ppe_conf.get()),
            imgsz=640,
            verbose=False
        )[0]

        output = frame.copy()

        # IMPORTANT FIX:
        # Count people in this frame, not across all video frames.
        person_count = 0

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            name = self.normalize_name(
                result.names.get(class_id, str(class_id))
            )

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )
            color = (40, 200, 120)

            if name == "person":
                person_count += 1

            elif name == "hardhat":
                self.stats["Hardhats"] += 1

            elif name == "no hardhat":
                self.stats["No Hardhats"] += 1
                color = (40, 40, 240)
                self.raise_alert(
                    "No Hardhat",
                    "Worker detected without a hardhat"
                )

            elif name == "mask":
                self.stats["Masks"] += 1

            elif name == "no mask":
                self.stats["No Masks"] += 1
                color = (40, 40, 240)
                self.raise_alert(
                    "No Mask",
                    "Worker detected without a mask"
                )

            elif name == "safety vest":
                self.stats["Safety Vests"] += 1

            elif name == "no safety vest":
                self.stats["No Safety Vests"] += 1
                color = (40, 40, 240)
                self.raise_alert(
                    "No Safety Vest",
                    "Worker detected without a safety vest"
                )

            cv2.rectangle(
                output, (x1, y1), (x2, y2), color, 2
            )
            cv2.putText(
                output,
                f"{name} {confidence:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # Replaces the previous frame's person count.
        self.stats["Persons"] = person_count

        return output

    # =====================================================
    # FALL DETECTION
    # =====================================================
    def detect_fall(self, frame):
        result = self.fall_model.predict(
            source=frame,
            conf=float(self.fall_conf.get()),
            imgsz=640,
            verbose=False
        )[0]

        output = frame.copy()

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            name = self.normalize_name(
                result.names.get(class_id, str(class_id))
            )

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )
            color = (40, 200, 120)

            if name in ("fallen", "fall", "falling"):
                self.stats["Falls"] += 1
                color = (40, 40, 240)
                self.raise_alert(
                    "Fall Detected",
                    "Possible fallen worker detected"
                )

            elif name == "pre falling":
                self.stats["Pre-Falls"] += 1
                color = (0, 165, 255)
                self.raise_alert(
                    "Pre-Fall",
                    "Possible pre-fall posture detected"
                )

            elif name == "standing":
                self.stats["Standing"] += 1

            elif name == "sitting":
                self.stats["Sitting"] += 1

            elif name == "bending":
                self.stats["Bending"] += 1

            elif name == "assisting":
                self.stats["Assisting"] += 1

            cv2.rectangle(
                output, (x1, y1), (x2, y2), color, 2
            )
            cv2.putText(
                output,
                f"{name} {confidence:.2f}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # This is a posture model, not a person-tracking model.
        self.stats["Persons"] = 0

        return output

    # =====================================================
    # ALERT MANAGEMENT
    # =====================================================
    def raise_alert(self, alert_type, message):
        now = time.time()

        if (
            now - self.last_alert_time.get(alert_type, 0)
            < ALERT_COOLDOWN
        ):
            return

        self.last_alert_time[alert_type] = now

        alert = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Type": alert_type,
            "Message": message
        }

        self.alert_history.append(alert)
        self.alert_history = self.alert_history[-500:]

        self.alert_text.set(f"{alert_type}: {message}")
        self.status_text.set(f"ALERT: {alert_type}")

        if self.sound_enabled.get() and winsound is not None:
            threading.Thread(
                target=self.play_sound,
                daemon=True
            ).start()

    @staticmethod
    def play_sound():
        try:
            winsound.Beep(1000, 300)
        except Exception:
            pass

    def show_alerts(self):
        self.set_active_nav("Alerts")
        self.clear_content("Alert Management")

        tk.Label(
            self.content,
            text="Recent PPE and fall-detection events",
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 12))

        self.make_button(
            self.content,
            "Export Alerts",
            self.export_report,
            True
        ).pack(anchor="w", pady=(0, 10))

        columns = ("Time", "Type", "Message")
        table = ttk.Treeview(
            self.content,
            columns=columns,
            show="headings"
        )

        for col, width in [
            ("Time", 180),
            ("Type", 180),
            ("Message", 550)
        ]:
            table.heading(col, text=col)
            table.column(col, width=width)

        table.pack(fill="both", expand=True)

        for alert in reversed(self.alert_history):
            table.insert(
                "",
                "end",
                values=(
                    alert["Time"],
                    alert["Type"],
                    alert["Message"]
                )
            )

        if not self.alert_history:
            tk.Label(
                self.content,
                text="No alerts recorded yet.",
                bg=APP_BG,
                fg=MUTED,
                font=("Segoe UI", 11)
            ).pack(anchor="w", pady=12)

    # =====================================================
    # REPORTS
    # =====================================================
    def show_reports(self):
        self.set_active_nav("Reports")
        self.clear_content("Reports")

        card = self.panel(self.content)
        card.pack(fill="x", anchor="n")

        self.heading(
            card, "Safety Report Centre", 17
        ).pack(anchor="w")

        tk.Label(
            card,
            text=(
                "Export current statistics and recorded alert history "
                "to a CSV file that can be opened in Excel."
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 10),
            wraplength=650,
            justify="left"
        ).pack(anchor="w", pady=12)

        self.make_button(
            card, "EXPORT CSV REPORT", self.export_report, True
        ).pack(anchor="w", pady=8)

        self.make_button(
            card, "VIEW ALERT HISTORY", self.show_alerts
        ).pack(anchor="w", pady=8)

        tk.Label(
            card,
            text=(
                "Person count is for the current frame. "
                "Other metrics are cumulative detections."
            ),
            bg=PANEL,
            fg=WARNING,
            font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(18, 0))

    # =====================================================
    # STATISTICS
    # =====================================================
    def show_statistics(self):
        self.set_active_nav("Statistics")
        self.clear_content("Detection Statistics")

        tk.Label(
            self.content,
            text=(
                "Person count = current frame; "
                "other detection counts = cumulative boxes."
            ),
            bg=APP_BG,
            fg=MUTED,
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(0, 12))

        table = ttk.Treeview(
            self.content,
            columns=("Metric", "Count"),
            show="headings"
        )

        table.heading("Metric", text="Metric")
        table.heading("Count", text="Count")
        table.column("Metric", width=350)
        table.column("Count", width=140)
        table.pack(fill="both", expand=True)

        for key, value in self.stats.items():
            table.insert("", "end", values=(key, value))

        self.make_button(
            self.content,
            "Refresh Statistics",
            self.show_statistics
        ).pack(anchor="w", pady=12)

    # =====================================================
    # SETTINGS
    # =====================================================
    def show_settings(self):
        self.set_active_nav("Settings")
        self.clear_content("Settings")

        card = self.panel(self.content)
        card.pack(fill="x", anchor="n")

        self.heading(
            card, "Detection Configuration", 16
        ).pack(anchor="w")

        tk.Label(
            card,
            text="PPE confidence threshold",
            bg=PANEL,
            fg=TEXT
        ).pack(anchor="w", pady=(18, 4))

        ttk.Scale(
            card,
            from_=0.10,
            to=0.90,
            variable=self.ppe_conf,
            orient="horizontal"
        ).pack(fill="x")

        tk.Label(
            card,
            text="Fall confidence threshold",
            bg=PANEL,
            fg=TEXT
        ).pack(anchor="w", pady=(18, 4))

        ttk.Scale(
            card,
            from_=0.10,
            to=0.90,
            variable=self.fall_conf,
            orient="horizontal"
        ).pack(fill="x")

        tk.Checkbutton(
            card,
            text="Enable alert sound",
            variable=self.sound_enabled,
            bg=PANEL,
            fg=TEXT,
            activebackground=PANEL,
            activeforeground=TEXT,
            selectcolor=PANEL_ALT
        ).pack(anchor="w", pady=(18, 4))

        tk.Label(
            card,
            text="Test different thresholds with representative videos.",
            bg=PANEL,
            fg=MUTED
        ).pack(anchor="w", pady=(10, 0))

    # =====================================================
    # ABOUT
    # =====================================================
    def show_about(self):
        self.set_active_nav("About")
        self.clear_content("About SafeVision")

        card = self.panel(self.content)
        card.pack(fill="x", anchor="n")

        self.heading(card, "SafeVision", 22).pack(anchor="w")

        tk.Label(
            card,
            text=(
                "AI-Based Real-Time Industrial Worker Safety Monitoring\n\n"
                "Core features\n"
                "• PPE detection using your trained PPE model\n"
                "• Fall and posture classification using your trained model\n"
                "• Video and image processing\n"
                "• Alert history and CSV report export\n\n"
                "Technology stack\n"
                "Python • Tkinter • OpenCV • Ultralytics YOLO"
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 11),
            justify="left"
        ).pack(anchor="w", pady=15)

    # =====================================================
    # CSV REPORT EXPORT
    # =====================================================
    def export_report(self):
        path = filedialog.asksaveasfilename(
            title="Export Safety Report",
            defaultextension=".csv",
            initialfile=(
                f"safety_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
            ),
            filetypes=[("CSV files", "*.csv")]
        )

        if not path:
            return

        try:
            with open(
                path, "w", newline="", encoding="utf-8-sig"
            ) as file:
                writer = csv.writer(file)

                writer.writerow(["SAFETY MONITORING REPORT"])
                writer.writerow([
                    "Generated at",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
                writer.writerow([])
                writer.writerow(["Metric", "Count"])

                for key, value in self.stats.items():
                    writer.writerow([key, value])

                writer.writerow([])
                writer.writerow(["Alert history"])
                writer.writerow(["Time", "Type", "Message"])

                for alert in self.alert_history:
                    writer.writerow([
                        alert["Time"],
                        alert["Type"],
                        alert["Message"]
                    ])

            messagebox.showinfo(
                "Export Successful",
                f"Report saved to:\n{path}"
            )

        except Exception as error:
            messagebox.showerror("Export Failed", str(error))

    # =====================================================
    # VIDEO DISPLAY
    # =====================================================
    def display_frame(self, frame):
        if not hasattr(self, "video_canvas"):
            return

        if not self.video_canvas.winfo_exists():
            return

        cw = self.video_canvas.winfo_width()
        ch = self.video_canvas.winfo_height()

        if cw < 2 or ch < 2:
            return

        height, width = frame.shape[:2]
        scale = min(cw / width, ch / height)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))

        resized = cv2.resize(
            frame, (new_width, new_height)
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        self.photo = ImageTk.PhotoImage(
            Image.fromarray(rgb)
        )

        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            cw // 2,
            ch // 2,
            image=self.photo,
            anchor="center"
        )

    def redraw_current_frame(self, _event=None):
        if self.current_image is not None:
            self.display_frame(self.current_image)

    # =====================================================
    # LOGOUT AND EXIT
    # =====================================================
    def logout(self):
        self.stop_video()
        self.current_image = None
        self.photo = None
        self.login_password.set("")
        self.show_login()

    def on_close(self):
        self.closing = True
        self.stop_video()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = WorkerSafetyApp(root)
    root.mainloop()
