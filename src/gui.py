"""
src/gui.py — Multi-Tab Tkinter GUI (MoE Edition)
Tabs:
  1. Real-Time Translator  — Camera, MoE Router Badge, Prediction, Translation, TTS
  2. Teach Custom Sign     — Record → Retrain (Word Expert) → Hot-Reload
  3. Prediction History    — SQLite ttk.Treeview logs
"""

import logging
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager
from src.predict import GesturePredictor, MODE_IDLE, MODE_ALPHABET, MODE_WORD
from src.retrain_dynamic import train_new_gesture
from src.translator import AudioTranslator
from src.utils import (
    draw_hand_landmarks, draw_router_badge,
    draw_prediction_banner, draw_status_bar, mirror_frame,
    extract_dual_hand_features,
)

logger = logging.getLogger(__name__)

# ─── Design Tokens ────────────────────────────────────────────────────────────
BG_DARK  = "#0d0d1a"
BG_MID   = "#16162a"
BG_CARD  = "#1e1e38"
ACCENT   = "#00e5cc"
ACCENT2  = "#7c3aed"
TEXT_PRI = "#e2e2f0"
TEXT_SEC = "#7878a8"
SUCCESS  = "#00c896"
WARN     = "#fbbf24"
DANGER   = "#f43f5e"
INFO_CLR = "#38bdf8"

F_BASE  = ("Segoe UI", 10)
F_H1    = ("Segoe UI Semibold", 18)
F_H2    = ("Segoe UI Semibold", 13)
F_MONO  = ("Consolas", 10)
F_GIANT = ("Segoe UI Semibold", 40)


# ─────────────────────────────────────────────────────────────────────────────
def _apply_styles(root: tk.Tk) -> ttk.Style:
    s = ttk.Style(root)
    s.theme_use(config.GUI_THEME)

    s.configure("TNotebook",           background=BG_DARK, borderwidth=0)
    s.configure("TNotebook.Tab",       background=BG_MID,  foreground=TEXT_SEC,
                                        padding=[18, 9], font=("Segoe UI", 10))
    s.map("TNotebook.Tab",
          background=[("selected", BG_CARD)],
          foreground=[("selected", ACCENT)])

    s.configure("Dark.TFrame",  background=BG_DARK)
    s.configure("Card.TFrame",  background=BG_CARD, relief="flat")
    s.configure("Mid.TFrame",   background=BG_MID)

    s.configure("TLabel",       background=BG_DARK, foreground=TEXT_PRI, font=F_BASE)
    s.configure("H1.TLabel",    background=BG_DARK, foreground=ACCENT,   font=F_H1)
    s.configure("H2.TLabel",    background=BG_CARD, foreground=ACCENT,   font=F_H2)
    s.configure("Muted.TLabel", background=BG_DARK, foreground=TEXT_SEC, font=F_BASE)
    s.configure("Card.TLabel",  background=BG_CARD, foreground=TEXT_PRI, font=F_BASE)
    s.configure("Mono.TLabel",  background=BG_CARD, foreground=ACCENT,   font=F_MONO)
    s.configure("Warn.TLabel",  background=BG_CARD, foreground=WARN,     font=F_BASE)

    s.configure("Accent.TButton",  background=ACCENT,  foreground=BG_DARK,
                                    font=("Segoe UI Semibold", 10), padding=[12, 6])
    s.map("Accent.TButton", background=[("active", "#00c4b0")])

    s.configure("Danger.TButton",  background=DANGER,  foreground="white",
                                    font=("Segoe UI Semibold", 10), padding=[12, 6])
    s.map("Danger.TButton", background=[("active", "#d4294d")])

    s.configure("Teal.Horizontal.TProgressbar",
                troughcolor=BG_MID, background=ACCENT, thickness=14, borderwidth=0)

    s.configure("TCombobox",    fieldbackground=BG_MID, background=BG_MID,
                                foreground=TEXT_PRI, selectbackground=ACCENT2)
    s.configure("TCheckbutton", background=BG_DARK, foreground=TEXT_PRI)

    s.configure("Treeview",
                background=BG_MID, foreground=TEXT_PRI,
                fieldbackground=BG_MID, rowheight=26, font=F_MONO)
    s.configure("Treeview.Heading",
                background=BG_CARD, foreground=ACCENT,
                font=("Segoe UI Semibold", 10))
    s.map("Treeview",
          background=[("selected", ACCENT2)],
          foreground=[("selected", "white")])
    return s


class ThreadedCameraStream:
    """
    Asynchronous background-threaded camera stream reader.
    Continuously fetches frames from VideoCapture and holds only the single latest frame,
    preventing hardware buffer accumulation and main GUI thread blocking.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._running = True
        self._ret = False
        self._frame: Optional[np.ndarray] = None

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        # Grab initial frame
        self._ret, self._frame = self._cap.read()

        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="CameraStreamWorker"
        )
        self._thread.start()

    def _worker(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                break
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._ret = ret
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._ret or self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=0.4)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


_WORKING_CAMERA_CONFIG: Optional[tuple[int, int]] = None

def open_robust_camera(preferred_index: int = config.CAMERA_INDEX) -> Optional[cv2.VideoCapture]:
    """Try preferred index and fallback backends/indices for robust camera capture."""
    global _WORKING_CAMERA_CONFIG
    if _WORKING_CAMERA_CONFIG is not None:
        idx, backend = _WORKING_CAMERA_CONFIG
        try:
            cap = cv2.VideoCapture(idx, backend)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    try:
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass
                return cap
        except Exception:
            _WORKING_CAMERA_CONFIG = None

    indices = [preferred_index] + [i for i in [0, 1, 2] if i != preferred_index]
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    
    for idx in indices:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                    _WORKING_CAMERA_CONFIG = (idx, backend)
                    return cap
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Real-Time Translator
# ─────────────────────────────────────────────────────────────────────────────

class LiveTranslatorTab(ttk.Frame):
    _PW, _PH = 520, 400

    def __init__(self, parent, predictor: GesturePredictor,
                 translator: AudioTranslator, db: DatabaseManager, **kw):
        super().__init__(parent, style="Dark.TFrame", **kw)
        self._pred   = predictor
        self._trans  = translator
        self._db     = db
        self._stream: Optional[ThreadedCameraStream] = None
        self._running = False
        self._after  = None

        # Cached state for selective UI redraws
        self._last_fill: int = -1
        self._last_badge_mode: str = ""

        # Prediction state
        self._label: str  = "—"
        self._conf: float = 0.0
        self._tx: str     = "—"
        self._mode: str   = MODE_IDLE

        self._build()
        self.after(500, self._start_camera)

    def _build(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Camera panel ────────────────────────────────────────────────────────
        left = ttk.Frame(self, style="Dark.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew")

        ttk.Label(left, text="📷  Live Camera Feed", style="H1.TLabel").pack(anchor="w", pady=(0, 8))

        self._cam_lbl = tk.Label(left, bg=BG_DARK, width=self._PW, height=self._PH)
        self._cam_lbl.pack()

        # Word buffer bar
        buf_row = ttk.Frame(left, style="Dark.TFrame")
        buf_row.pack(fill="x", pady=(6, 0))
        ttk.Label(buf_row, text="Word Buffer:", style="Muted.TLabel").pack(side="left")
        self._buf_bar = ttk.Progressbar(buf_row, maximum=config.SEQUENCE_LENGTH,
                                         style="Teal.Horizontal.TProgressbar", length=180)
        self._buf_bar.pack(side="left", padx=6)
        self._buf_lbl = ttk.Label(buf_row, text=f"0 / {config.SEQUENCE_LENGTH}",
                                   style="Muted.TLabel")
        self._buf_lbl.pack(side="left")

        # Buttons
        btn_row = ttk.Frame(left, style="Dark.TFrame")
        btn_row.pack(fill="x", pady=8)
        self._start_btn = ttk.Button(btn_row, text="▶  Start Camera",
                                      style="Accent.TButton",
                                      command=self._toggle_camera)
        self._start_btn.pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="🔄  Reset Buffer",
                   style="Danger.TButton",
                   command=self._reset_buffer).pack(side="left")

        # ── Controls + Prediction panel ────────────────────────────────────────
        right = ttk.Frame(self, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Controls", style="H2.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12))

        # Router badge (text)
        ttk.Label(right, text="MoE Router Mode:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w")
        self._mode_var = tk.StringVar(value="⬜  Idle")
        self._mode_lbl = tk.Label(right, textvariable=self._mode_var,
                                   bg=BG_CARD, fg=TEXT_SEC,
                                   font=("Segoe UI Semibold", 11), anchor="w")
        self._mode_lbl.grid(row=2, column=0, sticky="ew", pady=(2, 12))

        # Language
        ttk.Label(right, text="Target Language:", style="Card.TLabel").grid(
            row=3, column=0, sticky="w")
        self._lang_var = tk.StringVar(value=config.DEFAULT_LANGUAGE)
        cb = ttk.Combobox(right, textvariable=self._lang_var,
                          values=list(config.SUPPORTED_LANGUAGES.keys()),
                          state="readonly", width=18)
        cb.grid(row=4, column=0, sticky="ew", pady=(4, 10))
        cb.bind("<<ComboboxSelected>>", self._on_lang_change)

        # TTS
        self._tts_var = tk.BooleanVar(value=config.TTS_ENABLED_DEFAULT)
        ttk.Checkbutton(right, text="🔊  Text-to-Speech",
                         variable=self._tts_var,
                         command=lambda: self._trans.set_tts_enabled(self._tts_var.get())
                         ).grid(row=5, column=0, sticky="w", pady=(0, 14))

        ttk.Separator(right, orient="horizontal").grid(row=6, column=0, sticky="ew", pady=6)

        # Prediction display
        ttk.Label(right, text="Detected Sign", style="H2.TLabel").grid(
            row=7, column=0, sticky="w", pady=(0, 4))
        self._pred_lbl = tk.Label(right, text="—", font=F_GIANT,
                                   bg=BG_CARD, fg=ACCENT, anchor="center")
        self._pred_lbl.grid(row=8, column=0, sticky="ew", pady=(0, 4))

        # Confidence bar
        ttk.Label(right, text="Confidence:", style="Card.TLabel").grid(
            row=9, column=0, sticky="w")
        self._conf_bar = ttk.Progressbar(right, maximum=100,
                                          style="Teal.Horizontal.TProgressbar")
        self._conf_bar.grid(row=10, column=0, sticky="ew", pady=(4, 2))
        self._conf_lbl = ttk.Label(right, text="0.00%", style="Mono.TLabel")
        self._conf_lbl.grid(row=11, column=0, sticky="e")

        ttk.Separator(right, orient="horizontal").grid(row=12, column=0, sticky="ew", pady=10)

        # Translation box
        ttk.Label(right, text="Translation:", style="H2.TLabel").grid(
            row=13, column=0, sticky="w", pady=(0, 4))
        self._tx_var = tk.StringVar(value="—")
        tk.Label(right, textvariable=self._tx_var,
                 bg=BG_MID, fg=WARN,
                 font=("Segoe UI", 14), anchor="center",
                 wraplength=220, pady=10
                 ).grid(row=14, column=0, sticky="ew")

        # Status
        ttk.Separator(right, orient="horizontal").grid(row=15, column=0, sticky="ew", pady=10)
        self._status_var = tk.StringVar(value="Camera stopped.")
        ttk.Label(right, textvariable=self._status_var,
                  style="Muted.TLabel", wraplength=220).grid(row=16, column=0, sticky="w")

    # ── Camera loop ────────────────────────────────────────────────────────────

    def _toggle_camera(self) -> None:
        if self._running:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self) -> None:
        raw_cap = open_robust_camera(config.CAMERA_INDEX)
        if raw_cap is None:
            messagebox.showerror("Camera Error",
                                 f"Cannot open webcam (tried index {config.CAMERA_INDEX} & fallbacks).\n"
                                 "Check that a webcam is connected and not in use by another app.")
            return
        self._stream = ThreadedCameraStream(raw_cap)
        self._running = True
        self._start_btn.config(text="⏹  Stop Camera")
        self._status_var.set("Camera running…")
        self._tick()

    def _stop_camera(self) -> None:
        self._running = False
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if self._stream:
            self._stream.stop()
            self._stream = None
        self._cam_lbl.config(image="", text="Camera stopped", fg=TEXT_SEC)
        self._start_btn.config(text="▶  Start Camera")
        self._status_var.set("Camera stopped.")
        self._pred.reset_buffers()

    def _tick(self) -> None:
        if not self._running or self._stream is None:
            return

        t_start = time.monotonic()
        ret, frame = self._stream.read()
        if not ret or frame is None:
            self._status_var.set("⚠ Frame read error.")
            self._after = self.after(33, self._tick)
            return

        frame = mirror_frame(frame)

        # ── MoE inference (single pass) ──────────────────────────────────────
        result, left_lm, right_lm, n_hands = self._pred.update(frame)
        mode   = self._pred.current_mode

        # Update router badge only when state changes
        if mode != self._last_badge_mode:
            self._update_badge(mode)
            self._last_badge_mode = mode

        # Draw hand overlays using single-pass landmarks
        if left_lm:
            draw_hand_landmarks(frame, left_lm, color=(50, 200, 255))
        if right_lm:
            draw_hand_landmarks(frame, right_lm, color=(0, 230, 110))

        draw_router_badge(frame, mode)

        # Update word buffer bar only when fill changes
        fill = self._pred.word_buffer_fill
        if fill != self._last_fill:
            self._buf_bar["value"] = fill
            self._buf_lbl.config(text=f"{fill} / {config.SEQUENCE_LENGTH}")
            self._last_fill = fill

        if result:
            label, conf, _mode = result
            self._on_stable_prediction(label, conf)

        if self._label != "—":
            draw_prediction_banner(frame, self._label, self._conf, self._tx)

        draw_status_bar(frame, self._status_var.get())

        # Render
        display = cv2.resize(frame, (self._PW, self._PH))
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._cam_lbl.config(image=img, text="")
        self._cam_lbl.image = img

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        delay = max(1, 33 - elapsed_ms)
        self._after = self.after(delay, self._tick)

    def _update_badge(self, mode: str) -> None:
        badge_cfg = {
            MODE_IDLE:     ("⬜  Idle — No Hands",          TEXT_SEC),
            MODE_ALPHABET: ("🔤  Alphabet Mode — 1 Hand",   INFO_CLR),
            MODE_WORD:     ("💬  Word Mode — 2 Hands",       SUCCESS),
        }
        text, color = badge_cfg.get(mode, (mode, TEXT_PRI))
        self._mode_var.set(text)
        self._mode_lbl.config(fg=color)

    def _on_stable_prediction(self, label: str, confidence: float) -> None:
        self._label = label
        self._conf  = confidence
        tx = self._trans.process_prediction(label, confidence)
        self._tx    = tx

        self._pred_lbl.config(text=label)
        self._conf_bar["value"] = round(confidence * 100, 1)
        self._conf_lbl.config(text=f"{confidence * 100:.1f}%")
        self._tx_var.set(tx)
        self._status_var.set(f"Detected: {label}  ({confidence * 100:.1f}%)")

    def _reset_buffer(self) -> None:
        self._pred.reset_buffers()
        self._label = "—"
        self._conf  = 0.0
        self._tx    = "—"
        self._pred_lbl.config(text="—")
        self._conf_bar["value"] = 0
        self._conf_lbl.config(text="0.00%")
        self._tx_var.set("—")

    def _on_lang_change(self, _=None) -> None:
        code = config.SUPPORTED_LANGUAGES.get(self._lang_var.get(), "en")
        self._trans.target_language = code

    def on_activated(self) -> None:
        self._start_camera()

    def on_deactivated(self) -> None:
        self._stop_camera()

    def destroy(self) -> None:
        self._stop_camera()
        super().destroy()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Teach Custom Sign (Word Expert Continual Learning)
# ─────────────────────────────────────────────────────────────────────────────

class TeachNewSignTab(ttk.Frame):
    """Record new dual-hand word gesture → fine-tune Word Expert → hot-reload."""

    def __init__(self, parent, predictor: GesturePredictor,
                 db: DatabaseManager, **kw):
        super().__init__(parent, style="Dark.TFrame", **kw)
        self._pred  = predictor
        self._db    = db
        self._stream: Optional[ThreadedCameraStream] = None
        self._after = None
        self._running    = False
        self._recording  = False
        self._frame_buf: list[np.ndarray] = []
        self._recordings: list[np.ndarray] = []
        self._cur_label  = ""
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Camera preview ──────────────────────────────────────────────────
        left = ttk.Frame(self, style="Dark.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew")

        ttk.Label(left, text="🎓  Teach New Word Sign", style="H1.TLabel").pack(
            anchor="w", pady=(0, 8))

        self._cam_lbl = tk.Label(left, bg=BG_DARK, width=520, height=400)
        self._cam_lbl.pack()

        self._rec_status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._rec_status,
                  font=("Segoe UI Semibold", 13),
                  background=BG_DARK, foreground=WARN).pack(pady=4)

        ttk.Label(left,
                  text="ℹ  Use BOTH hands while recording for the Word Expert.",
                  style="Muted.TLabel").pack()

        # ── Controls ─────────────────────────────────────────────────────────
        right = ttk.Frame(self, style="Card.TFrame", padding=16)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="New Word Gesture", style="H2.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 12))

        ttk.Label(right, text="Gesture Label:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w")
        self._label_entry = ttk.Entry(right, font=("Segoe UI", 12), width=20)
        self._label_entry.grid(row=2, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(right, text="Recordings:", style="Card.TLabel").grid(
            row=3, column=0, sticky="w")
        self._rec_cnt = tk.StringVar(value=f"0 / {config.TEACH_NUM_RECORDINGS}")
        ttk.Label(right, textvariable=self._rec_cnt,
                  style="Mono.TLabel").grid(row=4, column=0, sticky="w")

        ttk.Label(right, text="Training Progress:", style="Card.TLabel").grid(
            row=5, column=0, sticky="w", pady=(12, 0))
        self._train_bar = ttk.Progressbar(right, maximum=7,
                                           style="Teal.Horizontal.TProgressbar")
        self._train_bar.grid(row=6, column=0, sticky="ew", pady=4)
        self._train_status = tk.StringVar(value="Idle")
        ttk.Label(right, textvariable=self._train_status,
                  style="Muted.TLabel", wraplength=200).grid(
            row=7, column=0, sticky="w")

        ttk.Separator(right, orient="horizontal").grid(
            row=8, column=0, sticky="ew", pady=12)

        self._rec_btn = ttk.Button(right, text="🎥  Start Recording",
                                    style="Accent.TButton",
                                    command=self._start_recording)
        self._rec_btn.grid(row=9, column=0, sticky="ew", pady=4)

        self._train_btn = ttk.Button(right, text="🧠  Train Model",
                                      style="Accent.TButton",
                                      command=self._start_train,
                                      state="disabled")
        self._train_btn.grid(row=10, column=0, sticky="ew", pady=4)

        ttk.Button(right, text="🔄  Reset All",
                   style="Danger.TButton",
                   command=self._reset).grid(row=11, column=0, sticky="ew", pady=(12, 0))

        ttk.Separator(right, orient="horizontal").grid(
            row=12, column=0, sticky="ew", pady=12)

        tips = ("Tips:\n"
                "• Use BOTH hands for word gestures.\n"
                "• Perform 10 clear recordings.\n"
                "• Good lighting improves accuracy.\n"
                "• After training, test in Live tab.")
        ttk.Label(right, text=tips, style="Muted.TLabel",
                  justify="left", wraplength=200).grid(row=13, column=0, sticky="w")

    def on_activated(self) -> None:
        self._start_camera()

    def on_deactivated(self) -> None:
        self._stop_camera()

    def _start_camera(self) -> None:
        if self._running:
            return
        raw_cap = open_robust_camera(config.CAMERA_INDEX)
        if raw_cap is None:
            self._rec_status.set("⚠ Cannot open webcam.")
            return
        self._stream = ThreadedCameraStream(raw_cap)
        self._running = True
        self._preview_tick()

    def _stop_camera(self) -> None:
        self._running = False
        self._recording = False
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if self._stream:
            self._stream.stop()
            self._stream = None
        self._cam_lbl.config(image="", text="Camera stopped", fg=TEXT_SEC)
        self._cam_lbl.image = None

    def _preview_tick(self) -> None:
        if not self._running or self._stream is None or self._recording:
            return

        t_start = time.monotonic()
        ret, frame = self._stream.read()
        if not ret or frame is None:
            self._after = self.after(33, self._preview_tick)
            return

        frame = mirror_frame(frame)
        left_lm, right_lm, _ = self._pred.detect_hands(frame)

        if left_lm:
            draw_hand_landmarks(frame, left_lm, color=(50, 200, 255))
        if right_lm:
            draw_hand_landmarks(frame, right_lm, color=(0, 230, 110))

        display = cv2.resize(frame, (520, 400))
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._cam_lbl.config(image=img, text="")
        self._cam_lbl.image = img

        elapsed_ms = int((time.monotonic() - t_start) * 1000)
        delay = max(1, 33 - elapsed_ms)
        self._after = self.after(delay, self._preview_tick)

    # ── Recording ──────────────────────────────────────────────────────────────

    def _start_recording(self) -> None:
        label = self._label_entry.get().strip().upper()
        if not label:
            messagebox.showwarning("Missing Label", "Enter a gesture label first.")
            return
        if len(self._recordings) >= config.TEACH_NUM_RECORDINGS:
            messagebox.showinfo("Done", "Already have 10 recordings. Click Train Model.")
            return

        if not self._running or self._stream is None:
            self._start_camera()
            if self._stream is None:
                messagebox.showerror("Camera Error", "Cannot open webcam.")
                return

        self._cur_label = label
        self._rec_btn.config(state="disabled")
        self._countdown_then_record()

    def _countdown_then_record(self) -> None:
        rem = [config.TEACH_COUNTDOWN_SECS]

        def tick():
            if rem[0] > 0:
                self._rec_status.set(f"Get ready… {rem[0]}")
                rem[0] -= 1
                self.after(1000, tick)
            else:
                self._rec_status.set(f"● Recording (0 / {config.SEQUENCE_LENGTH} frames)")
                self._frame_buf = []
                self._recording = True
                self._capture_tick()
        tick()

    def _capture_tick(self) -> None:
        if not self._recording or self._stream is None:
            return

        t_start = time.monotonic()
        ret, frame = self._stream.read()
        if not ret or frame is None:
            self._rec_status.set("⚠ Camera error.")
            self._stop_recording(success=False)
            return

        frame = mirror_frame(frame)
        left_lm, right_lm, _ = self._pred.detect_hands(frame)

        feat = extract_dual_hand_features(left_lm, right_lm)   # (123,) always
        self._frame_buf.append(feat)

        if left_lm:
            draw_hand_landmarks(frame, left_lm, color=(50, 200, 255))
        if right_lm:
            draw_hand_landmarks(frame, right_lm, color=(0, 230, 110))

        n = len(self._frame_buf)
        self._rec_status.set(f"● Recording ({n} / {config.SEQUENCE_LENGTH} frames)")

        display = cv2.resize(frame, (520, 400))
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img     = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self._cam_lbl.config(image=img, text="")
        self._cam_lbl.image = img

        if n >= config.SEQUENCE_LENGTH:
            self._stop_recording(success=True)
        else:
            elapsed_ms = int((time.monotonic() - t_start) * 1000)
            delay = max(1, 33 - elapsed_ms)
            self._after = self.after(delay, self._capture_tick)

    def _stop_recording(self, success: bool) -> None:
        self._recording = False
        if self._after:
            self.after_cancel(self._after)
            self._after = None
        if self._stream:
            self._stream.stop()
            self._stream = None

        if success and len(self._frame_buf) >= 20:
            from src.utils import pad_or_sample_sequence
            seq = pad_or_sample_sequence(
                self._frame_buf, config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES
            )
            self._recordings.append(seq)
            n = len(self._recordings)
            self._rec_cnt.set(f"{n} / {config.TEACH_NUM_RECORDINGS}")
            self._rec_status.set(
                f"✅ Recording {n} saved."
                + (" Train when ready!" if n >= 5 else "")
            )
            if n >= 5:
                self._train_btn.config(state="normal")
        else:
            self._rec_status.set("Recording discarded (too few frames).")

        self._rec_btn.config(state="normal")

    # ── Training ───────────────────────────────────────────────────────────────

    def _start_train(self) -> None:
        label = self._cur_label or self._label_entry.get().strip().upper()
        if not label or not self._recordings:
            messagebox.showwarning("Not Ready", "Record at least 5 samples first.")
            return

        # Save sequences to temp dir
        temp_dir = config.TEACH_TEMP_DIR / label
        temp_dir.mkdir(parents=True, exist_ok=True)
        for i, seq in enumerate(self._recordings):
            np.save(str(temp_dir / f"{i:04d}.npy"), seq)

        self._train_btn.config(state="disabled")
        self._rec_btn.config(state="disabled")
        self._train_status.set("Starting training thread…")
        self._train_bar["value"] = 0

        threading.Thread(
            target=self._run_train,
            args=(label, temp_dir),
            daemon=True, name="TrainThread",
        ).start()

    def _run_train(self, label: str, temp_dir: Path) -> None:
        try:
            def cb(step, total, msg):
                self.after(0, lambda s=step, t=total, m=msg: self._on_prog(s, t, m))

            train_new_gesture(
                new_word_label=label,
                new_samples_dir=temp_dir,
                progress_callback=cb,
            )
            self.after(0, self._pred.hot_reload)
            self.after(0, lambda l=label: self._on_done(l))
        except Exception as exc:           # noqa: BLE001
            self.after(0, lambda err=str(exc): self._on_err(err))

    def _on_prog(self, step, total, msg) -> None:
        self._train_bar["value"]   = step
        self._train_bar["maximum"] = total
        self._train_status.set(msg)

    def _on_done(self, label: str) -> None:
        self._train_status.set(f"✅ '{label}' trained & hot-reloaded!")
        messagebox.showinfo("Training Complete",
                            f"'{label}' has been trained successfully.\n"
                            "Switch to Live Translator to test it!")
        self._reset()

    def _on_err(self, msg: str) -> None:
        self._train_status.set(f"❌ Error: {msg}")
        self._train_btn.config(state="normal")
        self._rec_btn.config(state="normal")
        messagebox.showerror("Training Failed", msg)

    def _reset(self) -> None:
        self._recordings.clear()
        self._frame_buf.clear()
        self._cur_label = ""
        self._rec_cnt.set(f"0 / {config.TEACH_NUM_RECORDINGS}")
        self._rec_status.set("")
        self._train_status.set("Idle")
        self._train_bar["value"] = 0
        self._train_btn.config(state="disabled")
        self._rec_btn.config(state="normal")
        self._label_entry.delete(0, "end")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Prediction History & Analytics
# ─────────────────────────────────────────────────────────────────────────────

class HistoryTab(ttk.Frame):
    _COLS  = ("timestamp", "predicted_label", "confidence_score",
               "target_language", "translated_text")
    _HDRS  = ("Timestamp", "Sign", "Confidence", "Language", "Translation")
    _WIDTHS = (165, 80, 80, 90, 200)

    def __init__(self, parent, db: DatabaseManager, **kw):
        super().__init__(parent, style="Dark.TFrame", **kw)
        self._db = db
        self._build()
        self.refresh()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        hdr = ttk.Frame(self, style="Dark.TFrame", padding=(12, 8))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="📊  Prediction History & Analytics", style="H1.TLabel").pack(side="left")
        ttk.Button(hdr, text="🔄  Refresh",
                   style="Accent.TButton", command=self.refresh).pack(side="right", padx=(0, 4))
        ttk.Button(hdr, text="🗑  Clear All",
                   style="Danger.TButton", command=self._clear).pack(side="right", padx=(0, 8))

        tf = ttk.Frame(self, style="Dark.TFrame", padding=(12, 0))
        tf.grid(row=1, column=0, sticky="nsew")
        tf.columnconfigure(0, weight=1)
        tf.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(tf, columns=self._COLS,
                                   show="headings", selectmode="browse")
        for col, hdr, w in zip(self._COLS, self._HDRS, self._WIDTHS):
            self._tree.heading(col, text=hdr,
                               command=lambda c=col: self._sort(c))
            self._tree.column(col, width=w, anchor="center")

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        stats_frm = ttk.Frame(self, style="Card.TFrame", padding=8)
        stats_frm.grid(row=2, column=0, sticky="ew")
        self._stats_var = tk.StringVar(value="")
        ttk.Label(stats_frm, textvariable=self._stats_var,
                  style="Muted.TLabel").pack(side="left")

    def refresh(self) -> None:
        rows = self._db.get_history_summary(limit=500)
        for item in self._tree.get_children():
            self._tree.delete(item)
        for i, row in enumerate(rows):
            tag = "odd" if i % 2 else "even"
            self._tree.insert("", "end", values=(
                row["timestamp"],
                row["predicted_label"],
                f"{row['confidence_score'] * 100:.1f}%",
                row["target_language"],
                row["translated_text"],
            ), tags=(tag,))
        self._tree.tag_configure("odd",  background=BG_MID)
        self._tree.tag_configure("even", background=BG_CARD)

        s = self._db.get_stats()
        self._stats_var.set(
            f"Alphabets: {s.get('alphabet_gestures', 0)}  |  "
            f"Words: {s.get('word_gestures', 0)}  |  "
            f"Custom: {s.get('custom_gestures', 0)}  |  "
            f"Predictions: {s.get('history_entries', 0)}  |  "
            f"Cache: {s.get('cache_entries', 0)}"
        )

    def _sort(self, col: str) -> None:
        data = [(self._tree.set(i, col), i) for i in self._tree.get_children()]
        for idx, (_, item) in enumerate(sorted(data)):
            self._tree.move(item, "", idx)

    def _clear(self) -> None:
        if messagebox.askyesno("Confirm", "Clear ALL prediction history?"):
            self._db.clear_history()
            self.refresh()


# ─────────────────────────────────────────────────────────────────────────────
# Root Application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self, predictor: GesturePredictor,
                 db: DatabaseManager,
                 translator: AudioTranslator) -> None:
        super().__init__()
        self._pred  = predictor
        self._db    = db
        self._trans = translator

        self.title(config.GUI_TITLE)
        self.geometry("1120x740")
        self.minsize(900, 600)
        self.configure(bg=BG_DARK)

        _apply_styles(self)
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        # Title bar
        tb = tk.Frame(self, bg=BG_DARK, pady=8)
        tb.pack(fill="x", padx=16)
        tk.Label(tb, text="🤟  Multilingual Real-Time Sign Language Translator",
                 bg=BG_DARK, fg=ACCENT,
                 font=("Segoe UI Semibold", 16)).pack(side="left")
        tk.Label(tb, text="MoE Edition  |  Python 3.11  |  TensorFlow + MediaPipe",
                 bg=BG_DARK, fg=TEXT_SEC, font=("Segoe UI", 9)).pack(side="right", pady=(4, 0))

        ttk.Separator(self, orient="horizontal").pack(fill="x")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        self._live    = LiveTranslatorTab(nb, self._pred, self._trans, self._db)
        self._teach   = TeachNewSignTab(nb, self._pred, self._db)
        self._history = HistoryTab(nb, self._db)

        nb.add(self._live,    text="  📷  Real-Time Translator  ")
        nb.add(self._teach,   text="  🎓  Teach Custom Sign  ")
        nb.add(self._history, text="  📊  History & Analytics  ")

        def _tab_change(event):
            idx = nb.index(nb.select())
            if idx == 0:
                self._teach.on_deactivated()
                self._live.on_activated()
            elif idx == 1:
                self._live.on_deactivated()
                self._teach.on_activated()
            else:
                self._live.on_deactivated()
                self._teach.on_deactivated()
                if idx == 2:
                    self._history.refresh()

        nb.bind("<<NotebookTabChanged>>", _tab_change)

    def _on_close(self) -> None:
        self._live._stop_camera()
        self._teach._stop_camera()
        self._pred.close()
        self.destroy()
