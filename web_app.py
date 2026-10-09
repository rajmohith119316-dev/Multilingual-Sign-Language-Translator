"""
web_app.py — Flask Web UI for Multilingual Sign Language Translator (MoE Edition)

4-Page Architecture
───────────────────
  Page 1 — Recognition & Translation  (Live camera + MoE + multilingual)
  Page 2 — Teach Custom Sign          (Record + retrain + hot-reload)
  Page 3 — 3D Animation & Learn       (Three.js hand model + gesture gallery)
  Page 4 — Prediction History          (Full log + analytics)

How it works
────────────
1. Flask server opens the webcam once on startup.
2. /video_feed streams MJPEG frames via multipart/x-mixed-replace.
   → Each frame goes through MediaPipe hand detection + MoE inference.
   → Overlays (landmarks, router badge, prediction banner) are drawn with OpenCV.
3. /api/* JSON endpoints handle all interactive features.
4. The browser renders a stunning single-page app with client-side routing.

Run
───
    python web_app.py
    → Open http://127.0.0.1:5000 in a browser
    → Press Ctrl+C in the terminal to stop
"""

import base64
import json
import logging
import os
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, session

# ── Project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.database_manager import DatabaseManager
from src.predict import GesturePredictor, MODE_IDLE, MODE_ALPHABET, MODE_WORD
from src.retrain_dynamic import train_new_gesture
from src.translator import AudioTranslator
from src.utils import (
    draw_hand_landmarks,
    draw_router_badge,
    draw_prediction_banner,
    draw_status_bar,
    mirror_frame,
    extract_dual_hand_features,
    pad_or_sample_sequence,
)

# ── Logging ──────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "web_app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("web_app")

# ── Ensure directories ──────────────────────────────────────────────────────
for d in (config.DATA_DIR, config.MODEL_DIR, config.LOGS_DIR,
          config.RAW_ALPHABETS_DIR, config.RAW_WORDS_DIR,
          config.PROCESSED_ALPHABETS_DIR, config.PROCESSED_WORDS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# ASL Reference Data — finger joint angles for 3D hand model
# ═══════════════════════════════════════════════════════════════════════════

# Simplified ASL hand pose data for Three.js 3D hand model.
# Each letter maps to joint curl values (0 = extended, 1 = fully curled)
# and optional orientation hints.
ASL_HAND_POSES = {
    "A": {
        "description": "Fist with thumb alongside",
        "fingers": {"thumb": 0.3, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "B": {
        "description": "Flat hand, fingers together, thumb tucked",
        "fingers": {"thumb": 0.8, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
        "wrist_rotation": [0, 0, 0],
    },
    "C": {
        "description": "Curved hand like holding a cup",
        "fingers": {"thumb": 0.3, "index": 0.4, "middle": 0.4, "ring": 0.4, "pinky": 0.4},
        "wrist_rotation": [0, 0.2, 0],
    },
    "D": {
        "description": "Index up, others touching thumb",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 0.9, "ring": 0.9, "pinky": 0.9},
        "wrist_rotation": [0, 0, 0],
    },
    "E": {
        "description": "All fingers curled, thumb tucked under",
        "fingers": {"thumb": 0.7, "index": 0.8, "middle": 0.8, "ring": 0.8, "pinky": 0.8},
        "wrist_rotation": [0, 0, 0],
    },
    "F": {
        "description": "Thumb and index touching, others extended",
        "fingers": {"thumb": 0.5, "index": 0.6, "middle": 0.0, "ring": 0.0, "pinky": 0.0},
        "wrist_rotation": [0, 0, 0],
    },
    "G": {
        "description": "Index pointing sideways, thumb parallel",
        "fingers": {"thumb": 0.2, "index": 0.1, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, -0.7, 0],
    },
    "H": {
        "description": "Index and middle extended sideways",
        "fingers": {"thumb": 0.5, "index": 0.1, "middle": 0.1, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, -0.7, 0],
    },
    "I": {
        "description": "Pinky extended, others in fist",
        "fingers": {"thumb": 0.5, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
        "wrist_rotation": [0, 0, 0],
    },
    "J": {
        "description": "Like I, with a downward scoop motion",
        "fingers": {"thumb": 0.5, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
        "wrist_rotation": [0.3, 0, 0.3],
    },
    "K": {
        "description": "Index and middle up in V, thumb between them",
        "fingers": {"thumb": 0.3, "index": 0.0, "middle": 0.2, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "L": {
        "description": "Thumb and index form L shape",
        "fingers": {"thumb": 0.0, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "M": {
        "description": "Three fingers over thumb in fist",
        "fingers": {"thumb": 0.6, "index": 0.9, "middle": 0.9, "ring": 0.9, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "N": {
        "description": "Two fingers over thumb in fist",
        "fingers": {"thumb": 0.6, "index": 0.9, "middle": 0.9, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "O": {
        "description": "All fingers touch thumb in O shape",
        "fingers": {"thumb": 0.5, "index": 0.6, "middle": 0.6, "ring": 0.6, "pinky": 0.6},
        "wrist_rotation": [0, 0, 0],
    },
    "P": {
        "description": "Like K but pointing down",
        "fingers": {"thumb": 0.3, "index": 0.0, "middle": 0.2, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0.8, 0, 0],
    },
    "Q": {
        "description": "Like G but pointing down",
        "fingers": {"thumb": 0.2, "index": 0.1, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0.8, 0, 0],
    },
    "R": {
        "description": "Index and middle crossed",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "S": {
        "description": "Fist with thumb over fingers",
        "fingers": {"thumb": 0.5, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "T": {
        "description": "Thumb between index and middle",
        "fingers": {"thumb": 0.4, "index": 0.9, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "U": {
        "description": "Index and middle extended together",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "V": {
        "description": "Index and middle spread in V shape",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 0.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "W": {
        "description": "Index, middle, ring spread wide",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "X": {
        "description": "Index finger hooked",
        "fingers": {"thumb": 0.5, "index": 0.5, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0],
    },
    "Y": {
        "description": "Thumb and pinky extended (hang loose)",
        "fingers": {"thumb": 0.0, "index": 1.0, "middle": 1.0, "ring": 1.0, "pinky": 0.0},
        "wrist_rotation": [0, 0, 0],
    },
    "Z": {
        "description": "Index traces Z shape in air",
        "fingers": {"thumb": 0.5, "index": 0.0, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
        "wrist_rotation": [0, 0, 0.3],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Shared Application State
# ═══════════════════════════════════════════════════════════════════════════

# ── Components (initialized once at startup) ─────────────────────────────────
db: DatabaseManager = None          # type: ignore[assignment]
predictor: GesturePredictor = None  # type: ignore[assignment]
translator: AudioTranslator = None  # type: ignore[assignment]

# ── Auth ─────────────────────────────────────────────────────────────────────
from src.auth_models import UserManager
from src.auth_routes import auth_bp, init_auth
user_mgr: UserManager = None        # type: ignore[assignment]

# ── Single shared camera ─────────────────────────────────────────────────────
# (Cloud Inference: Camera is handled by the client browser)
cap = None
cap_lock = threading.Lock()

# ── Live prediction state ────────────────────────────────────────────────────
prediction_state = {
    "label": "—",
    "confidence": 0.0,
    "translation": "—",
    "mode": MODE_IDLE,
    "word_buffer_fill": 0,
}
pred_lock = threading.Lock()

# ── Recording state for "Teach Custom Sign" ──────────────────────────────────
rec_state = {
    "active": False,
    "label": "",
    "type": "Word",
    "frame_buf": [],
    "recordings": [],
    "status": "idle",
    "train_progress": "",
    "train_running": False,
}
rec_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
# Frame Processing (Cloud Inference)
# ═══════════════════════════════════════════════════════════════════════════


def _finish_recording():
    """
    Called (under rec_lock) when SEQUENCE_LENGTH frames have been captured.
    """
    buf = rec_state["frame_buf"]
    if len(buf) >= 15:
        if rec_state.get("type") == "Alphabet":
            rec_state["recordings"].append(buf.copy())
            n = len(rec_state["recordings"])
            rec_state["status"] = f"✅ Recording {n}/{config.TEACH_NUM_RECORDINGS} saved ({len(buf)} frames)."
        else:
            seq = pad_or_sample_sequence(buf, config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES)
            if not np.all(seq == 0):
                rec_state["recordings"].append(seq)
                n = len(rec_state["recordings"])
                rec_state["status"] = f"✅ Recording {n}/{config.TEACH_NUM_RECORDINGS} saved."
            else:
                rec_state["status"] = "Recording discarded (no hands detected)."
    else:
        rec_state["status"] = "Recording discarded (too few frames with hands detected)."

    rec_state["active"] = False
    rec_state["frame_buf"] = []


# ═══════════════════════════════════════════════════════════════════════════
# Flask App + Routes
# ═══════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32).hex())
app.permanent_session_lifetime = timedelta(hours=8)
app.register_blueprint(auth_bp)


# ── Page ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the multi-page web UI."""
    return render_template(
        "index.html",
        languages=list(config.SUPPORTED_LANGUAGES.keys()),
        num_recordings=config.TEACH_NUM_RECORDINGS,
        sequence_length=config.SEQUENCE_LENGTH,
    )


# ── Cloud Inference API ──────────────────────────────────────────────────────

@app.route("/api/process_frame", methods=["POST"])
def api_process_frame():
    """Receive base64 JPEG from client, run inference, return annotated JPEG and state JSON."""
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image"}), 400

    b64 = data["image"]
    if b64.startswith("data:image"):
        b64 = b64.split(",")[1]

    try:
        img_bytes = base64.b64decode(b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": "Bad image format"}), 400

    if frame is None:
        return jsonify({"error": "Bad image"}), 400

    # We do NOT mirror the frame here because the client browser already captures it mirrored!
    # Wait, the browser usually captures exactly what it sees. Let's let the frontend handle mirroring via CSS and flip it before drawing, OR we just mirror it here if needed.
    # We will just mirror it to keep MoE features consistent with how they were trained.
    frame = mirror_frame(frame)
    result, left_lm, right_lm, n_hands = predictor.update(frame)
    mode = predictor.current_mode

    if left_lm:
        draw_hand_landmarks(frame, left_lm, color=(50, 200, 255))
    if right_lm:
        draw_hand_landmarks(frame, right_lm, color=(0, 230, 110))

    draw_router_badge(frame, mode)

    if result:
        label, conf, _mode = result
        tx = translator.process_prediction(label, conf)
        draw_prediction_banner(frame, label, conf, tx)
        with pred_lock:
            prediction_state["label"] = label
            prediction_state["confidence"] = round(conf, 4)
            prediction_state["translation"] = tx

    with pred_lock:
        prediction_state["mode"] = mode
        prediction_state["word_buffer_fill"] = predictor.word_buffer_fill

    status_text = f"{mode}  |  Buffer: {predictor.word_buffer_fill}/{config.SEQUENCE_LENGTH}"
    draw_status_bar(frame, status_text)

    # Recording logic
    with rec_lock:
        if rec_state["active"]:
            if rec_state.get("type") == "Alphabet":
                active_lm = left_lm if left_lm is not None else right_lm
                if active_lm is not None:
                    from src.utils import extract_single_hand_features
                    feat = extract_single_hand_features(active_lm)
                    rec_state["frame_buf"].append(feat)
            else:
                if left_lm is not None or right_lm is not None:
                    from src.utils import extract_dual_hand_features
                    feat = extract_dual_hand_features(left_lm, right_lm)
                    rec_state["frame_buf"].append(feat)
            
            n = len(rec_state["frame_buf"])
            if n == 0:
                rec_state["status"] = "Waiting for hands in camera view…"
            else:
                rec_state["status"] = f"Recording ({n}/{config.SEQUENCE_LENGTH} frames)"

            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(frame, f"REC {n}/{config.SEQUENCE_LENGTH}",
                        (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

            if n >= config.SEQUENCE_LENGTH:
                _finish_recording()

    # Send the frame back. It is already mirrored once at the top (which makes it act like a mirror),
    # and the text was drawn forwards, so we leave it as is.
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    out_b64 = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode('utf-8')

    with pred_lock:
        pred_out = dict(prediction_state)
    with rec_lock:
        rec_out = {
            "recording_active": rec_state["active"],
            "recordings_count": len(rec_state["recordings"]),
            "rec_status": rec_state["status"],
            "train_running": rec_state["train_running"],
            "train_progress": rec_state["train_progress"],
        }
    
    state_out = {**pred_out, **rec_out}

    return jsonify({"image": out_b64, "state": state_out})


# ── Teach Custom Sign API ────────────────────────────────────────────────────

@app.route("/api/start_recording", methods=["POST"])
def api_start_recording():
    """Begin capturing frames for a new gesture recording."""
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip().upper()
    gesture_type = data.get("type", "Word")

    if not label:
        return jsonify({"error": "Enter a gesture label first."}), 400

    with rec_lock:
        if rec_state["active"]:
            return jsonify({"error": "A recording is already in progress."}), 400
        if len(rec_state["recordings"]) >= config.TEACH_NUM_RECORDINGS:
            return jsonify({"error": f"Already have {config.TEACH_NUM_RECORDINGS} recordings. Train the model."}), 400
        if rec_state["train_running"]:
            return jsonify({"error": "Training is in progress. Please wait."}), 400

        rec_state["label"] = label
        rec_state["type"] = gesture_type
        rec_state["frame_buf"] = []
        rec_state["active"] = True
        rec_state["status"] = f"Recording (0/{config.SEQUENCE_LENGTH} frames)"

    return jsonify({"ok": True, "message": f"Recording started for '{label}'."})


@app.route("/api/train", methods=["POST"])
def api_train():
    """Submit recorded samples for Admin review."""
    uid = session.get("user_id")
    if not uid:
        return jsonify({"error": "You must be logged in to submit training samples."}), 401

    with rec_lock:
        label = rec_state["label"]
        gesture_type = rec_state.get("type", "Word")
        recordings = list(rec_state["recordings"])
        if not label or len(recordings) < 5:
            return jsonify({"error": "Record at least 5 samples first."}), 400
        
        rec_state["train_progress"] = "Submitting for review..."

    try:
        # Create submission
        sub_id = user_mgr.create_submission(uid, label, gesture_type, len(recordings))
        if not sub_id:
            raise Exception("Failed to create submission record.")

        # Save arrays and link to submission
        save_dir = config.DATA_DIR / "pending_submissions" / str(sub_id)
        save_dir.mkdir(parents=True, exist_ok=True)

        sample_idx = 0
        for seq_or_frames in recordings:
            if gesture_type == "Alphabet":
                for frame in seq_or_frames:
                    path = str(save_dir / f"{sample_idx:04d}.npy")
                    np.save(path, frame)
                    with open(path, 'rb') as f:
                        file_data = f.read()
                    user_mgr.add_training_sample(sub_id, path, "npy", 1, sample_data=file_data)
                    sample_idx += 1
            else:
                path = str(save_dir / f"{sample_idx:04d}.npy")
                np.save(path, seq_or_frames)
                with open(path, 'rb') as f:
                    file_data = f.read()
                user_mgr.add_training_sample(sub_id, path, "npy", len(seq_or_frames), sample_data=file_data)
                sample_idx += 1

        user_mgr.log_activity(uid, "TRAINING_SUBMITTED", page="/teach", action=f"Submitted '{label}' ({len(recordings)} samples)", session_id=session.get("session_id"))

        with rec_lock:
            rec_state["train_progress"] = f"✅ '{label}' submitted successfully for admin review!"
            rec_state["recordings"] = []
            rec_state["status"] = "Submission pending review."
            
        return jsonify({"ok": True, "message": "Submission sent for review."})

    except Exception as exc:
        logger.error("Submission failed: %s", exc, exc_info=True)
        with rec_lock:
            rec_state["train_progress"] = f"❌ Error: {exc}"
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all recording state."""
    with rec_lock:
        if rec_state["train_running"]:
            return jsonify({"error": "Cannot reset while training is running."}), 400
        rec_state["active"] = False
        rec_state["label"] = ""
        rec_state["frame_buf"] = []
        rec_state["recordings"] = []
        rec_state["status"] = "idle"
        rec_state["train_progress"] = ""
    return jsonify({"ok": True})


@app.route("/api/set_language", methods=["POST"])
def api_set_language():
    """Change the target translation language."""
    data = request.get_json(silent=True) or {}
    lang_name = data.get("language", "English")
    code = config.SUPPORTED_LANGUAGES.get(lang_name, "en")
    translator.target_language = code
    return jsonify({"ok": True, "language": lang_name, "code": code})


@app.route("/api/set_tts", methods=["POST"])
def api_set_tts():
    """Toggle TTS on/off."""
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", True)
    translator.set_tts_enabled(enabled)
    return jsonify({"ok": True, "tts_enabled": enabled})


@app.route("/api/history")
def api_history():
    """Return recent prediction history from the database."""
    limit = request.args.get("limit", 200, type=int)
    rows = db.get_history_summary(limit=limit)
    return jsonify([dict(r) for r in rows])


@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    """Clear all prediction history."""
    db.clear_history()
    return jsonify({"ok": True})


@app.route("/api/stats")
def api_stats():
    """Return database statistics."""
    return jsonify(db.get_stats())


@app.route("/api/gestures")
def api_gestures():
    """Return all known gestures from the database."""
    return jsonify(db.get_all_gestures())


@app.route("/api/hand_poses")
def api_hand_poses():
    """Return ASL hand pose data for the 3D hand model."""
    return jsonify(ASL_HAND_POSES)


@app.route("/api/known_signs")
def api_known_signs():
    """Return lists of known alphabets and words the model can recognize."""
    return jsonify({
        "alphabets": predictor.known_alphabets if predictor else [],
        "words": predictor.known_words if predictor else [],
        "alphabet_ready": predictor.is_alphabet_ready if predictor else False,
        "word_ready": predictor.is_word_ready if predictor else False,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════════

def init_components():
    """Initialize all ML components and the camera. Called once at startup."""
    global db, predictor, translator, cap, user_mgr

    logger.info("=" * 60)
    logger.info("Starting %s (Web Edition — 4-Page UI)", config.GUI_TITLE)
    logger.info("=" * 60)

    # Database
    logger.info("Initializing database…")
    db = DatabaseManager()
    logger.info("DB stats: %s", db.get_stats())

    # Auth / User management (creates tables + seeds admin from env vars)
    logger.info("Initializing authentication system…")
    user_mgr = UserManager()
    init_auth(user_mgr)
    logger.info("Auth system ready.")

    # MoE Predictor
    logger.info("Initializing MoE Predictor…")
    predictor = GesturePredictor()

    # Translator + TTS
    logger.info("Initializing translator…")
    try:
        translator = AudioTranslator(
            db_manager=db,
            target_language_code=config.SUPPORTED_LANGUAGES[config.DEFAULT_LANGUAGE],
            tts_enabled=config.TTS_ENABLED_DEFAULT,
        )
        translator.wait_tts_ready(timeout=4.0)
    except Exception as exc:
        logger.warning("Translator init warning (TTS may be disabled): %s", exc)
        translator = AudioTranslator(db_manager=db, tts_enabled=False)

    # Camera
    logger.info("Camera handling delegated to client browser (Cloud Inference).")
    logger.info("All components ready.")


if __name__ == "__main__":
    init_components()

    print("\n" + "=" * 60)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")

    try:
        app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        predictor.close()
        logger.info("Application exited.")
