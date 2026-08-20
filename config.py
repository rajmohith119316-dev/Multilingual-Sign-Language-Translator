"""
config.py — Centralized Configuration for Multilingual Real-Time Sign Language Translator
Mixture-of-Experts (MoE) Edition — Python 3.11
"""

import os
from pathlib import Path

# ─── Base Directories ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
LOGS_DIR = BASE_DIR / "logs"

# ─── Data Directories ─────────────────────────────────────────────────────────
DATA_DIR            = BASE_DIR / "data"
RAW_DIR             = DATA_DIR / "raw"
RAW_ALPHABETS_DIR   = RAW_DIR / "alphabets"      # Static images A–Z
RAW_WORDS_DIR       = RAW_DIR / "words"           # Dynamic .mp4 videos per word

PROCESSED_DIR           = DATA_DIR / "processed"
PROCESSED_ALPHABETS_DIR = PROCESSED_DIR / "alphabets"   # .npy (60,)
PROCESSED_WORDS_DIR     = PROCESSED_DIR / "words"       # .npy (30, 123)

TEACH_TEMP_DIR = DATA_DIR / "teach_temp"

# ─── Model / Asset Paths ──────────────────────────────────────────────────────
MODEL_DIR = BASE_DIR / "models"

# MediaPipe task asset
TASK_PATH = BASE_DIR / "hand_landmarker.task"
TASK_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Expert model artefacts
ALPHABET_MODEL_PATH   = MODEL_DIR / "alphabet_expert.h5"
ALPHABET_ENCODER_PATH = MODEL_DIR / "alphabet_encoder.pkl"

WORD_MODEL_PATH   = MODEL_DIR / "word_expert.h5"
WORD_ENCODER_PATH = MODEL_DIR / "word_encoder.pkl"

# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = DATA_DIR / "sign_language_app.db"

# ─── Feature Dimensions ───────────────────────────────────────────────────────
# Single-hand alphabet: 20 landmarks × 3 = 60 wrist-relative features
ALPHABET_NUM_FEATURES = 60

# Dual-hand word: left(60) + right(60) + inter-wrist(3) = 123
WORD_NUM_FEATURES     = 123

# ─── Sequence Parameters (Word Expert) ───────────────────────────────────────
SEQUENCE_LENGTH = 30         # frames per word gesture sample

# ─── Inference / Stability Filter ─────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80  # minimum accepted confidence
STABILITY_FRAMES     = 5     # consecutive identical predictions needed

# ─── Training Hyperparameters ─────────────────────────────────────────────────
ALPHABET_EPOCHS   = 60
WORD_EPOCHS       = 50
FINETUNE_EPOCHS   = 20
BATCH_SIZE        = 32
VALIDATION_SPLIT  = 0.15
LEARNING_RATE     = 1e-3
FINETUNE_LR       = 1e-4
MIN_SAMPLES_CLASS = 3        # skip class if fewer samples

# ─── MediaPipe Settings ───────────────────────────────────────────────────────
MP_NUM_HANDS               = 2
MP_MIN_HAND_DETECTION_CONF = 0.5
MP_MIN_HAND_PRESENCE_CONF  = 0.5
MP_MIN_TRACKING_CONF       = 0.5

# ─── Continual Learning ───────────────────────────────────────────────────────
TEACH_NUM_RECORDINGS  = 10   # recordings per teach session
TEACH_COUNTDOWN_SECS  = 3

# ─── Supported Languages ──────────────────────────────────────────────────────
SUPPORTED_LANGUAGES: dict[str, str] = {
    "English":   "en",
    "Kannada":   "kn",
    "Hindi":     "hi",
    "Tamil":     "ta",
    "Telugu":    "te",
    "Malayalam": "ml",
    "French":    "fr",
    "German":    "de",
    "Spanish":   "es",
}
DEFAULT_LANGUAGE = "English"

# ─── TTS ──────────────────────────────────────────────────────────────────────
TTS_ENABLED_DEFAULT  = True
TTS_COOLDOWN_SECONDS = 3.0
TTS_SPEECH_RATE      = 150

# ─── Camera / GUI ─────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
GUI_TITLE     = "Multilingual Real-Time Sign Language Translator — MoE Edition"
GUI_THEME     = "clam"

# ─── Legacy / Backwards Compatibility Aliases ───────────────────────────────
RAW_VIDEOS_DIR = RAW_WORDS_DIR
SEQUENCES_DIR  = PROCESSED_WORDS_DIR
NUM_FEATURES   = WORD_NUM_FEATURES
MODEL_PATH     = WORD_MODEL_PATH
LABEL_ENC_PATH = WORD_ENCODER_PATH
EPOCHS_BASE    = WORD_EPOCHS

# ─── Static alphabet labels (for reference) ───────────────────────────────────
ALPHABET_LABELS: list[str] = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

# ─── Auto-create directories on import ────────────────────────────────────────
for _d in (
    DATA_DIR, RAW_DIR, RAW_ALPHABETS_DIR, RAW_WORDS_DIR,
    RAW_VIDEOS_DIR, SEQUENCES_DIR,
    PROCESSED_DIR, PROCESSED_ALPHABETS_DIR, PROCESSED_WORDS_DIR,
    MODEL_DIR, LOGS_DIR, TEACH_TEMP_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

