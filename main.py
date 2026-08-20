"""
main.py — Application Entry Point (MoE Edition)
Validates the environment, initialises all components, and launches the GUI.
"""

import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

# ── Logging setup (before any project imports) ────────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "app.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")

import config
from src.database_manager import DatabaseManager
from src.predict import GesturePredictor
from src.translator import AudioTranslator
from src.gui import App


# ─────────────────────────────────────────────────────────────────────────────
def _check_task_file() -> list[str]:
    """Auto-download hand_landmarker.task if missing. Returns warning strings."""
    warnings: list[str] = []
    if config.TASK_PATH.exists():
        return warnings

    logger.info("hand_landmarker.task not found — attempting auto-download...")
    try:
        import urllib.request
        logger.info("Downloading from: %s", config.TASK_URL)
        urllib.request.urlretrieve(config.TASK_URL, str(config.TASK_PATH))
        logger.info("Download complete: %s", config.TASK_PATH)
    except Exception as exc:                # noqa: BLE001
        warnings.append(
            f"[WARNING] Could not auto-download hand_landmarker.task:\n  {exc}\n\n"
            f"Please download it manually from:\n  {config.TASK_URL}\n"
            f"and place it at:\n  {config.TASK_PATH}"
        )
    return warnings


def _check_packages() -> list[str]:
    """Warn about any missing Python packages."""
    required = {
        "cv2":             "opencv-python",
        "mediapipe":       "mediapipe",
        "tensorflow":      "tf-nightly",
        "sklearn":         "scikit-learn",
        "joblib":          "joblib",
        "deep_translator": "deep-translator",
        "pyttsx3":         "pyttsx3",
        "PIL":             "Pillow",
        "pandas":          "pandas",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)

    if missing:
        return [
            "[WARNING] Missing packages:\n  " + "\n  ".join(missing) +
            "\n\nInstall with:\n  pip install " + " ".join(missing)
        ]
    return []


def _validate_environment() -> list[str]:
    warnings: list[str] = []
    warnings += _check_task_file()
    warnings += _check_packages()

    # Non-fatal: models may not exist yet (train first)
    if not config.ALPHABET_MODEL_PATH.exists():
        logger.info(
            f"[INFO] Alphabet Expert not found at {config.ALPHABET_MODEL_PATH}. "
            "Run:  python -m src.train_moe --expert alphabet"
        )
    if not config.WORD_MODEL_PATH.exists():
        logger.info(
            f"[INFO] Word Expert not found at {config.WORD_MODEL_PATH}. "
            "Run:  python -m src.train_moe --expert word"
        )
    return warnings


def _fatal(msg: str) -> None:
    _t = tk.Tk(); _t.withdraw()
    messagebox.showerror("Fatal Error", msg)
    _t.destroy()
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 65)
    logger.info("Starting %s", config.GUI_TITLE)
    logger.info("BASE_DIR : %s", config.BASE_DIR)
    logger.info("DB_PATH  : %s", config.DB_PATH)
    logger.info("=" * 65)

    # Ensure all directories exist
    for d in (config.DATA_DIR, config.MODEL_DIR, config.LOGS_DIR,
              config.RAW_ALPHABETS_DIR, config.RAW_WORDS_DIR,
              config.PROCESSED_ALPHABETS_DIR, config.PROCESSED_WORDS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Environment checks
    warnings = _validate_environment()
    if warnings:
        logger.warning("Setup Warnings:\n\n" + "\n\n".join(warnings) + 
                       "\n\nThe app will launch with limited functionality until setup is complete.")

    # ── Database ──────────────────────────────────────────────────────────────
    logger.info("Initialising database…")
    try:
        db = DatabaseManager(config.DB_PATH)
        stats = db.get_stats()
        logger.info("DB stats: %s", stats)
    except Exception as exc:
        logger.critical("Database init failed: %s", exc)
        _fatal(f"Database error:\n{exc}")
        return

    # ── MoE Predictor ─────────────────────────────────────────────────────────
    logger.info("Initialising MoE Predictor…")
    try:
        predictor = GesturePredictor()
    except Exception as exc:
        logger.error("GesturePredictor init error: %s", exc)
        _fatal(f"Predictor error:\n{exc}\n\nThe app cannot start.")
        return

    # ── Translator + TTS ──────────────────────────────────────────────────────
    logger.info("Initialising translator…")
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

    # ── Launch GUI ────────────────────────────────────────────────────────────
    logger.info("Launching GUI…")
    try:
        app = App(predictor=predictor, db=db, translator=translator)
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as exc:                # noqa: BLE001
        logger.critical("Unhandled exception: %s", exc, exc_info=True)
    finally:
        logger.info("Application exiting.")
        if predictor:
            predictor.close()


if __name__ == "__main__":
    main()
