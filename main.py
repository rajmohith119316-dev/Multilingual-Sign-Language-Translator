"""
main.py — Application Entry Point (MoE Edition, Flask Web UI)

Validates the environment, then starts the localhost Flask server.
The camera never opens a cv2.imshow / Tkinter window: frames are streamed
to the browser as MJPEG until you press Ctrl+C in this terminal.
"""

import logging
import sys
from pathlib import Path

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
import web_app


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
        "flask":           "Flask",
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


def main() -> None:
    logger.info("=" * 65)
    logger.info("Starting %s (Web Edition)", config.GUI_TITLE)
    logger.info("BASE_DIR : %s", config.BASE_DIR)
    logger.info("DATABASE : %s", config.DATABASE_URL)
    logger.info("=" * 65)

    for d in (config.DATA_DIR, config.MODEL_DIR, config.LOGS_DIR,
              config.RAW_ALPHABETS_DIR, config.RAW_WORDS_DIR,
              config.PROCESSED_ALPHABETS_DIR, config.PROCESSED_WORDS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    try:
        from src.cloud_sync import sync_active_models_from_db
        sync_active_models_from_db()
    except Exception as e:
        logger.error(f"Failed to sync models from cloud: {e}")

    warnings = _validate_environment()
    if warnings:
        logger.warning(
            "Setup Warnings:\n\n" + "\n\n".join(warnings)
            + "\n\nThe app will launch with limited functionality until setup is complete."
        )

    # Load models. (Cloud Inference mode: camera is handled by the client browser)
    web_app.init_components()

    print("\n" + "=" * 60)
    print("  Open http://127.0.0.1:5000 in your browser")
    print("  Press Ctrl+C in this terminal to stop")
    print("=" * 60 + "\n")

    try:
        web_app.app.run(
            host="127.0.0.1",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        logger.info("Application exiting.")
        if web_app.predictor:
            web_app.predictor.close()


if __name__ == "__main__":
    main()
