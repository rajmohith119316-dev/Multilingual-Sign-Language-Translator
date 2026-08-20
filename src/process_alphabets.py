"""
src/process_alphabets.py — Static Alphabet Dataset Processor
Reads images from data/raw/alphabets/{LABEL}/ using MediaPipe static_image_mode,
extracts 60-feature wrist-relative vectors, and saves as .npy files.

Expected input layout
─────────────────────
data/raw/alphabets/
    A/  img001.jpg  img002.jpg  ...
    B/  img001.jpg  ...
    ...
    Z/  ...

Output layout
─────────────
data/processed/alphabets/
    A/  0000.npy  0001.npy  ...
    B/  ...
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import extract_single_hand_features

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker, HandLandmarkerOptions,
    )
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.error("mediapipe not installed. Run: pip install mediapipe")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_landmarker() -> "HandLandmarker | None":
    if not _MP_OK or not config.TASK_PATH.exists():
        logger.error("hand_landmarker.task missing: %s", config.TASK_PATH)
        return None
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.TASK_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return HandLandmarker.create_from_options(options)


def _process_image(path: Path, landmarker: "HandLandmarker") -> np.ndarray | None:
    """Extract 60-D feature vector from one image. Returns None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning("Cannot read image: %s", path)
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)

    if not result.hand_landmarks:
        return None   # No hand found

    feat = extract_single_hand_features(result.hand_landmarks[0])
    return feat   # (60,) or None


def process_alphabets(
    raw_dir: Path      = config.RAW_ALPHABETS_DIR,
    out_dir: Path      = config.PROCESSED_ALPHABETS_DIR,
    max_per_class: int = 1000,
) -> dict[str, int]:
    """
    Process all alphabet image classes.

    Returns dict mapping label → number of .npy files saved.
    """
    landmarker = _build_landmarker()
    if landmarker is None:
        raise RuntimeError("Cannot build MediaPipe HandLandmarker.")

    label_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        logger.warning("No label directories in %s", raw_dir)
        return {}

    results: dict[str, int] = {}

    for label_dir in label_dirs:
        label   = label_dir.name.upper()
        out_cls = out_dir / label
        out_cls.mkdir(parents=True, exist_ok=True)

        images = sorted(
            [f for f in label_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        )[:max_per_class]

        if not images:
            logger.warning("No images in %s", label_dir)
            results[label] = 0
            continue

        saved = 0
        for img_path in images:
            feat = _process_image(img_path, landmarker)
            if feat is not None:
                np.save(str(out_cls / f"{saved:04d}.npy"), feat)
                saved += 1

        logger.info("Alphabet '%s': %d / %d images → .npy", label, saved, len(images))
        results[label] = saved

    landmarker.close()
    return results


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    summary = process_alphabets()
    print("\n── Alphabet Processing Summary ──")
    total = 0
    for lbl, cnt in sorted(summary.items()):
        print(f"  {lbl:4s}: {cnt:5d} samples")
        total += cnt
    print(f"\n  Total: {total} feature vectors saved.")
