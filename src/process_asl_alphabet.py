"""
src/process_asl_alphabet.py — ASL Alphabet Dataset Processor (Kaggle Dataset)
Reads images from grassknoted/asl-alphabet dataset,
extracts 60-D single-hand feature vectors via MediaPipe,
and populates data/processed/alphabets/{A..Z}/.
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
    logger.error("mediapipe not installed.")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

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
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_img)
    if not result.hand_landmarks:
        return None
    return extract_single_hand_features(result.hand_landmarks[0])

def process_asl_alphabet(
    asl_dataset_dir: Path,
    out_dir: Path = config.PROCESSED_ALPHABETS_DIR,
    samples_per_class: int = 40,
) -> dict[str, int]:
    landmarker = _build_landmarker()
    if landmarker is None:
        raise RuntimeError("Cannot build MediaPipe HandLandmarker.")

    # Locate inner asl_alphabet_train directory
    train_dir = asl_dataset_dir
    if (asl_dataset_dir / "asl_alphabet_train" / "asl_alphabet_train").exists():
        train_dir = asl_dataset_dir / "asl_alphabet_train" / "asl_alphabet_train"
    elif (asl_dataset_dir / "asl_alphabet_train").exists():
        train_dir = asl_dataset_dir / "asl_alphabet_train"

    label_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        logger.warning("No label directories in %s", train_dir)
        return {}

    results: dict[str, int] = {}

    for label_dir in label_dirs:
        raw_label = label_dir.name
        label = raw_label.upper()
        if label in ("SPACE", "DEL", "NOTHING"):
            continue

        out_cls = out_dir / label
        out_cls.mkdir(parents=True, exist_ok=True)

        images = sorted(
            [f for f in label_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        )[:samples_per_class * 2]  # fetch extra to guarantee samples_per_class valid hands

        saved = 0
        for img_path in images:
            if saved >= samples_per_class:
                break
            feat = _process_image(img_path, landmarker)
            if feat is not None:
                np.save(str(out_cls / f"{saved:04d}.npy"), feat)
                saved += 1

        logger.info("Alphabet '%s': %d / %d samples saved", label, saved, samples_per_class)
        results[label] = saved

    landmarker.close()
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger.info("Locating ASL Alphabet dataset...")
    local_path = Path(r"C:\Users\mohit\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1")
    if local_path.exists():
        path = local_path
    else:
        import kagglehub
        path = Path(kagglehub.dataset_download("grassknoted/asl-alphabet"))
    res = process_asl_alphabet(path)
    total = sum(res.values())
    print(f"\n--- ASL Alphabet Processing Summary ---")
    for lbl, cnt in sorted(res.items()):
        print(f"  {lbl:4s}: {cnt:4d} samples")
    print(f"\nTotal: {total} alphabet feature vectors saved across {len(res)} classes.")
