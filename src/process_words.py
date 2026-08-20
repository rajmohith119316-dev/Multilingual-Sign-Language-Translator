"""
src/process_words.py — Dynamic Word Video Processor (Dual-Hand, 123-D)
Reads .mp4 files from data/raw/words/{LABEL}/ via MediaPipe,
extracts 123-feature dual-hand vectors per frame,
standardises to SEQUENCE_LENGTH=30 frames, and saves (30,123) .npy files.

Expected input layout
─────────────────────
data/raw/words/
    hello/   001.mp4  002.mp4  ...
    thanks/  001.mp4  ...

Output layout
─────────────
data/processed/words/
    hello/   0000.npy  0001.npy  ...
    thanks/  ...
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import extract_dual_hand_features, pad_or_sample_sequence

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult,
    )
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.error("mediapipe not installed.")


# ─────────────────────────────────────────────────────────────────────────────

def _build_landmarker() -> Optional["HandLandmarker"]:
    if not _MP_OK or not config.TASK_PATH.exists():
        logger.error("hand_landmarker.task missing: %s", config.TASK_PATH)
        return None
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.TASK_PATH)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_MIN_HAND_DETECTION_CONF,
        min_hand_presence_confidence=config.MP_MIN_HAND_PRESENCE_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
    )
    return HandLandmarker.create_from_options(options)


def _detect_hands(
    landmarker: "HandLandmarker",
    bgr_frame: np.ndarray,
) -> tuple[Optional[list], Optional[list]]:
    """
    Run detection on a BGR frame.

    Returns (left_landmarks, right_landmarks).
    Each is a list of NormalizedLandmark or None.
    MediaPipe assigns handedness; we decode it here.
    """
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result: HandLandmarkerResult = landmarker.detect(mp_img)

    left_lm  = None
    right_lm = None

    for i, hand_landmarks in enumerate(result.hand_landmarks):
        if i >= len(result.handedness):
            break
        handedness_label = result.handedness[i][0].category_name.lower()
        # MediaPipe uses mirrored image convention — we swap for natural view
        if handedness_label == "left":
            right_lm = hand_landmarks
        else:
            left_lm = hand_landmarks

    return left_lm, right_lm


def _video_to_sequence(
    video_path: Path,
    landmarker: "HandLandmarker",
    seq_len: int      = config.SEQUENCE_LENGTH,
    num_feat: int     = config.WORD_NUM_FEATURES,
) -> Optional[np.ndarray]:
    """
    Extract a (seq_len, num_feat) array from one .mp4 file.

    Returns None if the video cannot be opened or has no usable frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open: %s", video_path)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    # Determine sample indices (evenly spaced)
    if total >= seq_len:
        sample_set = set(
            np.round(np.linspace(0, total - 1, seq_len)).astype(int).tolist()
        )
    else:
        sample_set = set(range(total))

    feats: list[np.ndarray] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in sample_set:
            left_lm, right_lm = _detect_hands(landmarker, frame)
            feat = extract_dual_hand_features(left_lm, right_lm)  # (123,)
            feats.append(feat)
        idx += 1

    cap.release()

    if not feats:
        return None

    return pad_or_sample_sequence(feats, seq_len, num_feat)   # (30, 123)


def process_words(
    raw_dir: Path  = config.RAW_WORDS_DIR,
    out_dir: Path  = config.PROCESSED_WORDS_DIR,
    seq_len: int   = config.SEQUENCE_LENGTH,
    num_feat: int  = config.WORD_NUM_FEATURES,
) -> dict[str, int]:
    """
    Process all word video classes.

    Returns dict mapping label → number of .npy sequences saved.
    """
    landmarker = _build_landmarker()
    if landmarker is None:
        raise RuntimeError("Cannot build MediaPipe HandLandmarker.")

    label_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        logger.warning("No label dirs in %s", raw_dir)
        return {}

    results: dict[str, int] = {}

    for label_dir in label_dirs:
        label   = label_dir.name.upper()
        out_cls = out_dir / label
        out_cls.mkdir(parents=True, exist_ok=True)

        videos = sorted(label_dir.glob("*.mp4"))
        if not videos:
            logger.warning("No .mp4 files in %s", label_dir)
            results[label] = 0
            continue

        saved = 0
        for vid_path in videos:
            seq = _video_to_sequence(vid_path, landmarker, seq_len, num_feat)
            if seq is not None:
                np.save(str(out_cls / f"{saved:04d}.npy"), seq)
                saved += 1
            else:
                logger.debug("Skipped (no data): %s", vid_path.name)

        logger.info("Word '%s': %d / %d videos → .npy", label, saved, len(videos))
        results[label] = saved

    landmarker.close()
    return results


# ─── CLI entry ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    summary = process_words()
    print("\n--- Word Processing Summary ---")
    total = 0
    for lbl, cnt in sorted(summary.items()):
        print(f"  {lbl:20s}: {cnt:4d} sequences")
        total += cnt
    print(f"\n  Total: {total} sequences saved.")
