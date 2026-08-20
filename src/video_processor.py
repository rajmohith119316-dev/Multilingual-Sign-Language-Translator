"""
src/video_processor.py — Bulk Video-to-Sequence Converter
Reads raw .mp4 files from data/raw_videos/{LABEL}/, extracts MediaPipe hand
landmarks for 30 frames, applies wrist-relative normalisation, and saves
processed .npy arrays under data/sequences/{LABEL}/{idx}.npy.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Resolve project root so we can import config
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import extract_normalized_features, pad_or_sample_sequence

logger = logging.getLogger(__name__)

# ─── MediaPipe Tasks (new Tasks API) ─────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision.hand_landmarker import (
        HandLandmarker,
        HandLandmarkerOptions,
        HandLandmarkerResult,
    )
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    logger.error("MediaPipe is not installed. Run: pip install mediapipe")


# ─────────────────────────────────────────────────────────────────────────────
def _build_landmarker(task_path: Path) -> Optional["HandLandmarker"]:
    """Construct and return a MediaPipe HandLandmarker in IMAGE run mode."""
    if not _MP_AVAILABLE:
        return None
    if not task_path.exists():
        logger.error(
            "MediaPipe task file not found: %s\n"
            "Download from: https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            task_path,
        )
        return None

    base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_MIN_HAND_DETECTION_CONF,
        min_hand_presence_confidence=config.MP_MIN_HAND_PRESENCE_CONF,
        min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
    )
    return HandLandmarker.create_from_options(options)


# ─────────────────────────────────────────────────────────────────────────────
def _extract_landmarks_from_frame(
    landmarker: "HandLandmarker",
    bgr_frame: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Run MediaPipe on a single BGR frame and return the 60-feature vector,
    or None if no hand is detected.
    """
    # Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result: HandLandmarkerResult = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None  # No hand detected in this frame

    # Use first detected hand
    return extract_normalized_features(result.hand_landmarks[0])


# ─────────────────────────────────────────────────────────────────────────────
def process_video(
    video_path: Path,
    landmarker: "HandLandmarker",
    sequence_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.NUM_FEATURES,
) -> Optional[np.ndarray]:
    """
    Extract a (sequence_length, num_features) array from a single video file.

    Algorithm
    ---------
    1. Evenly sample `sequence_length` frame indices from the video.
    2. For each sampled frame, detect hand landmarks and extract features.
    3. Missing detections are represented as zero-vectors.
    4. Return a padded/sampled array of shape (sequence_length, num_features).

    Returns None if the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None

    # Evenly spaced sample indices
    if total_frames >= sequence_length:
        indices = set(
            np.round(np.linspace(0, total_frames - 1, sequence_length)).astype(int).tolist()
        )
    else:
        indices = set(range(total_frames))

    feature_vectors: list[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices:
            feat = _extract_landmarks_from_frame(landmarker, frame)
            if feat is not None:
                feature_vectors.append(feat)
            else:
                # Placeholder zero for missing detection
                feature_vectors.append(np.zeros(num_features, dtype=np.float32))

        frame_idx += 1

    cap.release()

    if not feature_vectors:
        logger.warning("No frames extracted from %s", video_path)
        return None

    return pad_or_sample_sequence(feature_vectors, sequence_length, num_features)


# ─────────────────────────────────────────────────────────────────────────────
def process_label_directory(
    label_dir: Path,
    label: str,
    landmarker: "HandLandmarker",
    output_root: Path = config.SEQUENCES_DIR,
    sequence_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.NUM_FEATURES,
) -> int:
    """
    Process all .mp4 files in a label directory and save .npy sequences.

    Returns the number of successfully processed videos.
    """
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(label_dir.glob("*.mp4"))
    if not mp4_files:
        logger.warning("No .mp4 files found in %s", label_dir)
        return 0

    saved = 0
    for idx, video_path in enumerate(mp4_files):
        sequence = process_video(video_path, landmarker, sequence_length, num_features)
        if sequence is not None:
            save_path = output_dir / f"{idx:04d}.npy"
            np.save(str(save_path), sequence)
            saved += 1
            logger.debug("Saved sequence %s → %s", video_path.name, save_path.name)
        else:
            logger.warning("Skipped (no usable frames): %s", video_path.name)

    logger.info("Label '%s': %d / %d videos processed.", label, saved, len(mp4_files))
    return saved


# ─────────────────────────────────────────────────────────────────────────────
def process_all_videos(
    raw_videos_root: Path = config.RAW_VIDEOS_DIR,
    sequences_root: Path  = config.SEQUENCES_DIR,
    task_path: Path       = config.TASK_PATH,
) -> dict[str, int]:
    """
    Main entry point: process all label subdirectories under raw_videos_root.

    Expected folder structure
    -------------------------
    data/raw_videos/
        hello/
            001.mp4
            002.mp4
        thanks/
            001.mp4

    Returns a dict mapping label → number of sequences saved.
    """
    landmarker = _build_landmarker(task_path)
    if landmarker is None:
        raise RuntimeError("Could not initialise MediaPipe HandLandmarker.")

    label_dirs = [d for d in raw_videos_root.iterdir() if d.is_dir()]
    if not label_dirs:
        logger.warning("No label subdirectories found in %s", raw_videos_root)
        return {}

    results: dict[str, int] = {}
    for label_dir in sorted(label_dirs):
        label = label_dir.name.upper()
        logger.info("Processing label: %s", label)
        count = process_label_directory(
            label_dir, label, landmarker, sequences_root
        )
        results[label] = count

    landmarker.close()
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    summary = process_all_videos()
    print("\n─── Video Processing Summary ───")
    for label, count in summary.items():
        print(f"  {label:20s}: {count} sequences saved")
    total = sum(summary.values())
    print(f"\nTotal sequences saved: {total}")
