"""
src/utils.py — Feature Engineering & OpenCV Drawing Utilities
MoE Edition: supports both single-hand (60-D) and dual-hand (123-D) feature extraction.

Normalization Math
──────────────────
Single-hand (Alphabet Expert):
  • Subtract wrist (Landmark 0) from landmarks 1–20.
  • L2-normalise for scale invariance.
  • Output: (60,) float32 vector.

Dual-hand (Word Expert):
  • Left hand:  60 wrist-relative features (same as above).
  • Right hand: 60 wrist-relative features (same as above).
  • Inter-wrist: [x_left0 - x_right0, y_left0 - y_right0, z_left0 - z_right0]
  • Output: (123,) float32 vector per frame.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ─── MediaPipe hand skeleton connections (21-landmark spec) ───────────────────
_HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11),(11, 12),
    (0, 13), (13, 14),(14, 15),(15, 16),
    (0, 17), (17, 18),(18, 19),(19, 20),
    (5, 9),  (9, 13), (13, 17),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _hand_to_relative_vector(hand_landmarks) -> Optional[np.ndarray]:
    """
    Convert 21 MediaPipe NormalizedLandmarks into a wrist-relative,
    L2-normalised (60,) feature vector.

    Returns None if landmarks are missing or malformed.
    """
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return None
    try:
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
            dtype=np.float32,
        )  # (21, 3)
        wrist    = coords[0]                 # origin
        relative = coords[1:] - wrist        # (20, 3)  — landmarks 1–20
        norm = np.linalg.norm(relative)
        if norm > 1e-6:
            relative /= norm
        return relative.flatten()            # (60,)
    except Exception as exc:                 # noqa: BLE001
        logger.warning("_hand_to_relative_vector failed: %s", exc)
        return None


def extract_single_hand_features(hand_landmarks) -> Optional[np.ndarray]:
    """
    Single-hand 60-feature vector for the Alphabet Expert.

    Parameters
    ----------
    hand_landmarks : list[NormalizedLandmark]  (length 21)

    Returns
    -------
    np.ndarray shape (60,) or None.
    """
    return _hand_to_relative_vector(hand_landmarks)


# Backwards compatibility alias for legacy scripts
extract_normalized_features = extract_single_hand_features


def extract_dual_hand_features(
    left_landmarks,
    right_landmarks,
) -> np.ndarray:
    """
    Dual-hand 123-feature vector for the Word Expert.

    Components
    ----------
    [0:60]  — left-hand wrist-relative features  (zeros if absent)
    [60:120]— right-hand wrist-relative features (zeros if absent)
    [120:123]— inter-wrist distance [Δx, Δy, Δz] (zeros if either absent)

    Parameters
    ----------
    left_landmarks  : list[NormalizedLandmark] or None
    right_landmarks : list[NormalizedLandmark] or None

    Returns
    -------
    np.ndarray shape (123,)
    """
    left_feat  = _hand_to_relative_vector(left_landmarks)
    right_feat = _hand_to_relative_vector(right_landmarks)

    left_vec  = left_feat  if left_feat  is not None else np.zeros(60, dtype=np.float32)
    right_vec = right_feat if right_feat is not None else np.zeros(60, dtype=np.float32)

    # Inter-wrist distance (raw, not normalised — encodes spatial relationship)
    if (left_landmarks is not None and len(left_landmarks) >= 21 and
            right_landmarks is not None and len(right_landmarks) >= 21):
        lw = np.array([left_landmarks[0].x,  left_landmarks[0].y,  left_landmarks[0].z], dtype=np.float32)
        rw = np.array([right_landmarks[0].x, right_landmarks[0].y, right_landmarks[0].z], dtype=np.float32)
        inter = lw - rw   # (3,)
    else:
        inter = np.zeros(3, dtype=np.float32)

    return np.concatenate([left_vec, right_vec, inter])   # (123,)


# ─────────────────────────────────────────────────────────────────────────────
# Sequence Padding / Sampling
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_sample_sequence(
    frames: list[np.ndarray],
    target_length: int,
    num_features: int,
) -> np.ndarray:
    """
    Normalise a variable-length frame list to exactly `target_length`.

    - Fewer frames: zero-pad at the end.
    - More frames:  evenly subsample.

    Returns np.ndarray of shape (target_length, num_features).
    """
    n = len(frames)
    if n == 0:
        return np.zeros((target_length, num_features), dtype=np.float32)

    if n < target_length:
        pad = [np.zeros(num_features, dtype=np.float32)] * (target_length - n)
        frames = frames + pad
    elif n > target_length:
        indices = np.round(np.linspace(0, n - 1, target_length)).astype(int)
        frames  = [frames[i] for i in indices]

    return np.array(frames, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV Drawing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def draw_hand_landmarks(
    frame: np.ndarray,
    hand_landmarks,
    color: tuple[int, int, int] = (0, 220, 120),
    line_color: tuple[int, int, int] = (255, 255, 255),
    radius: int = 4,
    thickness: int = 2,
) -> None:
    """Draw skeleton and joint dots for one hand (in-place)."""
    if hand_landmarks is None or len(hand_landmarks) < 21:
        return
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for s, e in _HAND_CONNECTIONS:
        if s < len(pts) and e < len(pts):
            cv2.line(frame, pts[s], pts[e], line_color, thickness, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(frame, pt, radius, color, -1)
        cv2.circle(frame, pt, radius + 1, (20, 20, 20), 1)


def draw_router_badge(
    frame: np.ndarray,
    mode: str,            # "IDLE" | "ALPHABET" | "WORD"
    num_hands: int = 0,
) -> None:
    """Draw the MoE router mode badge in the top-right corner."""
    badge_text = {
        "IDLE":     "[IDLE] No Hands",
        "ALPHABET": "[ALPHABET] 1 Hand",
        "WORD":     "[WORD] 2 Hands",
    }.get(mode, mode)

    colour = {
        "IDLE":     (100, 100, 100),
        "ALPHABET": (0,   200, 255),
        "WORD":     (50,  230, 100),
    }.get(mode, (200, 200, 200))

    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.55
    thick = 1
    (tw, th), _ = cv2.getTextSize(badge_text, font, scale, thick)

    x = w - tw - 16
    y = 30
    # Background pill
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6),
                  (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6),
                  colour, 1)
    cv2.putText(frame, badge_text, (x, y), font, scale, colour, thick, cv2.LINE_AA)


def draw_prediction_banner(
    frame: np.ndarray,
    label: str,
    confidence: float,
    translation: str = "",
    banner_h: int = 72,
) -> None:
    """Semi-transparent prediction banner at the top of the frame."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, f"Sign: {label}  ({confidence * 100:.1f}%)",
                (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 230, 100), 2, cv2.LINE_AA)
    if translation and translation != label:
        cv2.putText(frame, f"-> {translation}",
                    (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 50), 1, cv2.LINE_AA)



def draw_status_bar(
    frame: np.ndarray,
    message: str,
    color: tuple[int, int, int] = (50, 50, 50),
) -> None:
    """Thin status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 28), (w, h), color, -1)
    cv2.putText(frame, message, (10, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def mirror_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.flip(frame, 1)


def resize_frame(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
