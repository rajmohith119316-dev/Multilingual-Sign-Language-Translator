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
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    
    if translation and translation != label:
        primary_text = translation
        secondary_text = f"English: {label}"
    else:
        primary_text = label
        secondary_text = ""

    try:
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font_primary = ImageFont.truetype("segoeui.ttf", 32)
        except IOError:
            try:
                font_primary = ImageFont.truetype("arial.ttf", 32)
            except IOError:
                font_primary = ImageFont.load_default()
                
        try:
            font_secondary = ImageFont.truetype("segoeui.ttf", 16)
        except IOError:
            try:
                font_secondary = ImageFont.truetype("arial.ttf", 16)
            except IOError:
                font_secondary = ImageFont.load_default()
        
        # Draw primary text (Translated language)
        draw.text((12, 5), f"Sign: {primary_text} ({confidence * 100:.1f}%)", font=font_primary, fill=(0, 230, 100))
        
        # Draw secondary text (English)
        if secondary_text:
            draw.text((12, 45), secondary_text, font=font_secondary, fill=(255, 220, 50))
            
        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.warning(f"Failed to draw unicode text: {e}")
        # Fallback to OpenCV putText
        cv2.putText(frame, f"Sign: {primary_text}  ({confidence * 100:.1f}%)",
                    (12, 38), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 230, 100), 2, cv2.LINE_AA)
        if secondary_text:
            cv2.putText(frame, secondary_text,
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


def render_sample_preview(
    sample_data: bytes,
    sample_type: str = "npy",
    width: int = 260,
    height: int = 260,
) -> tuple[Optional[bytes], str]:
    """
    Renders visual photo or video/animation preview for a training sample.

    Returns:
        (data_bytes, mime_type) e.g. (png_bytes, 'image/png') or (gif_bytes, 'image/gif')
    """
    if not sample_data:
        return None, "image/png"

    # If it's already an image binary (JPEG, PNG, GIF)
    if sample_data[:3] == b"\xff\xd8\xff":
        return sample_data, "image/jpeg"
    if sample_data[:4] == b"\x89PNG":
        return sample_data, "image/png"
    if sample_data[:4] == b"GIF8":
        return sample_data, "image/gif"
    if len(sample_data) > 8 and sample_data[4:8] in (b"ftyp", b"moov"):
        return sample_data, "video/mp4"

    # Otherwise, it's a .npy landmarks file
    import io
    from PIL import Image

    try:
        arr = np.load(io.BytesIO(sample_data))
    except Exception as exc:
        logger.error("Failed to load sample array: %s", exc)
        return None, "image/png"

    # 1-Frame Alphabet Gesture (shape: (60,))
    if arr.ndim == 1 and arr.shape[0] == 60:
        pts = np.vstack([[0.0, 0.0, 0.0], arr.reshape(20, 3)])
        xs, ys = pts[:, 0], pts[:, 1]
        pad = 28
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)

        px = np.int32(pad + (xs - min_x) * scale + (width - 2 * pad - span_x * scale) / 2)
        py = np.int32(pad + (ys - min_y) * scale + (height - 2 * pad - span_y * scale) / 2)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (18, 18, 30)

        # Draw bones
        for s, e in _HAND_CONNECTIONS:
            cv2.line(img, (px[s], py[s]), (px[e], py[e]), (230, 235, 255), 3, cv2.LINE_AA)

        # Draw joints
        for i in range(21):
            col = (0, 240, 255) if i == 0 else (0, 255, 140) if i in (4, 8, 12, 16, 20) else (120, 180, 255)
            r = 5 if i in (0, 4, 8, 12, 16, 20) else 4
            cv2.circle(img, (px[i], py[i]), r, col, -1)
            cv2.circle(img, (px[i], py[i]), r + 1, (255, 255, 255), 1)

        _, png = cv2.imencode(".png", img)
        return png.tobytes(), "image/png"

    # Multi-Frame Sequence / Word Gesture (shape: (N, 123) or (N, 60))
    if arr.ndim == 2:
        n_frames = arr.shape[0]
        feat_dim = arr.shape[1]
        gif_frames = []

        step = max(1, n_frames // 30)
        selected_indices = list(range(0, n_frames, step))[:30]

        for idx in selected_indices:
            frame_vec = arr[idx]
            frame_img = np.zeros((height, width, 3), dtype=np.uint8)
            frame_img[:] = (18, 18, 30)

            if feat_dim == 123:
                left_feat = frame_vec[:60]
                right_feat = frame_vec[60:120]
                inter = frame_vec[120:123]
                has_left = not np.all(left_feat == 0)
                has_right = not np.all(right_feat == 0)

                pts_list = []
                if has_left:
                    l_pts = np.vstack([[0.0, 0.0, 0.0], left_feat.reshape(20, 3)])
                    pts_list.append(l_pts)
                if has_right:
                    r_pts = np.vstack([[inter[0], inter[1], inter[2]], right_feat.reshape(20, 3) + inter])
                    pts_list.append(r_pts)

                if pts_list:
                    all_c = np.vstack(pts_list)
                    xs, ys = all_c[:, 0], all_c[:, 1]
                    pad = 28
                    min_x, max_x = xs.min(), xs.max()
                    min_y, max_y = ys.min(), ys.max()
                    span_x = max(max_x - min_x, 1e-5)
                    span_y = max(max_y - min_y, 1e-5)
                    scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)

                    def to_px(pts):
                        px = np.int32(pad + (pts[:, 0] - min_x) * scale + (width - 2 * pad - span_x * scale) / 2)
                        py = np.int32(pad + (pts[:, 1] - min_y) * scale + (height - 2 * pad - span_y * scale) / 2)
                        return px, py

                    if has_left:
                        lpx, lpy = to_px(l_pts)
                        for s, e in _HAND_CONNECTIONS:
                            cv2.line(frame_img, (lpx[s], lpy[s]), (lpx[e], lpy[e]), (100, 200, 255), 2, cv2.LINE_AA)
                        for i in range(21):
                            cv2.circle(frame_img, (lpx[i], lpy[i]), 4, (0, 240, 255), -1)

                    if has_right:
                        rpx, rpy = to_px(r_pts)
                        for s, e in _HAND_CONNECTIONS:
                            cv2.line(frame_img, (rpx[s], rpy[s]), (rpx[e], rpy[e]), (100, 255, 180), 2, cv2.LINE_AA)
                        for i in range(21):
                            cv2.circle(frame_img, (rpx[i], rpy[i]), 4, (0, 255, 140), -1)
            elif feat_dim == 60:
                pts = np.vstack([[0.0, 0.0, 0.0], frame_vec.reshape(20, 3)])
                xs, ys = pts[:, 0], pts[:, 1]
                pad = 28
                min_x, max_x = xs.min(), xs.max()
                min_y, max_y = ys.min(), ys.max()
                span_x = max(max_x - min_x, 1e-5)
                span_y = max(max_y - min_y, 1e-5)
                scale = min((width - 2 * pad) / span_x, (height - 2 * pad) / span_y)
                px = np.int32(pad + (xs - min_x) * scale + (width - 2 * pad - span_x * scale) / 2)
                py = np.int32(pad + (ys - min_y) * scale + (height - 2 * pad - span_y * scale) / 2)
                for s, e in _HAND_CONNECTIONS:
                    cv2.line(frame_img, (px[s], py[s]), (px[e], py[e]), (230, 235, 255), 2, cv2.LINE_AA)
                for i in range(21):
                    cv2.circle(frame_img, (px[i], py[i]), 4, (0, 240, 255), -1)

            rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            gif_frames.append(Image.fromarray(rgb))

        if gif_frames:
            buf = io.BytesIO()
            gif_frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=gif_frames[1:],
                duration=65,
                loop=0,
            )
            return buf.getvalue(), "image/gif"

    return None, "image/png"
