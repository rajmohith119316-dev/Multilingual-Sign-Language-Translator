"""
src/generate_samples.py — Generate UI Sample Photo & Video of the Dataset
Creates visual assets showcasing MediaPipe hand landmark tracking and dataset processing:
  1. videos/sample_dataset_photo.png — High-res annotated snapshot of hand sign detection
  2. videos/sample_dataset_video.mp4 — 30-frame MP4 video clip with live landmark tracking,
                                       router badge, and prediction banner.
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import draw_hand_landmarks, draw_router_badge, draw_prediction_banner

logger = logging.getLogger(__name__)

# MediaPipe Hand Connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

class SyntheticLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

def _create_hand_pose(base_x: float, base_y: float, scale: float = 0.25, is_left: bool = False) -> list[SyntheticLandmark]:
    """Generate 21 synthetic hand landmark points for clean visualization."""
    dir_mult = -1.0 if is_left else 1.0
    pts = [
        (base_x, base_y),  # 0: Wrist
        (base_x + 0.2 * dir_mult * scale, base_y - 0.1 * scale),  # 1: Thumb CMC
        (base_x + 0.35 * dir_mult * scale, base_y - 0.25 * scale), # 2: Thumb MCP
        (base_x + 0.45 * dir_mult * scale, base_y - 0.4 * scale),  # 3: Thumb IP
        (base_x + 0.55 * dir_mult * scale, base_y - 0.5 * scale),  # 4: Thumb Tip
        
        (base_x + 0.15 * dir_mult * scale, base_y - 0.5 * scale),  # 5: Index MCP
        (base_x + 0.18 * dir_mult * scale, base_y - 0.75 * scale), # 6: Index PIP
        (base_x + 0.20 * dir_mult * scale, base_y - 0.95 * scale), # 7: Index DIP
        (base_x + 0.21 * dir_mult * scale, base_y - 1.1 * scale),  # 8: Index Tip
        
        (base_x, base_y - 0.52 * scale),                          # 9: Middle MCP
        (base_x, base_y - 0.78 * scale),                          # 10: Middle PIP
        (base_x, base_y - 0.98 * scale),                          # 11: Middle DIP
        (base_x, base_y - 1.15 * scale),                          # 12: Middle Tip
        
        (base_x - 0.14 * dir_mult * scale, base_y - 0.5 * scale), # 13: Ring MCP
        (base_x - 0.16 * dir_mult * scale, base_y - 0.74 * scale),# 14: Ring PIP
        (base_x - 0.17 * dir_mult * scale, base_y - 0.92 * scale),# 15: Ring DIP
        (base_x - 0.18 * dir_mult * scale, base_y - 1.08 * scale),# 16: Ring Tip
        
        (base_x - 0.26 * dir_mult * scale, base_y - 0.45 * scale),# 17: Pinky MCP
        (base_x - 0.29 * dir_mult * scale, base_y - 0.65 * scale),# 18: Pinky PIP
        (base_x - 0.31 * dir_mult * scale, base_y - 0.82 * scale),# 19: Pinky DIP
        (base_x - 0.33 * dir_mult * scale, base_y - 0.95 * scale) # 20: Pinky Tip
    ]
    return [SyntheticLandmark(x, y) for x, y in pts]

def generate_sample_photo(out_path: Path = config.BASE_DIR / "videos" / "sample_dataset_photo.png") -> Path:
    """Generate a high-res sample image of the dataset sign gesture with landmark overlays."""
    h, w = 600, 800
    # Deep gradient background
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = int(13 + (y / h) * 10)
        g = int(13 + (y / h) * 15)
        b = int(26 + (y / h) * 20)
        img[y, :] = (b, g, r)

    # Grid pattern
    for x in range(0, w, 40):
        cv2.line(img, (x, 0), (x, h), (30, 30, 50), 1)
    for y in range(0, h, 40):
        cv2.line(img, (0, y), (w, y), (30, 30, 50), 1)

    # Generate Left and Right Hand landmarks
    left_hand = _create_hand_pose(0.35, 0.65, scale=0.35, is_left=True)
    right_hand = _create_hand_pose(0.65, 0.65, scale=0.35, is_left=False)

    draw_hand_landmarks(img, left_hand, color=(255, 200, 50))  # Cyan/Yellow
    draw_hand_landmarks(img, right_hand, color=(110, 230, 0)) # Green

    # Bounding boxes
    for hand, col, name in [(left_hand, (255, 200, 50), "Left Hand"), (right_hand, (110, 230, 0), "Right Hand")]:
        xs = [int(p.x * w) for p in hand]
        ys = [int(p.y * h) for p in hand]
        min_x, max_x = max(0, min(xs) - 20), min(w, max(xs) + 20)
        min_y, max_y = max(0, min(ys) - 20), min(h, max(ys) + 20)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), col, 2, cv2.LINE_AA)
        cv2.putText(img, name, (min_x, min_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    # UI Header & Card overlay
    draw_router_badge(img, "WORD")
    draw_prediction_banner(img, "HELLO", 0.985, "Hello / Greeting")

    # Metadata card
    cv2.rectangle(img, (20, 20), (320, 110), (30, 30, 56), -1)
    cv2.rectangle(img, (20, 20), (320, 110), (0, 229, 204), 1, cv2.LINE_AA)
    cv2.putText(img, "DATASET SAMPLE PREVIEW", (32, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 229, 204), 2, cv2.LINE_AA)
    cv2.putText(img, "Word Sign: HELLO", (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (226, 226, 240), 1, cv2.LINE_AA)
    cv2.putText(img, "Feature Vector: Dual-Hand (123-D)", (32, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 168), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    logger.info("Saved sample photo: %s", out_path)
    return out_path

def generate_sample_video(out_path: Path = config.BASE_DIR / "videos" / "sample_dataset_video.mp4") -> Path:
    """Generate a 30-frame MP4 video clip showing dynamic sign gesture tracking for the UI."""
    h, w = 480, 640
    fps = 25
    total_frames = 30

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    for frame_idx in range(total_frames):
        t = frame_idx / total_frames
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Dynamic gradient background
        for y in range(h):
            img[y, :] = (int(20 + y/h*10), int(15 + y/h*15), int(10 + y/h*10))

        # Dynamic motion of hands (simulating HELLO wave gesture)
        l_offset_y = np.sin(t * np.pi * 2) * 0.05
        r_offset_y = np.cos(t * np.pi * 2) * 0.08
        r_offset_x = np.sin(t * np.pi * 2) * 0.04

        left_hand = _create_hand_pose(0.35, 0.60 + l_offset_y, scale=0.32, is_left=True)
        right_hand = _create_hand_pose(0.65 + r_offset_x, 0.58 + r_offset_y, scale=0.32, is_left=False)

        draw_hand_landmarks(img, left_hand, color=(255, 200, 50))
        draw_hand_landmarks(img, right_hand, color=(110, 230, 0))

        # Hand bounding boxes
        for hand, col, name in [(left_hand, (255, 200, 50), "L-Hand"), (right_hand, (110, 230, 0), "R-Hand")]:
            xs = [int(p.x * w) for p in hand]
            ys = [int(p.y * h) for p in hand]
            min_x, max_x = max(0, min(xs) - 15), min(w, max(xs) + 15)
            min_y, max_y = max(0, min(ys) - 15), min(h, max(ys) + 15)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), col, 2, cv2.LINE_AA)
            cv2.putText(img, name, (min_x, min_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        # Overlays
        draw_router_badge(img, "WORD")
        conf = min(0.99, 0.85 + t * 0.14)
        draw_prediction_banner(img, "HELLO", conf, "Hello!")

        # Progress bar overlay
        buf_w = int((frame_idx + 1) / total_frames * 200)
        cv2.rectangle(img, (20, h - 30), (220, h - 15), (40, 40, 60), -1)
        cv2.rectangle(img, (20, h - 30), (20 + buf_w, h - 15), (0, 229, 204), -1)
        cv2.putText(img, f"Buffer: {frame_idx + 1}/{total_frames}", (230, h - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 240), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    logger.info("Saved sample video: %s", out_path)
    return out_path

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    p1 = generate_sample_photo()
    p2 = generate_sample_video()
    print(f"\n[OK] Dataset Sample Media Generated:")
    print(f"  Photo: {p1}")
    print(f"  Video: {p2}")

if __name__ == "__main__":
    main()
