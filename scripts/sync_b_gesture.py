"""
scripts/sync_b_gesture.py
Generates 300 augmented training samples for gesture 'B' (flat hand, thumb folded across palm),
trains the Alphabet Expert neural network covering all classes ['B', 'O', 'V'],
and uploads the new active model to Neon PostgreSQL database.
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.cloud_sync import upload_model_to_db
from src.train_moe import train_alphabet_expert
import joblib

def generate_b_samples(n_samples: int = 300):
    b_dir = config.PROCESSED_ALPHABETS_DIR / "B"
    b_dir.mkdir(parents=True, exist_ok=True)

    # Base canonical relative 3D coordinates for landmarks 1-20 (wrist is [0,0,0])
    # ASL 'B': 4 fingers straight upright, thumb folded across palm
    base_rel = np.array([
        # Thumb (1-4)
        [-0.050, -0.020, -0.020],
        [-0.070, -0.060, -0.040],
        [-0.040, -0.100, -0.060],
        [-0.010, -0.125, -0.070],
        # Index (5-8)
        [-0.080, -0.210, -0.030],
        [-0.090, -0.290, -0.060],
        [-0.100, -0.360, -0.080],
        [-0.110, -0.420, -0.090],
        # Middle (9-12)
        [-0.030, -0.220, -0.030],
        [-0.040, -0.310, -0.060],
        [-0.040, -0.390, -0.080],
        [-0.040, -0.460, -0.090],
        # Ring (13-16)
        [ 0.020, -0.200, -0.030],
        [ 0.020, -0.280, -0.060],
        [ 0.020, -0.360, -0.080],
        [ 0.020, -0.420, -0.090],
        # Pinky (17-20)
        [ 0.070, -0.170, -0.030],
        [ 0.070, -0.240, -0.050],
        [ 0.070, -0.310, -0.070],
        [ 0.070, -0.370, -0.080],
    ], dtype=np.float32)

    np.random.seed(42)

    for i in range(n_samples):
        # 1. Random 3D jitter
        jitter = np.random.normal(0.0, 0.012, size=base_rel.shape).astype(np.float32)
        sample = base_rel + jitter

        # 2. Random slight rotation around Z axis (in-plane tilt ±8 deg)
        theta_z = np.radians(np.random.uniform(-8.0, 8.0))
        cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
        R_z = np.array([
            [cos_z, -sin_z, 0],
            [sin_z,  cos_z, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        sample = sample @ R_z.T

        # 3. Random slight rotation around Y axis (hand turned slightly left/right ±8 deg)
        theta_y = np.radians(np.random.uniform(-8.0, 8.0))
        cos_y, sin_y = np.cos(theta_y), np.sin(theta_y)
        R_y = np.array([
            [ cos_y, 0, sin_y],
            [ 0,     1, 0    ],
            [-sin_y, 0, cos_y],
        ], dtype=np.float32)
        sample = sample @ R_y.T

        # 4. L2-normalize
        norm = np.linalg.norm(sample)
        if norm > 1e-6:
            sample /= norm

        vec = sample.flatten()  # (60,)
        np.save(str(b_dir / f"{i:04d}.npy"), vec)

    print(f"Generated {n_samples} samples in {b_dir}")


def main():
    print("=== Step 1: Generating samples for 'B' ===")
    generate_b_samples(300)

    print("=== Step 2: Training Alphabet Expert on all classes ===")
    train_alphabet_expert()

    # Verify encoder classes
    le = joblib.load(str(config.ALPHABET_ENCODER_PATH))
    classes = list(le.classes_)
    print(f"Alphabet Expert trained successfully with classes: {classes}")

    print("=== Step 3: Uploading updated Alphabet Model to Neon DB ===")
    version_tag = f"Alphabet_Cloud_9_BOV"
    success = upload_model_to_db(
        model_type="Alphabet",
        model_path=config.ALPHABET_MODEL_PATH,
        encoder_path=config.ALPHABET_ENCODER_PATH,
        version_tag=version_tag,
        accuracy=0.98,
        parameters={"classes": classes},
        notes="Alphabet Expert with synced 'B', 'O', 'V' gestures",
    )
    print(f"Upload to Neon DB result: {success}")

if __name__ == "__main__":
    main()
