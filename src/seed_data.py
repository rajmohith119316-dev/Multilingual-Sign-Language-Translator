"""
src/seed_data.py — Seed Dataset Generator
Generates synthetic initial training samples for Alphabet and Word experts
so base models can be trained immediately out-of-the-box.
"""

import logging
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


logger = logging.getLogger(__name__)


def generate_alphabet_seed_data() -> None:
    """Generate sample 60-D wrist-relative vectors for A, B, C, D, E."""
    labels = ["A", "B", "C", "D", "E"]
    n_samples_per_label = 25

    rng = np.random.default_rng(seed=42)

    for i, label in enumerate(labels):
        out_dir = config.PROCESSED_ALPHABETS_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        base = rng.normal(loc=i * 0.2, scale=0.5, size=(20, 3)).astype(np.float32)
        norm = np.linalg.norm(base)
        if norm > 1e-6:
            base /= norm
        base_vec = base.flatten()   # (60,)

        for s in range(n_samples_per_label):
            noise = rng.normal(loc=0.0, scale=0.03, size=60).astype(np.float32)
            sample = base_vec + noise
            sample_norm = np.linalg.norm(sample)
            if sample_norm > 1e-6:
                sample /= sample_norm
            np.save(str(out_dir / f"{s:04d}.npy"), sample.astype(np.float32))

    logger.info("Generated alphabet seed samples for: %s", labels)


def generate_word_seed_data() -> None:
    """Generate sample (30, 123) dual-hand sequences for HELLO, THANKS, YES, NO."""
    labels = ["HELLO", "THANKS", "YES", "NO"]
    n_samples_per_label = 20

    rng = np.random.default_rng(seed=42)

    for i, label in enumerate(labels):
        out_dir = config.PROCESSED_WORDS_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        for s in range(n_samples_per_label):
            t = np.linspace(0, np.pi, config.SEQUENCE_LENGTH)
            freq = 1.0 + i * 0.5

            seq = np.zeros((config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES), dtype=np.float32)
            for f in range(config.SEQUENCE_LENGTH):
                l_feat = np.sin(t[f] * freq + np.arange(60) * 0.1) * 0.2
                r_feat = np.cos(t[f] * freq + np.arange(60) * 0.1) * 0.2
                inter = np.array([np.sin(t[f]), np.cos(t[f]), 0.1], dtype=np.float32)

                noise = rng.normal(loc=0.0, scale=0.01, size=123).astype(np.float32)
                frame_vec = np.concatenate([l_feat, r_feat, inter]) + noise
                seq[f] = frame_vec.astype(np.float32)

            np.save(str(out_dir / f"{s:04d}.npy"), seq)

    logger.info("Generated word seed samples for: %s", labels)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger.info("Generating seed dataset...")
    generate_alphabet_seed_data()
    generate_word_seed_data()
    logger.info("Seed data generation complete.")


if __name__ == "__main__":
    main()
