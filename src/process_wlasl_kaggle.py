"""
src/process_wlasl_kaggle.py — Kaggle WLASL Processed Dataset Integrator
Processes risangbaskoro/wlasl-processed dataset and integrates video sequence files
into data/processed/words/{LABEL}/.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

def process_wlasl_kaggle(
    wlasl_dataset_dir: Path,
    out_dir: Path = config.PROCESSED_WORDS_DIR,
) -> dict[str, int]:
    """Inspect and format WLASL preprocessed files into PROCESSED_WORDS_DIR."""
    results: dict[str, int] = {}
    logger.info("Inspecting WLASL processed dataset at: %s", wlasl_dataset_dir)
    
    # Check for subdirectories or npy files in dataset directory
    files = list(wlasl_dataset_dir.rglob("*.npy"))
    logger.info("Found %d total .npy files in WLASL processed dataset", len(files))

    for fp in files:
        label = fp.parent.name.upper()
        if not label or label.startswith("."):
            continue
        out_cls = out_dir / label
        out_cls.mkdir(parents=True, exist_ok=True)

        try:
            arr = np.load(str(fp))
            # Standardize shape to (30, 123)
            if arr.ndim == 2 and arr.shape[1] == config.WORD_NUM_FEATURES:
                from src.utils import pad_or_sample_sequence
                seq = pad_or_sample_sequence(arr, config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES)
                count = results.get(label, 0)
                np.save(str(out_cls / f"{count:04d}.npy"), seq)
                results[label] = count + 1
        except Exception as exc:
            logger.debug("Load error %s: %s", fp.name, exc)

    return results

if __name__ == "__main__":
    import kagglehub
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    logger.info("Downloading/locating risangbaskoro/wlasl-processed from kagglehub...")
    path = Path(kagglehub.dataset_download("risangbaskoro/wlasl-processed"))
    res = process_wlasl_kaggle(path)
    total = sum(res.values())
    print(f"\n--- WLASL Kaggle Processing Summary ---")
    for lbl, cnt in sorted(res.items()):
        print(f"  {lbl:20s}: {cnt:4d} sequences")
    print(f"\nTotal: {total} sequences saved across {len(res)} word classes.")
