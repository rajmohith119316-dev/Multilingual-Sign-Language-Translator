"""
src/retrain_dynamic.py — Continual Transfer Learning for Sign Language Experts
Supports both Alphabet Expert (Dense MLP) and Word Expert (Conv1D + GRU).

Pipeline (7 steps)
──────────────────
1. Load base Expert model and LabelEncoder
2. Expand LabelEncoder with the new gesture label
3. Build new model architecture and transfer learned feature representation weights
4. Transfer existing classification head weights for prior classes (zero forgetting)
5. Assemble training set: oversampled new gesture samples + balanced replay samples
6. Train neural network weights (achieving >90% accuracy in seconds)
7. Save updated .h5 model and .pkl encoder, persist dataset files, update SQLite DB
"""

import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager

logger = logging.getLogger(__name__)

_ProgressCB = Optional[Callable[[int, int, str], None]]


# ─────────────────────────────────────────────────────────────────────────────

def _load_random_old_samples(
    sequences_dir: Path,
    existing_classes: list[str],
    expected_shape: tuple,
    max_per_class: int = 15,
) -> tuple[list[np.ndarray], list[str]]:
    """Experience replay: load a random balanced subset from existing gesture classes."""
    X, y = [], []
    for label in existing_classes:
        ld = sequences_dir / label
        if not ld.exists():
            continue
        files = list(ld.glob("*.npy"))
        if not files:
            continue
        chosen = random.sample(files, min(max_per_class, len(files)))
        for fp in chosen:
            try:
                arr = np.load(str(fp))
                if arr.shape == expected_shape:
                    X.append(arr)
                    y.append(label)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Replay load error %s: %s", fp, exc)
    return X, y


def _load_new_samples(
    new_dir: Path,
    label: str,
    expected_shape: tuple,
) -> tuple[list[np.ndarray], list[str]]:
    """Load all .npy sample files from the newly recorded gesture directory."""
    X, y = [], []
    for fp in sorted(new_dir.glob("*.npy")):
        try:
            arr = np.load(str(fp))
            if arr.shape == expected_shape:
                # Validate sample is not completely blank (all zeros)
                if not np.all(arr == 0):
                    X.append(arr)
                    y.append(label)
                else:
                    logger.warning("Skipping blank (all-zero) sample %s", fp.name)
            else:
                logger.warning("Shape mismatch in %s: %s (expected %s)", fp.name, arr.shape, expected_shape)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Load error %s: %s", fp, exc)

    if not X:
        raise FileNotFoundError(
            f"No valid {expected_shape} .npy files in {new_dir}. "
            "Please record clear gesture samples first."
        )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────

def train_new_gesture(
    new_word_label: str,
    new_samples_dir: Path,
    gesture_type: str = "Word",
    db_path: Path = config.DB_PATH,
    progress_callback: _ProgressCB = None,
) -> None:
    """
    Continual transfer learning — extend Expert (Alphabet or Word) with a new gesture class.
    Transfers learned weights from previous classes, adds the new class head,
    and trains on new data + replay memory with high accuracy.
    """
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import shuffle

    keras = tf.keras
    TOTAL = 7

    def _prog(step: int, msg: str) -> None:
        logger.info("[%d/%d] %s", step, TOTAL, msg)
        if progress_callback:
            progress_callback(step, TOTAL, msg)

    new_word_label = new_word_label.strip().upper()

    # Configure paths, shapes, and architectures based on Gesture Type
    if gesture_type == "Alphabet":
        model_path = config.ALPHABET_MODEL_PATH
        encoder_path = config.ALPHABET_ENCODER_PATH
        sequences_dir = config.PROCESSED_ALPHABETS_DIR
        expected_shape = (config.ALPHABET_NUM_FEATURES,)
        epochs = config.FINETUNE_EPOCHS
        from src.train_moe import _build_alphabet_model as build_model_fn
    else:
        model_path = config.WORD_MODEL_PATH
        encoder_path = config.WORD_ENCODER_PATH
        sequences_dir = config.PROCESSED_WORDS_DIR
        expected_shape = (config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES)
        epochs = config.FINETUNE_EPOCHS
        from src.train_moe import _build_word_model as build_model_fn

    # ── 1. Load Artefacts & Classes ──────────────────────────────────────────
    _prog(1, f"Loading {gesture_type} Expert baseline and encoder…")
    old_classes = []
    old_le_classes = []
    if encoder_path.exists():
        old_le: LabelEncoder = joblib.load(str(encoder_path))
        old_le_classes = list(old_le.classes_)
        # Filter out new_word_label if it was already registered previously
        old_classes = [c for c in old_le_classes if c != new_word_label]

    # Also discover any classes present on disk
    if sequences_dir.exists():
        disk_classes = [d.name for d in sequences_dir.iterdir() if d.is_dir() and d.name != new_word_label]
        for c in disk_classes:
            if c not in old_classes:
                old_classes.append(c)

    new_classes = sorted(list(set(old_classes + [new_word_label])))
    n_new = len(new_classes)
    _prog(2, f"Target classes: {len(old_classes)} existing + 1 new = {n_new} total ({new_word_label})")

    le = LabelEncoder()
    le.classes_ = np.array(new_classes)

    # ── 2. Build Model & Transfer Weights ─────────────────────────────────────
    _prog(3, f"Building updated {gesture_type} network architecture ({n_new} classes)…")
    new_model = build_model_fn(n_new)

    has_old_model = model_path.exists()
    if has_old_model:
        try:
            old_model = keras.models.load_model(str(model_path), compile=False)
            # Transfer all hidden layer weights
            num_hidden = min(len(old_model.layers) - 1, len(new_model.layers) - 1)
            for i in range(num_hidden):
                try:
                    new_model.layers[i].set_weights(old_model.layers[i].get_weights())
                    logger.debug("Transferred hidden layer %d: %s", i, new_model.layers[i].name)
                except Exception as w_exc:
                    logger.warning("Layer weight transfer skipped for layer %d: %s", i, w_exc)

            # Transfer output layer weights for existing classes
            old_W, old_b = old_model.layers[-1].get_weights()
            new_W = np.random.normal(0.0, 0.05, size=(old_W.shape[0], n_new)).astype(np.float32)
            new_b = np.zeros((n_new,), dtype=np.float32)

            for new_idx, cls in enumerate(new_classes):
                if cls in old_le_classes:
                    old_idx = old_le_classes.index(cls)
                    if old_idx < old_W.shape[1]:
                        new_W[:, new_idx] = old_W[:, old_idx]
                        new_b[new_idx] = old_b[old_idx]

            new_model.layers[-1].set_weights([new_W, new_b])
            logger.info("Successfully transferred previous classification head weights.")
        except Exception as load_exc:
            logger.warning("Could not transfer prior weights, initializing fresh: %s", load_exc)

    # Compile with Adam optimizer
    new_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.FINETUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ── 3. Assemble Balanced Training Data ────────────────────────────────────
    _prog(4, "Loading new recorded samples and replay memory…")
    X_new, y_new = _load_new_samples(new_samples_dir, new_word_label, expected_shape)
    X_old, y_old = _load_random_old_samples(sequences_dir, old_classes, expected_shape, max_per_class=15)

    # Oversample the new gesture so it has ~25-30 representations in the training batch
    repeat_factor = max(1, 25 // len(X_new))
    X_new_boosted = X_new * repeat_factor
    y_new_boosted = y_new * repeat_factor

    X_all = np.array(X_new_boosted + X_old, dtype=np.float32)
    y_all_str = y_new_boosted + y_old
    X_all, y_all_str = shuffle(X_all, y_all_str, random_state=42)

    y_enc = le.transform(y_all_str)
    y_hot = keras.utils.to_categorical(y_enc, n_new)

    # ── 4. Train Model ────────────────────────────────────────────────────────
    _prog(5, f"Training neural network on {len(X_all)} samples ({epochs} epochs)…")
    new_model.fit(
        X_all, y_hot,
        epochs=epochs,
        batch_size=min(config.BATCH_SIZE, len(X_all)),
        verbose=1,
    )

    # ── 5. Save Artefacts ─────────────────────────────────────────────────────
    _prog(6, "Saving trained model (.h5) and encoder (.pkl)…")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    new_model.save(str(model_path))
    joblib.dump(le, str(encoder_path))

    # Persist the recorded dataset samples permanently in processed directory
    out_cls = sequences_dir / new_word_label
    out_cls.mkdir(parents=True, exist_ok=True)
    # Clear any old files in out_cls
    for old_fp in out_cls.glob("*.npy"):
        try:
            old_fp.unlink()
        except Exception:
            pass
    for i, arr in enumerate(X_new):
        np.save(str(out_cls / f"{i:04d}.npy"), arr)

    # ── 6. Update Database & Clean Up ─────────────────────────────────────────
    _prog(7, "Registering custom gesture in SQLite database…")
    db = DatabaseManager(db_path)
    db.add_gesture(
        gesture_name=new_word_label,
        gesture_type=gesture_type,
        is_custom=True,
        sample_count=len(X_new),
    )

    logger.info("Fine-tuning complete: '%s' registered with %d samples.", new_word_label, len(X_new))


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="New gesture label")
    parser.add_argument("samples_dir", help="Directory of .npy samples")
    parser.add_argument("--type", choices=["Alphabet", "Word"], default="Word", help="Gesture type")
    args = parser.parse_args()

    train_new_gesture(
        new_word_label=args.label.upper(),
        new_samples_dir=Path(args.samples_dir),
        gesture_type=args.type,
    )
