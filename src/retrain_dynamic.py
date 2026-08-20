"""
src/retrain_dynamic.py — Continual Transfer Learning for Word Expert
Adds a new custom word gesture without catastrophic forgetting.

Pipeline (7 steps)
──────────────────
1. Load word_expert.h5 + word_encoder.pkl
2. Append new label to LabelEncoder
3. Freeze Conv1D and GRU layers (trainable = False)
4. Replace classification head → Dense(new_N, softmax)
5. Fine-tune on new samples + replay of old data
6. Save updated model + encoder
7. Update SQLite gestures table (is_custom = 1)
"""

import logging
import random
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
    max_per_class: int = 10,
) -> tuple[list[np.ndarray], list[str]]:
    """Experience replay: load a random subset from existing word classes."""
    X, y = [], []
    for label in existing_classes:
        ld = sequences_dir / label
        if not ld.exists():
            continue
        files  = list(ld.glob("*.npy"))
        chosen = random.sample(files, min(max_per_class, len(files)))
        for fp in chosen:
            try:
                arr = np.load(str(fp))
                if arr.shape == (config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES):
                    X.append(arr)
                    y.append(label)
            except Exception as exc:        # noqa: BLE001
                logger.warning("Replay load error %s: %s", fp, exc)
    return X, y


def _load_new_samples(
    new_dir: Path,
    label: str,
) -> tuple[list[np.ndarray], list[str]]:
    """Load all .npy sequences from the newly recorded gesture directory."""
    X, y = [], []
    for fp in sorted(new_dir.glob("*.npy")):
        try:
            arr = np.load(str(fp))
            if arr.shape == (config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES):
                X.append(arr)
                y.append(label)
            else:
                logger.warning("Shape mismatch in %s: %s", fp.name, arr.shape)
        except Exception as exc:            # noqa: BLE001
            logger.warning("Load error %s: %s", fp, exc)

    if not X:
        raise FileNotFoundError(
            f"No valid (30,123) .npy files in {new_dir}. "
            "Record samples in the Teach tab first."
        )
    return X, y


# ─────────────────────────────────────────────────────────────────────────────

def train_new_gesture(
    new_word_label: str,
    new_samples_dir: Path,
    model_path: Path     = config.WORD_MODEL_PATH,
    encoder_path: Path   = config.WORD_ENCODER_PATH,
    sequences_dir: Path  = config.PROCESSED_WORDS_DIR,
    db_path: Path        = config.DB_PATH,
    progress_callback: _ProgressCB = None,
) -> None:
    """
    Continual transfer learning — extend Word Expert with a new gesture class.
    """
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import LabelEncoder

    keras = tf.keras
    TOTAL = 7

    def _prog(step: int, msg: str) -> None:
        logger.info("[%d/%d] %s", step, TOTAL, msg)
        if progress_callback:
            progress_callback(step, TOTAL, msg)

    # ── 1. Load or Build artefacts ─────────────────────────────────────────────
    if not model_path.exists() or not encoder_path.exists():
        _prog(1, "Base Word Expert not found — initializing fresh Word Expert model...")
        X_new, y_new = _load_new_samples(new_samples_dir, new_word_label)
        X_old, y_old = _load_random_old_samples(sequences_dir, [])
        X_all = np.array(X_new + X_old, dtype=np.float32)
        y_all_str = y_new + y_old

        le = LabelEncoder()
        y_enc = le.fit_transform(y_all_str)
        n_new = len(le.classes_)
        y_hot = keras.utils.to_categorical(y_enc, n_new)

        from src.train_moe import _build_word_model
        new_model = _build_word_model(n_new)
        new_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    else:
        _prog(1, "Loading Word Expert + LabelEncoder…")
        old_model = keras.models.load_model(str(model_path), compile=False)
        le: LabelEncoder = joblib.load(str(encoder_path))

        old_classes = le.classes_.tolist()

        # ── 2. Expand LabelEncoder ────────────────────────────────────────────────
        _prog(2, f"Expanding encoder {len(old_classes)} → +1 class…")
        if new_word_label in old_classes:
            new_classes = old_classes
        else:
            new_classes = sorted(old_classes + [new_word_label])
        le.classes_ = np.array(new_classes)
        n_new = len(new_classes)

        # ── 3. Freeze Conv1D and GRU layers ───────────────────────────────────────
        _prog(3, "Freezing Conv1D + GRU layers…")
        for layer in old_model.layers:
            if layer.name.startswith(("conv1d", "gru", "maxpool")):
                layer.trainable = False
                logger.debug("Frozen: %s", layer.name)

        # ── 4. Replace classification head ────────────────────────────────────────
        _prog(4, "Replacing classification head…")
        penultimate_output = old_model.layers[-2].output   # Dropout layer
        new_out = keras.layers.Dense(
            n_new, activation="softmax", name="output_ft"
        )(penultimate_output)
        new_model = keras.Model(
            inputs=old_model.input,
            outputs=new_out,
            name="word_expert_finetuned",
        )
        new_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.FINETUNE_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # ── 5. Assemble training data ──────────────────────────────────────────────
        _prog(5, "Building replay + new sample training set…")
        X_new, y_new = _load_new_samples(new_samples_dir, new_word_label)
        X_old, y_old = _load_random_old_samples(sequences_dir, old_classes)

        X_all = np.array(X_new + X_old, dtype=np.float32)
        y_all_str = y_new + y_old

        if len(X_all) == 0:
            raise ValueError("Combined training set is empty.")

        y_enc = le.transform(y_all_str)
        y_hot = keras.utils.to_categorical(y_enc, n_new)

    val_split = 0.15 if len(X_all) >= 20 else 0.0

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy" if val_split > 0 else "accuracy",
            patience=5, restore_best_weights=True,
        ),
    ]

    _prog(5, f"Fine-tuning on {len(X_all)} samples ({config.FINETUNE_EPOCHS} epochs)…")
    new_model.fit(
        X_all, y_hot,
        epochs=config.FINETUNE_EPOCHS,
        batch_size=min(config.BATCH_SIZE, len(X_all)),
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 6. Save artefacts ─────────────────────────────────────────────────────
    _prog(6, "Saving updated model + encoder…")
    new_model.save(str(model_path))
    joblib.dump(le, str(encoder_path))

    # Copy new sequences to processed dir for future replays
    out_cls = sequences_dir / new_word_label
    out_cls.mkdir(parents=True, exist_ok=True)
    for i, arr in enumerate(X_new):
        np.save(str(out_cls / f"{i:04d}.npy"), arr)

    # ── 7. Update DB ──────────────────────────────────────────────────────────
    _prog(7, "Updating gesture registry…")
    db = DatabaseManager(db_path)
    db.add_gesture(
        gesture_name=new_word_label,
        gesture_type="Word",
        is_custom=True,
        sample_count=len(X_new),
    )
    logger.info("Fine-tuning complete. '%s' registered (is_custom=1).", new_word_label)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("label", help="New gesture label")
    parser.add_argument("samples_dir", help="Directory of .npy samples")
    args = parser.parse_args()

    train_new_gesture(
        new_word_label=args.label.upper(),
        new_samples_dir=Path(args.samples_dir),
    )
