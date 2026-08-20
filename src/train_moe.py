"""
src/train_moe.py — Hierarchical Mixture-of-Experts Training
Trains two specialist models:

  1. Alphabet Expert  — Dense MLP on (60,) feature vectors  → models/alphabet_expert.h5
  2. Word Expert      — Conv1D+GRU on (30,123) sequences    → models/word_expert.h5

Both encoders are saved as .pkl with sklearn LabelEncoder.
Trained classes are auto-populated into the SQLite gestures table.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load .npy dataset from a processed directory
# ─────────────────────────────────────────────────────────────────────────────

def _load_npy_dataset(
    root_dir: Path,
    expected_shape: tuple,
) -> tuple[np.ndarray, list[str]]:
    """
    Recursively load .npy files from root_dir/{LABEL}/*.npy.

    Parameters
    ----------
    root_dir       : path containing per-label subdirectories
    expected_shape : e.g. (60,) or (30, 123)

    Returns (X, y_str) or raises ValueError if empty.
    """
    X_list, y_list = [], []
    label_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        raise FileNotFoundError(f"No label dirs in {root_dir}. Run the processor first.")

    for ld in label_dirs:
        label     = ld.name
        npy_files = sorted(ld.glob("*.npy"))
        if len(npy_files) < config.MIN_SAMPLES_CLASS:
            logger.warning("Skipping '%s' — only %d samples.", label, len(npy_files))
            continue
        for fp in npy_files:
            try:
                arr = np.load(str(fp))
                if arr.shape == expected_shape:
                    X_list.append(arr)
                    y_list.append(label)
                else:
                    logger.debug("Shape mismatch %s: %s ≠ %s", fp.name, arr.shape, expected_shape)
            except Exception as exc:            # noqa: BLE001
                logger.warning("Load error %s: %s", fp, exc)

    if not X_list:
        raise ValueError(f"Empty dataset in {root_dir} after filtering.")

    logger.info("Dataset from %s: %d samples, %d classes.",
                root_dir.name, len(y_list), len(set(y_list)))
    return np.array(X_list, dtype=np.float32), y_list


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT 1 — Alphabet (Dense MLP, input 60-D)
# ─────────────────────────────────────────────────────────────────────────────

def _build_alphabet_model(num_classes: int):
    """
    Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dense(N, softmax)
    Input: (60,)
    """
    import tensorflow as tf
    keras = tf.keras
    model = keras.Sequential([
        keras.Input(shape=(config.ALPHABET_NUM_FEATURES,), name="alphabet_input"),
        keras.layers.Dense(128, activation="relu",  name="dense_1"),
        keras.layers.Dropout(0.3,                   name="dropout_1"),
        keras.layers.Dense(64,  activation="relu",  name="dense_2"),
        keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="alphabet_expert")
    return model


def train_alphabet_expert(
    processed_dir: Path  = config.PROCESSED_ALPHABETS_DIR,
    model_path: Path     = config.ALPHABET_MODEL_PATH,
    encoder_path: Path   = config.ALPHABET_ENCODER_PATH,
    db_path: Path        = config.DB_PATH,
) -> None:
    """Full training pipeline for the Alphabet Expert."""
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    keras = tf.keras

    logger.info("═══ Training Alphabet Expert ═══")
    X, y_str = _load_npy_dataset(processed_dir, (config.ALPHABET_NUM_FEATURES,))

    le = LabelEncoder()
    y_enc  = le.fit_transform(y_str)
    n_cls  = len(le.classes_)
    y_hot  = keras.utils.to_categorical(y_enc, n_cls)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y_hot, test_size=config.VALIDATION_SPLIT,
        random_state=42, stratify=y_enc,
    )
    logger.info("Classes(%d): %s", n_cls, list(le.classes_))
    logger.info("Train: %d  Val: %d", len(X_tr), len(X_v))

    model = _build_alphabet_model(n_cls)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy",
                                         save_best_only=True),
    ]

    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=config.ALPHABET_EPOCHS, batch_size=config.BATCH_SIZE,
              callbacks=cbs, verbose=1)

    if not model_path.exists():
        model.save(str(model_path))

    joblib.dump(le, str(encoder_path))
    val_loss, val_acc = model.evaluate(X_v, y_v, verbose=0)
    logger.info("Alphabet Expert - val_acc: %.4f", val_acc)
    print(f"\n[OK] Alphabet Expert trained - val_accuracy: {val_acc*100:.2f}%")

    _populate_db(le.classes_.tolist(), "Alphabet", processed_dir, db_path)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERT 2 — Word (Conv1D + GRU, input (30, 123))
# ─────────────────────────────────────────────────────────────────────────────

def _build_word_model(num_classes: int):
    """
    Conv1D(64, k=3, relu) → MaxPool(2) → GRU(64)
    → Dense(128, relu) → Dropout(0.5) → Dense(N, softmax)
    Input: (30, 123)
    """
    import tensorflow as tf
    keras = tf.keras
    model = keras.Sequential([
        keras.Input(shape=(config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES),
                    name="word_input"),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu",
                            padding="same", name="conv1d_1"),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.GRU(64, return_sequences=False, name="gru_1"),
        keras.layers.Dense(128, activation="relu",  name="dense_1"),
        keras.layers.Dropout(0.5,                   name="dropout_1"),
        keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="word_expert")
    return model


def train_word_expert(
    processed_dir: Path  = config.PROCESSED_WORDS_DIR,
    model_path: Path     = config.WORD_MODEL_PATH,
    encoder_path: Path   = config.WORD_ENCODER_PATH,
    db_path: Path        = config.DB_PATH,
) -> None:
    """Full training pipeline for the Word Expert."""
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    keras = tf.keras

    logger.info("═══ Training Word Expert ═══")
    X, y_str = _load_npy_dataset(processed_dir,
                                  (config.SEQUENCE_LENGTH, config.WORD_NUM_FEATURES))

    le = LabelEncoder()
    y_enc  = le.fit_transform(y_str)
    n_cls  = len(le.classes_)
    y_hot  = keras.utils.to_categorical(y_enc, n_cls)

    X_tr, X_v, y_tr, y_v = train_test_split(
        X, y_hot, test_size=config.VALIDATION_SPLIT,
        random_state=42, stratify=y_enc,
    )
    logger.info("Classes(%d): %s", n_cls, list(le.classes_))
    logger.info("Train: %d  Val: %d", len(X_tr), len(X_v))

    model = _build_word_model(n_cls)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                       restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                           patience=4, min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_accuracy",
                                         save_best_only=True),
    ]

    model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
              epochs=config.WORD_EPOCHS, batch_size=config.BATCH_SIZE,
              callbacks=cbs, verbose=1)

    if not model_path.exists():
        model.save(str(model_path))

    joblib.dump(le, str(encoder_path))
    val_loss, val_acc = model.evaluate(X_v, y_v, verbose=0)
    logger.info("Word Expert - val_acc: %.4f", val_acc)
    print(f"\n[OK] Word Expert trained - val_accuracy: {val_acc*100:.2f}%")

    _populate_db(le.classes_.tolist(), "Word", processed_dir, db_path)


# ─────────────────────────────────────────────────────────────────────────────
# Shared DB population helper
# ─────────────────────────────────────────────────────────────────────────────

def _populate_db(
    classes: list[str],
    gesture_type: str,      # 'Alphabet' | 'Word'
    processed_dir: Path,
    db_path: Path,
) -> None:
    db = DatabaseManager(db_path)
    for label in classes:
        label_dir    = processed_dir / label
        sample_count = len(list(label_dir.glob("*.npy"))) if label_dir.exists() else 0
        db.add_gesture(
            gesture_name=label,
            gesture_type=gesture_type,
            is_custom=False,
            sample_count=sample_count,
        )
    logger.info("DB updated with %d '%s' classes.", len(classes), gesture_type)


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    parser = argparse.ArgumentParser(description="Train MoE Sign Language Experts")
    parser.add_argument("--expert", choices=["alphabet", "word", "both"],
                        default="both", help="Which expert to train")
    args = parser.parse_args()

    if args.expert in ("alphabet", "both"):
        train_alphabet_expert()
    if args.expert in ("word", "both"):
        train_word_expert()
