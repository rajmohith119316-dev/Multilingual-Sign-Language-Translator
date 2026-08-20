"""
src/train_base_model.py — Base Model Training Script
Loads all .npy sequences from data/sequences/, builds a Conv1D + GRU hybrid
network, trains it, and saves the model and LabelEncoder.
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
# Lazy imports – TensorFlow can be slow; only load when needed
# ─────────────────────────────────────────────────────────────────────────────
def _import_tf():
    import tensorflow as tf
    return tf

def _import_keras():
    import tensorflow as tf
    return tf.keras

def _import_sklearn():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import joblib
    return LabelEncoder, train_test_split, joblib


# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(
    sequences_dir: Path = config.SEQUENCES_DIR,
    sequence_length: int = config.SEQUENCE_LENGTH,
    num_features: int = config.NUM_FEATURES,
) -> tuple[np.ndarray, list[str]]:
    """
    Walk sequences_dir looking for label sub-directories, each containing .npy files.

    Returns
    -------
    X : np.ndarray of shape (N, sequence_length, num_features)
    y : list of string labels, length N
    """
    X_list, y_list = [], []

    label_dirs = sorted([d for d in sequences_dir.iterdir() if d.is_dir()])
    if not label_dirs:
        raise FileNotFoundError(
            f"No label directories found in {sequences_dir}.\n"
            "Run src/video_processor.py first to generate sequences."
        )

    for label_dir in label_dirs:
        label    = label_dir.name
        npy_files = sorted(label_dir.glob("*.npy"))
        if not npy_files:
            logger.warning("Label '%s' has no .npy files – skipping.", label)
            continue

        for npy_path in npy_files:
            try:
                arr = np.load(str(npy_path))
                if arr.shape == (sequence_length, num_features):
                    X_list.append(arr)
                    y_list.append(label)
                else:
                    logger.warning(
                        "Shape mismatch in %s: expected (%d, %d), got %s",
                        npy_path, sequence_length, num_features, arr.shape,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load %s: %s", npy_path, exc)

    if not X_list:
        raise ValueError("Dataset is empty after loading. Check data/sequences/.")

    X = np.array(X_list, dtype=np.float32)
    logger.info("Dataset loaded: %d samples, %d classes.", len(y_list), len(set(y_list)))
    return X, y_list


# ─────────────────────────────────────────────────────────────────────────────
def build_model(num_classes: int, sequence_length: int, num_features: int):
    """
    Construct the Conv1D + GRU spatio-temporal hybrid model.

    Architecture
    ────────────
    Input (30, 60)
    → Conv1D(64, kernel=3, relu) + MaxPooling1D(2)
    → GRU(64)
    → Dense(128, relu) → Dropout(0.5)
    → Dense(num_classes, softmax)
    """
    keras = _import_keras()

    model = keras.Sequential([
        keras.Input(shape=(sequence_length, num_features), name="input_layer"),
        keras.layers.Conv1D(64, kernel_size=3, activation="relu",
                            padding="same", name="conv1d_1"),
        keras.layers.MaxPooling1D(pool_size=2, name="maxpool_1"),
        keras.layers.GRU(64, return_sequences=False, name="gru_1"),
        keras.layers.Dense(128, activation="relu", name="dense_1"),
        keras.layers.Dropout(0.5, name="dropout_1"),
        keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="sign_language_model")

    return model


# ─────────────────────────────────────────────────────────────────────────────
def train(
    sequences_dir: Path = config.SEQUENCES_DIR,
    model_path: Path    = config.MODEL_PATH,
    label_enc_path: Path = config.LABEL_ENC_PATH,
    db_path: Path       = config.DB_PATH,
) -> None:
    """
    Full training pipeline:
    load data → encode labels → build model → train → save artefacts → update DB.
    """
    tf = _import_tf()
    keras = _import_keras()
    LabelEncoder, train_test_split, joblib = _import_sklearn()

    # ── Load dataset ─────────────────────────────────────────────────────────
    X, y_str = load_dataset(sequences_dir)

    # ── Encode labels ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_str)
    num_classes = len(le.classes_)

    logger.info("Classes (%d): %s", num_classes, list(le.classes_))

    # One-hot encode
    y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

    # ── Train / validation split ─────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_onehot,
        test_size=config.VALIDATION_SPLIT,
        random_state=42,
        stratify=y_encoded,
    )

    logger.info("Train: %d  Val: %d", len(X_train), len(X_val))

    # ── Build & compile model ─────────────────────────────────────────────────
    model = build_model(num_classes, config.SEQUENCE_LENGTH, config.NUM_FEATURES)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS_BASE,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ModelCheckpoint already saved the best model; ensure model_path exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        model.save(str(model_path))

    # ── Save LabelEncoder ─────────────────────────────────────────────────────
    joblib.dump(le, str(label_enc_path))
    logger.info("LabelEncoder saved to %s", label_enc_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info("Final val_loss=%.4f  val_accuracy=%.4f", val_loss, val_acc)
    print(f"\n[OK] Training complete - val_accuracy: {val_acc * 100:.2f}%")

    # ── Populate SQLite gestures table ────────────────────────────────────────
    _populate_gesture_table(sequences_dir, le.classes_.tolist(), db_path)


# ─────────────────────────────────────────────────────────────────────────────
def _populate_gesture_table(
    sequences_dir: Path,
    classes: list[str],
    db_path: Path,
) -> None:
    """Insert all trained classes into the gestures table."""
    db = DatabaseManager(db_path)
    for label in classes:
        # Determine gesture type: single alphabets are 'static', rest 'dynamic'
        gesture_type = "static" if (len(label) == 1 and label.isalpha()) else "dynamic"

        # Count sample files
        label_dir    = sequences_dir / label
        sample_count = len(list(label_dir.glob("*.npy"))) if label_dir.exists() else 0

        db.add_gesture(
            gesture_name=label,
            gesture_type=gesture_type,
            is_custom=False,
            sample_count=sample_count,
        )

    logger.info("Gesture table updated with %d base classes.", len(classes))


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    train()
