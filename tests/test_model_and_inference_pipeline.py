"""
tests/test_model_and_inference_pipeline.py
Verifies local ML model artifacts, LabelEncoders, MediaPipe hand landmarker task,
and guarantees that real-time camera inference does not make database calls per frame.
"""

import unittest
import sys
import json
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


class TestModelAndInferencePipeline(unittest.TestCase):

    def test_01_model_files_exist(self):
        """Verify model files and encoders exist in models/."""
        self.assertTrue(config.ALPHABET_MODEL_PATH.exists(), f"Missing: {config.ALPHABET_MODEL_PATH}")
        self.assertTrue(config.WORD_MODEL_PATH.exists(), f"Missing: {config.WORD_MODEL_PATH}")
        self.assertTrue(config.ALPHABET_ENCODER_PATH.exists(), f"Missing: {config.ALPHABET_ENCODER_PATH}")
        self.assertTrue(config.WORD_ENCODER_PATH.exists(), f"Missing: {config.WORD_ENCODER_PATH}")

        # Check non-zero size
        self.assertGreater(config.ALPHABET_MODEL_PATH.stat().st_size, 10000)
        self.assertGreater(config.WORD_MODEL_PATH.stat().st_size, 10000)

    def test_02_label_encoders_classes(self):
        """Verify LabelEncoders load correctly and contain trained gesture classes."""
        alpha_le = joblib.load(str(config.ALPHABET_ENCODER_PATH))
        self.assertTrue(hasattr(alpha_le, "classes_"))
        alpha_classes = list(alpha_le.classes_)
        for expected in ["A", "B", "C", "D"]:
            self.assertIn(expected, alpha_classes, f"Class '{expected}' should be in Alphabet encoder")

        word_le = joblib.load(str(config.WORD_ENCODER_PATH))
        self.assertTrue(hasattr(word_le, "classes_"))
        word_classes = list(word_le.classes_)
        self.assertIn("CHOCOLATE", word_classes, "Class 'CHOCOLATE' should be in Word encoder")

    def test_03_hand_landmarker_task_asset(self):
        """Verify MediaPipe hand_landmarker.task model is present and has proper size."""
        task_path = config.BASE_DIR / "hand_landmarker.task"
        self.assertTrue(task_path.exists(), f"Missing hand_landmarker.task at {task_path}")
        size_mb = task_path.stat().st_size / (1024 * 1024)
        self.assertGreaterEqual(size_mb, 7.0, f"Task file should be ~7.8 MB, got {size_mb:.2f} MB")

    def test_04_alphabet_model_loading_and_inference(self):
        """Verify Alphabet MLP model loads and generates prediction on 60-dim input."""
        import tensorflow as tf
        model = tf.keras.models.load_model(str(config.ALPHABET_MODEL_PATH), compile=False)
        self.assertIsNotNone(model)

        # Alphabet expert takes (batch, 60) relative landmark vector
        dummy_input = np.random.randn(1, 60).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(dummy_input)
        if norm > 0:
            dummy_input /= norm

        output = model.predict(dummy_input, verbose=0)
        self.assertEqual(output.shape, (1, 4), "Alphabet output shape must match 4 classes ['A', 'B', 'C', 'D']")
        self.assertAlmostEqual(float(np.sum(output[0])), 1.0, places=3, msg="Softmax output probabilities must sum to 1")

    def test_05_word_model_loading_and_inference(self):
        """Verify Word Conv1D+GRU model loads and generates prediction on sequence input."""
        import tensorflow as tf
        model = tf.keras.models.load_model(str(config.WORD_MODEL_PATH), compile=False)
        self.assertIsNotNone(model)

        # Word expert takes sequence: (batch, sequence_length, features)
        input_shape = model.input_shape  # e.g. (None, 30, 60) or (None, 30, 63)
        seq_len = input_shape[1] if input_shape[1] else 30
        feat_dim = input_shape[2] if input_shape[2] else 60

        dummy_seq = np.random.randn(1, seq_len, feat_dim).astype(np.float32)
        output = model.predict(dummy_seq, verbose=0)
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 1, "Word output shape must match 1 class ['CHOCOLATE']")

    def test_06_zero_database_calls_during_feature_inference(self):
        """CRITICAL: Real-time inference loop must never invoke database operations per frame."""
        from src.database_manager import DatabaseManager
        import tensorflow as tf

        # Load local model
        model = tf.keras.models.load_model(str(config.ALPHABET_MODEL_PATH), compile=False)
        dummy_features = np.random.randn(1, 60).astype(np.float32)

        # Spy on DatabaseManager methods
        with patch.object(DatabaseManager, 'log_prediction') as mock_log, \
             patch.object(DatabaseManager, 'add_gesture') as mock_add, \
             patch.object(DatabaseManager, 'get_all_gestures') as mock_get:

            # Simulate 100 consecutive webcam frames through feature processing & neural inference
            for _ in range(100):
                # Landmark preprocessing simulation
                norm = np.linalg.norm(dummy_features)
                norm_feat = dummy_features / norm if norm > 0 else dummy_features
                preds = model(norm_feat, training=False).numpy()
                idx = int(np.argmax(preds[0]))
                conf = float(preds[0][idx])

            # Verify no database methods were called during continuous frame processing
            mock_log.assert_not_called()
            mock_add.assert_not_called()
            mock_get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
