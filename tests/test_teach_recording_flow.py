"""
tests/test_teach_recording_flow.py
Unit tests verifying the Teach Custom Sign recording workflow:
- Start recording validation & state transition from idle
- Frame collection and feature accumulation
- Recording completion and sample saving
- Reset endpoint behavior
"""

import unittest
import sys
import numpy as np
import base64
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from web_app import app, rec_state, rec_lock, _finish_recording


class TestTeachRecordingFlow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def setUp(self):
        # Reset state before each test
        with rec_lock:
            rec_state["active"] = False
            rec_state["label"] = ""
            rec_state["type"] = "Word"
            rec_state["frame_buf"] = []
            rec_state["recordings"] = []
            rec_state["status"] = "Ready — click 'Start Recording' to begin"
            rec_state["train_progress"] = ""
            rec_state["train_running"] = False

    def test_01_start_recording_requires_label(self):
        """Empty or whitespace-only label must return 400 Bad Request."""
        res = self.client.post("/api/start_recording", json={"label": "   ", "type": "Word"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    def test_02_start_recording_activates_state(self):
        """Valid start_recording request activates rec_state and initializes frame buffer."""
        res = self.client.post("/api/start_recording", json={"label": "HELLO", "type": "Word"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("ok"))

        with rec_lock:
            self.assertTrue(rec_state["active"])
            self.assertEqual(rec_state["label"], "HELLO")
            self.assertEqual(rec_state["type"], "Word")
            self.assertEqual(len(rec_state["frame_buf"]), 0)
            self.assertIn("Recording (0/", rec_state["status"])

    def test_03_cannot_start_recording_while_already_active(self):
        """Second start_recording call while already active must return 400."""
        self.client.post("/api/start_recording", json={"label": "HELLO", "type": "Word"})
        res = self.client.post("/api/start_recording", json={"label": "WORLD", "type": "Word"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("already in progress", res.get_json()["error"])

    def test_04_finish_recording_word_saves_sample_and_resets_active(self):
        """When 30 frames are captured for Word, _finish_recording appends to recordings."""
        with rec_lock:
            rec_state["active"] = True
            rec_state["label"] = "HELLO"
            rec_state["type"] = "Word"
            # Add 30 dummy feature vectors (123 dims each)
            rec_state["frame_buf"] = [np.ones(123, dtype=np.float32) for _ in range(30)]

            _finish_recording()

            self.assertFalse(rec_state["active"])
            self.assertEqual(len(rec_state["recordings"]), 1)
            self.assertEqual(rec_state["recordings"][0].shape, (30, 123))
            self.assertIn("✅ Recording 1/", rec_state["status"])

    def test_05_finish_recording_alphabet_saves_sample(self):
        """When frames are captured for Alphabet, _finish_recording saves frame list."""
        with rec_lock:
            rec_state["active"] = True
            rec_state["label"] = "A"
            rec_state["type"] = "Alphabet"
            # Add 30 dummy feature vectors (60 dims each)
            rec_state["frame_buf"] = [np.ones(60, dtype=np.float32) for _ in range(30)]

            _finish_recording()

            self.assertFalse(rec_state["active"])
            self.assertEqual(len(rec_state["recordings"]), 1)
            self.assertIn("✅ Recording 1/", rec_state["status"])

    def test_06_finish_recording_discards_if_too_few_frames(self):
        """If fewer than 15 frames are captured, sample must be discarded with feedback."""
        with rec_lock:
            rec_state["active"] = True
            rec_state["label"] = "TEST"
            rec_state["type"] = "Word"
            rec_state["frame_buf"] = [np.ones(123, dtype=np.float32) for _ in range(5)]

            _finish_recording()

            self.assertFalse(rec_state["active"])
            self.assertEqual(len(rec_state["recordings"]), 0)
            self.assertIn("Recording discarded", rec_state["status"])

    def test_07_reset_clears_all_recording_state(self):
        """POST /api/reset resets recordings count, frame buffer, and status."""
        with rec_lock:
            rec_state["active"] = True
            rec_state["label"] = "TEST"
            rec_state["recordings"] = [np.ones((30, 123))]

        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)

        with rec_lock:
            self.assertFalse(rec_state["active"])
            self.assertEqual(rec_state["label"], "")
            self.assertEqual(len(rec_state["recordings"]), 0)
            self.assertIn("Ready", rec_state["status"])


if __name__ == "__main__":
    unittest.main()
