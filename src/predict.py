"""
src/predict.py — Hierarchical MoE Real-Time Gesture Predictor
Router Logic
────────────
 0 hands detected  → status "IDLE"
 1 hand detected   → 60-D features → Alphabet Expert → 5-frame stability filter
 2 hands detected  → 123-D features → deque(30) → Word Expert → stability filter

Hot-reload supported after continual learning.
Thread-safe via RLock.
"""

import collections
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.utils import (
    extract_single_hand_features,
    extract_dual_hand_features,
)

logger = logging.getLogger(__name__)

# Router mode constants
MODE_IDLE     = "IDLE"
MODE_ALPHABET = "ALPHABET"
MODE_WORD     = "WORD"


# ─────────────────────────────────────────────────────────────────────────────
class GesturePredictor:
    """
    Mixture-of-Experts real-time predictor.

    Usage
    ─────
    predictor = GesturePredictor()

    Per frame:
        result = predictor.update(frame)
        # result is None, or (label, confidence, mode)
    """

    def __init__(
        self,
        alphabet_model_path: Path  = config.ALPHABET_MODEL_PATH,
        alphabet_enc_path: Path    = config.ALPHABET_ENCODER_PATH,
        word_model_path: Path      = config.WORD_MODEL_PATH,
        word_enc_path: Path        = config.WORD_ENCODER_PATH,
        task_path: Path            = config.TASK_PATH,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        stability_frames: int      = config.STABILITY_FRAMES,
        sequence_length: int       = config.SEQUENCE_LENGTH,
        word_num_features: int     = config.WORD_NUM_FEATURES,
    ) -> None:
        self._alpha_model_path = alphabet_model_path
        self._alpha_enc_path   = alphabet_enc_path
        self._word_model_path  = word_model_path
        self._word_enc_path    = word_enc_path
        self._task_path        = task_path
        self._conf_thresh      = confidence_threshold
        self._stab_frames      = stability_frames
        self._seq_len          = sequence_length
        self._word_feat        = word_num_features

        self._lock = threading.RLock()

        # Model artefacts
        self._alpha_model = None
        self._alpha_predict_fn = None
        self._alpha_le    = None
        self._word_model  = None
        self._word_predict_fn = None
        self._word_le     = None
        self._landmarker  = None
        self._last_timestamp_ms: int = 0

        # Sliding window for Word Expert
        self._word_buffer: collections.deque = collections.deque(maxlen=sequence_length)

        # Stability filter state (per expert)
        self._alpha_last:   str = ""
        self._alpha_stable: int = 0
        self._word_last:    str = ""
        self._word_stable:  int = 0

        # Current router mode (for GUI badge)
        self._current_mode: str = MODE_IDLE

        # Load everything
        self._load_all()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        self._load_artefacts()
        self._load_landmarker()

    def _load_artefacts(self) -> None:
        try:
            import tensorflow as tf
            import joblib

            with self._lock:
                # Alphabet Expert
                if self._alpha_model_path.exists():
                    self._alpha_model = tf.keras.models.load_model(
                        str(self._alpha_model_path), compile=False)
                    @tf.function(experimental_relax_shapes=True)
                    def _alpha_pred(x):
                        return self._alpha_model(x, training=False)
                    self._alpha_predict_fn = _alpha_pred

                    # Build ultra-fast NumPy evaluator for Alphabet Expert (Dense MLP)
                    try:
                        weights = self._alpha_model.get_weights()
                        if len(weights) == 6:
                            W1, b1, W2, b2, W3, b3 = [w.astype(np.float32) for w in weights]
                            def _fast_alpha_eval(feat_vec):
                                h1 = np.maximum(0.0, feat_vec @ W1 + b1)
                                h2 = np.maximum(0.0, h1 @ W2 + b2)
                                logits = h2 @ W3 + b3
                                exp_l = np.exp(logits - np.max(logits))
                                return exp_l / np.sum(exp_l)
                            self._fast_alpha_eval_fn = _fast_alpha_eval
                        else:
                            self._fast_alpha_eval_fn = None
                    except Exception:
                        self._fast_alpha_eval_fn = None

                    logger.info("Alphabet Expert loaded (%s classes).",
                                len(joblib.load(str(self._alpha_enc_path)).classes_)
                                if self._alpha_enc_path.exists() else "?")
                else:
                    self._alpha_model = None
                    self._alpha_predict_fn = None
                    self._fast_alpha_eval_fn = None
                    logger.warning("Alphabet model not found: %s", self._alpha_model_path)

                if self._alpha_enc_path.exists():
                    self._alpha_le = joblib.load(str(self._alpha_enc_path))
                else:
                    self._alpha_le = None
                    logger.warning("Alphabet encoder not found: %s", self._alpha_enc_path)

                # Word Expert
                if self._word_model_path.exists():
                    self._word_model = tf.keras.models.load_model(
                        str(self._word_model_path), compile=False)
                    @tf.function(experimental_relax_shapes=True)
                    def _word_pred(x):
                        return self._word_model(x, training=False)
                    self._word_predict_fn = _word_pred
                    logger.info("Word Expert loaded.")
                else:
                    self._word_model = None
                    self._word_predict_fn = None
                    logger.warning("Word model not found: %s", self._word_model_path)

                if self._word_enc_path.exists():
                    self._word_le = joblib.load(str(self._word_enc_path))
                else:
                    self._word_le = None
                    logger.warning("Word encoder not found: %s", self._word_enc_path)

        except Exception as exc:            # noqa: BLE001
            logger.error("Artefact load error: %s", exc)

    def _load_landmarker(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision.hand_landmarker import (
                HandLandmarker, HandLandmarkerOptions,
            )
            if not self._task_path.exists():
                logger.error("hand_landmarker.task not found: %s", self._task_path)
                return

            options = HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self._task_path)
                ),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=config.MP_NUM_HANDS,
                min_hand_detection_confidence=config.MP_MIN_HAND_DETECTION_CONF,
                min_hand_presence_confidence=config.MP_MIN_HAND_PRESENCE_CONF,
                min_tracking_confidence=config.MP_MIN_TRACKING_CONF,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._last_timestamp_ms = 0
            logger.info("MediaPipe HandLandmarker ready in VIDEO mode (num_hands=%d).",
                        config.MP_NUM_HANDS)
        except Exception as exc:            # noqa: BLE001
            logger.error("MediaPipe init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def hot_reload(self) -> None:
        """Reload both experts from disk. Thread-safe. Call after continual learning."""
        logger.info("Hot-reloading MoE artefacts…")
        self._load_artefacts()
        self.reset_buffers()
        logger.info("Hot-reload complete.")

    def reset_buffers(self) -> None:
        with self._lock:
            self._word_buffer.clear()
            self._alpha_last    = ""
            self._alpha_stable  = 0
            self._word_last     = ""
            self._word_stable   = 0
            self._current_mode  = MODE_IDLE

    @property
    def current_mode(self) -> str:
        with self._lock:
            return self._current_mode

    @property
    def word_buffer_fill(self) -> int:
        with self._lock:
            return len(self._word_buffer)

    @property
    def is_alphabet_ready(self) -> bool:
        with self._lock:
            return self._alpha_model is not None and self._alpha_le is not None

    @property
    def is_word_ready(self) -> bool:
        with self._lock:
            return self._word_model is not None and self._word_le is not None

    @property
    def known_alphabets(self) -> list[str]:
        with self._lock:
            return self._alpha_le.classes_.tolist() if (self._alpha_le and hasattr(self._alpha_le, 'classes_')) else []

    @property
    def known_words(self) -> list[str]:
        with self._lock:
            return self._word_le.classes_.tolist() if (self._word_le and hasattr(self._word_le, 'classes_')) else []

    def close(self) -> None:
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:               # noqa: BLE001
                pass

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_hands(self, bgr_frame: np.ndarray):
        """
        Run MediaPipe on a BGR frame using VIDEO mode temporal tracking.
        Pads non-square frames to 1:1 square aspect ratio (e.g. 640x640)
        so MediaPipe's C++ landmark_projection_calculator receives a 1:1 ROI,
        preventing fallback palm re-detections and eliminating C++ stderr warnings.

        Returns (left_landmarks, right_landmarks, num_hands)
        Each landmark list is None if that hand is not detected.
        """
        if self._landmarker is None:
            return None, None, 0

        try:
            import time
            import mediapipe as mp

            h, w = bgr_frame.shape[:2]
            if h != w:
                dim = max(h, w)
                top_pad = (dim - h) // 2
                left_pad = (dim - w) // 2
                padded = np.zeros((dim, dim, 3), dtype=bgr_frame.dtype)
                padded[top_pad:top_pad + h, left_pad:left_pad + w] = bgr_frame
                rgb = np.ascontiguousarray(padded[:, :, ::-1])
            else:
                top_pad = 0
                left_pad = 0
                dim = h
                rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            now_ms = int(time.monotonic() * 1000)
            if now_ms <= self._last_timestamp_ms:
                now_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = now_ms

            result = self._landmarker.detect_for_video(mp_img, now_ms)
            if result is None or not hasattr(result, "hand_landmarks") or not result.hand_landmarks:
                return None, None, 0
        except Exception as exc:            # noqa: BLE001
            logger.debug("detect_hands error: %s", exc)
            return None, None, 0

        # Unpad normalized coordinates back to original frame (w, h)
        def _unpad_lm_list(landmarks):
            if landmarks is None:
                return None
            unpadded = []
            for lm in landmarks:
                orig_x = (lm.x * dim - left_pad) / float(w)
                orig_y = (lm.y * dim - top_pad) / float(h)
                unpadded.append(mp.tasks.components.containers.NormalizedLandmark(
                    x=orig_x, y=orig_y, z=lm.z
                ))
            return unpadded

        left_lm  = None
        right_lm = None

        for i, lm_list in enumerate(result.hand_landmarks):
            if i >= len(result.handedness):
                break
            side = result.handedness[i][0].category_name.lower()
            unpadded = _unpad_lm_list(lm_list)
            # MediaPipe uses mirrored convention: swap for natural view
            if side == "left":
                right_lm = unpadded
            else:
                left_lm  = unpadded

        n_hands = (1 if left_lm is not None else 0) + (1 if right_lm is not None else 0)
        return left_lm, right_lm, n_hands

    # ── Main Per-Frame Update ─────────────────────────────────────────────────

    def update(
        self,
        bgr_frame: np.ndarray,
    ) -> tuple[Optional[tuple[str, float, str]], Optional[list], Optional[list], int]:
        """
        Process one BGR camera frame through the MoE router.

        Returns
        -------
        (result_tuple_or_None, left_lm, right_lm, n_hands)
        """
        with self._lock:
            left_lm, right_lm, n_hands = self.detect_hands(bgr_frame)

            # ── 0 hands ─────────────────────────────────────────────────────
            if n_hands == 0:
                self._current_mode = MODE_IDLE
                self._alpha_stable = 0
                self._word_stable  = 0
                self._word_buffer.clear()
                return None, left_lm, right_lm, n_hands

            # ── 1 hand → Alphabet Expert ─────────────────────────────────────
            if n_hands == 1:
                self._current_mode = MODE_ALPHABET
                self._word_stable  = 0

                # Use whichever hand is present
                active_lm = left_lm if left_lm is not None else right_lm
                feat = extract_single_hand_features(active_lm)
                if feat is None:
                    return None, left_lm, right_lm, n_hands

                return self._run_alphabet(feat), left_lm, right_lm, n_hands

            # ── 2 hands → Word Expert ─────────────────────────────────────────
            self._current_mode = MODE_WORD
            self._alpha_stable = 0

            feat = extract_dual_hand_features(left_lm, right_lm)   # (123,)
            self._word_buffer.append(feat)

            if len(self._word_buffer) < self._seq_len:
                return None, left_lm, right_lm, n_hands   # Still filling the buffer

            return self._run_word(), left_lm, right_lm, n_hands

    # ── Expert Runners ────────────────────────────────────────────────────────

    def _run_alphabet(self, feat: np.ndarray) -> Optional[tuple[str, float, str]]:
        """Run Alphabet Expert and apply stability filter."""
        if not self.is_alphabet_ready:
            return None
        try:
            if getattr(self, "_fast_alpha_eval_fn", None) is not None:
                probs = self._fast_alpha_eval_fn(feat)
            elif self._alpha_predict_fn is not None:
                inp = feat[np.newaxis, ...]
                probs = self._alpha_predict_fn(inp).numpy()[0]
            else:
                inp = feat[np.newaxis, ...]
                probs = self._alpha_model(inp, training=False).numpy()[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])

            if conf < self._conf_thresh:
                self._alpha_stable = 0
                return None

            label = str(self._alpha_le.inverse_transform([idx])[0])

            if label == self._alpha_last:
                self._alpha_stable += 1
            else:
                self._alpha_last   = label
                self._alpha_stable = 1

            if self._alpha_stable >= self._stab_frames:
                self._alpha_stable = 0
                return (label, conf, MODE_ALPHABET)

        except Exception as exc:            # noqa: BLE001
            logger.error("Alphabet prediction error: %s", exc)
        return None

    def _run_word(self) -> Optional[tuple[str, float, str]]:
        """Run Word Expert on the current 30-frame buffer and apply stability filter."""
        if not self.is_word_ready:
            return None
        try:
            sequence = np.array(self._word_buffer, dtype=np.float32)  # (30, 123)
            inp      = sequence[np.newaxis, ...]                       # (1, 30, 123)
            if self._word_predict_fn is not None:
                probs = self._word_predict_fn(inp).numpy()[0]
            else:
                probs = self._word_model(inp, training=False).numpy()[0]
            idx      = int(np.argmax(probs))
            conf     = float(probs[idx])

            if conf < self._conf_thresh:
                self._word_stable = 0
                return None

            label = str(self._word_le.inverse_transform([idx])[0])

            if label == self._word_last:
                self._word_stable += 1
            else:
                self._word_last   = label
                self._word_stable = 1

            if self._word_stable >= self._stab_frames:
                self._word_stable = 0
                return (label, conf, MODE_WORD)

        except Exception as exc:            # noqa: BLE001
            logger.error("Word prediction error: %s", exc)
        return None
