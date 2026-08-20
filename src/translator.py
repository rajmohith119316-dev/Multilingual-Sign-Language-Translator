"""
src/translator.py — Multilingual Translator + Threaded TTS
deep-translator with SQLite cache (0ms on cache hits) + pyttsx3 daemon thread.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class AudioTranslator:
    """
    Thread-safe multilingual translation and TTS.

    Translation strategy
    ─────────────────────
    1. Return as-is for English.
    2. SQLite cache lookup (0ms).
    3. Google Translate via deep-translator (network).
    4. Cache the result.
    5. Graceful fallback to English on network failure.

    TTS strategy
    ─────────────
    • Background daemon thread with pyttsx3.
    • Debounce: same text within cooldown window is suppressed.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        target_language_code: str = "en",
        tts_enabled: bool = config.TTS_ENABLED_DEFAULT,
        cooldown: float = config.TTS_COOLDOWN_SECONDS,
    ) -> None:
        self._db       = db_manager
        self._lang     = target_language_code
        self._tts_on   = tts_enabled
        self._cooldown = cooldown

        self._last_tts_text = ""
        self._last_tts_time = 0.0

        self._tts_lock   = threading.Lock()
        self._tts_queue: list[str] = []
        self._tts_engine = None
        self._tts_ready  = threading.Event()

        self._worker = threading.Thread(
            target=self._tts_worker, daemon=True, name="TTSWorker"
        )
        self._worker.start()

    # ── Language & TTS control ─────────────────────────────────────────────────

    @property
    def target_language(self) -> str:
        return self._lang

    @target_language.setter
    def target_language(self, code: str) -> None:
        self._lang = code

    def set_tts_enabled(self, enabled: bool) -> None:
        self._tts_on = enabled

    # ── Translation ────────────────────────────────────────────────────────────

    def translate(self, english_text: str) -> str:
        if not english_text:
            return english_text
        if self._lang == "en":
            return english_text

        # Cache hit (0ms)
        cached = self._db.get_cached_translation(english_text, self._lang)
        if cached:
            return cached

        # Async background translation to avoid UI freezing
        target_lang = self._lang
        def _bg_translate():
            try:
                from deep_translator import GoogleTranslator
                res = GoogleTranslator(source="en", target=target_lang).translate(english_text)
                if res:
                    self._db.cache_translation(english_text, target_lang, res)
                    if self._tts_on:
                        self.speak(res)
            except Exception as exc:        # noqa: BLE001
                logger.warning("Async translation failed for '%s': %s", english_text, exc)

        threading.Thread(target=_bg_translate, daemon=True, name="AsyncTranslator").start()
        return english_text   # immediate non-blocking response

    # ── TTS ────────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        if not self._tts_on:
            return
        now = time.monotonic()
        if text == self._last_tts_text and (now - self._last_tts_time) < self._cooldown:
            return   # debounce

        self._last_tts_text = text
        self._last_tts_time = now
        with self._tts_lock:
            self._tts_queue.append(text)

    def _tts_worker(self) -> None:
        try:
            if sys.platform.startswith("win"):
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                except Exception:
                    pass
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_SPEECH_RATE)
            engine.setProperty("volume", 1.0)
            self._tts_engine = engine
        except Exception as exc:            # noqa: BLE001
            logger.warning("pyttsx3 init failed: %s", exc)
        finally:
            self._tts_ready.set()

        while True:
            time.sleep(0.05)
            with self._tts_lock:
                text = self._tts_queue.pop(0) if self._tts_queue else None
            if text and self._tts_engine:
                try:
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                except Exception as exc:    # noqa: BLE001
                    logger.warning("TTS error: %s", exc)

    def wait_tts_ready(self, timeout: float = 5.0) -> bool:
        return self._tts_ready.wait(timeout=timeout)

    # ── Combined pipeline ─────────────────────────────────────────────────────

    def translate_and_speak(self, english_text: str) -> str:
        t = self.translate(english_text)
        self.speak(t)
        return t

    def process_prediction(
        self,
        predicted_label: str,
        confidence: float,
    ) -> str:
        """Translate + speak + log to DB. Returns translated text."""
        translated  = self.translate_and_speak(predicted_label)
        lang_name   = _code_to_name(self._lang)
        self._db.log_prediction(
            predicted_label=predicted_label,
            confidence_score=confidence,
            target_language=lang_name,
            translated_text=translated,
        )
        return translated


def _code_to_name(code: str) -> str:
    for name, c in config.SUPPORTED_LANGUAGES.items():
        if c == code:
            return name
    return code
