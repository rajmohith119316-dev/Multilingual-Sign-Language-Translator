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


import json
import urllib.parse
import urllib.request

MYMEMORY_LANG_MAP = {
    "en": "english",
    "kn": "kannada",
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "ml": "malayalam",
    "fr": "french",
    "de": "german",
    "es": "spanish",
}


def _online_translate(text: str, target_lang: str) -> str:
    """Multi-tier translation engine fallback."""
    if not text or target_lang == "en":
        return text

    # Tier 1: Google Translate GTX endpoint (fast 80-150ms, highly reliable, no web scraper blocks)
    try:
        q = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={q}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=3.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            res = "".join(chunk[0] for chunk in data[0] if chunk[0])
            if res:
                return res
    except Exception as exc:
        logger.debug("Google GTX translation failed: %s", exc)

    # Tier 2: MyMemoryTranslator with mapped language name
    try:
        from deep_translator import MyMemoryTranslator
        lang_name = MYMEMORY_LANG_MAP.get(target_lang, target_lang)
        res = MyMemoryTranslator(source="english", target=lang_name).translate(text)
        if res:
            return res
    except Exception as exc:
        logger.debug("MyMemoryTranslator failed: %s", exc)

    # Tier 3: deep_translator GoogleTranslator
    try:
        from deep_translator import GoogleTranslator
        res = GoogleTranslator(source="en", target=target_lang).translate(text)
        if res:
            return res
    except Exception as exc:
        logger.debug("deep_translator GoogleTranslator failed: %s", exc)

    return text


class AudioTranslator:
    """
    Thread-safe multilingual translation and TTS.

    Translation strategy
    ─────────────────────
    1. Return as-is for English.
    2. Built-in vocabulary & alphabet dictionary (0ms latency, 100% offline).
    3. Database cache lookup (0ms latency).
    4. Multi-engine online translator (GTX -> MyMemory -> deep_translator).
    5. Automatic database and memory caching.
    6. Graceful fallback on network failure.

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

        self._pending_translations = set()
        self._last_logged_label = ""
        self._last_logged_lang = ""

        # Load built-in instant translation dictionary
        self._builtin_dict: dict[str, str] = {}
        try:
            dict_file = Path(__file__).resolve().parent / "builtin_translations.json"
            if dict_file.exists():
                with open(dict_file, "r", encoding="utf-8") as f:
                    self._builtin_dict = json.load(f)
                logger.info("Loaded %d pre-cached translations into memory.", len(self._builtin_dict))
        except Exception as exc:
            logger.warning("Could not load builtin_translations.json: %s", exc)

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
        if not enabled:
            with self._tts_lock:
                self._tts_queue.clear()
        self.reset_last_spoken()

    # ── Translation ────────────────────────────────────────────────────────────

    def translate(self, english_text: str) -> str:
        if not english_text or english_text in ("—", "-"):
            return english_text
        if self._lang == "en":
            return english_text

        text_clean = english_text.strip().upper()

        # 1. Built-in instant dictionary lookup (0ms)
        builtin_key = f"{text_clean}_{self._lang}"
        if builtin_key in self._builtin_dict:
            return self._builtin_dict[builtin_key]

        # 2. Database cache hit (0ms)
        try:
            cached = self._db.get_cached_translation(english_text, self._lang)
            if cached:
                # Also store in local memory dict for instant next hit
                self._builtin_dict[builtin_key] = cached
                return cached
        except Exception as db_exc:
            logger.debug("Database cache lookup error: %s", db_exc)

        # 3. If already resolving in background, return current english text immediately
        if (english_text, self._lang) in self._pending_translations:
            return english_text

        self._pending_translations.add((english_text, self._lang))

        # 4. Async background translation to avoid UI freezing
        target_lang = self._lang
        def _bg_translate():
            try:
                res = _online_translate(english_text, target_lang)
                if res and res != english_text:
                    try:
                        self._db.cache_translation(english_text, target_lang, res)
                    except Exception:
                        pass
                    self._builtin_dict[f"{text_clean}_{target_lang}"] = res
                    if self._tts_on:
                        self.speak(res)
            except Exception as exc:        # noqa: BLE001
                logger.warning("Async translation failed for '%s' to '%s': %s", english_text, target_lang, exc)
            finally:
                self._pending_translations.discard((english_text, target_lang))

        threading.Thread(target=_bg_translate, daemon=True, name="AsyncTranslator").start()
        return english_text   # immediate non-blocking response

    # ── TTS ────────────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        if not self._tts_on:
            return
        if not text or text in ("—", "-", "Translating…"):
            return
        # Speak strictly once per gesture — do not repeat until a different gesture is shown
        if text == self._last_tts_text:
            return

        self._last_tts_text = text
        self._last_tts_time = time.monotonic()
        # Only queue to local SAPI5 pyttsx3 engine if text is ASCII (SAPI5 only has English voices on Windows)
        try:
            text.encode("ascii")
            with self._tts_lock:
                self._tts_queue.append(text)
        except UnicodeEncodeError:
            # Non-English/Unicode text is synthesized via high-fidelity get_tts_audio /api/tts endpoint
            pass

    def reset_last_spoken(self) -> None:
        """Reset deduplication state so the next gesture (or re-shown gesture) triggers speech once."""
        self._last_tts_text = ""
        self._last_logged_label = ""

    def get_tts_audio(self, text: str, lang_code: str) -> Optional[bytes]:
        """Fetch or return cached MP3 audio bytes for multilingual speech synthesis (Indic & European)."""
        if not text or text in ("—", "-", "Translating…"):
            return None
        clean_text = text.strip()
        clean_lang = (lang_code or "en").strip().lower()
        cache_key = (clean_text, clean_lang)

        if not hasattr(self, "_audio_cache"):
            self._audio_cache: dict[tuple[str, str], bytes] = {}

        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]

        if len(self._audio_cache) >= 500:
            try:
                oldest = next(iter(self._audio_cache))
                del self._audio_cache[oldest]
            except Exception:
                pass

        import urllib.request
        import urllib.parse
        try:
            url = (
                f"https://translate.google.com/translate_tts?ie=UTF-8"
                f"&tl={urllib.parse.quote(clean_lang)}"
                f"&client=tw-ob"
                f"&q={urllib.parse.quote(clean_text)}"
            )
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Referer": "https://translate.google.com/",
                },
            )
            with urllib.request.urlopen(req, timeout=4.0) as resp:
                audio_bytes = resp.read()
                if len(audio_bytes) > 200:
                    self._audio_cache[cache_key] = audio_bytes
                    return audio_bytes
        except Exception as exc:
            logger.warning("TTS audio fetch failed for '%s' (%s): %s", clean_text, clean_lang, exc)

        return None

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
        
        # Debounce DB logging
        if predicted_label != self._last_logged_label or lang_name != self._last_logged_lang:
            self._db.log_prediction(
                predicted_label=predicted_label,
                confidence_score=confidence,
                target_language=lang_name,
                translated_text=translated,
            )
            self._last_logged_label = predicted_label
            self._last_logged_lang = lang_name
            
        return translated


def _code_to_name(code: str) -> str:
    for name, c in config.SUPPORTED_LANGUAGES.items():
        if c == code:
            return name
    return code
