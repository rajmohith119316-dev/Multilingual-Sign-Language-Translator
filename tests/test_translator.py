"""
tests/test_translator.py
Verifies multilingual translation caching in memory and MongoDB Atlas,
language mapping, and audio synthesizer resilience.
"""

import unittest
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager
from src.translator import AudioTranslator, _code_to_name


class TestTranslator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dm = DatabaseManager()
        cls.translator = AudioTranslator(db_manager=cls.dm, tts_enabled=False)

    def test_01_language_code_mapping(self):
        """Verify _code_to_name maps language codes to human-readable names."""
        self.assertEqual(_code_to_name("es"), "Spanish")
        self.assertEqual(_code_to_name("fr"), "French")
        self.assertEqual(_code_to_name("de"), "German")
        self.assertEqual(_code_to_name("hi"), "Hindi")

    def test_02_database_translation_caching(self):
        """Verify translation queries database cache and retrieves stored translation."""
        if self.dm.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas live connection unavailable. Check Atlas IP access list.")
        unique_word = f"TESTWORD_{uuid.uuid4().hex[:6]}"
        expected_tx = f"TRADUCTION_{uuid.uuid4().hex[:4]}"

        # Pre-populate MongoDB cache
        self.dm.cache_translation(unique_word, "fr", expected_tx)

        # Clear in-memory dictionary on translator
        self.dm._in_memory_cache.clear()

        # Query translation
        cached = self.dm.get_cached_translation(unique_word, "fr")
        self.assertEqual(cached, expected_tx, "Translation must be retrieved from MongoDB Atlas cache")

        # Clean up
        self.dm.mongo_db.translation_cache.delete_many({"english_text": unique_word})

    def test_03_prediction_debouncing(self):
        """Verify process_prediction debounces repeat predictions to avoid duplicate database writes."""
        from unittest.mock import patch
        with patch.object(self.dm, "log_prediction") as mock_log:
            # First call logs
            t1 = self.translator.process_prediction("B", confidence=0.98)
            self.assertEqual(mock_log.call_count, 1)

            # Second consecutive call with same label does not duplicate log
            t2 = self.translator.process_prediction("B", confidence=0.99)
            self.assertEqual(mock_log.call_count, 1, "Debounce must prevent duplicate DB write for consecutive same label")


if __name__ == "__main__":
    unittest.main()
