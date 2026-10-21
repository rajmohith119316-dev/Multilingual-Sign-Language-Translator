"""
tests/test_database_manager_crud.py
Verifies CRUD operations, duplicate prevention, caching, and offline journal resilience in DatabaseManager.
"""

import unittest
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.database_manager import DatabaseManager


class TestDatabaseManagerCRUD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dm = DatabaseManager()
        cls.test_prefix = f"TEST_GESTURE_{uuid.uuid4().hex[:6].upper()}"

    def test_01_backend_is_mongodb(self):
        """Verify DatabaseManager is active with MongoDB Atlas backend."""
        self.assertEqual(self.dm.backend, "mongodb")
        if self.dm.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas live connection unavailable. Host IP needs addition to Atlas Network Access.")
        self.assertIsNotNone(self.dm.mongo_db)

    def test_02_gesture_crud_and_upsert(self):
        """Test adding, listing, duplicate updating, and deleting gestures in MongoDB."""
        if self.dm.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas live connection unavailable. Host IP needs addition to Atlas Network Access.")
        gesture_name = f"{self.test_prefix}_ALPHA"
        
        # 1. Add gesture
        self.dm.add_gesture(
            gesture_name=gesture_name,
            gesture_type="Alphabet",
            is_custom=True,
            sample_count=50,
        )

        # 2. Verify in list
        gestures = self.dm.get_all_gestures()
        found = [g for g in gestures if g.get("gesture_name") == gesture_name]
        self.assertEqual(len(found), 1, "Gesture should be present in get_all_gestures()")
        self.assertEqual(found[0]["sample_count"], 50)
        self.assertTrue(found[0]["is_custom"])

        # 3. Upsert test: Adding again with different sample count updates instead of creating duplicate
        self.dm.add_gesture(
            gesture_name=gesture_name,
            gesture_type="Alphabet",
            is_custom=True,
            sample_count=120,
        )
        gestures = self.dm.get_all_gestures()
        found = [g for g in gestures if g.get("gesture_name") == gesture_name]
        self.assertEqual(len(found), 1, "Upsert should not create duplicate gesture records")
        self.assertEqual(found[0]["sample_count"], 120)

        # 4. Clean up test gesture
        self.dm.delete_gesture(gesture_name)
        gestures_after = self.dm.get_all_gestures()
        found_after = [g for g in gestures_after if g.get("gesture_name") == gesture_name]
        self.assertEqual(len(found_after), 0, "Gesture should be deleted")

    def test_03_prediction_history_logging(self):
        """Test logging predictions to MongoDB prediction_history and retrieving recent items."""
        if self.dm.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas live connection unavailable. Host IP needs addition to Atlas Network Access.")
        test_label = f"PRED_{uuid.uuid4().hex[:6]}"
        self.dm.log_prediction(
            predicted_label=test_label,
            confidence_score=0.9876,
            target_language="es",
            translated_text="Prueba",
        )

        # Fetch recent history
        history = self.dm.get_recent_history(limit=20)
        self.assertIsInstance(history, list)
        matching = [h for h in history if h.get("predicted_label") == test_label]
        self.assertGreaterEqual(len(matching), 1, "Logged prediction must appear in recent history")
        self.assertAlmostEqual(matching[0]["confidence_score"], 0.9876, places=3)
        self.assertEqual(matching[0]["target_language"], "es")
        self.assertEqual(matching[0]["translated_text"], "Prueba")

        # Cleanup test document from MongoDB
        self.dm.mongo_db.prediction_history.delete_many({"predicted_label": test_label})

    def test_04_translation_cache(self):
        """Test in-memory and MongoDB persistent translation caching."""
        unique_text = f"HELLO_{uuid.uuid4().hex[:6]}"
        target_lang = "fr"
        expected_translation = "Bonjour Test"

        # Initially cache should miss
        cached = self.dm.get_cached_translation(unique_text, target_lang)
        self.assertIsNone(cached)

        # Cache the translation
        self.dm.cache_translation(unique_text, target_lang, expected_translation)

        # In-memory hit
        cached_hit = self.dm.get_cached_translation(unique_text, target_lang)
        self.assertEqual(cached_hit, expected_translation)

        if self.dm.mongo_db is None:
            return

        # Clear in-memory dictionary to force MongoDB collection lookup
        self.dm._in_memory_cache.clear()
        db_hit = self.dm.get_cached_translation(unique_text, target_lang)
        self.assertEqual(db_hit, expected_translation, "Translation must be retrieved from MongoDB cache")

        # Cleanup
        self.dm.mongo_db.translation_cache.delete_many({"english_text": unique_text})

    def test_05_get_stats(self):
        """Test get_stats returns correct structure with mongodb backend."""
        stats = self.dm.get_stats()
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats.get("backend"), "mongodb")
        for key in ["total_gestures", "alphabet_gestures", "word_gestures", "history_entries", "cache_entries"]:
            self.assertIn(key, stats)
            self.assertGreaterEqual(stats[key], 0)

    def test_06_offline_journal_resilience(self):
        """Verify prediction logging fails over to local offline journal during outage and flushes upon recovery."""
        journal_file = config.LOGS_DIR / "offline_prediction_journal.jsonl"
        # Ensure clean initial state
        if journal_file.exists():
            journal_file.unlink()

        offline_label = f"OFFLINE_{uuid.uuid4().hex[:6]}"
        real_mongo_db = self.dm.mongo_db
        real_dsn = self.dm.dsn

        try:
            # Simulate temporary database outage
            self.dm.mongo_db = None
            self.dm.dsn = None

            self.dm.log_prediction(
                predicted_label=offline_label,
                confidence_score=0.95,
                target_language="de",
                translated_text="Hallo Offline",
            )

            # Check journal file exists and has content
            self.assertTrue(journal_file.exists(), "Offline journal file must be created on outage")
            content = journal_file.read_text(encoding="utf-8")
            self.assertIn(offline_label, content)

            # Restore connection
            self.dm.mongo_db = real_mongo_db
            self.dm.dsn = real_dsn

            # Next log_prediction or explicit flush drains journal
            if real_mongo_db is not None:
                self.dm._flush_journal(journal_file)
                # Check journal is drained
                self.assertFalse(journal_file.exists(), "Journal file should be deleted or empty after flushing")
                # Check record is in MongoDB
                record = self.dm.mongo_db.prediction_history.find_one({"predicted_label": offline_label})
                self.assertIsNotNone(record, "Offline journaled prediction must appear in MongoDB after reconnection")
                self.assertEqual(record["translated_text"], "Hallo Offline")
                # Clean up
                self.dm.mongo_db.prediction_history.delete_many({"predicted_label": offline_label})
        finally:
            self.dm.mongo_db = real_mongo_db
            self.dm.dsn = real_dsn
            if journal_file.exists():
                journal_file.unlink()


if __name__ == "__main__":
    unittest.main()
