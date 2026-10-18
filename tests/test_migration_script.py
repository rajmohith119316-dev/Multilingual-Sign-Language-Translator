"""
tests/test_migration_script.py
Verifies data transformation routines, type conversions, and validation logic in migrate_postgres_to_mongodb.py.
"""

import unittest
import sys
import os
import base64
from datetime import datetime, timezone
from pathlib import Path
from bson.binary import Binary

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from scripts.migrate_postgres_to_mongodb import SafeMigrationManager


class TestMigrationScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        backup_dir = Path("C:/Users/mohit/OneDrive/Desktop/MajorProject/neon_backup_20261018")
        cls.manager = SafeMigrationManager(
            mongo_uri=config.MONGODB_URI,
            mongo_db=config.MONGODB_DATABASE,
            backup_dir=backup_dir if backup_dir.exists() else None,
            dry_run=True,
        )

    def test_01_id_standardization(self):
        """Test integer ID conversion in transform_record."""
        raw = {"id": "42", "gesture_name": "A"}
        transformed = self.manager.transform_record("gestures", raw)
        self.assertEqual(transformed["id"], 42)
        self.assertIsInstance(transformed["id"], int)

    def test_02_timestamp_iso_string_to_datetime(self):
        """Test ISO timestamp string conversion to datetime in transform_record."""
        raw = {
            "id": 1,
            "timestamp": "2026-10-18T12:30:45.123456+00:00",
            "date_added": datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc),
        }
        transformed = self.manager.transform_record("prediction_history", raw)
        self.assertIsInstance(transformed["timestamp"], datetime)
        self.assertEqual(transformed["timestamp"].tzinfo, timezone.utc)
        self.assertIsInstance(transformed["date_added"], datetime)

    def test_03_binary_model_data_transformation(self):
        """Test binary data bytes and base64 strings in model_versions."""
        mock_bytes = b"MODEL_H5_BINARY_TEST_DATA"
        # 1. Direct bytes
        raw_bytes = {"id": 1, "version_tag": "V1", "model_data": mock_bytes}
        transformed_bytes = self.manager.transform_record("model_versions", raw_bytes)
        self.assertIsInstance(transformed_bytes["model_data"], Binary)
        self.assertEqual(bytes(transformed_bytes["model_data"]), mock_bytes)

        # 2. Base64 encoded string
        b64_str = base64.b64encode(mock_bytes).decode("ascii")
        raw_b64 = {"id": 2, "version_tag": "V2", "model_data": b64_str}
        transformed_b64 = self.manager.transform_record("model_versions", raw_b64)
        self.assertIsInstance(transformed_b64["model_data"], Binary)
        self.assertEqual(bytes(transformed_b64["model_data"]), mock_bytes)

    def test_04_training_samples_binary_transformation(self):
        """Test binary sample_data conversion in training_samples."""
        mock_sample = b"NUMPY_LANDMARKS_ARRAY_BINARY"
        raw = {"id": 100, "sample_data": mock_sample}
        transformed = self.manager.transform_record("training_samples", raw)
        self.assertIsInstance(transformed["sample_data"], Binary)
        self.assertEqual(bytes(transformed["sample_data"]), mock_sample)

    def test_05_live_migration_validation(self):
        """Run validate_migration() against MongoDB Atlas and verify all 10 collections matched."""
        res = self.manager.validate_migration()
        if "error" in res:
            self.skipTest(f"BLOCKED: Live Atlas validation unavailable ({res['error']}). Check Atlas IP access list.")
        self.assertTrue(res.get("all_matched"), "All collections in MongoDB must match source counts")
        for table in self.manager.TABLE_COLLECTION_MAP:
            self.assertIn(table, res)
            self.assertTrue(res[table]["matched"], f"Table '{table}' counts must match: {res[table]}")
            self.assertGreaterEqual(res[table]["dest_count"], 0)


if __name__ == "__main__":
    unittest.main()
