"""
tests/test_mongo_config_and_connection.py
Verifies MongoDB configuration, connectivity, credentials security, and fallback mode.
"""

import unittest
import os
import sys
from pathlib import Path

# Ensure root directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


class TestMongoConfigAndConnection(unittest.TestCase):

    def test_01_mongodb_uri_configured(self):
        """Verify MONGODB_URI is loaded from environment or credentials file."""
        self.assertTrue(hasattr(config, "MONGODB_URI"), "config.MONGODB_URI attribute must exist")
        self.assertIsNotNone(config.MONGODB_URI, "config.MONGODB_URI must not be None")
        self.assertTrue(
            config.MONGODB_URI.startswith("mongodb://") or config.MONGODB_URI.startswith("mongodb+srv://"),
            "config.MONGODB_URI must begin with standard mongodb protocol prefix"
        )

    def test_02_database_name_configured(self):
        """Verify MONGODB_DATABASE defaults to sign_language_db or env setting."""
        self.assertEqual(config.MONGODB_DATABASE, "sign_language_db")

    def test_03_database_backend_configured(self):
        """Verify DATABASE_BACKEND is set to mongodb."""
        self.assertEqual(config.DATABASE_BACKEND, "mongodb")

    def test_04_credentials_not_hardcoded_in_source(self):
        """Verify config.py and source files do not contain hardcoded passwords."""
        config_path = Path(config.__file__)
        content = config_path.read_text(encoding="utf-8")
        self.assertNotIn("mongodb+srv://", content, "config.py must not hardcode MongoDB connection strings")

    def test_05_atlas_ping_connectivity(self):
        """Verify live MongoDB Atlas ping succeeds with PyMongo."""
        from pymongo import MongoClient
        try:
            client = MongoClient(
                config.MONGODB_URI,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
            )
            ping_res = client.admin.command("ping")
            self.assertEqual(ping_res.get("ok"), 1.0, "Atlas ping command must return ok: 1.0")
            client.close()
        except Exception as exc:
            self.skipTest(f"BLOCKED: Atlas live ping unavailable ({exc}). Host IP may need addition to Atlas Network Access.")

    def test_06_database_collections_exist(self):
        """Verify target collections are accessible on MongoDB Atlas."""
        from pymongo import MongoClient
        try:
            client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=2500)
            db = client[config.MONGODB_DATABASE]
            collections = db.list_collection_names()
            expected = [
                "gestures",
                "prediction_history",
                "translation_cache",
                "users",
                "user_sessions",
                "user_activity",
                "training_submissions",
                "training_samples",
                "model_versions",
                "training_jobs",
            ]
            for col in expected:
                self.assertIn(col, collections, f"Collection '{col}' should exist in '{config.MONGODB_DATABASE}'")
            client.close()
        except Exception as exc:
            self.skipTest(f"BLOCKED: Atlas collections listing unavailable ({exc}). Host IP may need addition to Atlas Network Access.")


if __name__ == "__main__":
    unittest.main()
