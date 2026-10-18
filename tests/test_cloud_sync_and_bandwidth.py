"""
tests/test_cloud_sync_and_bandwidth.py
Verifies MongoDB-backed cloud model synchronization, bandwidth reduction via metadata-only polling,
and local backup rollback mechanisms.
"""

import unittest
import sys
import json
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.cloud_sync import sync_active_models_from_db, _get_local_sync_versions, _save_local_sync_version
from src.auth_models import UserManager


class TestCloudSyncAndBandwidth(unittest.TestCase):

    def test_01_cloud_sync_state_file_valid(self):
        """Verify models/cloud_sync_state.json exists and contains active version tags."""
        state = _get_local_sync_versions()
        self.assertIsInstance(state, dict)
        self.assertIn("Alphabet", state)
        self.assertIn("Word", state)

    def test_02_sync_queries_metadata_without_downloading_when_up_to_date(self):
        """Verify polling checks version tags without downloading large model binary data."""
        with patch.object(UserManager, "get_model_versions", wraps=UserManager().get_model_versions) as spy_meta, \
             patch.object(UserManager, "get_active_model_version", wraps=UserManager().get_active_model_version) as spy_full:
            res = sync_active_models_from_db(force=False)
            self.assertIsInstance(res, dict)
            self.assertIn("active_tags", res)
            self.assertIn("synced", res)

            # When local models match cloud versions, synced is False (no redundant re-download)
            self.assertFalse(res["synced"], "Unchanged model must not trigger redundant download")
            self.assertEqual(len(res["updated"]), 0, "No models should be updated when versions match")

            # Verify that get_model_versions was called with include_data=False during polling
            self.assertGreater(spy_meta.call_count, 0)
            kwargs = spy_meta.call_args[1]
            self.assertFalse(kwargs.get("include_data", True), "Metadata polling must omit binary data")

            # Verify get_active_model_version (which downloads binaries) was NOT called
            spy_full.assert_not_called()

    def test_03_active_tags_in_mongodb(self):
        """Verify active model tags exist in MongoDB Atlas model_versions collection."""
        um = UserManager()
        if um.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas live connection unavailable. Check Atlas IP access list.")
        alpha_v = um.get_active_model_version("Alphabet", include_data=False)
        self.assertIsNotNone(alpha_v, "Active Alphabet model version must exist in MongoDB")
        self.assertTrue(alpha_v.get("is_active"))
        self.assertIn("Alphabet", alpha_v.get("version_tag", ""))

        word_v = um.get_active_model_version("Word", include_data=False)
        self.assertIsNotNone(word_v, "Active Word model version must exist in MongoDB")
        self.assertTrue(word_v.get("is_active"))
        self.assertIn("Word", word_v.get("version_tag", ""))


if __name__ == "__main__":
    unittest.main()
