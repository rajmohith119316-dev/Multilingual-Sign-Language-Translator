"""
tests/test_user_manager_crud.py
Verifies UserManager operations on MongoDB Atlas:
- User authentication & password hashing
- Session management
- User activity audit trail
- Submissions and dataset samples
- Model versions metadata
- Training jobs tracking
"""

import unittest
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.auth_models import UserManager


class TestUserManagerCRUD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.um = UserManager()
        cls.test_uid = uuid.uuid4().hex[:8]
        cls.test_username = f"test_user_{cls.test_uid}"
        cls.test_email = f"test_{cls.test_uid}@example.com"
        cls.test_password = "SecurePassword123!"

    def setUp(self):
        if self._testMethodName != "test_01_backend_is_mongodb" and self.um.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas cluster is not reachable from current IP. Whitelist current IP in MongoDB Atlas Network Access.")

    def test_01_backend_is_mongodb(self):
        """Verify UserManager connects to MongoDB Atlas."""
        self.assertEqual(self.um.backend, "mongodb")
        if self.um.mongo_db is None:
            self.skipTest("BLOCKED: MongoDB Atlas cluster is not reachable from current IP. Whitelist current IP in MongoDB Atlas Network Access.")
        self.assertIsNotNone(self.um.mongo_db)

    def test_02_user_create_auth_and_duplicate(self):
        """Test creating user, password verification, and duplicate rejection."""
        # 1. Create user
        user = self.um.create_user(
            full_name="Test Suite User",
            username=self.test_username,
            email=self.test_email,
            password=self.test_password,
            role="USER",
        )
        self.assertIsNotNone(user, "User creation must succeed")
        user_id = user["id"]

        # 2. Duplicate username rejection
        dup = self.um.create_user(
            full_name="Duplicate User",
            username=self.test_username,
            email=f"other_{self.test_uid}@example.com",
            password="OtherPassword",
        )
        self.assertIsNone(dup, "Duplicate username must be rejected")

        # 3. Retrieve by username
        fetched = self.um.get_user_by_username(self.test_username)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["email"], self.test_email)
        self.assertEqual(fetched["role"], "USER")

        # 4. Verify password
        authenticated = self.um.authenticate(self.test_username, self.test_password)
        self.assertIsNotNone(authenticated, "Valid credentials should authenticate successfully")
        wrong = self.um.authenticate(self.test_username, "WrongPassword!")
        self.assertIsNone(wrong, "Invalid password should return None")

    def test_03_user_sessions(self):
        """Test creating, validating, touching, and ending user sessions."""
        user = self.um.get_user_by_username(self.test_username)
        self.assertIsNotNone(user)
        user_id = user["id"]
        sess_token = f"sess_{uuid.uuid4().hex}"

        # 1. Create session
        self.um.create_session(
            user_id=user_id,
            session_id=sess_token,
            ip_address="127.0.0.1",
            user_agent="UnitTest/1.0",
        )

        # 2. Verify in collection
        sess_doc = self.um.mongo_db.user_sessions.find_one({"session_id": sess_token})
        self.assertIsNotNone(sess_doc)
        self.assertEqual(sess_doc["user_id"], user_id)
        self.assertIsNone(sess_doc["logout_time"])

        # 3. Touch session
        self.um.touch_session(sess_token)

        # 4. End session
        self.um.end_session(sess_token)
        sess_after = self.um.mongo_db.user_sessions.find_one({"session_id": sess_token})
        self.assertIsNotNone(sess_after["logout_time"], "logout_time must be set when ending session")

    def test_04_user_activity_logging(self):
        """Test logging and retrieving user audit activities."""
        user = self.um.get_user_by_username(self.test_username)
        self.assertIsNotNone(user)

        # Log activity
        self.um.log_activity(
            user_id=user["id"],
            activity_type="test_action",
            page="/test_page",
            action="click_test",
        )

        # Retrieve activity
        logs = self.um.get_user_activity(user_id=user["id"], limit=5)
        self.assertIsInstance(logs, list)
        self.assertGreaterEqual(len(logs), 1, "At least one activity log should exist for test user")
        self.assertEqual(logs[0]["activity_type"], "test_action")
        self.assertEqual(logs[0]["page"], "/test_page")

    def test_05_submissions_and_samples(self):
        """Test training submissions, sample insertion, and review workflow."""
        user = self.um.get_user_by_username(self.test_username)
        self.assertIsNotNone(user)

        # 1. Create submission
        sub_id = self.um.create_submission(
            user_id=user["id"],
            gesture_label="TEST_SIGN",
            gesture_type="Alphabet",
            sample_count=2,
        )
        self.assertIsNotNone(sub_id, "Submission creation must return an ID")

        # 2. Add sample
        sample_bytes = b"MOCK_NUMPY_ARRAY_BYTES_FOR_TESTING"
        self.um.add_training_sample(
            submission_id=sub_id,
            sample_path=f"data/samples/test_{sub_id}_0.npy",
            sample_type="npy",
            frame_count=1,
            sample_data=sample_bytes,
        )

        # 3. Retrieve submission & samples
        submission = self.um.get_submission_by_id(sub_id)
        self.assertIsNotNone(submission)
        self.assertEqual(submission["status"], "PENDING")

        samples = self.um.get_submission_samples(sub_id, include_data=True)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sample_data"], sample_bytes)

        # 4. Review submission (approve)
        self.um.review_submission(
            submission_id=sub_id,
            status="APPROVED",
            reviewed_by=user["id"],
            rejection_reason=None,
        )
        sub_updated = self.um.get_submission_by_id(sub_id)
        self.assertEqual(sub_updated["status"], "APPROVED")

        # Clean up submission & samples
        self.um.mongo_db.training_samples.delete_many({"submission_id": sub_id})
        self.um.mongo_db.training_submissions.delete_one({"id": sub_id})

    def test_06_model_versions_and_training_jobs(self):
        """Test model version metadata and training job tracking in MongoDB."""
        user = self.um.get_user_by_username(self.test_username)
        self.assertIsNotNone(user)

        tag = f"Test_Model_V{self.test_uid}"
        # 1. Save model version metadata
        saved = self.um.save_model_version(
            version_tag=tag,
            model_type="Alphabet",
            model_path="models/test_model.h5",
            accuracy=0.992,
            parameters={"test": True, "classes": ["A", "B"]},
            notes="Unit test model version",
            model_data=b"MOCK_H5_BYTES",
            encoder_data=b"MOCK_PKL_BYTES",
            is_active=False,
            created_by=user["id"],
        )
        self.assertTrue(saved, "Saving model version metadata must succeed")

        # 2. Fetch versions
        versions = self.um.get_model_versions(include_data=False)
        matched = [v for v in versions if v.get("version_tag") == tag]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["accuracy"], 0.992)
        # Ensure model_data was excluded by default for network transfer reduction
        self.assertNotIn("model_data", matched[0])

        # 3. Training jobs tracking
        job_id = self.um.create_training_job(
            job_type="RETRAIN_ALPHABET",
            created_by=user["id"],
        )
        self.assertIsNotNone(job_id)

        self.um.update_training_job_status(
            job_id=job_id,
            status="COMPLETED",
            completed=True,
        )
        job_doc = self.um.mongo_db.training_jobs.find_one({"id": job_id})
        self.assertEqual(job_doc["status"], "COMPLETED")

        # Clean up
        self.um.mongo_db.model_versions.delete_many({"version_tag": tag})
        self.um.mongo_db.training_jobs.delete_one({"id": job_id})

    @classmethod
    def tearDownClass(cls):
        """Clean up test user, sessions, and activity logs."""
        if cls.um.mongo_db is not None:
            user = cls.um.get_user_by_username(cls.test_username)
            if user:
                uid = user["id"]
                cls.um.mongo_db.user_sessions.delete_many({"user_id": uid})
                cls.um.mongo_db.user_activity.delete_many({"user_id": uid})
                cls.um.mongo_db.users.delete_one({"id": uid})


if __name__ == "__main__":
    unittest.main()
