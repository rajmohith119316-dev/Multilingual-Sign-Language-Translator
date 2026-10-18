"""
tests/test_flask_routes.py
Verifies Flask application startup, routing, template rendering, and REST API endpoints.
Ensures JSON responses conform to frontend expectations and reflect MongoDB Atlas backend.
"""

import unittest
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from web_app import app


class TestFlaskRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        cls.client = app.test_client()

    def test_01_index_redirects_to_login(self):
        """Verify root route '/' redirects unauthenticated users to '/auth/login' (302)."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/auth/login", res.headers.get("Location", ""))

    def test_02_auth_login_page(self):
        """Verify login page renders with 200 OK."""
        res = self.client.get("/auth/login")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("login", html.lower())

    def test_03_auth_register_page(self):
        """Verify register page renders with 200 OK."""
        res = self.client.get("/auth/register")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn("register", html.lower())

    def test_04_api_stats_mongodb_backend(self):
        """Verify /api/stats returns status 200 and reports MongoDB backend."""
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertIn("backend", data)
        self.assertEqual(data["backend"], "mongodb")
        for metric in ["total_gestures", "alphabet_gestures", "word_gestures", "history_entries", "cache_entries"]:
            self.assertIn(metric, data)
            self.assertGreaterEqual(data[metric], 0)

    def test_05_api_gestures(self):
        """Verify /api/gestures returns list of gestures."""
        res = self.client.get("/api/gestures")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertIsInstance(data, list)

    def test_06_api_history(self):
        """Verify /api/history returns prediction history log."""
        res = self.client.get("/api/history?limit=10")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertIsInstance(data, list)

    def test_07_api_hand_poses(self):
        """Verify /api/hand_poses returns ASL poses dictionary."""
        res = self.client.get("/api/hand_poses")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertIn("A", data)
        self.assertIn("B", data)

    def test_08_api_known_signs(self):
        """Verify /api/known_signs returns model status and categories."""
        res = self.client.get("/api/known_signs")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.is_json)
        data = res.get_json()
        self.assertIn("alphabets", data)
        self.assertIn("words", data)

    def test_09_authenticated_translator_view(self):
        """Verify authenticated session can access /translator (200 OK with UI)."""
        with self.client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "admin"
            sess["role"] = "ADMIN"

        res = self.client.get("/translator")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertTrue(
            "Sign Language" in html or "translator" in html.lower() or "canvas" in html.lower(),
            "Translator template should render for authenticated user"
        )


if __name__ == "__main__":
    unittest.main()
