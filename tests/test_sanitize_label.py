import unittest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import sanitize_label


class TestSanitizeLabel(unittest.TestCase):

    def test_question_mark_removed(self):
        self.assertEqual(sanitize_label("HOW ARE YOU?"), "HOW ARE YOU")
        self.assertEqual(sanitize_label("how are you?"), "HOW ARE YOU")
        self.assertEqual(sanitize_label("What?"), "WHAT")

    def test_windows_illegal_characters_removed(self):
        # <>:"/\|?*
        self.assertEqual(sanitize_label("Test<sign>"), "TESTSIGN")
        self.assertEqual(sanitize_label('A"B:C*D?E/F\\G|H'), "ABCDEFGH")

    def test_whitespace_and_trailing_dots(self):
        self.assertEqual(sanitize_label("  hello   world  "), "HELLO WORLD")
        self.assertEqual(sanitize_label("sign..."), "SIGN")
        self.assertEqual(sanitize_label(""), "")
        self.assertEqual(sanitize_label(None), "")

    def test_valid_characters_preserved(self):
        self.assertEqual(sanitize_label("SEE YOU LATER"), "SEE YOU LATER")
        self.assertEqual(sanitize_label("SO-SO"), "SO-SO")
        self.assertEqual(sanitize_label("HELLO_WORLD"), "HELLO_WORLD")

    def test_directory_creation_with_sanitized_label(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_label = "HOW ARE YOU?"
            clean_label = sanitize_label(raw_label)
            target = Path(tmpdir) / clean_label
            target.mkdir(parents=True, exist_ok=True)
            self.assertTrue(target.exists())
            self.assertTrue(target.is_dir())


if __name__ == "__main__":
    unittest.main()
