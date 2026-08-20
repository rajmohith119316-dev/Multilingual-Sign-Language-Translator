"""
src/database_manager.py — SQLite Database Manager (MoE Edition)
Three tables: gestures, prediction_history, translation_cache.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

# ─── DDL ──────────────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS gestures (
    id           INTEGER   PRIMARY KEY AUTOINCREMENT,
    gesture_name TEXT      NOT NULL UNIQUE,
    gesture_type TEXT      NOT NULL CHECK(gesture_type IN ('Alphabet','Word')),
    is_custom    INTEGER   NOT NULL DEFAULT 0,
    sample_count INTEGER   NOT NULL DEFAULT 0,
    date_added   TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id               INTEGER   PRIMARY KEY AUTOINCREMENT,
    predicted_label  TEXT      NOT NULL,
    confidence_score REAL      NOT NULL,
    target_language  TEXT      NOT NULL,
    translated_text  TEXT      NOT NULL,
    timestamp        TIMESTAMP NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS translation_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    english_text    TEXT    NOT NULL,
    target_language TEXT    NOT NULL,
    translated_text TEXT    NOT NULL,
    UNIQUE(english_text, target_language)
);
"""


class DatabaseManager:
    """Thread-compatible SQLite manager (per-call connection pattern)."""

    def __init__(self, db_path: Path = config.DB_PATH) -> None:
        self.db_path = str(db_path)
        self._init_schema()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.executescript(_SCHEMA_SQL)
            logger.info("Database ready: %s", self.db_path)
        except sqlite3.Error as exc:
            logger.error("Schema init failed: %s", exc)
            raise

    # ── Gestures ──────────────────────────────────────────────────────────────

    def add_gesture(
        self,
        gesture_name: str,
        gesture_type: str,          # 'Alphabet' | 'Word'
        is_custom: bool = False,
        sample_count: int = 0,
    ) -> None:
        sql = """
            INSERT INTO gestures (gesture_name, gesture_type, is_custom, sample_count)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(gesture_name) DO UPDATE SET
                sample_count = excluded.sample_count,
                is_custom    = excluded.is_custom
        """
        try:
            with self._connect() as conn:
                conn.execute(sql, (gesture_name, gesture_type, int(is_custom), sample_count))
        except sqlite3.Error as exc:
            logger.error("add_gesture('%s') failed: %s", gesture_name, exc)
            raise

    def get_all_gestures(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM gestures ORDER BY gesture_name").fetchall()
        return [dict(r) for r in rows]

    def get_gesture_names(self) -> list[str]:
        return [g["gesture_name"] for g in self.get_all_gestures()]

    def delete_gesture(self, gesture_name: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM gestures WHERE gesture_name = ?", (gesture_name,))

    # ── Prediction History ─────────────────────────────────────────────────────

    def log_prediction(
        self,
        predicted_label: str,
        confidence_score: float,
        target_language: str,
        translated_text: str,
    ) -> None:
        sql = """
            INSERT INTO prediction_history
                (predicted_label, confidence_score, target_language, translated_text)
            VALUES (?, ?, ?, ?)
        """
        try:
            with self._connect() as conn:
                conn.execute(sql, (predicted_label, round(confidence_score, 4),
                                   target_language, translated_text))
        except sqlite3.Error as exc:
            logger.error("log_prediction failed: %s", exc)

    def get_history_summary(self, limit: int = 200) -> list[dict]:
        sql = """
            SELECT id, predicted_label, confidence_score, target_language,
                   translated_text, timestamp
            FROM prediction_history
            ORDER BY id DESC LIMIT ?
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def clear_history(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM prediction_history")

    # ── Translation Cache ──────────────────────────────────────────────────────

    def get_cached_translation(
        self, english_text: str, target_language: str
    ) -> Optional[str]:
        sql = """
            SELECT translated_text FROM translation_cache
            WHERE english_text = ? AND target_language = ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (english_text, target_language)).fetchone()
        return row["translated_text"] if row else None

    def cache_translation(
        self, english_text: str, target_language: str, translated_text: str
    ) -> None:
        sql = """
            INSERT INTO translation_cache (english_text, target_language, translated_text)
            VALUES (?, ?, ?)
            ON CONFLICT(english_text, target_language) DO NOTHING
        """
        try:
            with self._connect() as conn:
                conn.execute(sql, (english_text, target_language, translated_text))
        except sqlite3.Error as exc:
            logger.error("cache_translation failed: %s", exc)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        try:
            with self._connect() as conn:
                n_total   = conn.execute("SELECT COUNT(*) FROM gestures").fetchone()[0]
                n_alpha   = conn.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Alphabet'").fetchone()[0]
                n_word    = conn.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Word'").fetchone()[0]
                n_custom  = conn.execute("SELECT COUNT(*) FROM gestures WHERE is_custom=1").fetchone()[0]
                n_history = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
                n_cache   = conn.execute("SELECT COUNT(*) FROM translation_cache").fetchone()[0]
            return {
                "total_gestures":    n_total,
                "alphabet_gestures": n_alpha,
                "word_gestures":     n_word,
                "custom_gestures":   n_custom,
                "history_entries":   n_history,
                "cache_entries":     n_cache,
            }
        except sqlite3.Error as exc:
            logger.error("get_stats failed: %s", exc)
            return {}
