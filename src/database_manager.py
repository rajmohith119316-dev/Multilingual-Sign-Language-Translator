"""
src/database_manager.py — PostgreSQL Database Manager (MoE Edition)
Three tables: gestures, prediction_history, translation_cache.
"""

import logging
import psycopg2
import psycopg2.extras
from datetime import datetime
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

# ─── DDL ──────────────────────────────────────────────────────────────────────
_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS gestures (
        id           SERIAL    PRIMARY KEY,
        gesture_name TEXT      NOT NULL UNIQUE,
        gesture_type TEXT      NOT NULL CHECK(gesture_type IN ('Alphabet','Word')),
        is_custom    BOOLEAN   NOT NULL DEFAULT FALSE,
        sample_count INTEGER   NOT NULL DEFAULT 0,
        date_added   TIMESTAMP NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_history (
        id               SERIAL    PRIMARY KEY,
        predicted_label  TEXT      NOT NULL,
        confidence_score REAL      NOT NULL,
        target_language  TEXT      NOT NULL,
        translated_text  TEXT      NOT NULL,
        timestamp        TIMESTAMP NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS translation_cache (
        id              SERIAL  PRIMARY KEY,
        english_text    TEXT    NOT NULL,
        target_language TEXT    NOT NULL,
        translated_text TEXT    NOT NULL,
        UNIQUE(english_text, target_language)
    )
    """,
]


class DatabaseManager:
    """Thread-compatible PostgreSQL manager (per-call connection pattern)."""

    def __init__(self) -> None:
        self.dsn = config.DATABASE_URL
        self._init_schema()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for stmt in _SCHEMA_STATEMENTS:
                        cur.execute(stmt)
                conn.commit()
            logger.info("Database ready: %s", self.dsn)
        except psycopg2.Error as exc:
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
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(gesture_name) DO UPDATE SET
                sample_count = EXCLUDED.sample_count,
                is_custom    = EXCLUDED.is_custom
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (gesture_name, gesture_type, is_custom, sample_count))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("add_gesture('%s') failed: %s", gesture_name, exc)
            raise

    def get_all_gestures(self) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM gestures ORDER BY gesture_name")
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def get_gesture_names(self) -> list[str]:
        return [g["gesture_name"] for g in self.get_all_gestures()]

    def delete_gesture(self, gesture_name: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM gestures WHERE gesture_name = %s", (gesture_name,))
            conn.commit()

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
            VALUES (%s, %s, %s, %s)
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (predicted_label, round(confidence_score, 4),
                                     target_language, translated_text))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("log_prediction failed: %s", exc)

    def get_history_summary(self, limit: int = 200) -> list[dict]:
        sql = """
            SELECT id, predicted_label, confidence_score, target_language,
                   translated_text, timestamp
            FROM prediction_history
            ORDER BY id DESC LIMIT %s
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def clear_history(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM prediction_history")
            conn.commit()

    # ── Translation Cache ──────────────────────────────────────────────────────

    def get_cached_translation(
        self, english_text: str, target_language: str
    ) -> Optional[str]:
        sql = """
            SELECT translated_text FROM translation_cache
            WHERE english_text = %s AND target_language = %s
        """
        with self._connect() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql, (english_text, target_language))
                row = cur.fetchone()
        return row["translated_text"] if row else None

    def cache_translation(
        self, english_text: str, target_language: str, translated_text: str
    ) -> None:
        sql = """
            INSERT INTO translation_cache (english_text, target_language, translated_text)
            VALUES (%s, %s, %s)
            ON CONFLICT(english_text, target_language) DO NOTHING
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (english_text, target_language, translated_text))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("cache_translation failed: %s", exc)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM gestures"); n_total   = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Alphabet'"); n_alpha   = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Word'"); n_word    = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM gestures WHERE is_custom=TRUE"); n_custom  = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM prediction_history"); n_history = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM translation_cache"); n_cache   = cur.fetchone()[0]
            return {
                "total_gestures":    n_total,
                "alphabet_gestures": n_alpha,
                "word_gestures":     n_word,
                "custom_gestures":   n_custom,
                "history_entries":   n_history,
                "cache_entries":     n_cache,
            }
        except psycopg2.Error as exc:
            logger.error("get_stats failed: %s", exc)
            return {}
