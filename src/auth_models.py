"""
src/auth_models.py — PostgreSQL Auth & Activity Schema + User Manager

Tables
──────
  users                – authentication + roles
  user_activity        – application event log
  user_sessions        – login/logout session tracking
  training_submissions – user-submitted teaching sessions
  training_samples     – individual samples within submissions
  model_versions       – trained model version registry
  training_jobs        – background training job log

All passwords are stored as Werkzeug-hashed digests — NEVER plaintext.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
from werkzeug.security import generate_password_hash, check_password_hash

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# DDL — Table Definitions
# ═══════════════════════════════════════════════════════════════════════════

_AUTH_SCHEMA = [
    # ── Users ─────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS users (
        id            SERIAL      PRIMARY KEY,
        full_name     TEXT        NOT NULL,
        username      TEXT        NOT NULL UNIQUE,
        email         TEXT        NOT NULL UNIQUE,
        password_hash TEXT        NOT NULL,
        role          TEXT        NOT NULL DEFAULT 'USER'
                                  CHECK (role IN ('USER', 'ADMIN')),
        is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_login    TIMESTAMPTZ,
        updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_users_username ON users (username)",
    "CREATE INDEX IF NOT EXISTS idx_users_email    ON users (email)",
    "CREATE INDEX IF NOT EXISTS idx_users_role     ON users (role)",

    # ── User Activity ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS user_activity (
        id            SERIAL      PRIMARY KEY,
        user_id       INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        activity_type TEXT        NOT NULL,
        page          TEXT,
        action        TEXT,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        session_id    TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_activity_user    ON user_activity (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_activity_type    ON user_activity (activity_type)",
    "CREATE INDEX IF NOT EXISTS idx_activity_created ON user_activity (created_at)",

    # ── User Sessions ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS user_sessions (
        id             SERIAL      PRIMARY KEY,
        user_id        INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        session_id     TEXT        NOT NULL UNIQUE,
        login_time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        logout_time    TIMESTAMPTZ,
        last_activity  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        ip_address     TEXT,
        user_agent     TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_sid  ON user_sessions (session_id)",

    # ── Training Submissions ──────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS training_submissions (
        id              SERIAL      PRIMARY KEY,
        user_id         INTEGER     NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        gesture_label   TEXT        NOT NULL,
        gesture_type    TEXT        NOT NULL DEFAULT 'Word'
                                    CHECK (gesture_type IN ('Alphabet', 'Word')),
        status          TEXT        NOT NULL DEFAULT 'PENDING'
                                    CHECK (status IN ('PENDING','APPROVED','REJECTED')),
        rejection_reason TEXT,
        sample_count    INTEGER     NOT NULL DEFAULT 0,
        submitted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        reviewed_at     TIMESTAMPTZ,
        reviewed_by     INTEGER     REFERENCES users(id),
        notes           TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_submissions_user   ON training_submissions (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_submissions_status ON training_submissions (status)",

    # ── Training Samples ──────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS training_samples (
        id              SERIAL      PRIMARY KEY,
        submission_id   INTEGER     NOT NULL REFERENCES training_submissions(id)
                                    ON DELETE CASCADE,
        sample_path     TEXT        NOT NULL,
        sample_data     BYTEA,
        sample_type     TEXT        NOT NULL DEFAULT 'npy'
                                    CHECK (sample_type IN ('npy','image','video')),
        frame_count     INTEGER,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        is_valid        BOOLEAN     DEFAULT TRUE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_samples_submission ON training_samples (submission_id)",

    # ── Model Versions ────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS model_versions (
        id              SERIAL      PRIMARY KEY,
        version_tag     TEXT        NOT NULL UNIQUE,
        model_type      TEXT        NOT NULL CHECK (model_type IN ('Alphabet','Word','MoE')),
        model_path      TEXT        NOT NULL,
        model_data      BYTEA,
        encoder_data    BYTEA,
        accuracy        REAL,
        parameters      JSONB,
        is_active       BOOLEAN     NOT NULL DEFAULT FALSE,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_by      INTEGER     REFERENCES users(id),
        notes           TEXT
    )
    """,

    # ── Training Jobs ─────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS training_jobs (
        id              SERIAL      PRIMARY KEY,
        job_type        TEXT        NOT NULL,
        status          TEXT        NOT NULL DEFAULT 'QUEUED'
                                    CHECK (status IN ('QUEUED','RUNNING','COMPLETED','FAILED')),
        started_at      TIMESTAMPTZ,
        completed_at    TIMESTAMPTZ,
        created_by      INTEGER     REFERENCES users(id),
        model_version_id INTEGER    REFERENCES model_versions(id),
        parameters      JSONB,
        log_path        TEXT,
        error_message   TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
]


# ═══════════════════════════════════════════════════════════════════════════
# UserManager — CRUD + Auth + Activity
# ═══════════════════════════════════════════════════════════════════════════

class UserManager:
    """Manages user authentication, activity tracking, and session management."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or config.DATABASE_URL
        self._init_schema()
        self._seed_admin()

    # ── DB helpers ────────────────────────────────────────────────────────

    def _connect(self):
        conn = psycopg2.connect(self.dsn)
        conn.autocommit = False
        return conn

    def _init_schema(self):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    for stmt in _AUTH_SCHEMA:
                        cur.execute(stmt)
                conn.commit()
            logger.info("Auth schema ready.")
        except psycopg2.Error as exc:
            logger.error("Auth schema init failed: %s", exc)
            raise

    # ── Admin Seed ────────────────────────────────────────────────────────

    def _seed_admin(self):
        """Create the first admin from environment variables if not exists."""
        admin_user = os.getenv("ADMIN_USERNAME")
        admin_email = os.getenv("ADMIN_EMAIL")
        admin_pass = os.getenv("ADMIN_PASSWORD")

        if not all([admin_user, admin_email, admin_pass]):
            logger.info("No ADMIN_USERNAME/ADMIN_EMAIL/ADMIN_PASSWORD env vars — skipping admin seed.")
            return

        existing = self.get_user_by_username(admin_user)
        if existing:
            logger.info("Admin user '%s' already exists — skipping seed.", admin_user)
            return

        pw_hash = generate_password_hash(admin_pass)
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO users (full_name, username, email, password_hash, role)
                        VALUES (%s, %s, %s, %s, 'ADMIN')
                        ON CONFLICT (username) DO NOTHING
                    """, (admin_user, admin_user, admin_email, pw_hash))
                conn.commit()
            logger.info("Admin user '%s' seeded from environment variables.", admin_user)
        except psycopg2.Error as exc:
            logger.error("Admin seed failed: %s", exc)

    # ── User CRUD ─────────────────────────────────────────────────────────

    def create_user(self, full_name: str, username: str, email: str,
                    password: str, role: str = "USER") -> dict | None:
        """Register a new user. Returns the user dict or None on conflict."""
        pw_hash = generate_password_hash(password)
        sql = """
            INSERT INTO users (full_name, username, email, password_hash, role)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, full_name, username, email, role, is_active, created_at
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (full_name, username.lower(), email.lower(), pw_hash, role))
                    user = dict(cur.fetchone())
                conn.commit()
            return user
        except psycopg2.errors.UniqueViolation:
            return None
        except psycopg2.Error as exc:
            logger.error("create_user failed: %s", exc)
            return None

    def get_user_by_username(self, username: str) -> dict | None:
        sql = "SELECT * FROM users WHERE username = %s"
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (username.lower(),))
                    row = cur.fetchone()
            return dict(row) if row else None
        except psycopg2.Error as exc:
            logger.error("get_user_by_username failed: %s", exc)
            return None

    def get_user_by_email(self, email: str) -> dict | None:
        sql = "SELECT * FROM users WHERE email = %s"
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (email.lower(),))
                    row = cur.fetchone()
            return dict(row) if row else None
        except psycopg2.Error as exc:
            logger.error("get_user_by_email failed: %s", exc)
            return None

    def get_user_by_id(self, user_id: int) -> dict | None:
        sql = "SELECT * FROM users WHERE id = %s"
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (user_id,))
                    row = cur.fetchone()
            return dict(row) if row else None
        except psycopg2.Error as exc:
            logger.error("get_user_by_id failed: %s", exc)
            return None

    def authenticate(self, username_or_email: str, password: str) -> dict | None:
        """Validate credentials. Returns user dict on success, None on failure."""
        user = self.get_user_by_username(username_or_email)
        if not user:
            user = self.get_user_by_email(username_or_email)
        if not user:
            return None
        if not user.get("is_active", True):
            return None
        if check_password_hash(user["password_hash"], password):
            return user
        return None

    def update_last_login(self, user_id: int):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET last_login = NOW(), updated_at = NOW() WHERE id = %s",
                        (user_id,),
                    )
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("update_last_login failed: %s", exc)

    def get_all_users(self) -> list[dict]:
        sql = """
            SELECT id, full_name, username, email, role, is_active,
                   created_at, last_login, updated_at
            FROM users ORDER BY created_at DESC
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_all_users failed: %s", exc)
            return []

    def toggle_user_active(self, user_id: int, is_active: bool):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET is_active = %s, updated_at = NOW() WHERE id = %s",
                        (is_active, user_id),
                    )
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("toggle_user_active failed: %s", exc)

    def update_user_role(self, user_id: int, role: str):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE users SET role = %s, updated_at = NOW() WHERE id = %s",
                        (role, user_id),
                    )
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("update_user_role failed: %s", exc)

    def delete_user(self, user_id: int):
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("delete_user failed: %s", exc)

    # ── Activity Tracking ─────────────────────────────────────────────────

    def log_activity(self, user_id: int, activity_type: str,
                     page: str = None, action: str = None,
                     session_id: str = None):
        sql = """
            INSERT INTO user_activity (user_id, activity_type, page, action, session_id)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, activity_type, page, action, session_id))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("log_activity failed: %s", exc)

    def get_user_activity(self, user_id: int, limit: int = 50) -> list[dict]:
        sql = """
            SELECT id, activity_type, page, action, created_at, session_id
            FROM user_activity WHERE user_id = %s
            ORDER BY created_at DESC LIMIT %s
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (user_id, limit))
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_user_activity failed: %s", exc)
            return []

    def get_all_activity(self, limit: int = 200, user_id: int = None,
                         activity_type: str = None, page: str = None,
                         date_from: str = None, date_to: str = None) -> list[dict]:
        """Admin: query activity with optional filters."""
        sql = """
            SELECT ua.id, ua.user_id, u.full_name, u.username, u.email,
                   ua.activity_type, ua.page, ua.action, ua.created_at, ua.session_id
            FROM user_activity ua
            JOIN users u ON u.id = ua.user_id
            WHERE 1=1
        """
        params = []
        if user_id:
            sql += " AND ua.user_id = %s"
            params.append(user_id)
        if activity_type:
            sql += " AND ua.activity_type = %s"
            params.append(activity_type)
        if page:
            sql += " AND ua.page = %s"
            params.append(page)
        if date_from:
            sql += " AND ua.created_at >= %s"
            params.append(date_from)
        if date_to:
            sql += " AND ua.created_at <= %s"
            params.append(date_to)
        sql += " ORDER BY ua.created_at DESC LIMIT %s"
        params.append(limit)

        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_all_activity failed: %s", exc)
            return []

    # ── Session Management ────────────────────────────────────────────────

    def create_session(self, user_id: int, session_id: str,
                       ip_address: str = None, user_agent: str = None):
        sql = """
            INSERT INTO user_sessions (user_id, session_id, ip_address, user_agent)
            VALUES (%s, %s, %s, %s)
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, session_id, ip_address, user_agent))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("create_session failed: %s", exc)

    def end_session(self, session_id: str):
        sql = "UPDATE user_sessions SET logout_time = NOW() WHERE session_id = %s"
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (session_id,))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("end_session failed: %s", exc)

    def touch_session(self, session_id: str):
        sql = "UPDATE user_sessions SET last_activity = NOW() WHERE session_id = %s"
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (session_id,))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("touch_session failed: %s", exc)

    # ── Training Submissions ──────────────────────────────────────────────

    def create_submission(self, user_id: int, gesture_label: str,
                          gesture_type: str = "Word",
                          sample_count: int = 0) -> int | None:
        sql = """
            INSERT INTO training_submissions
                (user_id, gesture_label, gesture_type, sample_count)
            VALUES (%s, %s, %s, %s)
            RETURNING id
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, gesture_label, gesture_type, sample_count))
                    sid = cur.fetchone()[0]
                conn.commit()
            return sid
        except psycopg2.Error as exc:
            logger.error("create_submission failed: %s", exc)
            return None

    def get_user_submissions(self, user_id: int) -> list[dict]:
        sql = """
            SELECT id, gesture_label, gesture_type, status, rejection_reason,
                   sample_count, submitted_at, reviewed_at, notes
            FROM training_submissions WHERE user_id = %s
            ORDER BY submitted_at DESC
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (user_id,))
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_user_submissions failed: %s", exc)
            return []

    def get_all_submissions(self, status: str = None) -> list[dict]:
        sql = """
            SELECT ts.*, u.full_name, u.username, u.email
            FROM training_submissions ts
            JOIN users u ON u.id = ts.user_id
        """
        params = []
        if status:
            sql += " WHERE ts.status = %s"
            params.append(status)
        sql += " ORDER BY ts.submitted_at DESC"
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_all_submissions failed: %s", exc)
            return []

    def review_submission(self, submission_id: int, status: str,
                          reviewed_by: int, rejection_reason: str = None):
        sql = """
            UPDATE training_submissions
            SET status = %s, rejection_reason = %s,
                reviewed_at = NOW(), reviewed_by = %s
            WHERE id = %s
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (status, rejection_reason, reviewed_by, submission_id))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("review_submission failed: %s", exc)

    def get_submission_by_id(self, submission_id: int) -> dict | None:
        sql = """
            SELECT ts.*, u.full_name, u.username
            FROM training_submissions ts
            JOIN users u ON u.id = ts.user_id
            WHERE ts.id = %s
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (submission_id,))
                    row = cur.fetchone()
            return dict(row) if row else None
        except psycopg2.Error as exc:
            logger.error("get_submission_by_id failed: %s", exc)
            return None

    # ── Training Samples ──────────────────────────────────────────────────

    def add_training_sample(self, submission_id: int, sample_path: str,
                            sample_type: str = "npy", frame_count: int = None,
                            sample_data: bytes = None):
        sql = """
            INSERT INTO training_samples (submission_id, sample_path, sample_type, frame_count, sample_data)
            VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (submission_id, sample_path, sample_type, frame_count, sample_data))
                conn.commit()
        except psycopg2.Error as exc:
            logger.error("add_training_sample failed: %s", exc)

    def get_submission_samples(self, submission_id: int) -> list[dict]:
        sql = """
            SELECT * FROM training_samples WHERE submission_id = %s
            ORDER BY created_at
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql, (submission_id,))
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_submission_samples failed: %s", exc)
            return []

    # ── Model Versions ────────────────────────────────────────────────────

    def get_model_versions(self) -> list[dict]:
        sql = "SELECT * FROM model_versions ORDER BY created_at DESC"
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_model_versions failed: %s", exc)
            return []

    # ── Training Jobs ─────────────────────────────────────────────────────

    def get_training_jobs(self) -> list[dict]:
        sql = """
            SELECT tj.*, mv.version_tag
            FROM training_jobs tj
            LEFT JOIN model_versions mv ON mv.id = tj.model_version_id
            ORDER BY tj.created_at DESC
        """
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(sql)
                    rows = cur.fetchall()
            return [dict(r) for r in rows]
        except psycopg2.Error as exc:
            logger.error("get_training_jobs failed: %s", exc)
            return []

    def create_training_job(self, job_type: str, created_by: int) -> int | None:
        sql = """
            INSERT INTO training_jobs (job_type, created_by)
            VALUES (%s, %s)
            RETURNING id
        """
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (job_type, created_by))
                    job_id = cur.fetchone()[0]
                conn.commit()
            return job_id
        except psycopg2.Error as exc:
            logger.error("create_training_job failed: %s", exc)
            return None

    # ── Dashboard Stats ───────────────────────────────────────────────────

    def get_admin_stats(self) -> dict:
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM users")
                    total_users = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM users WHERE role = 'ADMIN'")
                    total_admins = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'PENDING'")
                    pending = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'APPROVED'")
                    approved = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'REJECTED'")
                    rejected = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM user_activity")
                    total_activity = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM model_versions")
                    total_models = cur.fetchone()[0]
            return {
                "total_users": total_users,
                "total_admins": total_admins,
                "pending_submissions": pending,
                "approved_submissions": approved,
                "rejected_submissions": rejected,
                "total_activity": total_activity,
                "total_models": total_models,
            }
        except psycopg2.Error as exc:
            logger.error("get_admin_stats failed: %s", exc)
            return {}
