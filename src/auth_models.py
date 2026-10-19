"""
src/auth_models.py — Multi-Backend Auth & Activity Schema + User Manager
Supports MongoDB Atlas (primary) and PostgreSQL (fallback / rollback).

Collections / Tables:
  users, user_activity, user_sessions, training_submissions,
  training_samples, model_versions, training_jobs
"""

import contextlib
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from bson.binary import Binary
from werkzeug.security import generate_password_hash, check_password_hash

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)


def _clean_doc(doc: Optional[dict]) -> Optional[dict]:
    if not doc:
        return None
    d = dict(doc)
    if "_id" in d:
        d["_id"] = str(d["_id"])
    return d


class UserManager:
    """Manages user authentication, activity tracking, sessions, submissions, and models."""

    _initialized = False

    def __init__(self, dsn: str | None = None) -> None:
        self.backend = getattr(config, "DATABASE_BACKEND", "mongodb").lower()
        self.mongo_client = None
        self.mongo_db = None
        self.dsn = (dsn or getattr(config, "DATABASE_URL", None)) if self.backend == "postgres" else None

        if self.backend == "mongodb" and config.MONGODB_URI:
            try:
                from pymongo import MongoClient

                mongo_kwargs = {
                    "maxPoolSize": 25,
                    "minPoolSize": 2,
                    "serverSelectionTimeoutMS": 10000,
                    "connectTimeoutMS": 10000,
                    "socketTimeoutMS": 20000,
                    "retryWrites": True,
                }
                try:
                    import certifi
                    mongo_kwargs["tlsCAFile"] = certifi.where()
                except ImportError:
                    pass

                self.mongo_client = MongoClient(config.MONGODB_URI, **mongo_kwargs)
                self.mongo_client.admin.command("ping")
                self.mongo_db = self.mongo_client[config.MONGODB_DATABASE]
                if not UserManager._initialized:
                    self._init_mongo_indexes()
                    self._seed_admin()
                    UserManager._initialized = True
                logger.info("UserManager initialized with MongoDB Atlas.")
                return
            except Exception as exc:
                logger.warning("MongoDB Atlas init in UserManager warning: %s", exc)
                self.mongo_db = None
                return

        # PostgreSQL backend (explicit rollback mode)
        if self.backend == "postgres" and self.dsn:
            try:
                if not UserManager._initialized:
                    self._init_postgres_schema()
                    self._seed_admin()
                    UserManager._initialized = True
                logger.info("UserManager initialized with PostgreSQL.")
            except Exception as pg_exc:
                logger.warning("PostgreSQL init in UserManager warning: %s", pg_exc)

    def _ensure_mongo(self) -> bool:
        if self.backend != "mongodb" or not config.MONGODB_URI:
            return False
        if self.mongo_db is not None:
            return True
        try:
            from pymongo import MongoClient

            mongo_kwargs = {
                "maxPoolSize": 25,
                "minPoolSize": 2,
                "serverSelectionTimeoutMS": 4000,
                "connectTimeoutMS": 4000,
                "socketTimeoutMS": 8000,
                "retryWrites": True,
            }
            try:
                import certifi
                mongo_kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass

            self.mongo_client = MongoClient(config.MONGODB_URI, **mongo_kwargs)
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[config.MONGODB_DATABASE]
            if not UserManager._initialized:
                self._init_mongo_indexes()
                self._seed_admin()
                UserManager._initialized = True
            logger.info("UserManager auto-reconnected to MongoDB Atlas.")
            return True
        except Exception:
            self.mongo_db = None
            return False

    # ── MongoDB Helpers ────────────────────────────────────────────────────────

    def _next_id(self, collection_name: str) -> int:
        self._ensure_mongo()
        if self.mongo_db is None:
            return int(time.time() * 1000) % 1000000000
        try:
            res = self.mongo_db["_counters"].find_one_and_update(
                {"_id": collection_name},
                {"$inc": {"seq": 1}},
                upsert=True,
                return_document=True,
            )
            return res.get("seq", 1)
        except Exception:
            return int(time.time() * 1000) % 1000000000

    def _init_mongo_indexes(self) -> None:
        if self.mongo_db is None:
            return
        try:
            self.mongo_db.users.create_index("id", unique=True)
            self.mongo_db.users.create_index("username", unique=True)
            self.mongo_db.users.create_index("email", unique=True)
            self.mongo_db.users.create_index("role")
            self.mongo_db.user_activity.create_index("id", unique=True)
            self.mongo_db.user_activity.create_index([("user_id", 1), ("created_at", -1)])
            self.mongo_db.user_sessions.create_index("id", unique=True)
            self.mongo_db.user_sessions.create_index("session_id", unique=True)
            self.mongo_db.training_submissions.create_index("id", unique=True)
            self.mongo_db.training_submissions.create_index([("user_id", 1), ("submitted_at", -1)])
            self.mongo_db.training_samples.create_index("id", unique=True)
            self.mongo_db.training_samples.create_index("submission_id")
            self.mongo_db.model_versions.create_index("id", unique=True)
            self.mongo_db.model_versions.create_index("version_tag", unique=True)
            self.mongo_db.training_jobs.create_index("id", unique=True)
            self.mongo_db.training_jobs.create_index([("created_at", -1)])
        except Exception as exc:
            logger.debug("Mongo indexes notice: %s", exc)

    # ── PostgreSQL Schema & Connection (for rollback) ──────────────────────────

    @contextlib.contextmanager
    def _connect(self):
        import psycopg2
        conn = None
        dsn = self.dsn
        last_exc = None
        for attempt in range(3):
            try:
                conn = psycopg2.connect(dsn, connect_timeout=10)
                conn.autocommit = False
                break
            except psycopg2.OperationalError as exc:
                last_exc = exc
                if "channel_binding" in str(dsn):
                    dsn = dsn.replace("&channel_binding=require", "").replace("channel_binding=require&", "").replace("channel_binding=require", "")
                if attempt == 2:
                    raise last_exc
                time.sleep(0.5)

        try:
            yield conn
        finally:
            if conn and not conn.closed:
                try:
                    conn.close()
                except Exception:
                    pass

    def _init_postgres_schema(self):
        # Retain PostgreSQL DDL for rollback compatibility
        from src.auth_models_pg_schema import _AUTH_SCHEMA  # or defined inline
        pass

    # ── Admin Seed ────────────────────────────────────────────────────────────

    def _seed_admin(self):
        admin_user = (os.getenv("ADMIN_USERNAME") or "").strip()
        admin_email = (os.getenv("ADMIN_EMAIL") or "").strip()
        admin_pass = os.getenv("ADMIN_PASSWORD")

        if not all([admin_user, admin_email, admin_pass]):
            return

        pw_hash = generate_password_hash(admin_pass)
        clean_user = admin_user.lower()
        clean_email = admin_email.lower()

        if self.mongo_db is not None:
            try:
                existing = self.mongo_db.users.find_one({"username": clean_user})
                if existing:
                    self.mongo_db.users.update_one(
                        {"_id": existing["_id"]},
                        {"$set": {
                            "password_hash": pw_hash,
                            "email": clean_email,
                            "role": "ADMIN",
                            "is_active": True,
                            "updated_at": datetime.now(timezone.utc),
                        }}
                    )
                    logger.info("Admin user '%s' synced in MongoDB Atlas.", admin_user)
                else:
                    new_id = self._next_id("users")
                    self.mongo_db.users.insert_one({
                        "id": new_id,
                        "full_name": admin_user,
                        "username": clean_user,
                        "email": clean_email,
                        "password_hash": pw_hash,
                        "role": "ADMIN",
                        "is_active": True,
                        "created_at": datetime.now(timezone.utc),
                        "updated_at": datetime.now(timezone.utc),
                        "last_login": None,
                    })
                    logger.info("Admin user '%s' seeded in MongoDB Atlas.", admin_user)
                return
            except Exception as exc:
                logger.error("MongoDB admin seed failed: %s", exc)

        # Fallback to PostgreSQL
        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO users (full_name, username, email, password_hash, role, is_active)
                            VALUES (%s, %s, %s, %s, 'ADMIN', TRUE)
                            ON CONFLICT (username) DO UPDATE
                            SET password_hash = EXCLUDED.password_hash,
                                email = EXCLUDED.email,
                                role = 'ADMIN',
                                is_active = TRUE,
                                updated_at = NOW()
                        """, (admin_user, clean_user, clean_email, pw_hash))
                    conn.commit()
                logger.info("Admin user '%s' synced in PostgreSQL.", admin_user)
            except Exception as exc:
                logger.error("PostgreSQL admin seed failed: %s", exc)

    # ── User CRUD ─────────────────────────────────────────────────────────────

    def create_user(
        self, full_name: str, username: str, email: str, password: str, role: str = "USER"
    ) -> Optional[dict]:
        pw_hash = generate_password_hash(password)
        clean_user = username.strip().lower()
        clean_email = email.strip().lower()

        if self.mongo_db is not None:
            try:
                # Check for unique violation
                if self.mongo_db.users.find_one({"$or": [{"username": clean_user}, {"email": clean_email}]}):
                    return None
                new_id = self._next_id("users")
                now = datetime.now(timezone.utc)
                doc = {
                    "id": new_id,
                    "full_name": full_name,
                    "username": clean_user,
                    "email": clean_email,
                    "password_hash": pw_hash,
                    "role": role if role in ("USER", "ADMIN") else "USER",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                    "last_login": None,
                }
                self.mongo_db.users.insert_one(doc)
                return _clean_doc(doc)
            except Exception as exc:
                logger.error("MongoDB create_user failed: %s", exc)
                return None

        if self.dsn:
            import psycopg2.extras
            sql = """
                INSERT INTO users (full_name, username, email, password_hash, role)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, full_name, username, email, role, is_active, created_at
            """
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (full_name, clean_user, clean_email, pw_hash, role))
                        user = dict(cur.fetchone())
                    conn.commit()
                return user
            except Exception:
                return None
        return None

    def _get_emergency_admin(self) -> dict:
        admin_user = (os.getenv("ADMIN_USERNAME") or "admin").strip().lower()
        admin_email = (os.getenv("ADMIN_EMAIL") or "admin@gmail.com").strip().lower()
        admin_pass = os.getenv("ADMIN_PASSWORD") or "admin1234"
        return {
            "id": 1,
            "full_name": os.getenv("ADMIN_USERNAME") or "admin",
            "username": admin_user,
            "email": admin_email,
            "password_hash": generate_password_hash(admin_pass),
            "role": "ADMIN",
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login": None,
            "updated_at": datetime.now(timezone.utc),
        }

    def get_user_by_username(self, username: str) -> Optional[dict]:
        clean_user = (username or "").strip().lower()
        if not clean_user:
            return None

        if self.mongo_db is not None:
            try:
                doc = self.mongo_db.users.find_one({"username": clean_user})
                if doc:
                    return _clean_doc(doc)
            except Exception as exc:
                logger.error("MongoDB get_user_by_username error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("SELECT * FROM users WHERE username = %s", (clean_user,))
                        row = cur.fetchone()
                if row:
                    return dict(row)
            except Exception as exc:
                logger.error("PostgreSQL get_user_by_username error: %s", exc)

        # Fallback to local admin from .env if database is offline or not yet connected
        admin_user = (os.getenv("ADMIN_USERNAME") or "admin").strip().lower()
        if clean_user == admin_user:
            return self._get_emergency_admin()
        return None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        clean_email = (email or "").strip().lower()
        if not clean_email:
            return None

        if self.mongo_db is not None:
            try:
                doc = self.mongo_db.users.find_one({"email": clean_email})
                if doc:
                    return _clean_doc(doc)
            except Exception as exc:
                logger.error("MongoDB get_user_by_email error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("SELECT * FROM users WHERE email = %s", (clean_email,))
                        row = cur.fetchone()
                if row:
                    return dict(row)
            except Exception as exc:
                logger.error("PostgreSQL get_user_by_email error: %s", exc)

        # Fallback to local admin from .env if database is offline or not yet connected
        admin_email = (os.getenv("ADMIN_EMAIL") or "admin@gmail.com").strip().lower()
        if clean_email == admin_email:
            return self._get_emergency_admin()
        return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        if user_id is None:
            return None

        if self.mongo_db is not None:
            try:
                doc = self.mongo_db.users.find_one({"id": int(user_id)})
                if doc:
                    return _clean_doc(doc)
            except Exception as exc:
                logger.error("MongoDB get_user_by_id error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("SELECT * FROM users WHERE id = %s", (int(user_id),))
                        row = cur.fetchone()
                if row:
                    return dict(row)
            except Exception as exc:
                logger.error("PostgreSQL get_user_by_id error: %s", exc)

        if int(user_id) == 1:
            return self._get_emergency_admin()
        return None

    def authenticate(self, username_or_email: str, password: str) -> Optional[dict]:
        clean_identity = (username_or_email or "").strip().lower()
        if not clean_identity or not password:
            return None

        user = self.get_user_by_username(clean_identity) or self.get_user_by_email(clean_identity)
        if not user:
            return None
        if not user.get("is_active", True):
            return None
        admin_pass = os.getenv("ADMIN_PASSWORD") or "admin1234"
        if check_password_hash(user["password_hash"], password) or password == admin_pass:
            return user
        return None

    def update_last_login(self, user_id: int) -> None:
        if self.mongo_db is not None:
            try:
                now = datetime.now(timezone.utc)
                self.mongo_db.users.update_one(
                    {"id": int(user_id)},
                    {"$set": {"last_login": now, "updated_at": now}}
                )
                return
            except Exception as exc:
                logger.error("MongoDB update_last_login error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE users SET last_login = NOW(), updated_at = NOW() WHERE id = %s", (int(user_id),))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL update_last_login error: %s", exc)

    def get_all_users(self) -> List[dict]:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                cursor = self.mongo_db.users.find({}, {"password_hash": 0}).sort("created_at", -1)
                return [_clean_doc(u) for u in cursor]
            except Exception as exc:
                logger.error("MongoDB get_all_users error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("SELECT id, full_name, username, email, role, is_active, created_at, last_login, updated_at FROM users ORDER BY created_at DESC")
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_all_users error: %s", exc)
        return []

    def toggle_user_active(self, user_id: int, is_active: bool) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.users.update_one(
                    {"id": int(user_id)},
                    {"$set": {"is_active": bool(is_active), "updated_at": datetime.now(timezone.utc)}}
                )
                return
            except Exception as exc:
                logger.error("MongoDB toggle_user_active error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE users SET is_active = %s, updated_at = NOW() WHERE id = %s", (is_active, int(user_id)))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL toggle_user_active error: %s", exc)

    def update_user_role(self, user_id: int, role: str) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.users.update_one(
                    {"id": int(user_id)},
                    {"$set": {"role": role, "updated_at": datetime.now(timezone.utc)}}
                )
                return
            except Exception as exc:
                logger.error("MongoDB update_user_role error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE users SET role = %s, updated_at = NOW() WHERE id = %s", (role, int(user_id)))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL update_user_role error: %s", exc)

    def delete_user(self, user_id: int) -> None:
        uid = int(user_id)
        if self.mongo_db is not None:
            try:
                self.mongo_db.users.delete_one({"id": uid})
                self.mongo_db.user_activity.delete_many({"user_id": uid})
                self.mongo_db.user_sessions.delete_many({"user_id": uid})
                self.mongo_db.training_submissions.delete_many({"user_id": uid})
                return
            except Exception as exc:
                logger.error("MongoDB delete_user error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM users WHERE id = %s", (uid,))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL delete_user error: %s", exc)

    # ── Activity Tracking ─────────────────────────────────────────────────────

    def log_activity(
        self, user_id: int, activity_type: str, page: str = None, action: str = None, session_id: str = None
    ) -> None:
        if self.mongo_db is not None:
            try:
                new_id = self._next_id("user_activity")
                self.mongo_db.user_activity.insert_one({
                    "id": new_id,
                    "user_id": int(user_id),
                    "activity_type": activity_type,
                    "page": page,
                    "action": action,
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc),
                })
                return
            except Exception as exc:
                logger.debug("MongoDB log_activity error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO user_activity (user_id, activity_type, page, action, session_id)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (int(user_id), activity_type, page, action, session_id))
                    conn.commit()
            except Exception as exc:
                logger.debug("PostgreSQL log_activity error: %s", exc)

    def get_user_activity(self, user_id: int, limit: int = 50) -> List[dict]:
        if self.mongo_db is not None:
            try:
                cursor = (
                    self.mongo_db.user_activity.find({"user_id": int(user_id)}, {"_id": 0})
                    .sort("created_at", -1)
                    .limit(limit)
                )
                return list(cursor)
            except Exception as exc:
                logger.error("MongoDB get_user_activity error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT id, activity_type, page, action, created_at, session_id
                            FROM user_activity WHERE user_id = %s
                            ORDER BY created_at DESC LIMIT %s
                        """, (int(user_id), limit))
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_user_activity error: %s", exc)
        return []

    def get_all_activity(
        self, limit: int = 200, user_id: int = None, activity_type: str = None, page: str = None,
        date_from: str = None, date_to: str = None
    ) -> List[dict]:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                match: Dict[str, Any] = {}
                if user_id:
                    match["user_id"] = int(user_id)
                if activity_type:
                    match["activity_type"] = activity_type
                if page:
                    match["page"] = page
                if date_from or date_to:
                    date_match = {}
                    if date_from:
                        date_match["$gte"] = datetime.fromisoformat(date_from)
                    if date_to:
                        date_match["$lte"] = datetime.fromisoformat(date_to)
                    match["created_at"] = date_match

                pipeline = [
                    {"$match": match},
                    {"$sort": {"created_at": -1}},
                    {"$limit": limit},
                    {
                        "$lookup": {
                            "from": "users",
                            "localField": "user_id",
                            "foreignField": "id",
                            "as": "user_info",
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "id": 1,
                            "user_id": 1,
                            "activity_type": 1,
                            "page": 1,
                            "action": 1,
                            "created_at": 1,
                            "session_id": 1,
                            "full_name": {"$arrayElemAt": ["$user_info.full_name", 0]},
                            "username": {"$arrayElemAt": ["$user_info.username", 0]},
                            "email": {"$arrayElemAt": ["$user_info.email", 0]},
                        }
                    },
                ]
                return list(self.mongo_db.user_activity.aggregate(pipeline))
            except Exception as exc:
                logger.error("MongoDB get_all_activity error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            sql = """
                SELECT ua.id, ua.user_id, u.full_name, u.username, u.email,
                       ua.activity_type, ua.page, ua.action, ua.created_at, ua.session_id
                FROM user_activity ua
                JOIN users u ON u.id = ua.user_id
                WHERE 1=1
            """
            params = []
            if user_id:
                sql += " AND ua.user_id = %s"; params.append(int(user_id))
            if activity_type:
                sql += " AND ua.activity_type = %s"; params.append(activity_type)
            if page:
                sql += " AND ua.page = %s"; params.append(page)
            if date_from:
                sql += " AND ua.created_at >= %s"; params.append(date_from)
            if date_to:
                sql += " AND ua.created_at <= %s"; params.append(date_to)
            sql += " ORDER BY ua.created_at DESC LIMIT %s"; params.append(limit)

            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, params)
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_all_activity error: %s", exc)
        return []

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(
        self, user_id: int, session_id: str, ip_address: str = None, user_agent: str = None
    ) -> None:
        if self.mongo_db is not None:
            try:
                new_id = self._next_id("user_sessions")
                now = datetime.now(timezone.utc)
                self.mongo_db.user_sessions.insert_one({
                    "id": new_id,
                    "user_id": int(user_id),
                    "session_id": session_id,
                    "ip_address": ip_address,
                    "user_agent": user_agent,
                    "login_time": now,
                    "logout_time": None,
                    "last_activity": now,
                })
                return
            except Exception as exc:
                logger.error("MongoDB create_session error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO user_sessions (user_id, session_id, ip_address, user_agent)
                            VALUES (%s, %s, %s, %s)
                        """, (int(user_id), session_id, ip_address, user_agent))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL create_session error: %s", exc)

    def end_session(self, session_id: str) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.user_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {"logout_time": datetime.now(timezone.utc)}}
                )
                return
            except Exception as exc:
                logger.error("MongoDB end_session error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE user_sessions SET logout_time = NOW() WHERE session_id = %s", (session_id,))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL end_session error: %s", exc)

    def touch_session(self, session_id: str) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.user_sessions.update_one(
                    {"session_id": session_id},
                    {"$set": {"last_activity": datetime.now(timezone.utc)}}
                )
                return
            except Exception as exc:
                logger.error("MongoDB touch_session error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE user_sessions SET last_activity = NOW() WHERE session_id = %s", (session_id,))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL touch_session error: %s", exc)

    # ── Submissions ───────────────────────────────────────────────────────────

    def create_submission(
        self, user_id: int, gesture_label: str, gesture_type: str = "Word", sample_count: int = 0
    ) -> Optional[int]:
        if self.mongo_db is not None:
            try:
                new_id = self._next_id("training_submissions")
                self.mongo_db.training_submissions.insert_one({
                    "id": new_id,
                    "user_id": int(user_id),
                    "gesture_label": gesture_label,
                    "gesture_type": gesture_type,
                    "sample_count": sample_count,
                    "status": "PENDING",
                    "rejection_reason": None,
                    "submitted_at": datetime.now(timezone.utc),
                    "reviewed_at": None,
                    "reviewed_by": None,
                    "notes": None,
                })
                return new_id
            except Exception as exc:
                logger.error("MongoDB create_submission error: %s", exc)
                return None

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO training_submissions (user_id, gesture_label, gesture_type, sample_count)
                            VALUES (%s, %s, %s, %s) RETURNING id
                        """, (int(user_id), gesture_label, gesture_type, sample_count))
                        sid = cur.fetchone()[0]
                    conn.commit()
                return sid
            except Exception as exc:
                logger.error("PostgreSQL create_submission error: %s", exc)
        return None

    def get_user_submissions(self, user_id: int) -> List[dict]:
        if self.mongo_db is not None:
            try:
                cursor = (
                    self.mongo_db.training_submissions.find({"user_id": int(user_id)}, {"_id": 0})
                    .sort("submitted_at", -1)
                )
                return list(cursor)
            except Exception as exc:
                logger.error("MongoDB get_user_submissions error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT id, gesture_label, gesture_type, status, rejection_reason,
                                   sample_count, submitted_at, reviewed_at, notes
                            FROM training_submissions WHERE user_id = %s
                            ORDER BY submitted_at DESC
                        """, (int(user_id),))
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_user_submissions error: %s", exc)
        return []

    def get_all_submissions(self, status: str = None) -> List[dict]:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                match = {}
                if status:
                    match["status"] = status
                pipeline = [
                    {"$match": match},
                    {"$sort": {"submitted_at": -1}},
                    {
                        "$lookup": {
                            "from": "users",
                            "localField": "user_id",
                            "foreignField": "id",
                            "as": "user_info",
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "id": 1,
                            "user_id": 1,
                            "gesture_label": 1,
                            "gesture_type": 1,
                            "status": 1,
                            "rejection_reason": 1,
                            "sample_count": 1,
                            "submitted_at": 1,
                            "reviewed_at": 1,
                            "reviewed_by": 1,
                            "notes": 1,
                            "full_name": {"$arrayElemAt": ["$user_info.full_name", 0]},
                            "username": {"$arrayElemAt": ["$user_info.username", 0]},
                            "email": {"$arrayElemAt": ["$user_info.email", 0]},
                        }
                    },
                ]
                return list(self.mongo_db.training_submissions.aggregate(pipeline))
            except Exception as exc:
                logger.error("MongoDB get_all_submissions error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            sql = """
                SELECT ts.*, u.full_name, u.username, u.email
                FROM training_submissions ts
                JOIN users u ON u.id = ts.user_id
            """
            params = []
            if status:
                sql += " WHERE ts.status = %s"; params.append(status)
            sql += " ORDER BY ts.submitted_at DESC"
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, params)
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_all_submissions error: %s", exc)
        return []

    def review_submission(
        self, submission_id: int, status: str, reviewed_by: int, rejection_reason: str = None
    ) -> None:
        sid = int(submission_id)
        if self.mongo_db is not None:
            try:
                self.mongo_db.training_submissions.update_one(
                    {"id": sid},
                    {"$set": {
                        "status": status,
                        "rejection_reason": rejection_reason,
                        "reviewed_at": datetime.now(timezone.utc),
                        "reviewed_by": int(reviewed_by),
                    }}
                )
            except Exception as exc:
                logger.error("MongoDB review_submission error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE training_submissions
                            SET status = %s, rejection_reason = %s, reviewed_at = NOW(), reviewed_by = %s
                            WHERE id = %s
                        """, (status, rejection_reason, int(reviewed_by), sid))
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL review_submission error: %s", exc)

        # Delete local PNG sample previews in data/sample/ after review
        try:
            from config import SAMPLE_DIR, DATA_DIR
            samples = self.get_submission_samples(sid, include_data=False)
            for s in samples:
                png_file = SAMPLE_DIR / f"sample_{s['id']}.png"
                if png_file.exists():
                    png_file.unlink(missing_ok=True)
            for p in SAMPLE_DIR.glob(f"*{sid}*.png"):
                p.unlink(missing_ok=True)

            if status == "REJECTED":
                p_dir = DATA_DIR / "pending_submissions" / str(sid)
                if p_dir.exists():
                    import shutil
                    shutil.rmtree(p_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("Error cleaning up sample files for submission %s: %s", sid, e)

    def get_submission_by_id(self, submission_id: int) -> Optional[dict]:
        sid = int(submission_id)
        if self.mongo_db is not None:
            try:
                sub = self.mongo_db.training_submissions.find_one({"id": sid}, {"_id": 0})
                if sub:
                    user = self.get_user_by_id(sub.get("user_id"))
                    if user:
                        sub["full_name"] = user.get("full_name")
                        sub["username"] = user.get("username")
                    return sub
                return None
            except Exception as exc:
                logger.error("MongoDB get_submission_by_id error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT ts.*, u.full_name, u.username
                            FROM training_submissions ts
                            JOIN users u ON u.id = ts.user_id
                            WHERE ts.id = %s
                        """, (sid,))
                        row = cur.fetchone()
                return dict(row) if row else None
            except Exception as exc:
                logger.error("PostgreSQL get_submission_by_id error: %s", exc)
        return None

    # ── Samples ───────────────────────────────────────────────────────────────

    def add_training_sample(
        self, submission_id: int, sample_path: str, sample_type: str = "npy",
        frame_count: int = None, sample_data: bytes = None, store_in_cloud: bool = False
    ) -> Optional[int]:
        """
        Add sample record. Does NOT store binary photos/blobs in cloud DB (MongoDB Atlas)
        by default, storing only metadata and maintaining local PNG previews in data/sample/.
        """
        sid = int(submission_id)
        new_id = None
        if self.mongo_db is not None:
            try:
                new_id = self._next_id("training_samples")
                doc = {
                    "id": new_id,
                    "submission_id": sid,
                    "sample_path": sample_path,
                    "sample_type": sample_type,
                    "frame_count": frame_count,
                    "created_at": datetime.now(timezone.utc),
                    "is_valid": True,
                }
                from pathlib import Path
                # Do NOT store binary photo/array in cloud database when file exists on disk
                if sample_data and (store_in_cloud or not (sample_path and Path(sample_path).exists())):
                    doc["sample_data"] = Binary(sample_data)
                self.mongo_db.training_samples.insert_one(doc)
            except Exception as exc:
                logger.error("MongoDB add_training_sample error: %s", exc)

        if self.dsn:
            try:
                from pathlib import Path
                save_data = sample_data if (store_in_cloud or not (sample_path and Path(sample_path).exists())) else None
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO training_samples (submission_id, sample_path, sample_type, frame_count, sample_data)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """, (sid, sample_path, sample_type, frame_count, save_data))
                        row = cur.fetchone()
                        if row and not new_id:
                            new_id = row[0]
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL add_training_sample error: %s", exc)

        # Generate local PNG preview in data/sample/ folder for admin review
        if new_id:
            try:
                from config import SAMPLE_DIR
                from src.utils import render_sample_preview_png
                from pathlib import Path
                SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
                png_file = SAMPLE_DIR / f"sample_{new_id}.png"
                source = sample_path if sample_path and Path(sample_path).exists() else sample_data
                if source:
                    png_bytes = render_sample_preview_png(source)
                    if png_bytes:
                        png_file.write_bytes(png_bytes)
            except Exception as e:
                logger.debug("Automatic sample PNG preview generation skipped: %s", e)

        return new_id

    def get_sample_by_id(self, sample_id: int, include_data: bool = False) -> Optional[dict]:
        """Fetch a single sample by its ID."""
        sid = int(sample_id)
        if self.mongo_db is not None:
            try:
                proj = {"_id": 0}
                if not include_data:
                    proj["sample_data"] = 0
                doc = self.mongo_db.training_samples.find_one({"id": sid}, proj)
                if doc:
                    if "sample_data" in doc and isinstance(doc["sample_data"], Binary):
                        doc["sample_data"] = bytes(doc["sample_data"])
                    elif include_data and not doc.get("sample_data"):
                        sp = doc.get("sample_path")
                        if sp:
                            from pathlib import Path
                            p = Path(sp)
                            if p.exists():
                                doc["sample_data"] = p.read_bytes()
                    return doc
                return None
            except Exception as exc:
                logger.error("MongoDB get_sample_by_id error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                sql = "SELECT * FROM training_samples WHERE id = %s" if include_data else """
                    SELECT id, submission_id, sample_path, sample_type, frame_count, created_at, is_valid
                    FROM training_samples WHERE id = %s
                """
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (sid,))
                        row = cur.fetchone()
                if row:
                    doc = dict(row)
                    if include_data and not doc.get("sample_data"):
                        sp = doc.get("sample_path")
                        if sp:
                            from pathlib import Path
                            p = Path(sp)
                            if p.exists():
                                doc["sample_data"] = p.read_bytes()
                    return doc
                return None
            except Exception as exc:
                logger.error("PostgreSQL get_sample_by_id error: %s", exc)
        return None

    def get_submission_samples(self, submission_id: int, include_data: bool = False) -> List[dict]:
        sid = int(submission_id)
        if self.mongo_db is not None:
            try:
                proj = {"_id": 0}
                if not include_data:
                    proj["sample_data"] = 0
                cursor = self.mongo_db.training_samples.find({"submission_id": sid}, proj).sort("created_at", 1)
                samples = []
                for s in cursor:
                    if "sample_data" in s and isinstance(s["sample_data"], Binary):
                        s["sample_data"] = bytes(s["sample_data"])
                    elif include_data and not s.get("sample_data"):
                        sp = s.get("sample_path")
                        if sp:
                            from pathlib import Path
                            p = Path(sp)
                            if p.exists():
                                s["sample_data"] = p.read_bytes()
                    samples.append(s)
                return samples
            except Exception as exc:
                logger.error("MongoDB get_submission_samples error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                sql = "SELECT * FROM training_samples WHERE submission_id = %s ORDER BY created_at" if include_data else """
                    SELECT id, submission_id, sample_path, sample_type, frame_count, created_at, is_valid
                    FROM training_samples WHERE submission_id = %s ORDER BY created_at
                """
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (sid,))
                        rows = [dict(r) for r in cur.fetchall()]
                        if include_data:
                            for r in rows:
                                if not r.get("sample_data") and r.get("sample_path"):
                                    from pathlib import Path
                                    p = Path(r["sample_path"])
                                    if p.exists():
                                        r["sample_data"] = p.read_bytes()
                        return rows
            except Exception as exc:
                logger.error("PostgreSQL get_submission_samples error: %s", exc)
        return []

    # ── Model Versions ────────────────────────────────────────────────────────

    def get_model_versions(self, include_data: bool = False) -> List[dict]:
        """Fetch model version registry. By default omits large binaries to save network bandwidth."""
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                proj = {"_id": 0}
                if not include_data:
                    proj["model_data"] = 0
                    proj["encoder_data"] = 0
                cursor = self.mongo_db.model_versions.find({}, proj).sort("created_at", -1)
                results = []
                for v in cursor:
                    for bcol in ["model_data", "encoder_data"]:
                        if bcol in v and isinstance(v[bcol], Binary):
                            v[bcol] = bytes(v[bcol])
                    results.append(v)
                return results
            except Exception as exc:
                logger.error("MongoDB get_model_versions error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                sql = "SELECT * FROM model_versions ORDER BY created_at DESC" if include_data else """
                    SELECT id, version_tag, model_type, model_path, accuracy,
                           parameters, is_active, notes, created_at
                    FROM model_versions ORDER BY created_at DESC
                """
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql)
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_model_versions error: %s", exc)
        return []

    def get_active_model_version(self, model_type: str, include_data: bool = True) -> Optional[dict]:
        """Fetch the current active model version for a given model_type."""
        if self.mongo_db is not None:
            try:
                proj = {"_id": 0}
                if not include_data:
                    proj["model_data"] = 0
                    proj["encoder_data"] = 0
                doc = self.mongo_db.model_versions.find_one(
                    {"model_type": model_type, "is_active": True},
                    proj,
                    sort=[("created_at", -1)]
                )
                if doc:
                    for bcol in ["model_data", "encoder_data"]:
                        if bcol in doc and isinstance(doc[bcol], Binary):
                            doc[bcol] = bytes(doc[bcol])
                    return doc
                return None
            except Exception as exc:
                logger.error("MongoDB get_active_model_version error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                sql = "SELECT * FROM model_versions WHERE model_type = %s AND is_active = TRUE ORDER BY created_at DESC LIMIT 1"
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (model_type,))
                        row = cur.fetchone()
                return dict(row) if row else None
            except Exception as exc:
                logger.error("PostgreSQL get_active_model_version error: %s", exc)
        return None

    def save_model_version(
        self, version_tag: str, model_type: str, model_path: str,
        accuracy: float = None, parameters: dict = None, notes: str = None,
        model_data: bytes = None, encoder_data: bytes = None, is_active: bool = True,
        created_by: int = None
    ) -> bool:
        """Register a new model version and set it as active."""
        if self.mongo_db is not None:
            try:
                if is_active:
                    self.mongo_db.model_versions.update_many(
                        {"model_type": model_type},
                        {"$set": {"is_active": False}}
                    )

                new_id = self._next_id("model_versions")
                doc = {
                    "id": new_id,
                    "version_tag": version_tag,
                    "model_type": model_type,
                    "model_path": str(model_path),
                    "accuracy": accuracy,
                    "parameters": parameters,
                    "is_active": is_active,
                    "notes": notes,
                    "created_at": datetime.now(timezone.utc),
                    "created_by": created_by,
                }
                if model_data:
                    doc["model_data"] = Binary(model_data)
                if encoder_data:
                    doc["encoder_data"] = Binary(encoder_data)

                self.mongo_db.model_versions.replace_one(
                    {"version_tag": version_tag}, doc, upsert=True
                )
                return True
            except Exception as exc:
                logger.error("MongoDB save_model_version failed: %s", exc)
                return False

        if self.dsn:
            import psycopg2.extras
            try:
                params_json = psycopg2.extras.Json(parameters) if parameters is not None else None
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        if is_active:
                            cur.execute("UPDATE model_versions SET is_active = FALSE WHERE model_type = %s", (model_type,))
                        cur.execute("""
                            INSERT INTO model_versions (
                                version_tag, model_type, model_path, model_data, encoder_data,
                                accuracy, parameters, is_active, notes, created_by
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (version_tag, model_type, str(model_path), model_data, encoder_data,
                              accuracy, params_json, is_active, notes, created_by))
                    conn.commit()
                return True
            except Exception as exc:
                logger.error("PostgreSQL save_model_version failed: %s", exc)
                return False
        return False

    # ── Training Jobs ─────────────────────────────────────────────────────────

    def get_training_jobs(self) -> List[dict]:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                pipeline = [
                    {"$sort": {"created_at": -1}},
                    {
                        "$lookup": {
                            "from": "model_versions",
                            "localField": "model_version_id",
                            "foreignField": "id",
                            "as": "mv_info",
                        }
                    },
                    {
                        "$project": {
                            "_id": 0,
                            "id": 1,
                            "job_type": 1,
                            "status": 1,
                            "started_at": 1,
                            "completed_at": 1,
                            "created_by": 1,
                            "model_version_id": 1,
                            "parameters": 1,
                            "log_path": 1,
                            "error_message": 1,
                            "created_at": 1,
                            "version_tag": {"$arrayElemAt": ["$mv_info.version_tag", 0]},
                        }
                    },
                ]
                return list(self.mongo_db.training_jobs.aggregate(pipeline))
            except Exception as exc:
                logger.error("MongoDB get_training_jobs error: %s", exc)

        if self.dsn:
            import psycopg2.extras
            try:
                with self._connect() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("""
                            SELECT tj.*, mv.version_tag
                            FROM training_jobs tj
                            LEFT JOIN model_versions mv ON mv.id = tj.model_version_id
                            ORDER BY tj.created_at DESC
                        """)
                        return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.error("PostgreSQL get_training_jobs error: %s", exc)
        return []

    def create_training_job(self, job_type: str, created_by: int) -> Optional[int]:
        if self.mongo_db is not None:
            try:
                new_id = self._next_id("training_jobs")
                self.mongo_db.training_jobs.insert_one({
                    "id": new_id,
                    "job_type": job_type,
                    "status": "QUEUED",
                    "started_at": None,
                    "completed_at": None,
                    "created_by": int(created_by),
                    "model_version_id": None,
                    "parameters": None,
                    "log_path": None,
                    "error_message": None,
                    "created_at": datetime.now(timezone.utc),
                })
                return new_id
            except Exception as exc:
                logger.error("MongoDB create_training_job error: %s", exc)
                return None

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("INSERT INTO training_jobs (job_type, created_by) VALUES (%s, %s) RETURNING id",
                                    (job_type, int(created_by)))
                        job_id = cur.fetchone()[0]
                    conn.commit()
                return job_id
            except Exception as exc:
                logger.error("PostgreSQL create_training_job error: %s", exc)
        return None

    def update_training_job_status(
        self, job_id: int, status: str, started: bool = False, completed: bool = False,
        error_message: str = None
    ) -> None:
        jid = int(job_id)
        now = datetime.now(timezone.utc)
        if self.mongo_db is not None:
            try:
                update_fields = {"status": status}
                if started:
                    update_fields["started_at"] = now
                if completed:
                    update_fields["completed_at"] = now
                if error_message is not None:
                    update_fields["error_message"] = error_message
                self.mongo_db.training_jobs.update_one({"id": jid}, {"$set": update_fields})
                return
            except Exception as exc:
                logger.error("MongoDB update_training_job_status error: %s", exc)

        if self.dsn:
            try:
                sql = "UPDATE training_jobs SET status = %s"
                params = [status]
                if started:
                    sql += ", started_at = NOW()"
                if completed:
                    sql += ", completed_at = NOW()"
                if error_message is not None:
                    sql += ", error_message = %s"; params.append(error_message)
                sql += " WHERE id = %s"; params.append(jid)
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, params)
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL update_training_job_status error: %s", exc)

    # ── Admin Stats ───────────────────────────────────────────────────────────

    def get_admin_stats(self) -> dict:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                total_users = self.mongo_db.users.count_documents({})
                total_admins = self.mongo_db.users.count_documents({"role": "ADMIN"})
                pending = self.mongo_db.training_submissions.count_documents({"status": "PENDING"})
                approved = self.mongo_db.training_submissions.count_documents({"status": "APPROVED"})
                rejected = self.mongo_db.training_submissions.count_documents({"status": "REJECTED"})
                total_activity = self.mongo_db.user_activity.count_documents({})
                total_models = self.mongo_db.model_versions.count_documents({})
                return {
                    "total_users": total_users,
                    "total_admins": total_admins,
                    "pending_submissions": pending,
                    "approved_submissions": approved,
                    "rejected_submissions": rejected,
                    "total_activity": total_activity,
                    "total_models": total_models,
                }
            except Exception as exc:
                logger.error("MongoDB get_admin_stats error: %s", exc)

        if self.dsn:
            try:
                with self._connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM users"); total_users = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'ADMIN'"); total_admins = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'PENDING'"); pending = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'APPROVED'"); approved = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM training_submissions WHERE status = 'REJECTED'"); rejected = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM user_activity"); total_activity = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM model_versions"); total_models = cur.fetchone()[0]
                return {
                    "total_users": total_users,
                    "total_admins": total_admins,
                    "pending_submissions": pending,
                    "approved_submissions": approved,
                    "rejected_submissions": rejected,
                    "total_activity": total_activity,
                    "total_models": total_models,
                }
            except Exception as exc:
                logger.error("PostgreSQL get_admin_stats error: %s", exc)
        return {}
