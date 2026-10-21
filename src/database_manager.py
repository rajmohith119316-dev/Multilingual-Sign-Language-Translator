"""
src/database_manager.py — Multi-Backend Database Manager (MoE Edition)
Supports MongoDB Atlas (primary) and PostgreSQL (fallback / rollback).
Collections / Tables: gestures, prediction_history, translation_cache.
"""

import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)


def _sanitize_doc(doc: Optional[dict]) -> Optional[dict]:
    """Helper to convert ObjectId or datetime to JSON-friendly representations."""
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["_id"] = str(d["_id"])
    return d


class DatabaseManager:
    """Thread-safe database manager supporting MongoDB Atlas with PostgreSQL rollback."""

    def __init__(self) -> None:
        self.backend = getattr(config, "DATABASE_BACKEND", "mongodb").lower()
        self.mongo_client = None
        self.mongo_db = None
        self.dsn = getattr(config, "DATABASE_URL", None) if self.backend == "postgres" else None
        self._in_memory_cache: Dict[tuple, str] = {}

        if self.backend == "mongodb" and config.MONGODB_URI:
            try:
                try:
                    import dns.resolver
                    _res = dns.resolver.get_default_resolver()
                    _res.nameservers = ["8.8.8.8", "1.1.1.1"] + [ns for ns in _res.nameservers if ns not in ("8.8.8.8", "1.1.1.1")]
                    _res.timeout = 2.0
                    _res.lifetime = 5.0
                except Exception:
                    pass

                logging.getLogger("pymongo.client").setLevel(logging.CRITICAL)
                logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
                from pymongo import MongoClient

                mongo_kwargs = {
                    "maxPoolSize": 25,
                    "minPoolSize": 0,
                    "maxIdleTimeMS": 45000,
                    "serverSelectionTimeoutMS": 15000,
                    "connectTimeoutMS": 25000,
                    "socketTimeoutMS": 30000,
                    "retryWrites": True,
                    "retryReads": True,
                }
                try:
                    import certifi
                    mongo_kwargs["tlsCAFile"] = certifi.where()
                except ImportError:
                    pass

                self.mongo_client = MongoClient(config.MONGODB_URI, **mongo_kwargs)
                # Verify connectivity with ping
                self.mongo_client.admin.command("ping")
                self.mongo_db = self.mongo_client[config.MONGODB_DATABASE]
                self._init_mongo_indexes()
                logger.info("DatabaseManager initialized with MongoDB Atlas (db='%s').", config.MONGODB_DATABASE)
                return
            except Exception as exc:
                logger.warning(
                    "MongoDB Atlas connection unavailable (%s). Running in resilient offline mode.",
                    exc,
                )
                self.mongo_db = None
                return

        # PostgreSQL backend (explicit rollback mode)
        if self.backend == "postgres" and self.dsn:
            try:
                self._init_postgres_schema()
                logger.info("DatabaseManager initialized with PostgreSQL.")
            except Exception as pg_exc:
                logger.warning("PostgreSQL initialization warning: %s.", pg_exc)
        else:
            logger.warning("No live database connected. DatabaseManager running in resilient offline/cache mode.")

    def _ensure_mongo(self) -> bool:
        if self.backend != "mongodb" or not config.MONGODB_URI:
            return False
        if self.mongo_db is not None:
            return True
        try:
            from pymongo import MongoClient

            mongo_kwargs = {
                "maxPoolSize": 25,
                "minPoolSize": 0,
                "maxIdleTimeMS": 45000,
                "serverSelectionTimeoutMS": 10000,
                "connectTimeoutMS": 20000,
                "socketTimeoutMS": 25000,
                "retryWrites": True,
                "retryReads": True,
            }
            try:
                import certifi
                mongo_kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass

            self.mongo_client = MongoClient(config.MONGODB_URI, **mongo_kwargs)
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[config.MONGODB_DATABASE]
            self._init_mongo_indexes()
            logger.info("DatabaseManager auto-reconnected to MongoDB Atlas.")
            return True
        except Exception:
            self.mongo_db = None
            return False

    # ── MongoDB Helpers ────────────────────────────────────────────────────────

    def _next_id(self, collection_name: str) -> int:
        """Atomically generate the next sequential integer ID."""
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
            self.mongo_db.gestures.create_index("id", unique=True)
            self.mongo_db.gestures.create_index("gesture_name", unique=True)
            self.mongo_db.prediction_history.create_index("id", unique=True)
            self.mongo_db.prediction_history.create_index([("timestamp", -1)])
            self.mongo_db.prediction_history.create_index("predicted_label")
            self.mongo_db.translation_cache.create_index("id", unique=True)
            self.mongo_db.translation_cache.create_index(
                [("english_text", 1), ("target_language", 1)], unique=True
            )
        except Exception as exc:
            logger.debug("Mongo index notice: %s", exc)

    # ── PostgreSQL Internals (for rollback compatibility) ─────────────────────

    @contextlib.contextmanager
    def _connect_pg(self):
        import psycopg2
        conn = None
        dsn = self.dsn
        last_exc = None
        for attempt in range(3):
            try:
                conn = psycopg2.connect(dsn, connect_timeout=8)
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

    def _init_postgres_schema(self) -> None:
        statements = [
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
        with self._connect_pg() as conn:
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
            conn.commit()

    # ── Gestures ──────────────────────────────────────────────────────────────

    def add_gesture(
        self,
        gesture_name: str,
        gesture_type: str,          # 'Alphabet' | 'Word'
        is_custom: bool = False,
        sample_count: int = 0,
    ) -> None:
        from src.utils import sanitize_label
        gesture_name = sanitize_label(gesture_name)
        if self.mongo_db is not None:
            try:
                now = datetime.now(timezone.utc)
                existing = self.mongo_db.gestures.find_one({"gesture_name": gesture_name})
                if existing:
                    self.mongo_db.gestures.update_one(
                        {"gesture_name": gesture_name},
                        {"$set": {
                            "gesture_type": gesture_type,
                            "is_custom": is_custom,
                            "sample_count": sample_count,
                            "updated_at": now,
                        }}
                    )
                else:
                    new_id = self._next_id("gestures")
                    self.mongo_db.gestures.insert_one({
                        "id": new_id,
                        "gesture_name": gesture_name,
                        "gesture_type": gesture_type,
                        "is_custom": is_custom,
                        "sample_count": sample_count,
                        "date_added": now,
                        "updated_at": now,
                    })
                return
            except Exception as exc:
                logger.error("MongoDB add_gesture('%s') failed: %s", gesture_name, exc)

        # Fallback (PostgreSQL explicit rollback mode)
        if self.backend == "postgres" and self.dsn:
            sql = """
                INSERT INTO gestures (gesture_name, gesture_type, is_custom, sample_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(gesture_name) DO UPDATE SET
                    sample_count = EXCLUDED.sample_count,
                    is_custom    = EXCLUDED.is_custom
            """
            with self._connect_pg() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (gesture_name, gesture_type, is_custom, sample_count))
                conn.commit()

    def get_all_gestures(self) -> List[dict]:
        self._ensure_mongo()
        if self.mongo_db is not None:
            try:
                cursor = self.mongo_db.gestures.find({}, {"_id": 0}).sort("gesture_name", 1)
                return list(cursor)
            except Exception as exc:
                logger.error("MongoDB get_all_gestures failed: %s", exc)

        if self.backend == "postgres" and self.dsn:
            import psycopg2.extras
            with self._connect_pg() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT * FROM gestures ORDER BY gesture_name")
                    return [dict(r) for r in cur.fetchall()]
        return []

    def get_gesture_names(self) -> List[str]:
        return [g["gesture_name"] for g in self.get_all_gestures() if "gesture_name" in g]

    def delete_gesture(self, gesture_name: str) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.gestures.delete_one({"gesture_name": gesture_name})
                return
            except Exception as exc:
                logger.error("MongoDB delete_gesture failed: %s", exc)

        if self.backend == "postgres" and self.dsn:
            with self._connect_pg() as conn:
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
        """Non-blocking, fault-tolerant write to prediction_history with offline journal."""
        doc = {
            "predicted_label": predicted_label,
            "confidence_score": round(float(confidence_score), 4),
            "target_language": target_language,
            "translated_text": translated_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        journal_file = config.LOGS_DIR / "offline_prediction_journal.jsonl"

        if self.mongo_db is not None:
            try:
                # First drain any previously journaled records
                if journal_file.exists():
                    self._flush_journal(journal_file)

                new_id = self._next_id("prediction_history")
                doc["id"] = new_id
                doc["timestamp"] = datetime.now(timezone.utc)
                self.mongo_db.prediction_history.insert_one(doc)
                return
            except Exception as exc:
                logger.debug("MongoDB log_prediction error: %s. Journaling locally.", exc)
                self._append_to_journal(journal_file, doc)
                return

        # Fallback offline journaling if MongoDB is configured but temporarily disconnected
        if self.backend == "mongodb":
            self._append_to_journal(journal_file, doc)
            return

        if self.backend == "postgres" and self.dsn:
            sql = """
                INSERT INTO prediction_history
                    (predicted_label, confidence_score, target_language, translated_text)
                VALUES (%s, %s, %s, %s)
            """
            try:
                with self._connect_pg() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, (predicted_label, round(confidence_score, 4),
                                         target_language, translated_text))
                    conn.commit()
            except Exception as exc:
                logger.debug("PostgreSQL log_prediction failed: %s", exc)

    def _append_to_journal(self, journal_file: Path, doc: dict) -> None:
        try:
            journal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(journal_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc) + "\n")
        except Exception as j_err:
            logger.debug("Could not write to prediction journal: %s", j_err)

    def _flush_journal(self, journal_file: Path) -> None:
        if self.mongo_db is None or not journal_file.exists():
            return
        try:
            lines = journal_file.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                journal_file.unlink(missing_ok=True)
                return

            buffered_docs = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if "id" not in d:
                        d["id"] = self._next_id("prediction_history")
                    if isinstance(d.get("timestamp"), str):
                        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
                    buffered_docs.append(d)
                except Exception:
                    pass

            if buffered_docs:
                self.mongo_db.prediction_history.insert_many(buffered_docs, ordered=False)
                logger.info("Successfully flushed %d offline journaled predictions to MongoDB.", len(buffered_docs))
            journal_file.unlink(missing_ok=True)
        except Exception as flush_err:
            logger.debug("Journal flush notice: %s", flush_err)

    def get_history_summary(self, limit: int = 200) -> List[dict]:
        if self.mongo_db is not None:
            try:
                cursor = (
                    self.mongo_db.prediction_history.find({}, {"_id": 0})
                    .sort("id", -1)
                    .limit(limit)
                )
                rows = list(cursor)
                # Ensure timestamp is formatted cleanly for frontend display
                for r in rows:
                    if isinstance(r.get("timestamp"), datetime):
                        r["timestamp"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                return rows
            except Exception as exc:
                logger.error("MongoDB get_history_summary failed: %s", exc)

        if self.backend == "postgres" and self.dsn:
            import psycopg2.extras
            sql = """
                SELECT id, predicted_label, confidence_score, target_language,
                       translated_text, timestamp
                FROM prediction_history
                ORDER BY id DESC LIMIT %s
            """
            try:
                with self._connect_pg() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (limit,))
                        rows = cur.fetchall()
                res = []
                for r in rows:
                    d = dict(r)
                    if isinstance(d.get("timestamp"), datetime):
                        d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                    res.append(d)
                return res
            except Exception as exc:
                logger.error("PostgreSQL get_history_summary failed: %s", exc)
        return []

    # Alias for compatibility
    get_recent_history = get_history_summary

    def clear_history(self) -> None:
        if self.mongo_db is not None:
            try:
                self.mongo_db.prediction_history.delete_many({})
                return
            except Exception as exc:
                logger.error("MongoDB clear_history failed: %s", exc)

        if self.backend == "postgres" and self.dsn:
            try:
                with self._connect_pg() as conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM prediction_history")
                    conn.commit()
            except Exception as exc:
                logger.error("PostgreSQL clear_history failed: %s", exc)

    # ── Translation Cache ──────────────────────────────────────────────────────

    def get_cached_translation(
        self, english_text: str, target_language: str
    ) -> Optional[str]:
        cache_key = (english_text.strip().upper(), target_language.strip().lower())
        if cache_key in self._in_memory_cache:
            return self._in_memory_cache[cache_key]

        if self.mongo_db is not None:
            try:
                doc = self.mongo_db.translation_cache.find_one(
                    {"english_text": english_text, "target_language": target_language},
                    {"translated_text": 1, "_id": 0},
                )
                if doc and "translated_text" in doc:
                    val = doc["translated_text"]
                    self._in_memory_cache[cache_key] = val
                    return val
                return None
            except Exception as exc:
                logger.debug("MongoDB translation cache lookup error: %s", exc)

        if self.backend == "postgres" and self.dsn:
            import psycopg2.extras
            sql = """
                SELECT translated_text FROM translation_cache
                WHERE english_text = %s AND target_language = %s
            """
            try:
                with self._connect_pg() as conn:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute(sql, (english_text, target_language))
                        row = cur.fetchone()
                if row:
                    val = row["translated_text"]
                    self._in_memory_cache[cache_key] = val
                    return val
            except Exception as exc:
                logger.debug("PostgreSQL translation cache lookup error: %s", exc)
        return None

    def cache_translation(
        self, english_text: str, target_language: str, translated_text: str
    ) -> None:
        cache_key = (english_text.strip().upper(), target_language.strip().lower())
        self._in_memory_cache[cache_key] = translated_text

        if self.mongo_db is not None:
            try:
                new_id = self._next_id("translation_cache")
                self.mongo_db.translation_cache.update_one(
                    {"english_text": english_text, "target_language": target_language},
                    {"$setOnInsert": {"id": new_id}, "$set": {"translated_text": translated_text}},
                    upsert=True,
                )
                return
            except Exception as exc:
                logger.debug("MongoDB cache_translation error: %s", exc)
                return

        if self.backend == "postgres" and self.dsn:
            sql = """
                INSERT INTO translation_cache (english_text, target_language, translated_text)
                VALUES (%s, %s, %s)
                ON CONFLICT(english_text, target_language) DO NOTHING
            """
            try:
                with self._connect_pg() as conn:
                    with conn.cursor() as cur:
                        cur.execute(sql, (english_text, target_language, translated_text))
                    conn.commit()
            except Exception as exc:
                logger.debug("PostgreSQL cache_translation failed: %s", exc)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        if self.backend == "mongodb":
            self._ensure_mongo()
            if self.mongo_db is not None:
                try:
                    n_total = self.mongo_db.gestures.count_documents({})
                    n_alpha = self.mongo_db.gestures.count_documents({"gesture_type": "Alphabet"})
                    n_word = self.mongo_db.gestures.count_documents({"gesture_type": "Word"})
                    n_custom = self.mongo_db.gestures.count_documents({"is_custom": True})
                    n_history = self.mongo_db.prediction_history.count_documents({})
                    n_cache = self.mongo_db.translation_cache.count_documents({})
                    return {
                        "total_gestures": n_total,
                        "alphabet_gestures": n_alpha,
                        "word_gestures": n_word,
                        "custom_gestures": n_custom,
                        "history_entries": n_history,
                        "cache_entries": n_cache,
                        "backend": "mongodb",
                    }
                except Exception as exc:
                    logger.error("MongoDB get_stats failed: %s", exc)
            return {
                "total_gestures": 0,
                "alphabet_gestures": 0,
                "word_gestures": 0,
                "custom_gestures": 0,
                "history_entries": 0,
                "cache_entries": 0,
                "backend": "mongodb",
                "status": "offline_resilient",
            }

        if self.dsn:
            try:
                with self._connect_pg() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT COUNT(*) FROM gestures"); n_total = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Alphabet'"); n_alpha = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM gestures WHERE gesture_type='Word'"); n_word = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM gestures WHERE is_custom=TRUE"); n_custom = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM prediction_history"); n_history = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM translation_cache"); n_cache = cur.fetchone()[0]
                return {
                    "total_gestures": n_total,
                    "alphabet_gestures": n_alpha,
                    "word_gestures": n_word,
                    "custom_gestures": n_custom,
                    "history_entries": n_history,
                    "cache_entries": n_cache,
                    "backend": "postgres",
                }
            except Exception as exc:
                logger.error("PostgreSQL get_stats failed: %s", exc)
        return {}
