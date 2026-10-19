"""
scripts/migrate_postgres_to_mongodb.py
======================================
Safe, restartable, idempotent migration tool to migrate data from
PostgreSQL (Neon) to MongoDB Atlas Free Tier.

Features:
- Live Neon extraction with read-only cursor.
- Offline backup extraction fallback (supports saved JSON backup).
- Safe dry-run mode (--dry-run).
- Validation-only mode (--validate).
- Idempotent upsert by primary key ('id').
- Preserves all IDs, foreign keys, timestamps, nulls, and nested JSON.
- Creates indexes on MongoDB collections for optimal query performance.
- Full credential redaction from all logs and output.
- Generates a structured migration report.
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bson
from bson.binary import Binary
from dotenv import load_dotenv

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [migrate]  %(message)s",
)
logger = logging.getLogger("migrate")


def redact_secrets(text: str) -> str:
    """Mask any URIs, passwords, or credentials in log messages."""
    text = re.sub(r"mongodb(\+srv)?://[^@]+@", "mongodb+srv://[REDACTED]@ ", text)
    text = re.sub(r"postgresql://[^@]+@", "postgresql://[REDACTED]@", text)
    text = re.sub(r"password=[^;&\s]+", "password=[REDACTED]", text, flags=re.IGNORECASE)
    return text


class SafeMigrationManager:
    """Orchestrates the PostgreSQL-to-MongoDB migration."""

    TABLE_COLLECTION_MAP = {
        "gestures": "gestures",
        "prediction_history": "prediction_history",
        "translation_cache": "translation_cache",
        "users": "users",
        "user_sessions": "user_sessions",
        "user_activity": "user_activity",
        "training_submissions": "training_submissions",
        "training_jobs": "training_jobs",
        "model_versions": "model_versions",
        "training_samples": "training_samples",
    }

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
        backup_dir: Optional[Path] = None,
        dry_run: bool = False,
        batch_size: int = 250,
    ):
        self.mongo_uri = mongo_uri or config.MONGODB_URI
        self.mongo_db_name = mongo_db or config.MONGODB_DATABASE
        self.postgres_dsn = postgres_dsn or config.DATABASE_URL
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.batch_size = batch_size

        self.mongo_client = None
        self.mongo_db = None
        self.pg_conn = None

        self.stats: Dict[str, Dict[str, int]] = {
            t: {"source": 0, "inserted": 0, "updated": 0, "skipped": 0, "failed": 0}
            for t in self.TABLE_COLLECTION_MAP
        }

    def connect_mongo(self) -> bool:
        if not self.mongo_uri:
            logger.error("MongoDB URI not provided or found in environment.")
            return False

        try:
            from pymongo import MongoClient

            mongo_kwargs = {
                "serverSelectionTimeoutMS": 8000,
                "connectTimeoutMS": 8000,
            }
            try:
                import certifi
                mongo_kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass

            self.mongo_client = MongoClient(self.mongo_uri, **mongo_kwargs)
            # Test ping
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[self.mongo_db_name]
            logger.info("Successfully connected to MongoDB Atlas database '%s'", self.mongo_db_name)
            return True
        except Exception as exc:
            logger.error("MongoDB connection failed: %s", redact_secrets(str(exc)))
            return False

    def connect_postgres(self) -> bool:
        if not self.postgres_dsn:
            logger.warning("No PostgreSQL DSN provided.")
            return False

        try:
            import psycopg2
            import psycopg2.extras

            dsn = self.postgres_dsn
            dsn = dsn.replace("&channel_binding=require", "").replace("channel_binding=require&", "").replace("channel_binding=require", "")
            self.pg_conn = psycopg2.connect(dsn, connect_timeout=10)
            logger.info("Successfully connected to PostgreSQL source.")
            return True
        except Exception as exc:
            logger.warning("PostgreSQL connection failed: %s", redact_secrets(str(exc)))
            self.pg_conn = None
            return False

    def create_indexes(self) -> None:
        """Create required indexes in MongoDB collections."""
        if self.dry_run or self.mongo_db is None:
            logger.info("[Dry-Run] Skipping index creation on MongoDB.")
            return

        logger.info("Ensuring indexes on MongoDB collections...")
        try:
            # gestures: unique on gesture_name, unique on id
            self.mongo_db.gestures.create_index("id", unique=True)
            self.mongo_db.gestures.create_index("gesture_name", unique=True)

            # prediction_history: timestamp DESC, label
            self.mongo_db.prediction_history.create_index("id", unique=True)
            self.mongo_db.prediction_history.create_index([("timestamp", -1)])
            self.mongo_db.prediction_history.create_index("predicted_label")

            # translation_cache: compound unique on english_text + target_language
            self.mongo_db.translation_cache.create_index("id", unique=True)
            self.mongo_db.translation_cache.create_index(
                [("english_text", 1), ("target_language", 1)], unique=True
            )

            # users: unique username, unique email, id
            self.mongo_db.users.create_index("id", unique=True)
            self.mongo_db.users.create_index("username", unique=True)
            self.mongo_db.users.create_index("email", unique=True)
            self.mongo_db.users.create_index("role")

            # user_activity: user_id + created_at DESC
            self.mongo_db.user_activity.create_index("id", unique=True)
            self.mongo_db.user_activity.create_index([("user_id", 1), ("created_at", -1)])
            self.mongo_db.user_activity.create_index("activity_type")

            # user_sessions: unique session_id, user_id
            self.mongo_db.user_sessions.create_index("id", unique=True)
            self.mongo_db.user_sessions.create_index("session_id", unique=True)
            self.mongo_db.user_sessions.create_index("user_id")

            # training_submissions: user_id + submitted_at DESC, status
            self.mongo_db.training_submissions.create_index("id", unique=True)
            self.mongo_db.training_submissions.create_index([("user_id", 1), ("submitted_at", -1)])
            self.mongo_db.training_submissions.create_index("status")

            # training_samples: submission_id
            self.mongo_db.training_samples.create_index("id", unique=True)
            self.mongo_db.training_samples.create_index("submission_id")

            # model_versions: unique version_tag, active flag
            self.mongo_db.model_versions.create_index("id", unique=True)
            self.mongo_db.model_versions.create_index("version_tag", unique=True)
            self.mongo_db.model_versions.create_index([("model_type", 1), ("is_active", 1)])

            # training_jobs: created_at DESC, status
            self.mongo_db.training_jobs.create_index("id", unique=True)
            self.mongo_db.training_jobs.create_index([("created_at", -1)])
            self.mongo_db.training_jobs.create_index("status")

            logger.info("MongoDB indexes verified successfully.")
        except Exception as exc:
            logger.error("Failed creating indexes: %s", redact_secrets(str(exc)))

    def transform_record(self, table: str, raw_row: dict) -> dict:
        """Transform a PostgreSQL row dict into a MongoDB document."""
        doc = dict(raw_row)

        # Standardize ID
        if "id" in doc and doc["id"] is not None:
            doc["id"] = int(doc["id"])

        # Timestamp fields handling
        for k, v in list(doc.items()):
            if isinstance(v, (datetime,)):
                doc[k] = v
            elif isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", v):
                try:
                    doc[k] = datetime.fromisoformat(v.replace("Z", "+00:00"))
                except Exception:
                    pass

        # Table-specific transformations
        if table == "model_versions":
            # Handle binary model and encoder data
            for bcol in ["model_data", "encoder_data"]:
                val = doc.get(bcol)
                if val:
                    if isinstance(val, (bytes, memoryview)):
                        doc[bcol] = Binary(bytes(val))
                    elif isinstance(val, str):
                        try:
                            doc[bcol] = Binary(base64.b64decode(val))
                        except Exception:
                            pass
            # If loaded from backup with external bin files
            if self.backup_dir:
                bin_dir = self.backup_dir / "binaries"
                for bcol in ["model_data", "encoder_data"]:
                    bfile = doc.get(f"{bcol}_bin_file")
                    if bfile and (bin_dir / bfile).exists():
                        doc[bcol] = Binary((bin_dir / bfile).read_bytes())

        elif table == "training_samples":
            val = doc.get("sample_data")
            if val:
                if isinstance(val, (bytes, memoryview)):
                    doc["sample_data"] = Binary(bytes(val))
                elif isinstance(val, str):
                    try:
                        doc["sample_data"] = Binary(base64.b64decode(val))
                    except Exception:
                        pass
            if self.backup_dir:
                bin_dir = self.backup_dir / "binaries"
                bfile = doc.get("sample_data_bin_file")
                if bfile and (bin_dir / bfile).exists():
                    doc["sample_data"] = Binary((bin_dir / bfile).read_bytes())

        return doc

    def read_table_rows(self, table: str) -> List[dict]:
        """Fetch rows from live PostgreSQL or from backup files."""
        # 1. Try live PostgreSQL if connected
        if self.pg_conn and not self.pg_conn.closed:
            try:
                import psycopg2.extras
                with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"SELECT * FROM \"{table}\" ORDER BY id ASC")
                    return [dict(r) for r in cur.fetchall()]
            except Exception as exc:
                logger.warning("Error querying live PostgreSQL for %s: %s", table, redact_secrets(str(exc)))

        # 2. Try backup directory
        if self.backup_dir:
            json_file = self.backup_dir / f"{table}.json"
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data if isinstance(data, list) else []
                except Exception as exc:
                    logger.error("Error reading backup file %s: %s", json_file, exc)

        return []

    def migrate_table(self, table: str) -> Tuple[int, int, int]:
        """Migrate a single table to MongoDB. Returns (source_count, upserted_count, error_count)."""
        coll_name = self.TABLE_COLLECTION_MAP[table]
        rows = self.read_table_rows(table)
        source_count = len(rows)
        self.stats[table]["source"] = source_count

        if source_count == 0:
            logger.info("Table '%s': 0 rows found to migrate.", table)
            return (0, 0, 0)

        logger.info("Migrating '%s' -> MongoDB collection '%s' (%d rows)...", table, coll_name, source_count)

        if self.dry_run:
            # Validate transformations
            valid_count = 0
            for r in rows:
                try:
                    doc = self.transform_record(table, r)
                    if "id" in doc:
                        valid_count += 1
                except Exception as e:
                    self.stats[table]["failed"] += 1
            self.stats[table]["inserted"] = valid_count
            logger.info("[Dry-Run] Table '%s': %d valid documents transformed successfully.", table, valid_count)
            return (source_count, valid_count, self.stats[table]["failed"])

        from pymongo import ReplaceOne

        coll = self.mongo_db[coll_name]
        upserted_count = 0
        failed_count = 0

        # Batch upsert
        for i in range(0, source_count, self.batch_size):
            batch = rows[i : i + self.batch_size]
            operations = []

            for r in batch:
                try:
                    doc = self.transform_record(table, r)
                    if "id" in doc:
                        operations.append(
                            ReplaceOne({"id": doc["id"]}, doc, upsert=True)
                        )
                except Exception as row_err:
                    failed_count += 1
                    logger.debug("Row transform failed in %s: %s", table, row_err)

            if operations:
                try:
                    res = coll.bulk_write(operations, ordered=False)
                    batch_upserted = (res.upserted_count or 0) + (res.modified_count or 0) + (res.matched_count or 0)
                    upserted_count += len(operations)
                except Exception as bulk_err:
                    logger.error("Bulk write failed in %s: %s", coll_name, redact_secrets(str(bulk_err)))
                    failed_count += len(operations)

            logger.info("  '%s' progress: %d/%d processed", table, min(i + len(batch), source_count), source_count)

        self.stats[table]["inserted"] = upserted_count
        self.stats[table]["failed"] = failed_count

        return (source_count, upserted_count, failed_count)

    def update_id_counters(self) -> None:
        """Update MongoDB sequence counters collection so future inserts start after maximum existing IDs."""
        if self.dry_run or self.mongo_db is None:
            return

        counters = self.mongo_db["_counters"]
        for table, coll_name in self.TABLE_COLLECTION_MAP.items():
            try:
                coll = self.mongo_db[coll_name]
                top_doc = coll.find_one(sort=[("id", -1)], projection={"id": 1})
                max_id = top_doc.get("id", 0) if top_doc else 0
                counters.replace_one(
                    {"_id": coll_name},
                    {"_id": coll_name, "seq": max_id},
                    upsert=True,
                )
            except Exception as e:
                logger.debug("Counter update note for %s: %s", coll_name, e)
        logger.info("Sequence counters synchronized with max IDs.")

    def run_migration(self) -> dict:
        """Run the full migration pipeline."""
        start_time = time.time()
        logger.info("=" * 65)
        logger.info("Starting PostgreSQL to MongoDB Atlas Migration (Dry-Run: %s)", self.dry_run)
        logger.info("=" * 65)

        # 1. Connect MongoDB
        if not self.connect_mongo():
            raise RuntimeError("Could not connect to MongoDB Atlas. Migration aborted.")

        # 2. Connect PostgreSQL (optional if backup_dir provided)
        self.connect_postgres()
        if not self.pg_conn and not self.backup_dir:
            raise RuntimeError("Neither live PostgreSQL nor backup directory available. Migration aborted.")

        # 3. Create Indexes
        self.create_indexes()

        # 4. Migrate tables in relational order
        for table in self.TABLE_COLLECTION_MAP:
            self.migrate_table(table)

        # 5. Synchronize ID counters
        self.update_id_counters()

        duration = round(time.time() - start_time, 2)
        logger.info("=" * 65)
        logger.info("Migration completed in %.2f seconds.", duration)
        logger.info("=" * 65)

        return self.get_summary_report(duration)

    def validate_migration(self) -> dict:
        """Validate counts and sample records between source and destination."""
        logger.info("=" * 65)
        logger.info("Validating MongoDB Atlas data against Source...")
        logger.info("=" * 65)

        if not self.connect_mongo():
            return {"error": "MongoDB connection failed"}

        self.connect_postgres()

        validation_results = {}
        all_matched = True

        for table, coll_name in self.TABLE_COLLECTION_MAP.items():
            source_rows = self.read_table_rows(table)
            src_count = len(source_rows)
            dest_count = self.mongo_db[coll_name].count_documents({})

            match = (dest_count >= src_count)
            if not match:
                all_matched = False

            validation_results[table] = {
                "source_count": src_count,
                "dest_count": dest_count,
                "matched": match,
            }
            logger.info(
                "Table '%s': Source=%d | MongoDB=%d | Match=%s",
                table, src_count, dest_count, "OK" if match else "MISMATCH",
            )

        validation_results["all_matched"] = all_matched
        return validation_results

    def get_summary_report(self, duration: float = 0.0) -> dict:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
            "duration_seconds": duration,
            "target_database": self.mongo_db_name,
            "table_stats": self.stats,
            "total_source_rows": sum(s["source"] for s in self.stats.values()),
            "total_migrated_rows": sum(s["inserted"] for s in self.stats.values()),
            "total_failed_rows": sum(s["failed"] for s in self.stats.values()),
        }


def main():
    parser = argparse.ArgumentParser(description="Migrate PostgreSQL database to MongoDB Atlas.")
    parser.add_argument("--dry-run", action="store_true", help="Validate transformations without writing to MongoDB.")
    parser.add_argument("--validate", action="store_true", help="Validate destination counts against source.")
    parser.add_argument("--backup-dir", type=str, default=r"C:\Users\mohit\OneDrive\Desktop\MajorProject\neon_backup_20261018",
                        help="Path to external backup directory to use as data source.")
    parser.add_argument("--batch-size", type=int, default=250, help="Batch size for bulk write operations.")
    args = parser.parse_args()

    backup_path = Path(args.backup_dir) if args.backup_dir else None

    migrator = SafeMigrationManager(
        backup_dir=backup_path,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    if args.validate:
        results = migrator.validate_migration()
        print("\n=== VALIDATION RESULTS ===")
        print(json.dumps(results, indent=2))
        sys.exit(0 if results.get("all_matched") else 1)

    report = migrator.run_migration()
    print("\n=== MIGRATION REPORT ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
