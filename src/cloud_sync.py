"""
src/cloud_sync.py — Model Versioning, Cloud Sync & Background Retraining
Supports MongoDB Atlas (primary) and PostgreSQL (fallback).
Optimized to avoid unnecessary network transfer: polls metadata first and
downloads binaries only when an updated active version tag is detected.
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.auth_models import UserManager

logger = logging.getLogger("cloud_sync")

_SYNC_STATE_FILE = config.MODEL_DIR / "cloud_sync_state.json"
_sync_poller_thread = None
_sync_poller_running = False
_sync_lock = threading.Lock()


def _get_local_sync_versions() -> dict:
    """Read local version tracking dictionary."""
    if _SYNC_STATE_FILE.exists():
        try:
            with open(_SYNC_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not read cloud_sync_state: %s", exc)
            return {}
    return {}


def _save_local_sync_version(model_type: str, version_tag: str) -> None:
    """Record current model version tag locally."""
    try:
        state = _get_local_sync_versions()
        state[model_type] = version_tag
        _SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_SYNC_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as exc:
        logger.warning("Could not persist cloud_sync_state: %s", exc)


def _file_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def upload_model_to_db(
    model_type: str,
    model_path: Path,
    encoder_path: Path,
    version_tag: str,
    accuracy: float = None,
    parameters: dict = None,
    notes: str = None,
    created_by: int = None,
) -> bool:
    """
    Upload trained model and encoder binary data to database model_versions collection/table
    and set as active. Deactivates previous active versions of this model_type.
    """
    if not model_path.exists():
        logger.error("Model path %s does not exist for upload.", model_path)
        return False

    try:
        with open(model_path, "rb") as f:
            model_bytes = f.read()

        encoder_bytes = None
        if encoder_path and encoder_path.exists():
            with open(encoder_path, "rb") as f:
                encoder_bytes = f.read()

        params = dict(parameters or {})
        params["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
        params["model_size_bytes"] = len(model_bytes)
        if encoder_bytes:
            params["encoder_sha256"] = hashlib.sha256(encoder_bytes).hexdigest()
            params["encoder_size_bytes"] = len(encoder_bytes)

        mgr = UserManager()
        success = mgr.save_model_version(
            version_tag=version_tag,
            model_type=model_type,
            model_path=str(model_path),
            accuracy=accuracy,
            parameters=params,
            notes=notes,
            model_data=model_bytes,
            encoder_data=encoder_bytes,
            is_active=True,
            created_by=created_by,
        )

        if success:
            _save_local_sync_version(model_type, version_tag)
            logger.info(
                "Successfully uploaded %s model (version_tag=%s, size=%d bytes) to database.",
                model_type, version_tag, len(model_bytes),
            )
            return True
        else:
            logger.error("Database save_model_version returned False for %s.", version_tag)
            return False
    except Exception as exc:
        logger.error("Failed to upload %s model to database: %s", model_type, exc, exc_info=True)
        return False


def sync_active_models_from_db(predictor=None, force: bool = False) -> dict:
    """
    Check database for active model versions.
    Optimized bandwidth: queries metadata without binary blobs first.
    Only downloads binary data if local version differs or local files are missing.
    Maintains a .bak rollback copy in case download is corrupted.
    """
    with _sync_lock:
        result = {"synced": False, "updated": [], "active_tags": {}}
        try:
            mgr = UserManager()
            # Fetch metadata only to preserve network allowance
            meta_versions = mgr.get_model_versions(include_data=False)
        except Exception as exc:
            logger.warning("Could not query model_versions metadata: %s", exc)
            return result

        local_state = _get_local_sync_versions()
        updated_any = False

        for v in meta_versions:
            if not v.get("is_active"):
                continue

            mtype = v.get("model_type")
            vtag = v.get("version_tag")
            if not mtype or not vtag:
                continue

            result["active_tags"][mtype] = vtag

            target_model_path = config.ALPHABET_MODEL_PATH if mtype == "Alphabet" else config.WORD_MODEL_PATH
            target_encoder_path = config.ALPHABET_ENCODER_PATH if mtype == "Alphabet" else config.WORD_ENCODER_PATH

            local_vtag = local_state.get(mtype)
            files_exist = target_model_path.exists() and target_encoder_path.exists()

            needs_download = force or (local_vtag != vtag) or (not files_exist)

            if not needs_download:
                logger.debug("%s model is already up to date with cloud version %s.", mtype, vtag)
                continue

            logger.info("New or missing active version detected for %s: %s (local=%s). Downloading...",
                        mtype, vtag, local_vtag)

            # Fetch binary data for this specific model type
            try:
                full_v = mgr.get_active_model_version(mtype, include_data=True)
                if not full_v or not full_v.get("model_data"):
                    logger.warning("No binary data available in DB for %s (tag=%s).", mtype, vtag)
                    continue

                mdata = full_v["model_data"]
                edata = full_v.get("encoder_data")

                # Backup existing files for rollback
                bak_model = target_model_path.with_suffix(".h5.bak")
                bak_encoder = target_encoder_path.with_suffix(".pkl.bak")

                if target_model_path.exists():
                    shutil.copy2(target_model_path, bak_model)
                if target_encoder_path.exists():
                    shutil.copy2(target_encoder_path, bak_encoder)

                # Write downloaded bytes
                target_model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_model_path, "wb") as f:
                    f.write(mdata)
                if edata:
                    with open(target_encoder_path, "wb") as f:
                        f.write(edata)

                # Integrity validation: check non-empty
                if target_model_path.stat().st_size == 0:
                    raise IOError(f"Downloaded model file for {mtype} is empty!")

                _save_local_sync_version(mtype, vtag)
                updated_any = True
                result["updated"].append(mtype)
                logger.info("Downloaded and synced %s model (version_tag=%s, size=%d bytes).",
                            mtype, vtag, len(mdata))

            except Exception as dl_err:
                logger.error("Failed downloading/activating %s model: %s. Rolling back.", mtype, dl_err)
                if bak_model.exists():
                    shutil.copy2(bak_model, target_model_path)
                if bak_encoder.exists():
                    shutil.copy2(bak_encoder, target_encoder_path)

        if updated_any:
            result["synced"] = True
            p = predictor
            if p is None:
                try:
                    import web_app
                    p = getattr(web_app, "predictor", None)
                except Exception:
                    p = None

            if p is not None:
                try:
                    p.hot_reload()
                    logger.info("Successfully hot-reloaded predictor with synced models.")
                except Exception as hr_err:
                    logger.error("Predictor hot_reload failed: %s", hr_err)

        return result


def start_cloud_sync_poller(predictor=None, interval_seconds: int = 45):
    """
    Launch a background daemon thread that periodically checks for new active models
    trained by other instances.
    """
    global _sync_poller_thread, _sync_poller_running
    if _sync_poller_running:
        return

    def _poll_worker():
        global _sync_poller_running
        _sync_poller_running = True
        logger.info("Cloud model sync poller started (interval=%ds).", interval_seconds)
        while _sync_poller_running:
            time.sleep(interval_seconds)
            try:
                sync_active_models_from_db(predictor=predictor, force=False)
            except Exception as exc:
                logger.warning("Cloud model sync poll check warning: %s", exc)

    _sync_poller_thread = threading.Thread(target=_poll_worker, daemon=True, name="CloudModelPoller")
    _sync_poller_thread.start()


def run_cloud_retrain(job_id: int):
    """
    Background job to:
    1. Mark training job as RUNNING.
    2. Fetch approved samples from DB.
    3. Write them to processed disk.
    4. Retrain both models.
    5. Upload resulting models to database.
    6. Mark job as COMPLETED.
    """
    mgr = UserManager()
    mgr.update_training_job_status(job_id, status="RUNNING", started=True)

    try:
        submissions = mgr.get_all_submissions(status="APPROVED")
        if not submissions:
            logger.warning("No approved submissions to train on.")
            raise Exception("No approved submissions")

        for sub in submissions:
            label = sub["gesture_label"].upper()
            gtype = sub["gesture_type"]

            target_dir = config.PROCESSED_ALPHABETS_DIR if gtype == "Alphabet" else config.PROCESSED_WORDS_DIR
            label_dir = target_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)

            samples = mgr.get_submission_samples(sub["id"], include_data=True)
            for idx, s in enumerate(samples):
                data = s.get("sample_data")
                if data:
                    file_path = label_dir / f"sub_{sub['id']}_s_{idx}.npy"
                    with open(file_path, "wb") as f:
                        f.write(data)

        # Run training
        from src.train_moe import train_alphabet_expert, train_word_expert
        import joblib

        train_alphabet_expert()
        train_word_expert()

        # Upload Alphabet model
        if config.ALPHABET_MODEL_PATH.exists():
            alpha_classes = []
            if config.ALPHABET_ENCODER_PATH.exists():
                try:
                    le = joblib.load(str(config.ALPHABET_ENCODER_PATH))
                    alpha_classes = list(le.classes_)
                except Exception:
                    pass
            upload_model_to_db(
                model_type="Alphabet",
                model_path=config.ALPHABET_MODEL_PATH,
                encoder_path=config.ALPHABET_ENCODER_PATH,
                version_tag=f"Alphabet_Cloud_{job_id}",
                parameters={"classes": alpha_classes},
                notes=f"Cloud retraining job #{job_id}",
            )

        # Upload Word model
        if config.WORD_MODEL_PATH.exists():
            word_classes = []
            if config.WORD_ENCODER_PATH.exists():
                try:
                    le = joblib.load(str(config.WORD_ENCODER_PATH))
                    word_classes = list(le.classes_)
                except Exception:
                    pass
            upload_model_to_db(
                model_type="Word",
                model_path=config.WORD_MODEL_PATH,
                encoder_path=config.WORD_ENCODER_PATH,
                version_tag=f"Word_Cloud_{job_id}",
                parameters={"classes": word_classes},
                notes=f"Cloud retraining job #{job_id}",
            )

        mgr.update_training_job_status(job_id, status="COMPLETED", completed=True)
        logger.info("Cloud retrain job #%d completed successfully!", job_id)

        try:
            import web_app
            if web_app.predictor:
                web_app.predictor.hot_reload()
                logger.info("MoE Predictor hot reloaded with new models!")
        except Exception as e:
            logger.warning("Failed to hot reload predictor: %s", e)

    except Exception as e:
        logger.error("Cloud retrain failed: %s", e, exc_info=True)
        mgr.update_training_job_status(job_id, status="FAILED", completed=True, error_message=str(e))


def start_cloud_retrain_thread(job_id: int):
    t = threading.Thread(target=run_cloud_retrain, args=(job_id,), daemon=True)
    t.start()
