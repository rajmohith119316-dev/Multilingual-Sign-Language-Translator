import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.auth_models import UserManager
import psycopg2.extras

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
        except Exception:
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


def upload_model_to_db(
    model_type: str,
    model_path: Path,
    encoder_path: Path,
    version_tag: str,
    accuracy: float = None,
    parameters: dict = None,
    notes: str = None,
) -> bool:
    """
    Upload trained model and encoder binary data to Neon DB model_versions table and set as active.
    Deactivates any previous active version of this model_type.
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

        mgr = UserManager()
        sql_insert = """
            INSERT INTO model_versions (
                version_tag, model_type, model_path, model_data, encoder_data,
                accuracy, parameters, is_active, notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE, %s)
        """
        params_json = psycopg2.extras.Json(parameters) if parameters is not None else None

        with mgr._connect() as conn:
            with conn.cursor() as cur:
                # Deactivate older active versions of this model type
                cur.execute(
                    "UPDATE model_versions SET is_active = FALSE WHERE model_type = %s",
                    (model_type,),
                )
                cur.execute(
                    sql_insert,
                    (
                        version_tag,
                        model_type,
                        str(model_path),
                        model_bytes,
                        encoder_bytes,
                        accuracy,
                        params_json,
                        notes,
                    ),
                )
            conn.commit()

        # Update local sync state tracker so local doesn't re-download what it just uploaded
        _save_local_sync_version(model_type, version_tag)

        logger.info(
            "Successfully uploaded %s model to Neon DB (version_tag=%s, size=%d bytes)",
            model_type, version_tag, len(model_bytes),
        )
        return True
    except Exception as exc:
        logger.error("Failed to upload %s model to Neon DB: %s", model_type, exc, exc_info=True)
        return False


def sync_active_models_from_db(predictor=None, force: bool = False) -> dict:
    """
    Check Neon Cloud DB for active model versions.
    If a new active model version exists or force=True or local files are missing,
    download the model and encoder, write to disk, update tracking, and hot-reload.
    """
    with _sync_lock:
        result = {"synced": False, "updated": [], "active_tags": {}}
        try:
            mgr = UserManager()
            versions = mgr.get_model_versions(include_data=True)
        except Exception as exc:
            logger.error("Failed to connect or query model_versions from DB: %s", exc)
            return result

        local_state = _get_local_sync_versions()
        updated_any = False

        for v in versions:
            if not v.get("is_active"):
                continue

            mtype = v.get("model_type")
            vtag = v.get("version_tag")
            result["active_tags"][mtype] = vtag

            mdata = v.get("model_data")
            edata = v.get("encoder_data")
            if not mdata:
                continue

            target_model_path = config.ALPHABET_MODEL_PATH if mtype == "Alphabet" else config.WORD_MODEL_PATH
            target_encoder_path = config.ALPHABET_ENCODER_PATH if mtype == "Alphabet" else config.WORD_ENCODER_PATH

            local_vtag = local_state.get(mtype)
            files_exist = target_model_path.exists() and (target_encoder_path.exists() if edata else True)

            needs_download = force or (local_vtag != vtag) or (not files_exist)

            if needs_download:
                target_model_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_model_path, "wb") as f:
                    f.write(mdata)
                if edata:
                    with open(target_encoder_path, "wb") as f:
                        f.write(edata)

                _save_local_sync_version(mtype, vtag)
                updated_any = True
                result["updated"].append(mtype)
                logger.info("Downloaded and synced %s model (version_tag=%s) from Neon DB.", mtype, vtag)
            else:
                logger.debug("%s model is already up to date with cloud version %s.", mtype, vtag)

        if updated_any:
            result["synced"] = True
            # Hot-reload predictor if provided or accessible
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
                    logger.info("Successfully hot-reloaded predictor with synced cloud models.")
                except Exception as hr_err:
                    logger.error("Predictor hot_reload failed: %s", hr_err)

        return result


def start_cloud_sync_poller(predictor=None, interval_seconds: int = 45):
    """
    Launch a background daemon thread that periodically checks Neon DB for new models
    trained by other devices/systems.
    """
    global _sync_poller_thread, _sync_poller_running
    if _sync_poller_running:
        return

    def _poll_worker():
        global _sync_poller_running
        _sync_poller_running = True
        logger.info("Neon cloud model sync poller started (interval=%ds).", interval_seconds)
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
    1. Fetch approved samples from DB
    2. Write them to processed disk
    3. Retrain both models
    4. Upload the resulting models to Neon DB
    """
    mgr = UserManager()

    # Mark job as running
    sql_running = "UPDATE training_jobs SET status = 'RUNNING', started_at = NOW() WHERE id = %s"
    try:
        with mgr._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_running, (job_id,))
            conn.commit()
    except Exception as e:
        logger.error("Error updating job status: %s", e)
        return

    try:
        submissions = mgr.get_all_submissions(status="APPROVED")
        if not submissions:
            logger.warning("No approved submissions to train on.")
            raise Exception("No approved submissions")

        # Group and write sample data to processed directory
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

        # Run the training scripts
        from src.train_moe import train_alphabet_expert, train_word_expert
        import joblib

        train_alphabet_expert()
        train_word_expert()

        # Upload Alphabet Model to DB
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

        # Upload Word Model to DB
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

        # Update job as completed
        sql_completed = "UPDATE training_jobs SET status = 'COMPLETED', completed_at = NOW() WHERE id = %s"
        with mgr._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_completed, (job_id,))
            conn.commit()

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
        sql_fail = "UPDATE training_jobs SET status = 'FAILED', error_message = %s, completed_at = NOW() WHERE id = %s"
        with mgr._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_fail, (str(e), job_id))
            conn.commit()


def start_cloud_retrain_thread(job_id: int):
    t = threading.Thread(target=run_cloud_retrain, args=(job_id,), daemon=True)
    t.start()
