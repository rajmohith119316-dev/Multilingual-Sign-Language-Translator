import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.auth_models import UserManager
from src.retrain_dynamic import train_new_gesture

def sync_active_models_from_db():
    """Download the active models from the database if they exist."""
    print("Checking cloud database for active models...")
    mgr = UserManager()
    versions = mgr.get_model_versions()
    
    for v in versions:
        if v.get("is_active"):
            mtype = v["model_type"]
            mdata = v.get("model_data")
            edata = v.get("encoder_data")
            
            if not mdata:
                continue
                
            model_path = config.ALPHABET_MODEL_PATH if mtype == "Alphabet" else config.WORD_MODEL_PATH
            encoder_path = config.ALPHABET_ENCODER_PATH if mtype == "Alphabet" else config.WORD_ENCODER_PATH
            
            # Save the model
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(mdata)
            
            # Save the encoder if present
            if edata:
                with open(encoder_path, 'wb') as f:
                    f.write(edata)
            
            print(f"Downloaded and synced active {mtype} model from cloud (Version: {v['version_tag']}).")


def run_cloud_retrain(job_id: int):
    """
    Background job to:
    1. Fetch approved samples from DB
    2. Write them to temp disk
    3. Retrain
    4. Upload the resulting model to DB
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
        print(f"Error updating job status: {e}")
        return

    try:
        submissions = mgr.get_all_submissions(status="APPROVED")
        if not submissions:
            print("No approved submissions to train on.")
            raise Exception("No approved submissions")
            
        # For simplicity, we just group them by type and use retrain_dynamic or move them to processed
        # Here we will write the binary sample data to the correct processed directory
        for sub in submissions:
            label = sub["gesture_label"].upper()
            gtype = sub["gesture_type"]
            
            target_dir = config.PROCESSED_ALPHABETS_DIR if gtype == "Alphabet" else config.PROCESSED_WORDS_DIR
            label_dir = target_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            
            samples = mgr.get_submission_samples(sub["id"])
            for idx, s in enumerate(samples):
                data = s.get("sample_data")
                if data:
                    file_path = label_dir / f"sub_{sub['id']}_s_{idx}.npy"
                    with open(file_path, 'wb') as f:
                        f.write(data)

        # Run the full training script
        # train_moe.py trains BOTH alphabet and word models
        from src.train_moe import train_alphabet_expert, train_word_expert
        train_alphabet_expert()
        train_word_expert()
        
        # After training, the models are at ALPHABET_MODEL_PATH and WORD_MODEL_PATH
        
        # Upload Alphabet Model to DB
        if config.ALPHABET_MODEL_PATH.exists():
            with open(config.ALPHABET_MODEL_PATH, 'rb') as f:
                alpha_mdata = f.read()
            alpha_edata = None
            if config.ALPHABET_ENCODER_PATH.exists():
                with open(config.ALPHABET_ENCODER_PATH, 'rb') as f:
                    alpha_edata = f.read()
            
            vtag_alpha = f"Alphabet_Cloud_{job_id}"
            sql_insert_model = """
                INSERT INTO model_versions (version_tag, model_type, model_path, model_data, encoder_data, is_active)
                VALUES (%s, %s, %s, %s, %s, TRUE)
            """
            with mgr._connect() as conn:
                with conn.cursor() as cur:
                    # Deactivate old ones
                    cur.execute("UPDATE model_versions SET is_active = FALSE WHERE model_type = 'Alphabet'")
                    cur.execute(sql_insert_model, (vtag_alpha, "Alphabet", str(config.ALPHABET_MODEL_PATH), alpha_mdata, alpha_edata))
                conn.commit()

        # Upload Word Model to DB
        if config.WORD_MODEL_PATH.exists():
            with open(config.WORD_MODEL_PATH, 'rb') as f:
                word_mdata = f.read()
            word_edata = None
            if config.WORD_ENCODER_PATH.exists():
                with open(config.WORD_ENCODER_PATH, 'rb') as f:
                    word_edata = f.read()
            
            vtag_word = f"Word_Cloud_{job_id}"
            sql_insert_model = """
                INSERT INTO model_versions (version_tag, model_type, model_path, model_data, encoder_data, is_active)
                VALUES (%s, %s, %s, %s, %s, TRUE)
            """
            with mgr._connect() as conn:
                with conn.cursor() as cur:
                    # Deactivate old ones
                    cur.execute("UPDATE model_versions SET is_active = FALSE WHERE model_type = 'Word'")
                    cur.execute(sql_insert_model, (vtag_word, "Word", str(config.WORD_MODEL_PATH), word_mdata, word_edata))
                conn.commit()

        # Update job as completed
        sql_completed = "UPDATE training_jobs SET status = 'COMPLETED', completed_at = NOW() WHERE id = %s"
        with mgr._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_completed, (job_id,))
            conn.commit()
            
        print("Cloud retrain completed successfully!")
        
        try:
            import web_app
            if web_app.predictor:
                web_app.predictor.hot_reload()
                print("MoE Predictor hot reloaded with new models!")
        except Exception as e:
            print(f"Failed to hot reload predictor: {e}")

    except Exception as e:
        print(f"Cloud retrain failed: {e}")
        sql_fail = "UPDATE training_jobs SET status = 'FAILED', completed_at = NOW() WHERE id = %s"
        with mgr._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_fail, (job_id,))
            conn.commit()

def start_cloud_retrain_thread(job_id: int):
    t = threading.Thread(target=run_cloud_retrain, args=(job_id,))
    t.start()
