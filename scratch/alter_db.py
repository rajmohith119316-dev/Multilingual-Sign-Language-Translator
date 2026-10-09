import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("No DATABASE_URL found.")
    exit(1)

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("ALTER TABLE training_samples ADD COLUMN IF NOT EXISTS sample_data BYTEA;")
    cur.execute("ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS model_data BYTEA;")
    cur.execute("ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS encoder_data BYTEA;")
    conn.commit()
    print("Successfully added BYTEA columns to database.")
except Exception as e:
    print("Error:", e)
finally:
    if 'conn' in locals():
        conn.close()
