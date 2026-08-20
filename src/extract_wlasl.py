"""
src/extract_wlasl.py — Extract WLASL Video Clips from archive (2).zip
Extracts raw mp4 videos for target glosses (words) into data/raw/words/{GLOSS}/.
"""

import json
import logging
import zipfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

ZIP_PATH = config.BASE_DIR / "videos" / "archive (2).zip"
JSON_PATH = config.BASE_DIR / "videos" / "WLASL_v0.3.json"

TARGET_GLOSSES = [
    "book", "drink", "computer", "before", "chair", "go", "clothes", "who",
    "candy", "cousin", "deaf", "fine", "help", "no", "thin", "hello", "thanks", "yes"
]

def extract_wlasl(
    zip_path: Path = ZIP_PATH,
    json_path: Path = JSON_PATH,
    raw_words_dir: Path = config.RAW_WORDS_DIR,
    max_videos_per_gloss: int = 15,
) -> dict[str, int]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON metadata not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    logger.info("Opening archive: %s", zip_path.name)
    zf = zipfile.ZipFile(zip_path, "r")
    available_files = set(zf.namelist())

    extracted_counts: dict[str, int] = {}

    for entry in wlasl_data:
        gloss = entry.get("gloss", "").strip().upper()
        if not gloss:
            continue
        
        # Filter for target glosses if specified
        if TARGET_GLOSSES and gloss.lower() not in TARGET_GLOSSES:
            continue

        instances = entry.get("instances", [])
        gloss_dir = raw_words_dir / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for inst in instances:
            video_id = inst.get("video_id", "")
            zip_member = f"videos/{video_id}.mp4"

            if zip_member in available_files:
                out_path = gloss_dir / f"{video_id}.mp4"
                if not out_path.exists():
                    try:
                        data = zf.read(zip_member)
                        out_path.write_bytes(data)
                        count += 1
                    except Exception as exc:
                        logger.warning("Failed to extract %s: %s", zip_member, exc)
                else:
                    count += 1

                if count >= max_videos_per_gloss:
                    break

        if count > 0:
            extracted_counts[gloss] = count
            logger.info("Extracted %d videos for gloss '%s'", count, gloss)

    zf.close()
    return extracted_counts

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    counts = extract_wlasl()
    total = sum(counts.values())
    print(f"\n[OK] WLASL Extraction Complete: {len(counts)} glosses, {total} videos total.")
