"""
check_db.py — Diagnostic utility to verify MongoDB Atlas connection & environment setup.
Run this on any machine to pinpoint why the database or metrics are not showing:
    python check_db.py
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent

def run_diagnostics():
    print("=" * 65)
    print("[DIAGNOSTIC] MULTILINGUAL SIGN LANGUAGE TRANSLATOR -- DB CHECK")
    print("=" * 65)

    # 1. Check .env file existence
    env_path = ROOT / ".env"
    env_txt_path = ROOT / ".env.txt"

    if env_txt_path.exists() and not env_path.exists():
        print("[!] CRITICAL ERROR: Found '.env.txt' instead of '.env'!")
        print("    Windows hid the file extension. Rename '.env.txt' to '.env'.")
        return

    if not env_path.exists():
        print(f"[!] CRITICAL ERROR: '.env' file not found at: {env_path}")
        print("    Please place the .env file in the project root directory.")
        return

    print("[OK] Found '.env' file in project root.")

    # 2. Test dotenv loading
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)

    backend = os.getenv("DATABASE_BACKEND", "mongodb")
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_db_name = os.getenv("MONGODB_DATABASE", "sign_language_db")

    print(f"[INFO] DATABASE_BACKEND: {backend}")
    print(f"[INFO] MONGODB_DATABASE: {mongo_db_name}")

    if not mongo_uri:
        print("[!] CRITICAL ERROR: 'MONGODB_URI' is missing or empty inside .env!")
        return
    print(f"[OK] 'MONGODB_URI' is present: {mongo_uri[:25]}...***")

    # 3. Check Python dependencies
    missing_deps = []
    try:
        import pymongo
        print(f"[OK] pymongo installed (v{pymongo.__version__})")
    except ImportError:
        missing_deps.append("pymongo")

    try:
        import dns.resolver
        print("[OK] dnspython installed")
    except ImportError:
        missing_deps.append("dnspython")

    try:
        import certifi
        print(f"[OK] certifi installed (v{certifi.__version__})")
    except ImportError:
        missing_deps.append("certifi")

    if missing_deps:
        print(f"[!] MISSING PACKAGES: {', '.join(missing_deps)}")
        print(f"    Run: pip install {' '.join(missing_deps)}")
        return

    # 4. Check DNS Resolution
    print("\nTesting DNS resolution for MongoDB Atlas cluster...")
    try:
        res = dns.resolver.get_default_resolver()
        for ns in ["8.8.8.8", "1.1.1.1"]:
            if ns not in res.nameservers:
                res.nameservers.append(ns)
        res.timeout = 3.0
        res.lifetime = 5.0
        # Resolve SRV
        answers = res.resolve("_mongodb._tcp.cluster0.tttcyvz.mongodb.net", "SRV")
        print(f"[OK] DNS SRV resolved successfully ({len(answers)} shard targets found).")
    except Exception as dns_err:
        print(f"[WARN] DNS resolution warning: {dns_err}")
        print("       If your ISP / local router blocks SRV lookups, ensure your DNS is set to 8.8.8.8.")

    # 5. Test MongoDB Atlas Connection & Ping
    print("\nTesting connection to MongoDB Atlas...")
    try:
        from pymongo import MongoClient

        mongo_kwargs = {
            "serverSelectionTimeoutMS": 8000,
            "connectTimeoutMS": 8000,
            "socketTimeoutMS": 10000,
            "retryWrites": True,
            "tlsCAFile": certifi.where(),
        }
        client = MongoClient(mongo_uri, **mongo_kwargs)
        res = client.admin.command("ping")
        print(f"[OK] PING SUCCESSFUL! Atlas responded: {res}")

        # 6. Check Document Counts
        db = client[mongo_db_name]
        cols = db.list_collection_names()
        print(f"\nConnected to database: '{mongo_db_name}' ({len(cols)} collections)")
        for col_name in sorted(cols):
            count = db[col_name].count_documents({})
            print(f"   * {col_name:<24}: {count} records")

        print("\n[SUCCESS] ALL TESTS PASSED! Your system is fully connected to MongoDB Atlas.")
        print("          When you run 'python main.py', the Admin Dashboard will display all metrics.")

    except Exception as exc:
        err_str = str(exc)
        print(f"\n[!] CONNECTION FAILED: {err_str}\n")

        if "TLSV1_ALERT_INTERNAL_ERROR" in err_str or "SSL handshake failed" in err_str:
            print("[CAUSE] IP ACCESS RESTRICTION OR TLS REJECTION BY ATLAS!")
            print("        MongoDB Atlas terminates the SSL handshake if your IP address is not whitelisted.")
            print("\n        >>> HOW TO FIX IN 1 MINUTE <<<")
            print("        1. Log into your MongoDB Atlas account (https://cloud.mongodb.com).")
            print("        2. In the left sidebar, click 'Network Access' (under Security).")
            print("        3. Click 'Add IP Address'.")
            print("        4. Click 'ALLOW ACCESS FROM ANYWHERE' (0.0.0.0/0) and confirm.")
            print("        5. Wait 60 seconds and run this script again!")
        elif "ServerSelectionTimeoutError" in err_str:
            print("[CAUSE] TIMEOUT CONNECTING TO ATLAS!")
            print("        Your network / firewall / antivirus might be blocking outbound port 27017,")
            print("        or your router cannot resolve Atlas DNS. Try using a mobile hotspot.")
        elif "ConfigurationError" in err_str:
            print("[CAUSE] CONFIGURATION ERROR!")
            print("        Run: pip install dnspython certifi")
        else:
            print(f"[DETAILS] {err_str}")

    print("=" * 65)

if __name__ == "__main__":
    run_diagnostics()
