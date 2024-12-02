import os
import logging
from global_variables import LANDING_DIR, LUKES_DATASET_NAME, LUKES_DATASET_REF


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)

def is_landing_dir_exists() -> bool:
    if not LANDING_DIR:
        raise ValueError("🚨'LANDING_DIR' is empty. HINT: Check `.env` file in the working directory.")
    
    if os.path.isdir(LANDING_DIR):
        logging.info(f"✅ Landing directory found: {LANDING_DIR}")
        return True
    else:
        logging.info(f"❌ Landing directory not found: {LANDING_DIR}")
        return False
    
def is_file_exists() -> bool:
    if not LUKES_DATASET_NAME:
        raise ValueError("🚨'LUKES_DATASET_NAME' is empty. HINT: Check `.env` file in the working directory.")
    
    if os.path.isfile( os.path.join(LANDING_DIR, LUKES_DATASET_NAME)):
        logging.info(f"📄 File exists: {LUKES_DATASET_NAME}")
        return True
    else:
        logging.info(f"❌ File not found: {LUKES_DATASET_NAME}")
        return False

def ingest():
    import subprocess
    
    dataset_ref = LUKES_DATASET_REF
    download_path = LANDING_DIR

    logging.info(f"⚡ Starting dataset download: {dataset_ref}")
    os.makedirs(download_path, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_ref, "-p", download_path, "--unzip"])
    logging.info(f"✅ Dataset downloaded and saved in: {download_path}")
    
if __name__ == "__main__":
    if not is_landing_dir_exists():
        logging.info(f"🛠️ Creating directory: {LANDING_DIR}")
        os.mkdir(LANDING_DIR)
        logging.info(f"✅ Directory created: {LANDING_DIR}")
    if not is_file_exists():
        logging.info(f"⚡ File '{LUKES_DATASET_NAME}' not found. Triggering ingestion...")
        ingest()
    else:
        logging.info(f"🎉 File '{LUKES_DATASET_NAME}' already exists. No need to re-ingest.")
