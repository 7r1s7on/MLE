import os
import logging
import json # For setting up Kaggle credentials via env vars
from zipfile import ZipFile


# Ensure constants can be imported if this script is run directly or as part of a module
# try:
from .constants import RAW_DATA_DIR, MAIN_RAW_DATA_FILE
# except ImportError:
#     # Fallback for direct execution if src is not in PYTHONPATH
#     from constants import RAW_DATA_DIR, MAIN_RAW_DATA_FILE


KAGGLE_DATASET_ID = "devansodariya/student-performance-data"
KAGGLE_FILE_NAME = "student_data.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_kaggle_api():
    """Initializes and returns KaggleApi. Handles credential setup."""
    # Check for KAGGLE_USERNAME and KAGGLE_KEY environment variables
    # This allows passing credentials without mounting kaggle.json, useful for some CI/CD
    kaggle_dir_container = os.path.expanduser("~/.kaggle")
    kaggle_json_path_container = os.path.join(kaggle_dir_container, "kaggle.json")

    if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
        logger.info("Found KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        os.makedirs(kaggle_dir_container, exist_ok=True)
        with open(kaggle_json_path_container, "w") as f:
            json.dump({"username": os.environ["KAGGLE_USERNAME"], "key": os.environ["KAGGLE_KEY"]}, f)
        os.chmod(kaggle_json_path_container, 0o600)
        logger.info(f"Kaggle API credentials written to {kaggle_json_path_container}")
    elif not os.path.exists(kaggle_json_path_container):
        error_msg = (
            f"Kaggle API credentials not found. "
            f"Ensure '{kaggle_json_path_container}' exists (e.g., via Docker volume mount) "
            f"or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    else:
        logger.info(f"Using existing Kaggle credentials at {kaggle_json_path_container}")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

def download_student_data_from_kaggle():
    """
    Downloads the student performance dataset from Kaggle.
    It renames the downloaded file to 'data.csv' in the RAW_DATA_DIR.
    """
    api = _get_kaggle_api()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    logger.info(f"Ensured raw data directory exists: {RAW_DATA_DIR}")

    logger.info(f"Downloading {KAGGLE_FILE_NAME} from dataset {KAGGLE_DATASET_ID} to {RAW_DATA_DIR}...")
    
    api.dataset_download_file(KAGGLE_DATASET_ID, KAGGLE_FILE_NAME, path=RAW_DATA_DIR, force=True, quiet=False)
    
    # The file is downloaded as KAGGLE_FILE_NAME (e.g., student-mat.csv)
    # No zipping for single file downloads usually, but good to check.
    downloaded_file_path = os.path.join(RAW_DATA_DIR, KAGGLE_FILE_NAME)
    downloaded_zip_path = downloaded_file_path + ".zip"

    if os.path.exists(downloaded_zip_path):
        logger.info(f"Unzipping {downloaded_zip_path}...")
        with ZipFile(downloaded_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        os.remove(downloaded_zip_path)
        logger.info(f"Unzipped and removed {downloaded_zip_path}.")
        # Ensure the extracted file path is now set to downloaded_file_path
        if not os.path.exists(downloaded_file_path):
             raise FileNotFoundError(f"File {KAGGLE_FILE_NAME} not found after unzipping {downloaded_zip_path}.")

    if not os.path.exists(downloaded_file_path):
         raise FileNotFoundError(f"Expected file {downloaded_file_path} not found after Kaggle download attempt.")

    # Rename to data.csv as expected by generate_data function
    if os.path.exists(downloaded_file_path):
        os.rename(downloaded_file_path, MAIN_RAW_DATA_FILE)
        logger.info(f"Renamed {downloaded_file_path} to {MAIN_RAW_DATA_FILE}")
    else:
        # This case should ideally be caught above
        raise FileNotFoundError(f"Downloaded file {downloaded_file_path} not found for renaming.")

    logger.info(f"Student data downloaded and prepared as {MAIN_RAW_DATA_FILE} successfully.")

if __name__ == "__main__":
    # Allows testing the download script independently
    # Ensure KAGGLE_USERNAME and KAGGLE_KEY are set as env vars or ~/.kaggle/kaggle.json is present
    logger.info("Running download_student_data_from_kaggle() standalone test...")
    download_student_data_from_kaggle()
