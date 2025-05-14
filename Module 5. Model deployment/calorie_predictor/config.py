import pathlib
import logging

TRAIN_FILE_ID = "1y5SpZ1nufIasYhBuxf1wUyT0-6bUVrOh"
TEST_FILE_ID = "1aygopxpXublz-3-nCAoLvr6ZfT4MPPGJ"

# --- PATHS ---
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "calorie_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
OUTPUT_PREDICTIONS_PATH = ARTIFACTS_DIR / "batch_predictions.csv"

# ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# --- MODEL CONFIG ---
TARGET_COLUMN = "Calories"
FEATURES = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
CATEGORICAL_FEATURES = ["Sex"]
NUMERICAL_FEATURES = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

# --- LOGGING ---
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

def get_logger(name):
    return logging.getLogger(name)

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DOCKER_IMAGE_NAME = "calorie_api_service"
API_DOCKER_IMAGE_TAG = "latest"