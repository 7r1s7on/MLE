import pandas as pd
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

from .config import (
    TRAIN_FILE_ID, TEST_FILE_ID, DATA_DIR, PREPROCESSOR_PATH,
    TARGET_COLUMN, FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, get_logger
)
from .utils import ensure_directories_exist

logger = get_logger(__name__)

def download_data():
    """Downloads data from Google Drive."""
    ensure_directories_exist()
    train_csv_path = DATA_DIR / "train.csv"
    test_csv_path = DATA_DIR / "test.csv"

    if not train_csv_path.exists():
        logger.info(f"Downloading train.csv to {train_csv_path}...")
        gdown.download(id=TRAIN_FILE_ID, output=str(train_csv_path), quiet=False)
        logger.info("train.csv downloaded.")
    else:
        logger.info("train.csv already exists.")

    if not test_csv_path.exists():
        logger.info(f"Downloading test.csv to {test_csv_path}...")
        gdown.download(id=TEST_FILE_ID, output=str(test_csv_path), quiet=False)
        logger.info("test.csv downloaded.")
    else:
        logger.info("test.csv already exists.")
    return train_csv_path, test_csv_path

def load_data():
    """Loads data from CSV files."""
    train_csv_path, test_csv_path = download_data()
    logger.info("Loading data...")
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path) # for batch prediction example
    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape (for batch prediction): {test_df.shape}")
    return train_df, test_df

def get_preprocessor():
    """Creates and returns a scikit-learn ColumnTransformer for preprocessing."""
    logger.info("Creating preprocessor...")
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # changed sparse to sparse_output

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough' # keep other columns if any, though we select FEATURES
    )
    logger.info("Preprocessor created.")
    return preprocessor

def preprocess_data_and_fit_preprocessor(df: pd.DataFrame):
    """
    Fits the preprocessor on the training data and transforms it.
    Saves the fitted preprocessor.
    """
    logger.info("Preprocessing data and fitting preprocessor...")
    X = df[FEATURES]
    y = df[TARGET_COLUMN]

    preprocessor = get_preprocessor()
    X_processed = preprocessor.fit_transform(X)
    
    logger.info(f"Saving preprocessor to {PREPROCESSOR_PATH}")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info("Preprocessor saved.")
    
    return X_processed, y, preprocessor

def load_preprocessor():
    """Loads the saved preprocessor."""
    if not PREPROCESSOR_PATH.exists():
        logger.error(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train model first.")
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    logger.info(f"Loading preprocessor from {PREPROCESSOR_PATH}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    logger.info("Preprocessor loaded.")
    return preprocessor

def transform_data(df: pd.DataFrame, preprocessor):
    """Transforms data using a loaded preprocessor."""
    logger.info("Transforming data with loaded preprocessor...")
    X = df[FEATURES]
    X_processed = preprocessor.transform(X)
    logger.info("Data transformed.")
    return X_processed

if __name__ == '__main__':
    # example usage:
    logger.info("Running data_processing.py standalone for testing.")
    train_df, _ = load_data()
    X_processed, y, fitted_preprocessor = preprocess_data_and_fit_preprocessor(train_df)
    logger.info(f"Processed X shape: {X_processed.shape}")
    logger.info(f"y shape: {y.shape}")

    # test loading preprocessor
    loaded_preprocessor = load_preprocessor()
    test_sample = train_df.sample(5)[FEATURES]
    transformed_sample = transform_data(test_sample, loaded_preprocessor)
    logger.info(f"Transformed sample shape: {transformed_sample.shape}")
    logger.info("Data processing module test completed.")
