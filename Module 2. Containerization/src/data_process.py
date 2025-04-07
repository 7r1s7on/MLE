import os
import logging
import numpy as np
import pandas as pd
import gdown
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

# ensure logs directory exists
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../logs')
os.makedirs(log_dir, exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'data_processing.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def download_data(url, download_path):
    gdown.download(url, download_path, quiet=False)

def load_data(data_path):
    """Load and preprocess the data."""
    try:
        logger.info(f"Attempting to load data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(df):
    """Preprocess the data."""
    try:
        logger.info("Starting data preprocessing")
        
        if 'id' in df.columns:
            df.drop(columns='id', inplace=True)
            logger.info("Dropped ID column")
        
        # convert target columns to int
        target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 
                      'Dirtiness', 'Bumps', 'Other_Faults']
        for col in target_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
                logger.debug(f"Converted {col} to int")
        
        logger.info("Data preprocessing completed")
        return df
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        raise

def prepare_datasets(df, test_size=0.2, random_state=42):
    """Split data into features and targets, then train-test split."""
    try:
        logger.info("Preparing datasets")
        
        target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 
                      'Dirtiness', 'Bumps', 'Other_Faults']
        
        X = df.drop(target_cols, axis=1)
        y = df[target_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train/test split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        
        # handle class imbalance
        sample_weights = np.zeros(len(y_train))
        for i in range(y_train.shape[1]):
            sample_weights += compute_sample_weight('balanced', y_train.iloc[:,i])
        sample_weights /= y_train.shape[1]
        logger.info("Sample weights calculated")
        
        # scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, sample_weights, scaler
    except Exception as e:
        logger.error(f"Error preparing datasets: {str(e)}")
        raise

def main():
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_PATH = os.path.join(BASE_DIR, 'data')

        # Create data directory if it doesn't exist
        os.makedirs(DATA_PATH, exist_ok=True)
        
        data_sources = [
            ("https://drive.google.com/uc?export=download&id=1v9uryJuwquza5r3XOMVVYameUPpXhsPQ", "train.csv"),
            ("https://drive.google.com/uc?export=download&id=16s0enePorjXtJOfXsJVVXcYpqUcln_6D", "test.csv")
        ]
        
        for url, filename in data_sources:
            filepath = os.path.join(DATA_PATH, filename)
            download_data(url, filepath) 

        TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train.csv')
        TEST_DATA_PATH = os.path.join(DATA_PATH, 'test.csv')
        OUTPUT_PATH = os.path.join(DATA_PATH, 'preprocessed_train.csv')

        logger.info("Starting data processing pipeline")
        df = load_data(TRAIN_DATA_PATH)
        df = preprocess_data(df)
        
        # save preprocessed data
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Saved preprocessed data to {OUTPUT_PATH}")
        
        X_train, X_test, y_train, y_test, sample_weights, scaler = prepare_datasets(df)
        logger.info("Data processing completed successfully")
        
        return df, X_train, X_test, y_train, y_test, sample_weights, scaler
    except Exception as e:
        logger.exception("Failed to process data")
        raise

if __name__ == "__main__":
    main()