import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report

def setup_logging():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(BASE_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # file handler
    file_handler = logging.FileHandler(os.path.join(log_dir, 'inference.log'))
    file_handler.setLevel(logging.INFO)
    
    # fonsole handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()


def load_model_and_scaler():
    """Load trained model and scaler."""
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tabnet_model.pkl')
        SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
        
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Contents of /app/models: {os.listdir('/app/models')}")
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model/scaler: {e}")
        raise

def predict_new_data(model, scaler, data_path):
    """Make predictions on new data."""
    try:
        # load and preprocess new data
        df = pd.read_csv(data_path)
        if 'id' in df.columns:
            ids = df['id']
            df.drop(columns='id', inplace=True)
        else:
            ids = None
        
        # scale features
        X_scaled = scaler.transform(df)
        
        # make predictions
        y_probabilities = model.predict_proba(X_scaled)
        y_pred_proba = np.column_stack([p[:, 1] for p in y_probabilities])
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # create output DataFrame
        target_cols = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 
                      'Dirtiness', 'Bumps', 'Other_Faults']
        results = pd.DataFrame(y_pred_binary, columns=target_cols)
        
        if ids is not None:
            results.insert(0, 'id', ids)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

if __name__ == "__main__":
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
        OUTPUT_PATH = os.path.join(BASE_DIR, 'output', 'predictions.csv')
        
        model, scaler = load_model_and_scaler()
        predictions = predict_new_data(model, scaler, TEST_DATA_PATH)
        
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        predictions.to_csv(OUTPUT_PATH, index=False)
        logger.info(f"Predictions saved to {OUTPUT_PATH}")
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
