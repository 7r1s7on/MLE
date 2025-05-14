import pandas as pd
from sklearn.ensemble import RandomForestRegressor # lightweight and effective
import joblib

from .config import MODEL_PATH, PREPROCESSOR_PATH, TARGET_COLUMN, FEATURES, get_logger
from .data_processing import preprocess_data_and_fit_preprocessor, load_preprocessor, transform_data, load_data
from .utils import root_mean_squared_log_error, ensure_directories_exist

logger = get_logger(__name__)

def train_model():
    """Trains the model and saves it along with the preprocessor."""
    ensure_directories_exist()
    logger.info("Starting model training process...")
    train_df, _ = load_data() # we only need train_df for training
    
    X_processed, y, _ = preprocess_data_and_fit_preprocessor(train_df) # preprocessor is saved here

    logger.info("Training RandomForestRegressor model...")
    # using a simple RandomForestRegressor, keeping parameters minimal for speed
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10, min_samples_split=10)
    model.fit(X_processed, y)
    logger.info("Model training complete.")

    logger.info(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    logger.info("Model saved.")

    # example evaluation on training data
    predictions = model.predict(X_processed)
    rmsle = root_mean_squared_log_error(y, predictions)
    logger.info(f"RMSLE on training data: {rmsle:.4f}")
    
    return model

def load_model_and_preprocessor():
    """Loads the trained model and preprocessor."""
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Train model first.")
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        logger.error(f"Preprocessor not found at {PREPROCESSOR_PATH}. Train model first.")
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")

    logger.info(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")
    
    preprocessor = load_preprocessor()
    
    return model, preprocessor

def predict(input_data: pd.DataFrame, model=None, preprocessor=None):
    """
    Makes predictions on new data.
    If model and preprocessor are not provided, they are loaded.
    """
    logger.info("Making predictions...")
    if model is None or preprocessor is None:
        logger.info("Model or preprocessor not provided, loading them...")
        model, preprocessor = load_model_and_preprocessor()

    if not isinstance(input_data, pd.DataFrame):
        # assuming input_data is a dictionary or list of dictionaries for API
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data

    # ensure all required features are present
    missing_cols = [col for col in FEATURES if col not in input_df.columns]
    if missing_cols:
        logger.error(f"Missing columns in input data: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
        
    X_to_predict = input_df[FEATURES]
    X_processed = transform_data(X_to_predict, preprocessor)
    
    predictions = model.predict(X_processed)
    # ensure predictions are non-negative as calories cannot be negative
    predictions_clipped = [max(0, p) for p in predictions]
    logger.info(f"Predictions made. Example: {predictions_clipped[:3]}")
    return predictions_clipped


if __name__ == '__main__':
    # example usage:
    logger.info("Running model.py standalone for testing.")
    
    # 1. train model (this also downloads data and saves preprocessor)
    train_model()
    
    # 2. load model and preprocessor
    model, preprocessor = load_model_and_preprocessor()
    
    # 3. make a test prediction
    # create a sample dataFrame similar to what the API might receive or batch might process
    sample_data_dict = {
        "Sex": ["male", "female"],
        "Age": [25, 30],
        "Height": [175.0, 160.0],
        "Weight": [70.0, 60.0],
        "Duration": [30.0, 45.0],
        "Heart_Rate": [120.0, 130.0],
        "Body_Temp": [38.0, 38.5]
    }
    sample_df = pd.DataFrame(sample_data_dict)
    
    predictions = predict(sample_df, model, preprocessor)
    logger.info(f"Sample predictions: {predictions}")
    logger.info("Model module test completed.")
