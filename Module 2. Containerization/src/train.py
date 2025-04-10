import os
import logging
import joblib
import numpy as np
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier
from sklearn.metrics import roc_auc_score, classification_report
from data_process import load_data, preprocess_data, prepare_datasets, download_data

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
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    
    # console handler
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

def train_model():
    try:
        # paths
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
        MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'tabnet_model.pkl')
        SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
        DATA_PATH = os.path.join(BASE_DIR, 'data')

        # create data directory if it doesn't exist
        os.makedirs(DATA_PATH, exist_ok=True)
        
        data_sources = [
            ("https://drive.google.com/uc?export=download&id=1v9uryJuwquza5r3XOMVVYameUPpXhsPQ", "train.csv"),
            ("https://drive.google.com/uc?export=download&id=16s0enePorjXtJOfXsJVVXcYpqUcln_6D", "test.csv")
        ]
        
        logger.info("Downloading data")
        for url, filename in data_sources:
            filepath = os.path.join(DATA_PATH, filename)
            download_data(url, filepath) 

        # load and preprocess data
        logger.info("Loading and preprocessing data")
        df = load_data(TRAIN_PATH)
        df = preprocess_data(df)
        X_train, X_test, y_train, y_test, sample_weights, scaler = prepare_datasets(df)
        
        # train model
        logger.info("Training TabNetMultiTaskClassifier")
        clf = TabNetMultiTaskClassifier()
        clf.fit(
            X_train, y_train.to_numpy(),
            eval_set=[(X_test, y_test.to_numpy())],
            eval_metric=['logloss'],
            max_epochs=50, patience=10,
            batch_size=128, virtual_batch_size=64,
            weights=sample_weights
        )
        
        # evaluate model
        logger.info("Evaluating model")
        y_probabilities = clf.predict_proba(X_test)
        y_pred_proba = np.column_stack([p[:, 1] for p in y_probabilities])
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        for i in range(y_test.shape[1]):
            logger.info(f"\nDefect {i} Report:")
            logger.info(f"\n{classification_report(y_test.iloc[:, i], y_pred_binary[:, i])}")
        
        logger.info(f"\nMicro-average ROC-AUC: {roc_auc_score(y_test, y_pred_proba, average='micro')}")
        
        # save model and scaler
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        joblib.dump(clf, MODEL_SAVE_PATH)
        joblib.dump(scaler, SCALER_SAVE_PATH)
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")
        logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")
        
        return clf, scaler
    
    except Exception as e:
        logger.error(f"Error in training: {e}")
        raise

if __name__ == "__main__":
    train_model()
