import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import logging
from data_loader import load_data

import warnings
warnings.filterwarnings('ignore')

# Defining paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, 'training.log')

# Setting basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='w'
)

# Loading dataset
try:
    logging.info("Loading dataset...")
    df = load_data()
    logging.info(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    logging.error(f"Failed to load data: {e}")
    raise

def split_and_scale(df):
    """Function to split and scale dataset"""
    logging.info("Splitting data into train/test sets and scaling...")
    try:
        X = df.drop(columns='target')
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        logging.info(f"Data split complete. Train samples: {len(X_train)}, Test samples: {len(X_test)}")

        scaler = MinMaxScaler()
        # Fit on training data only
        X_train_scaled = scaler.fit_transform(X_train)
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        logging.info("Data scaling complete using MinMaxScaler.")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    except Exception as e:
        logging.error(f"Error during split and scale: {e}")
        raise

def train_model(X_train, X_test, y_train, y_test):
    """Function to train the random forests model"""
    logging.info("Starting RandomForest model training with GridSearchCV...")
    try:
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_params = {
            'max_depth': [2, 3, 4],
            'min_samples_leaf': [3, 4, 5]
            }

        rf_clf = GridSearchCV(
            estimator=rf,
            param_grid=rf_params,
            n_jobs=-1,
            cv=5,
            refit=True,
            verbose=1
            )

        rf_clf.fit(X_train, y_train) # Train on scaled data
        logging.info(f"GridSearchCV completed. Best parameters found: {rf_clf.best_params_}")
        logging.info(f"Best score achieved during CV: {rf_clf.best_score_:.4f}")

        y_pred_rf = rf_clf.predict(X_test) # Predict on scaled test data
        report = classification_report(y_test, y_pred_rf)
        logging.info(f"Classification report on test set:\n{report}")
        print("--- Classification Report on Test Set ---\n") # Keep print for immediate console view
        print(report)
        print("-----------------------------------------")
        return rf_clf
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def main():
    logging.info("Starting training script execution...")
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'rf_model.joblib')
    SCALER_SAVE_PATH = os.path.join(MODELS_DIR, 'minmax_scaler.joblib')

    # Ensure the models directory exists
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        logging.info(f"Ensured models directory exists: {MODELS_DIR}")
    except OSError as e:
        logging.error(f"Could not create models directory {MODELS_DIR}: {e}")
        return # Exit if we can't create the directory

    try:
        # Split data and get the fitted scaler
        X_train_scaled, X_test_scaled, y_train, y_test, fitted_scaler = split_and_scale(df)

        # Train the model
        rf_gridsearch_model = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

        # --- Save the best model ---
        best_rf_model = rf_gridsearch_model.best_estimator_
        logging.info(f"Saving the best RandomForest model to: {MODEL_SAVE_PATH}")
        joblib.dump(best_rf_model, MODEL_SAVE_PATH)
        logging.info("Best model saved successfully.")

        # --- Save the fitted scaler ---
        logging.info(f"Saving the fitted scaler to: {SCALER_SAVE_PATH}")
        joblib.dump(fitted_scaler, SCALER_SAVE_PATH)
        logging.info("Scaler saved successfully.")

        logging.info("Training script finished successfully.")

    except Exception as e:
        # Log any exception that occurred during the main process
        logging.error("An error occurred during the main execution:")
        logging.exception(e) # Logs the error message and traceback


if __name__ == "__main__":
    main()