import os
import sys
import json
import logging
from datetime import datetime, timedelta

import pandas as pd # For read_pickle
from joblib import dump, load # For scaler
from sklearn.preprocessing import StandardScaler

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago # Useful for dynamic start_dates if not fixed

# --- Path Setup ---
# Assuming DAG file is in /opt/airflow/dags and src is in /opt/airflow/src
DAG_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(DAG_DIR, "..")) # /opt/airflow
sys.path.append(ROOT_DIR)

from src.constants import (
    MAIN_RAW_DATA_FILE, SOURCE_PATH, EXTERNAL_SOURCE_PATH, PROCESSED_DATA_PATH,
    INTERMEDIATE_MERGED_DATA_PKL, INTERMEDIATE_CLEANED_DATA_PKL,
    SPLIT_X_TRAIN_PKL, SPLIT_X_TEST_PKL, SPLIT_Y_TRAIN_PKL, SPLIT_Y_TEST_PKL, SPLIT_FEATURE_NAMES_JSON,
    SCALED_X_TRAIN_PKL, SCALED_X_TEST_PKL, SCALER_JOBLIB,
    FINAL_X_TRAIN_CSV, FINAL_X_TEST_CSV, FINAL_Y_TRAIN_CSV, FINAL_Y_TEST_CSV
)
from src.data_generate import generate_data
from src.download_data import download_student_data_from_kaggle
from src.data_process import StudentPerfomanceDataPrep

# --- Configuration ---
with open(os.path.join(DAG_DIR, "config.json"), "r") as f:
    config = json.load(f)

raw_dag_args = config["student_perfomance_dag"]["default_args"]
processing_params = config["student_perfomance_dag"]["processing_params"]

default_args = {
    "owner": raw_dag_args["owner"],
    "depends_on_past": raw_dag_args["depends_on_past"],
    "email": raw_dag_args.get("email"), # Use .get for optional fields
    "email_on_failure": raw_dag_args.get("email_on_failure", False),
    "email_on_retry": raw_dag_args.get("email_on_retry", False),
    "retries": raw_dag_args["retries"],
    "retry_delay": timedelta(minutes=raw_dag_args["retry_delay_minutes"]),
    "retry_exponential_backoff": raw_dag_args.get("retry_exponential_backoff", False),
    "max_retry_delay": timedelta(hours=raw_dag_args["max_retry_delay_hours"]),
    "start_date": datetime.strptime(raw_dag_args["start_date_str"], "%Y-%m-%d"),
}

# --- Task Functions (Wrappers for statelessness) ---

def download_and_prepare_raw_data_fn():
    logging.info("Starting data download from Kaggle...")
    download_student_data_from_kaggle() # Downloads and renames to MAIN_RAW_DATA_FILE
    logging.info(f"Data downloaded. Preparing (splitting) {MAIN_RAW_DATA_FILE}...")
    if not os.path.exists(MAIN_RAW_DATA_FILE):
        raise FileNotFoundError(f"Main raw data file {MAIN_RAW_DATA_FILE} not found after download.")
    generate_data(MAIN_RAW_DATA_FILE) # Splits data.csv into human_factor and edu_factor csvs
    logging.info("Raw data downloaded and prepared into factor files.")

def merge_external_source_fn():
    # This task uses the human_factor_data.csv and edu_factor_data.csv
    # No specific scaler or test_size/random_state needed for StudentPerfomanceDataPrep here.
    stpd_instance = StudentPerfomanceDataPrep()
    merged_data = stpd_instance.merge_external_source(SOURCE_PATH, EXTERNAL_SOURCE_PATH)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    merged_data.to_pickle(INTERMEDIATE_MERGED_DATA_PKL)
    logging.info(f"Merged data saved to {INTERMEDIATE_MERGED_DATA_PKL}")

def clean_data_fn():
    merged_data = pd.read_pickle(INTERMEDIATE_MERGED_DATA_PKL)
    stpd_instance = StudentPerfomanceDataPrep() # Default params are fine
    cleaned_data = stpd_instance.clean_data(input_data=merged_data)
    cleaned_data.to_pickle(INTERMEDIATE_CLEANED_DATA_PKL)
    logging.info(f"Cleaned data saved to {INTERMEDIATE_CLEANED_DATA_PKL}")

def split_data_fn():
    cleaned_data = pd.read_pickle(INTERMEDIATE_CLEANED_DATA_PKL)
    stpd_instance = StudentPerfomanceDataPrep(
        test_size=processing_params["test_size"],
        random_state=processing_params["random_state"]
    )
    X_train, X_test, y_train, y_test, feature_names = stpd_instance.split_data(input_data=cleaned_data)
    
    X_train.to_pickle(SPLIT_X_TRAIN_PKL)
    X_test.to_pickle(SPLIT_X_TEST_PKL)
    y_train.to_pickle(SPLIT_Y_TRAIN_PKL)
    y_test.to_pickle(SPLIT_Y_TEST_PKL)
    with open(SPLIT_FEATURE_NAMES_JSON, 'w') as f:
        json.dump(feature_names, f)
    logging.info("Data split and intermediate files saved.")

def scale_data_fn():
    X_train = pd.read_pickle(SPLIT_X_TRAIN_PKL)
    X_test = pd.read_pickle(SPLIT_X_TEST_PKL)
    with open(SPLIT_FEATURE_NAMES_JSON, 'r') as f:
        feature_names = json.load(f)

    # Initialize scaler here, as it's specific to this task's logic
    scaler = StandardScaler()
    stpd_instance = StudentPerfomanceDataPrep(scaler=scaler) # Pass the scaler instance

    X_train_scaled, X_test_scaled, fitted_scaler = stpd_instance.scale_data(
        X_train_df=X_train, X_test_df=X_test, feature_names=feature_names
    )
    
    X_train_scaled.to_pickle(SCALED_X_TRAIN_PKL)
    X_test_scaled.to_pickle(SCALED_X_TEST_PKL)
    dump(fitted_scaler, SCALER_JOBLIB) # Save the fitted scaler
    logging.info("Data scaled and scaled components saved.")

def save_final_data_fn():
    X_train_scaled = pd.read_pickle(SCALED_X_TRAIN_PKL)
    X_test_scaled = pd.read_pickle(SCALED_X_TEST_PKL)
    y_train = pd.read_pickle(SPLIT_Y_TRAIN_PKL) # y values are not scaled
    y_test = pd.read_pickle(SPLIT_Y_TEST_PKL)   # y values are not scaled

    # For save_clean_data, it doesn't need an stpd_instance if paths are passed.
    # Let's make a dummy call or directly save.
    # The StudentPerfomanceDataPrep.save_clean_data can be used if it's adapted,
    # or just save directly here.
    
    # Path configuration for the save_clean_data method
    path_config = {
        "X_train_csv": FINAL_X_TRAIN_CSV,
        "X_test_csv": FINAL_X_TEST_CSV,
        "y_train_csv": FINAL_Y_TRAIN_CSV,
        "y_test_csv": FINAL_Y_TEST_CSV,
    }
    
    stpd_instance = StudentPerfomanceDataPrep() # Not strictly needed if save_clean_data is static-like
    stpd_instance.save_clean_data(X_train_scaled, X_test_scaled, y_train, y_test, path_config)
    logging.info("Final processed data saved to CSV files.")

# --- DAG Definition ---
with DAG(
    "Student_Performance_DAG",
    default_args=default_args,
    description="DAG to download, process, and prepare student performance data",
    schedule_interval=timedelta(days=1), # Changed from 5 minutes
    catchup=False, # Typically good for production DAGs unless backfills are intended
    tags=['student_data', 'ml_pipeline'],
) as dag:

    download_and_prepare_task = PythonOperator(
        task_id="download_and_prepare_raw_data",
        python_callable=download_and_prepare_raw_data_fn,
    )

    merge_external_source_task = PythonOperator(
        task_id="merge_external_source_data",
        python_callable=merge_external_source_fn,
    )

    clean_data_task = PythonOperator(
        task_id="clean_merged_data",
        python_callable=clean_data_fn,
    )

    split_data_task = PythonOperator(
        task_id="split_data_into_train_test",
        python_callable=split_data_fn,
    )

    scale_data_task = PythonOperator(
        task_id="scale_features",
        python_callable=scale_data_fn,
    )

    save_data_task = PythonOperator(
        task_id="save_processed_data",
        python_callable=save_final_data_fn,
    )

    # --- Task Dependencies ---
    (
        download_and_prepare_task 
        >> merge_external_source_task 
        >> clean_data_task 
        >> split_data_task 
        >> scale_data_task 
        >> save_data_task
    )

logging.info("Student_Performance_DAG created successfully.")
