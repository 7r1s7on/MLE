import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(path) # This might be redundant if ROOT_DIR is added and is the same

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data/")
RAW_DATA_DIR = os.path.join(DATA_PATH, "raw/")
# This is the target name for the main dataset after download and potential renaming
MAIN_RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "data.csv") 

# These are the files generated by data_generate.py
SOURCE_PATH = os.path.join(RAW_DATA_DIR, "human_factor_data.csv")
EXTERNAL_SOURCE_PATH = os.path.join(RAW_DATA_DIR, "edu_factor_data.csv")

PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed/")

# Intermediate file paths for DAG tasks
INTERMEDIATE_MERGED_DATA_PKL = os.path.join(PROCESSED_DATA_PATH, "merged_data.pkl")
INTERMEDIATE_CLEANED_DATA_PKL = os.path.join(PROCESSED_DATA_PATH, "cleaned_data.pkl")

# Paths for split data components (before scaling)
# Using .pkl for DataFrames/Series to preserve types and column names easily
SPLIT_X_TRAIN_PKL = os.path.join(PROCESSED_DATA_PATH, "temp_X_train.pkl")
SPLIT_X_TEST_PKL = os.path.join(PROCESSED_DATA_PATH, "temp_X_test.pkl")
SPLIT_Y_TRAIN_PKL = os.path.join(PROCESSED_DATA_PATH, "temp_y_train.pkl")
SPLIT_Y_TEST_PKL = os.path.join(PROCESSED_DATA_PATH, "temp_y_test.pkl")
SPLIT_FEATURE_NAMES_JSON = os.path.join(PROCESSED_DATA_PATH, "temp_feature_names.json")

# Paths for scaled data components
SCALED_X_TRAIN_PKL = os.path.join(PROCESSED_DATA_PATH, "scaled_X_train.pkl")
SCALED_X_TEST_PKL = os.path.join(PROCESSED_DATA_PATH, "scaled_X_test.pkl")
SCALER_JOBLIB = os.path.join(PROCESSED_DATA_PATH, "scaler.joblib")

# Final output CSV paths (as in original save_clean_data)
FINAL_X_TRAIN_CSV = os.path.join(PROCESSED_DATA_PATH, "X_train.csv")
FINAL_X_TEST_CSV = os.path.join(PROCESSED_DATA_PATH, "X_test.csv")
FINAL_Y_TRAIN_CSV = os.path.join(PROCESSED_DATA_PATH, "y_train.csv")
FINAL_Y_TEST_CSV = os.path.join(PROCESSED_DATA_PATH, "y_test.csv")
