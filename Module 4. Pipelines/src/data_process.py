import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Ensure constants can be imported
# try:
from .constants import PROCESSED_DATA_PATH
# except ImportError:
#     from constants import PROCESSED_DATA_PATH


class StudentPerfomanceDataPrep:
    def __init__(self, scaler=None, test_size=0.2, random_state=42):
        self.scaler = scaler # Scaler object, e.g., StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        # Internal state variables - these will be managed by task wrappers in Airflow
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names_after_split = None # To store column names after get_dummies

    def merge_external_source(self, source_path1, source_path2):
        source_df = pd.read_csv(source_path1, index_col=0)
        ext_source_df = pd.read_csv(source_path2, index_col=0)
        self.data = pd.concat([source_df, ext_source_df], axis=1)
        print("Data merged with an external source successfully.")
        print(f"Merged data columns: {self.data.columns.tolist()}")
        return self.data # Return data for stateless operation

    def clean_data(self, input_data=None):
        if input_data is not None:
            self.data = input_data.copy() # Work on a copy if input_data is provided
        
        if self.data is None:
            raise ValueError("Data not loaded. Call merge_external_source or provide data.")

        categorical_col = self.data.select_dtypes(include=["object"]).columns.tolist()
        numerical_col = self.data.select_dtypes(include=["number"]).columns.tolist()

        # Handle NaNs in categorical columns (dropna might be too aggressive, consider mode imputation)
        # Original: self.data[categorical_col] = self.data[categorical_col].dropna()
        # This drops rows if ANY categorical column has NaN. Better to impute or be specific.
        # For now, let's fill with 'Unknown' or mode. Let's use mode.
        for col in categorical_col:
            if self.data[col].isnull().any():
                mode_val = self.data[col].mode()
                if not mode_val.empty:
                    self.data[col] = self.data[col].fillna(mode_val[0])
                else: # All values are NaN or column is empty
                    self.data[col] = self.data[col].fillna("Unknown")


        # Handle NaNs in numerical columns: fill with median of each column
        for col in numerical_col:
            if self.data[col].isnull().any():
                median_val = self.data[col].median()
                self.data[col] = self.data[col].fillna(median_val)
        
        print("Data cleaned successfully.")
        return self.data # Return data

    def split_data(self, input_data=None):
        if input_data is not None:
            self.data = input_data.copy()

        if self.data is None:
            raise ValueError("Data not loaded/cleaned. Provide data to split.")

        categorical_cols_to_encode = self.data.select_dtypes(include=["object"]).columns.tolist()
        temp_df = pd.get_dummies(self.data, columns=categorical_cols_to_encode, dummy_na=False) # dummy_na=False is default

        if "G3" not in temp_df.columns:
            raise ValueError("Target column 'G3' not found in the data after pre-processing.")
            
        X_df = temp_df.drop("G3", axis=1)
        y_series = temp_df["G3"]
        self.feature_names_after_split = X_df.columns.tolist() # Save feature names

        # Convert to numpy arrays for scikit-learn compatibility if needed, but keeping as DF/Series is fine
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_df, y_series, test_size=self.test_size, random_state=self.random_state
        )
        print("Data split successfully.")
        # Return all components for stateless operation
        return self.X_train, self.X_test, self.y_train, self.y_test, self.feature_names_after_split

    def scale_data(self, X_train_df=None, X_test_df=None, feature_names=None):
        if self.scaler is None:
            raise ValueError("Scaler not provided in constructor.")
        if X_train_df is None or X_test_df is None: # Use internal state if not provided
            X_train_df = self.X_train
            X_test_df = self.X_test
        if feature_names is None:
            feature_names = self.feature_names_after_split
            if feature_names is None and isinstance(X_train_df, pd.DataFrame):
                feature_names = X_train_df.columns.tolist()


        if X_train_df is None or X_test_df is None:
            raise ValueError("X_train or X_test data not available. Run split_data or provide them.")

        # Fit scaler on X_train and transform both X_train and X_test
        X_train_scaled_array = self.scaler.fit_transform(X_train_df)
        X_test_scaled_array = self.scaler.transform(X_test_df)

        # Get feature names for the scaled data
        # If scaler has get_feature_names_out and original feature names were passed to it
        try:
            scaled_feature_names = self.scaler.get_feature_names_out(input_features=feature_names)
        except Exception:
            # Fallback if get_feature_names_out is not available or fails
            if feature_names:
                scaled_feature_names = feature_names
            else: # If input was numpy array without feature names
                scaled_feature_names = [f"feature_{i}" for i in range(X_train_scaled_array.shape[1])]


        self.X_train = pd.DataFrame(X_train_scaled_array, columns=scaled_feature_names, index=X_train_df.index)
        self.X_test = pd.DataFrame(X_test_scaled_array, columns=scaled_feature_names, index=X_test_df.index)
        
        # y_train and y_test are not changed by scaling, assumed to be already set or passed if needed
        # self.y_train = pd.Series(self.y_train) # Ensure Series type if they were numpy arrays
        # self.y_test = pd.Series(self.y_test)

        print("Data scaled successfully.")
        return self.X_train, self.X_test, self.scaler # Return scaled data and fitted scaler

    def save_clean_data(self, X_train_df, X_test_df, y_train_series, y_test_series, processed_path_config):
        if not os.path.exists(PROCESSED_DATA_PATH): # Global PROCESSED_DATA_PATH
            os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        
        X_train_df.to_csv(processed_path_config["X_train_csv"], index=False)
        X_test_df.to_csv(processed_path_config["X_test_csv"], index=False)
        y_train_series.to_csv(processed_path_config["y_train_csv"], index=False, header=True) # Series to CSV
        y_test_series.to_csv(processed_path_config["y_test_csv"], index=False, header=True) # Series to CSV
        print("Final data saved successfully to CSV files.")
