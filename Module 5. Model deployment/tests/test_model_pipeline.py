import unittest
import pandas as pd
import os
from calorie_predictor.config import ARTIFACTS_DIR, MODEL_PATH, PREPROCESSOR_PATH, FEATURES, DATA_DIR, OUTPUT_PREDICTIONS_PATH
from calorie_predictor.model import train_model, load_model_and_preprocessor, predict
from calorie_predictor.data_processing import download_data, load_data
from calorie_predictor.batch_pipeline import run_batch_predictions


class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure data is downloaded once for all tests."""
        print("Setting up test class: Downloading data...")
        download_data() # ensures data is present in DATA_DIR
        print("Data download attempt complete.")

    def test_01_train_and_artifacts_exist(self):
        """Test if training runs and artifacts are created."""
        print("Running test_01_train_and_artifacts_exist...")
        train_model() # this also saves artifacts
        self.assertTrue(MODEL_PATH.exists(), "Model file not found after training.")
        self.assertTrue(PREPROCESSOR_PATH.exists(), "Preprocessor file not found after training.")
        print("test_01_train_and_artifacts_exist: PASSED")

    def test_02_load_model_and_predict(self):
        """Test loading the model and making a prediction."""
        print("Running test_02_load_model_and_predict...")
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            print("Model/Preprocessor not found, attempting to train first for this test...")
            train_model() # ensure artifacts are there if test is run standalone

        model, preprocessor = load_model_and_preprocessor()
        self.assertIsNotNone(model, "Loaded model is None.")
        self.assertIsNotNone(preprocessor, "Loaded preprocessor is None.")

        # create a dummy input DataFrame
        sample_data = {
            "Sex": ["male"], "Age": [30], "Height": [180.0], "Weight": [75.0],
            "Duration": [60.0], "Heart_Rate": [150.0], "Body_Temp": [39.0]
        }
        # ensure all features are in sample_data, fill with defaults if not
        for f in FEATURES:
            if f not in sample_data:
                if f == "Sex": sample_data[f] = ["unknown"] # or a common category
                else: sample_data[f] = [0]


        input_df = pd.DataFrame(sample_data)
        
        predictions = predict(input_df, model, preprocessor)
        self.assertIsInstance(predictions, list, "Predictions should be a list.")
        self.assertEqual(len(predictions), 1, "Should predict for one sample.")
        self.assertGreaterEqual(predictions[0], 0, "Prediction should be non-negative.")
        print(f"test_02_load_model_and_predict: PASSED (Prediction: {predictions[0]})")

    def test_03_batch_prediction_output(self):
        """Test if batch prediction runs and creates an output file."""
        print("Running test_03_batch_prediction_output...")
        if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
            print("Model/Preprocessor not found, attempting to train first for this test...")
            train_model() # ensure artifacts for batch prediction

        # clean up previous batch prediction file if it exists
        if OUTPUT_PREDICTIONS_PATH.exists():
            os.remove(OUTPUT_PREDICTIONS_PATH)
        
        output_file = run_batch_predictions() # this loads test data internally
        
        self.assertIsNotNone(output_file, "Batch prediction did not return an output file path.")
        self.assertTrue(os.path.exists(output_file), "Batch prediction output file not found.")
        
        # check if the output file has content
        output_df = pd.read_csv(output_file)
        self.assertFalse(output_df.empty, "Batch prediction output CSV is empty.")
        self.assertIn("predicted_calories", output_df.columns, "'predicted_calories' column missing in batch output.")
        print(f"test_03_batch_prediction_output: PASSED (Output file: {output_file})")


if __name__ == '__main__':
    unittest.main()
