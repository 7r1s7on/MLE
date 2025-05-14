import pandas as pd
import subprocess

from .config import (
    ARTIFACTS_DIR, OUTPUT_PREDICTIONS_PATH, BASE_DIR,
    API_DOCKER_IMAGE_NAME, API_DOCKER_IMAGE_TAG, get_logger
)
from .model import train_model, load_model_and_preprocessor, predict
from .data_processing import load_data # to get test_df for batch prediction
from .utils import ensure_directories_exist

logger = get_logger(__name__)

def run_batch_predictions():
    """
    Loads the test data, makes predictions using the trained model,
    and saves the predictions.
    """
    ensure_directories_exist()
    logger.info("Starting batch prediction process...")
    
    # load the model and preprocessor
    try:
        model, preprocessor = load_model_and_preprocessor()
    except FileNotFoundError:
        logger.error("Model/Preprocessor not found. Please run training first (e.g., by calling train_model()).")
        logger.info("Attempting to train model now...")
        train_model() # train if not found
        model, preprocessor = load_model_and_preprocessor()


    # load test data
    _, test_df = load_data() # download_data is called within load_data

    if test_df.empty:
        logger.warning("Test data is empty. No batch predictions to make.")
        return

    logger.info(f"Making batch predictions on {len(test_df)} records...")
    # the 'predict' function expects a DataFrame with the feature columns
    # if 'test_df' has an 'id' column and no 'Calories' column, it's fine for prediction input
    predictions = predict(test_df, model, preprocessor)
    
    # add predictions to the test_df or create a new DataFrame
    # it's good practice to include original identifiers if available
    output_df = test_df.copy()
    if 'id' in output_df.columns:
         output_df = output_df[['id']] # keep only id for output clarity
    else: # if no id, create a placeholder
        output_df = pd.DataFrame(index=test_df.index)

    output_df['predicted_calories'] = predictions
    
    logger.info(f"Saving batch predictions to {OUTPUT_PREDICTIONS_PATH}")
    output_df.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    logger.info("Batch predictions saved.")
    return OUTPUT_PREDICTIONS_PATH

def build_api_docker_image():
    """
    Builds the Docker image for the REST API service.
    """
    logger.info("Starting Docker image build for API service...")
    image_full_name = f"{API_DOCKER_IMAGE_NAME}:{API_DOCKER_IMAGE_TAG}"
    
    # dockerfile.api should be in the project root (BASE_DIR)
    dockerfile_path = BASE_DIR / "dockerfile.api"
    if not dockerfile_path.exists():
        logger.error(f"dockerfile.api not found at {dockerfile_path}")
        return False

    command = [
        "docker", "build",
        "-t", image_full_name,
        "-f", str(dockerfile_path),
        str(BASE_DIR) # context is the project root
    ]
    
    logger.info(f"Executing Docker build command: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line.strip()) # print docker build output in real-time
        process.wait()

        if process.returncode == 0:
            logger.info(f"Docker image {image_full_name} built successfully.")
            return True
        else:
            logger.error(f"Docker image build failed with return code {process.returncode}.")
            return False
    except Exception as e:
        logger.error(f"Exception during Docker image build: {e}", exc_info=True)
        return False

def main_batch_pipeline():
    logger.info("===== STARTING BATCH PIPELINE =====")
    
    # step 1: ensure artifacts exist by training the model (if not already done)
    # train_model() will download data, preprocess, train, and save artifacts.
    logger.info("--- Step 1: Training model and preparing artifacts ---")
    train_model() 
    logger.info("--- Model training and artifact preparation complete ---")

    # step 2: run batch predictions
    logger.info("--- Step 2: Running batch predictions ---")
    predictions_file = run_batch_predictions()
    if predictions_file:
        logger.info(f"--- Batch predictions complete. Output: {predictions_file} ---")
    else:
        logger.warning("--- Batch predictions did not produce an output file. ---")

    # step 3: build docker image for online prediction API (last task)
    logger.info("--- Step 3: Building Docker image for API service ---")
    build_success = build_api_docker_image()
    if build_success:
        logger.info("--- Docker image for API service built successfully. ---")
    else:
        logger.error("--- Failed to build Docker image for API service. ---")
        
    logger.info("===== BATCH PIPELINE FINISHED =====")

if __name__ == '__main__':
    main_batch_pipeline()
