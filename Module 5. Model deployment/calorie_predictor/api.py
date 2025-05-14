from flask import Flask, request, jsonify
import pandas as pd
import time

from .config import API_HOST, API_PORT, get_logger
from .model import load_model_and_preprocessor, predict

logger = get_logger(__name__)

app = Flask(__name__)

# load model and preprocessor at startup to minimize prediction latency
try:
    model, preprocessor = load_model_and_preprocessor()
    logger.info("Model and preprocessor loaded successfully for API.")
except Exception as e:
    logger.error(f"Failed to load model/preprocessor at startup: {e}")
    model, preprocessor = None, None # allow app to start, but /predict will fail

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running."}), 200

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    start_time = time.time()
    if model is None or preprocessor is None:
        logger.error("Model or preprocessor not loaded. API cannot make predictions.")
        return jsonify({"error": "Model not available. Check server logs."}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # expecting a list of records or a single record
        if isinstance(data, dict): #sSingle record
            input_df = pd.DataFrame([data])
        elif isinstance(data, list): # sist of records
            input_df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Expecting JSON object or list of objects."}), 400

        logger.info(f"Received prediction request for {len(input_df)} records.")
        
        predictions = predict(input_df, model, preprocessor)
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # in milliseconds
        logger.info(f"Prediction successful. Response time: {response_time:.2f} ms")
        
        # if a single record was passed, return a single prediction value
        if isinstance(data, dict) and len(predictions) == 1:
            return jsonify({"calories_prediction": predictions[0], "response_time_ms": response_time})
        else:
            return jsonify({"calories_predictions": predictions, "response_time_ms": response_time})

    except ValueError as ve: # specific error from our predict function (e.g. missing columns)
        logger.warning(f"ValueError during prediction: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during prediction.", "details": str(e)}), 500

if __name__ == '__main__':
    if model is None or preprocessor is None:
        logger.warning("API starting without model/preprocessor. /predict endpoint will not work.")
    logger.info(f"Starting Flask API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)
