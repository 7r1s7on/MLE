# Calorie Predictor Deployment Assignment

This project demonstrates deploying a machine learning model for calorie prediction in two modes:
1.  **Online Mode**: As a REST API using Flask and Gunicorn.
2.  **Batch Mode**: As a scheduled pipeline (simulated with a script, intended for cron).

The batch pipeline trains the model, saves artifacts, makes batch predictions, and finally builds a Docker image for the REST API service. Both modes use the same model and preprocessor artifacts.

## Prerequisites

*   Docker and Docker Compose (optional, but helpful)
*   Python 3.8+
*   Access to Google Drive for dataset download

## Project Structure
```
calorie_predictor_project/
├── calorie_predictor/ # Main package
├── tests/ # Unit/Integration tests
├── artifacts/ # Stores model, preprocessor, batch predictions (generated)
├── data/ # Stores downloaded CSVs (generated)
├── Dockerfile.api # For the REST API service
├── Dockerfile.dev_batch # For development and running the batch pipeline
├── requirements.txt
├── setup.py
├── cron_job.sh # Example script to trigger batch pipeline
└── README.md
```

## Setup & Usage

**1. Build the Development/Batch Docker Image:**

This image is used for running the batch pipeline, tests, and general development. It includes Docker CLI to build the API image.

```bash
docker build -t calorie_dev_batch:latest -f Dockerfile.dev_batch .
```

**2. Run the Batch Pipeline:**

The batch pipeline will:
a. Download data (if not present).
b. Train the model and save artifacts (model.joblib, preprocessor.joblib) to ./artifacts/.
c. Make batch predictions on test.csv and save them to ./artifacts/batch_predictions.csv.
d. Build the Docker image for the REST API (calorie_api_service:latest).

Execute this from the project root:
```bash
# Ensure artifacts and data directories exist on host for volume mounting
mkdir -p ./artifacts ./data

docker run --rm \
    -v "$(pwd)/artifacts:/app/artifacts" \
    -v "$(pwd)/data:/app/data" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "$(pwd):/app" \
    --workdir /app \
    calorie_dev_batch:latest \
    python -m calorie_predictor.batch_pipeline
```

This command mounts the current directory's artifacts and data folders into the container, mounts the Docker socket, and then runs the batch pipeline. The API Docker image (calorie_api_service:latest) will be built and available on your host.

**3. Run the REST API Service (Online Mode):**

After the batch pipeline has run and built calorie_api_service:latest:
```bash
docker run -d -p 8000:8000 --name calorie_api calorie_api_service:latest
```

The API will be available at http://localhost:8000.

Endpoints:

* GET /health: Health check.
* POST /predict: Make predictions. Send JSON payload:
    * Single prediction:
    ```json
    {
    "Sex": "male",
    "Age": 30,
    "Height": 180.0,
    "Weight": 75.0,
    "Duration": 60.0,
    "Heart_Rate": 150.0,
    "Body_Temp": 39.0
    }
    ```
    * Batch prediction (list of records):
    ```json
    [
        { "Sex": "male", "Age": 30, ... },
        { "Sex": "female", "Age": 25, ... }
    ]
    ```
**4. Run Tests:**

Tests can be run inside the `calorie_dev_batch` container:
```bash
docker run --rm \
    -v "$(pwd)/artifacts:/app/artifacts" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd):/app" \
    --workdir /app \
    calorie_dev_batch:latest \
    python -m unittest tests.test_model_pipeline
```

**5. Scheduling Batch Pipeline (Cron):**

The `cron_job.sh` script provides an example of how to trigger the batch pipeline using the `calorie_dev_batch` docker container.
To set up a real cron job:
1. Make cron_job.sh executable: chmod +x cron_job.sh
2. Edit your crontab: crontab -e
3. Add a line like this (e.g., to run daily at 2 AM):
    ```bash
    0 2 * * * /full/path/to/calorie_predictor_project/cron_job.sh >> /full/path/to/calorie_predictor_project/cron_log.log 2>&1
    ```
    *Replace /full/path/to/ with the actual absolute path to your project.*

## Notes
* **Shared Artifacts**: The artifacts/ directory is key. The batch pipeline produces model.joblib and preprocessor.joblib here. The Dockerfile.api copies these artifacts into the API image.
* **Simplicity**: The model is a RandomForestRegressor with basic parameters. Preprocessing uses StandardScaler and OneHotEncoder.
* **Logging**: Basic logging is implemented across modules. Docker container logs can be viewed with docker logs <container_name_or_id>.
* **Response Time**: The API loads the model at startup. Predictions should be fast. The RandomForest model chosen is generally quick for inference.
* **Error Handling**: Basic error handling is in the API.

---

**Final Steps & How to Run:**

1.  **Save all these files** into the specified project structure.
2.  **Build the `dev_batch` Docker image**:
    ```bash
    cd /path/to/calorie_predictor_project
    docker build -t calorie_dev_batch:latest -f dockerfile.dev_batch .
    ```
3.  **Run the Batch Pipeline**:
    ```bash
    # From calorie_predictor_project directory
    mkdir -p ./artifacts ./data # ensure these exist on host
    docker run --rm \
        -v "$(pwd)/artifacts:/app/artifacts" \
        -v "$(pwd)/data:/app/data" \
        -v "/var/run/docker.sock:/var/run/docker.sock" \
        -v "$(pwd):/app" \
        --workdir /app \
        calorie_dev_batch \
        python -m calorie_predictor.batch_pipeline
    ```
    This will:
    *   Download `train.csv` and `test.csv` to `./data/`.
    *   Train the model, save `model.joblib` and `preprocessor.joblib` to `./artifacts/`.
    *   Create `batch_predictions.csv` in `./artifacts/`.
    *   Build a Docker image named `calorie_api_service`.

4.  **Run the API Service**:
    ```bash
    docker run -d -p 8000:8000 --name calorie_api calorie_api_service
    ```
5.  **Test the API**:
    *   Health: `curl http://localhost:8000/health`
    *   Predict:
        ```bash
        curl -X POST -H "Content-Type: application/json" -d '{
            "Sex": "female",
            "Age": 28,
            "Height": 165.0,
            "Weight": 58.0,
            "Duration": 40.0,
            "Heart_Rate": 125.0,
            "Body_Temp": 38.2
        }' http://localhost:8000/predict
        ```
        Expected output (will vary slightly based on model randomness, but structure should be similar):
        `{"calories_prediction":SOME_NUMBER,"response_time_ms":SOME_TIME}`

6.  **Run Tests**:
    ```bash
    docker run --rm \
        -v "$(pwd)/artifacts:/app/artifacts" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd):/app" \
        --workdir /app \
        calorie_dev_batch \
        python -m unittest tests.test_model_pipeline
    ```

7.  **Simulate Cron Job**:
    Make `cron_job.sh` executable (`chmod +x cron_job.sh`) and then run it:
    ```bash
    ./cron_job.sh
    ```
    This will again run the batch pipeline using the Docker container. Check `batch_pipeline_run.log`.