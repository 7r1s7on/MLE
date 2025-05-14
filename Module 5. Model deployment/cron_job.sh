#!/bin/bash

# This script is an example of how you might trigger the batch pipeline.
# For a real cron job, you'd add a line to your crontab.
# Example crontab entry (runs daily at 2 AM):
# 0 2 * * * /path/to/your/calorie_predictor_project/cron_job.sh >> /path/to/your/calorie_predictor_project/cron_log.log 2>&1

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${PROJECT_DIR}/batch_pipeline_run.log"
PYTHON_EXECUTABLE="python" # rr specify full path, e.g., /usr/bin/python3 or venv python

echo "===== Cron Job Triggered: $(date) ====="

# ensure artifacts and data directories exist on host for volume mounting
mkdir -p "${PROJECT_DIR}/artifacts"
mkdir -p "${PROJECT_DIR}/data"

echo "Running batch pipeline inside Docker container..."
docker run --rm \
    -v "${PROJECT_DIR}/artifacts:/app/artifacts" \
    -v "${PROJECT_DIR}/data:/app/data" \
    -v "/var/run/docker.sock:/var/run/docker.sock" \
    -v "${PROJECT_DIR}:/app" \
    --workdir /app \
    calorie_dev_batch:latest \
    python -m calorie_predictor.batch_pipeline # assuming dev_batch image is built

echo "Batch pipeline execution attempt finished: $(date)"
echo "Check ${LOG_FILE} and Docker logs for details."