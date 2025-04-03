# --------base stage--------
FROM ubuntu:22.04 AS base
    
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    LOG_DIR=/app/logs \
    VENV_PATH=/opt/venv
    
# install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
    
# create virtual environment
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"
    
# directory structure
RUN mkdir -p ${MODEL_DIR} ${DATA_DIR} ${LOG_DIR} /app/src
WORKDIR /app
    
# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
    
# --------data preprocessing stage--------
FROM base AS preprocessing
        
# copy all source files
COPY src/ /app/src/

# copy csv files
COPY data/ /app/data/
        
# set permissions
RUN chmod -R a+rwx ${LOG_DIR} ${DATA_DIR}
        
VOLUME ${DATA_DIR}
VOLUME ${LOG_DIR}
CMD ["python", "src/data_preprocess.py"]

# --------training stage--------
FROM preprocessing AS trainer

CMD ["python", "src/train.py"]

# --------inference stage--------
FROM ubuntu:22.04 AS inference

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    VENV_PATH=/opt/venv
            
# minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*
            
# copy virtual environment from base
COPY --from=base ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:$PATH"

# copy only necessary files
COPY --from=trainer /app/src/inference.py /app/src/
COPY --from=trainer /app/src/data_process.py /app/src/
COPY --from=trainer ${MODEL_DIR} ${MODEL_DIR}
COPY --from=trainer /app/models /app/models

# create directories
RUN mkdir -p ${DATA_DIR} /app/src
WORKDIR /app
            
VOLUME ${DATA_DIR}

RUN pip uninstall -y numpy && pip install numpy==1.24.4

CMD ["python", "src/inference.py"]