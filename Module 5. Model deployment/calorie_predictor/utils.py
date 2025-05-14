import numpy as np
from sklearn.metrics import mean_squared_log_error
from .config import get_logger

logger = get_logger(__name__)

def root_mean_squared_log_error(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    Handles negative predictions by clipping them to a small positive value.
    """
    y_pred_clipped = np.maximum(1e-6, y_pred) # Clip to avoid log(0) or log(negative)
    y_true_clipped = np.maximum(1e-6, y_true)
    return np.sqrt(mean_squared_log_error(y_true_clipped, y_pred_clipped))

def ensure_directories_exist():
    from .config import DATA_DIR, ARTIFACTS_DIR
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured data directory exists: {DATA_DIR}")
    logger.info(f"Ensured artifacts directory exists: {ARTIFACTS_DIR}")
