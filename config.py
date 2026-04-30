import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.resolve()

# Dataset paths - Using the organized dataset structure
DATA_PATHS = {
    "node_1": BASE_DIR / "dataset" / "organized" / "device_1_danmini",
    "node_2": BASE_DIR / "dataset" / "organized" / "device_2_ecobee",
    "node_3": BASE_DIR / "dataset" / "organized" / "device_3_ennio",
    "test": BASE_DIR / "dataset" / "raw" / "unsw_nb15"
}

# Federated Learning constants
NUM_NODES = 3
FL_ROUNDS = 10

# Data split configurations
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Seed for reproducibility
RANDOM_SEED = 42

# Output paths
MODEL_SAVE_PATH = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results"
LOG_PATH = BASE_DIR / "logs"
