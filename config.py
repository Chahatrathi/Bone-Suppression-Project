import torch
import os

# Device Selection
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_PATH, "dataset", "augmented", "augmented", "source")
TARGET_DIR = os.path.join(BASE_PATH, "dataset", "augmented", "augmented", "target")

# Hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
LAMBDA_L1 = 200
NUM_EPOCHS = 100

# Set to 0 for stability on MacOS
NUM_WORKERS = 0