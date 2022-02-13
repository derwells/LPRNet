import os

# CCPD Path
CCPD_PATH = "/home/derick/Downloads/CCPD/CCPD2019"

# List of file splits
TRAIN_PATH = os.path.join(CCPD_PATH, "splits/train.txt")
VAL_PATH = os.path.join(CCPD_PATH, "splits/val.txt")
TEST_PATH = os.path.join(CCPD_PATH, "splits/test_easy.txt")
RAW_SPLIT_PATHS = {
    "test": TEST_PATH,
    "val": VAL_PATH,
    "train": TRAIN_PATH
}

# Clean data directory
DATA_PATH = "data/"

# Training
INPUT_DIMS = (24, 94, 3)
