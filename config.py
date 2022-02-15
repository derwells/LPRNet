import os

# CCPD Path
CCPD_PATH = "/home/derick/Downloads/CCPD/CCPD2019"

# List of file splits
RAW_TRAIN_PATH = os.path.join(CCPD_PATH, "splits/train.txt")
RAW_VAL_PATH = os.path.join(CCPD_PATH, "splits/val.txt")
RAW_TEST_PATH = os.path.join(CCPD_PATH, "splits/test_easy.txt")
RAW_SPLIT_PATHS = {
    "test": RAW_TEST_PATH,
    "val": RAW_VAL_PATH,
    "train": RAW_TRAIN_PATH
}

# Clean data directory
DATA_PATH = "data/"

# Clean data split
CLEAN_TRAIN_PATH = os.path.join(DATA_PATH, "train")
CLEAN_VAL_PATH = os.path.join(DATA_PATH, "val")
CLEAN_TEST_PATH = os.path.join(DATA_PATH, "test")
CLEAN_SPLIT_PATHS = {
    "test": CLEAN_TEST_PATH,
    "val": CLEAN_VAL_PATH,
    "train": CLEAN_TRAIN_PATH
}

# Training
INPUT_DIMS = (24, 94, 3)

# Model Name
MODEL_NAME = "lprnet_ccpd_base_custom_split.tflite"
MODEL_TARGET_DIR = "trained_models/"
MODEL_TARGET_PATH = os.path.join(
    MODEL_TARGET_DIR, MODEL_NAME
)
