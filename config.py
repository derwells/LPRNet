import os

# CCPD Path
CCPD_PATH = "/home/derick/Downloads/CCPD/CCPD2019"

# List of file splits
TRAIN_PATH = os.path.join(CCPD_PATH, "splits/train.txt")
VAL_PATH = os.path.join(CCPD_PATH, "splits/val.txt")
TEST_PATH = os.path.join(CCPD_PATH, "splits/test_easy.txt")
