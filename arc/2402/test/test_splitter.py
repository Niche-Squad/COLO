"""
This script is to test the YOLO splitter function
"""

import os
ROOT = os.path.dirname(os.path.dirname(__file__))
os.chdir(ROOT)

# local imports
from data.splitter.yolo import YOLO_Splitter

# constants
DIR_DATA = os.path.join(ROOT, "data")
DIR_COW200 = os.path.join(DIR_DATA, "cow100", "yolov5")
PATH_DATA = os.path.join(DIR_COW200, "data.yaml")
DIR_TRAIN = os.path.join(DIR_COW200, "train", "images")
DIR_VAL = os.path.join(DIR_COW200, "val", "images")
# model configuration
N_INTER = 100

# every iteration, shuffle the dataset
splitter = YOLO_Splitter(DIR_COW200, classes=["cow"])
splitter.shuffle_train_test()
splitter.write_dataset()

# every iteration, shuffle the dataset
n_train = 20
splitter = YOLO_Splitter(DIR_COW200, classes=["cow"])
splitter.shuffle_train_val(n_included=n_train)
splitter.write_dataset()

# os.listdir(DIR_TRAIN)
os.listdir(DIR_VAL)


test
1-1
1-5
1-11
46
49
val
1-2
1-10
1-21