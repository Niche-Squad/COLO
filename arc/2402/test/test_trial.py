"""
Date: 2023-06-08
Author: James Chen

This trial aims to compare the performance difference based on two factors:
    1. the number of images in the dataset
        - 20
        - 50
        - 80
    2. size of the model
        - YOLOv8n: mAP 0.5:0.95 = 37.3; params = 3.2M;
        - YOLOv8m: mAP 0.5:0.95 = 50.2; params = 25.9M;
        - YOLOv8x: mAP 0.5:0.95 = 53.9; params = 68.2M;
"""

import os
import argparse

# local imports
from models.yolo import Niche_YOLO
from data.splitter.yolo import YOLO_Splitter

# constants
ROOT = os.path.dirname(__file__)
DIR_OUT = os.path.join(ROOT, "out")
DIR_MODEL = os.path.join(ROOT, "models")
DIR_DATA = os.path.join(ROOT, "data")
DIR_COW200 = os.path.join(DIR_DATA, "cow100", "yolov5")
PATH_DATA = os.path.join(DIR_COW200, "data.yaml")

# model configuration
BATCH = 16
N_INTER = 100

def main(i):
        
    n_train = 20
    yolo_base = "yolov8n.pt"

    splitter = YOLO_Splitter(DIR_COW200, classes=["cow"])
    splitter.shuffle_train_val(n_included=n_train)
    splitter.write_dataset()

    # log
    print("-----------------------------------")
    print("n_train: %d, yolo_base: %s, i: %d" % (n_train, yolo_base, i))
    print("-----------------------------------")
    # define paths
    dir_out = "n%d_%s_i%d" % (n_train, yolo_base[:-3], i)
    path_yolo = os.path.join(DIR_MODEL, yolo_base)

    # configure model
    yolo = Niche_YOLO(
        path_model=path_yolo,
        dir_train=os.path.join(DIR_OUT, "train"),
        dir_val=os.path.join(DIR_OUT, "val"),
    )

    # train
    yolo.train(PATH_DATA, dir_out, BATCH, epochs=1)

    # evaluate
    yolo.evaluate(dir_out)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=0)
    args = parser.parse_args()
    i = args.iter
    main(i)
