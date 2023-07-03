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
DIR_COW200 = os.path.join(DIR_DATA, "cow200", "yolov5")

# model configuration
BATCH = 16
EPOCHS = 100


def main(args):
    # parse arguments
    i = args.iter
    n_train = args.n_train
    yolo_base = args.yolo_base
    suffix = args.suffix

    # shuffle dataset
    splitter = YOLO_Splitter(DIR_COW200, classes=["cow"], suffix=suffix)
    splitter.shuffle_train_val(n_included=n_train)
    path_data = splitter.write_dataset()

    # log
    print("-----------------------------------")
    print("n_train: %d, yolo_base: %s, i: %d, %s" % (n_train, yolo_base, i, suffix))
    print("-----------------------------------")
    # define paths
    name_task = "n%d_%s_i%d_%s" % (n_train, yolo_base[:-3], i, suffix)

    # configure model
    yolo = Niche_YOLO(
        path_model=os.path.join(DIR_MODEL, yolo_base),
        dir_train=os.path.join(DIR_OUT, "train"),
        dir_val=os.path.join(DIR_OUT, "val"),
        name_task=name_task,
    )

    # train
    yolo.train(path_data, BATCH, EPOCHS)

    # evaluate
    yolo.evaluate()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, help="iteration number")
    parser.add_argument("--n_train", type=int, help="number of images in training set")
    parser.add_argument("--yolo_base", type=str, help="e.g., yolo8n, yolo8m, yolo8x")
    parser.add_argument("--suffix", type=str, help="suffix for folder name")
    args = parser.parse_args()
    main(args)
