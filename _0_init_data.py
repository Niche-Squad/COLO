"""
Download data from HuggingFace to local machine
"""

# 1. Imports -------------------------------------------------------------------
# native imports
import os
import sys
import argparse

# sys.path.insert(0, os.path.join("/home", "niche", "pyniche"))

# custom imports
from pyniche.data.yolo.API import YOLO_API
from pyniche.data.huggingface.detection import hf_to_yolo

# huggingface imports
import datasets

# 2. Global Variables ----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = "Niche-Squad/COLO"
CONFIGS = [
    "0_all",
    "1_top",
    "2_side",
    "3_external",
    "a1_t2s",
    "a2_s2t",
    "b_light",
    "c_external",
]


# 3. Download Data -------------------------------------------------------------
def main(args):
    DIR_DATA = args.dir_data
    THREADS_YOLO = args.threads

    for config in CONFIGS:
        dir_config = os.path.join(DIR_DATA, config)
        hf_dataset = datasets.load_dataset(
            DATASET,
            config,
            download_mode="force_redownload",
            cache_dir=os.path.join(DIR_DATA, ".huggingface"),
        )
        # convert to YOLO for YOLOv8
        hf_to_yolo(
            hf_dataset,
            dir_config,
            classes=["cow"],
            size_new=(640, 640),
        )
        # # clone the datasets for multi-threading
        # yolo_api = YOLO_API(dir_config)
        # for thread in range(THREADS_YOLO):
        #     yolo_api.clone("run_%d" % thread)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data")
    parser.add_argument("--threads", default=2)
    args = parser.parse_args()
    main(args)
