"""
Download data from HuggingFace to local machine
"""

# 1. Imports -------------------------------------------------------------------
# native imports
import os
import sys
sys.path.insert(0, os.path.join("/home", "niche", "pyniche"))

# custom imports
from pyniche.data.yolo.API import YOLO_API
from pyniche.data.huggingface.detection import hf_to_yolo

# huggingface imports
import datasets

# 2. Global Variables ----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.join(ROOT, "data")
DIR_YOLO = os.path.join(DIR_DATA, "yolo")
DATASET = "Niche-Squad/cowsformer"
THREADS_YOLO = 4
CONFIGS = [
    "1a_angle_t2s",
    "1b_angle_s2t",
    "2_light",
    "3_breed",
    "4_all",
]

# 3. Download Data -------------------------------------------------------------
for config in CONFIGS:
    dir_config = os.path.join(DIR_YOLO, config)
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
    # clone the datasets for multi-threading
    yolo_api = YOLO_API(dir_config)
    for thread in range(THREADS_YOLO):
        yolo_api.clone("run_%d" % thread)
