LOAD_LOCAL = True

# imports
import os
import sys
import random
import datasets

if LOAD_LOCAL:
    LOCAL_PYNICHE = os.path.join(
        "/Users",
        "niche",
        "_04_Software",
        "pyniche",
    )
    sys.path.insert(0, LOCAL_PYNICHE)

import pyniche
from pyniche.data.coco.API import COCO_API
from pyniche.data.yolo.API import YOLO_API
from pyniche.data.huggingface.detection import hf_to_coco, hf_to_yolo
from pyniche.visualization.supervision import vis_detections

print(pyniche.__file__)

# CONSTANTS
ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_VERIFY = os.path.join(ROOT, "verify")
PATH_COCO = os.path.join(PATH_VERIFY, "coco")
PATH_YOLO = os.path.join(PATH_VERIFY, "yolo")
CONFIG = "0_all"  # 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all
N_VERIFY = 20
RESIZE = (640, 640)


# download
print("------- DOWNLOADING DATASET -------")
cowsformer = datasets.load_dataset(
    "Niche-Squad/cowsformer",
    CONFIG,
    # download_mode="force_redownload",
)
print(cowsformer)


# COCO
print("------- VERIFYING COCO -------")
hf_to_coco(
    cowsformer,
    PATH_COCO,
    classes=["cow"],
    size_new=RESIZE,
)
path_json = os.path.join(PATH_COCO, "test", "data.json")
coco_api = COCO_API(path_json)
detections = coco_api.get_detections()
n_detect = len(detections)
for _ in range(N_VERIFY):
    i = random.randint(0, n_detect)
    image = coco_api.get_PIL(i)
    vis_detections(
        image,
        detections[i],
        text="cow",
        thickness=1,
        save=os.path.join(PATH_COCO, f"verify_{i}.png"),
    )

# YOLO
print("------- VERIFYING YOLO -------")
hf_to_yolo(
    cowsformer,
    PATH_YOLO,
    classes=["cow"],
    size_new=RESIZE,
)
yolo_api = YOLO_API(PATH_YOLO)
detections = yolo_api.get_detections("test")
n_detect = len(detections)
for _ in range(N_VERIFY):
    i = random.randint(0, n_detect)
    image = yolo_api.get_PIL("test", i)
    vis_detections(
        image,
        detections[i],
        text="cow",
        thickness=1,
        save=os.path.join(PATH_YOLO, f"verify_{i}.png"),
    )
