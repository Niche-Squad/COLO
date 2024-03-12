"""
- YOLOv8n: mAP 0.5:0.95 = 37.3; params = 3.2M;
- YOLOv8m: mAP 0.5:0.95 = 50.2; params = 25.9M;
- YOLOv8x: mAP 0.5:0.95 = 53.9; params = 68.2M;
"""

# 1. Imports -------------------------------------------------------------------
# native imports
import os
import sys
import argparse

# custom imports
sys.path.insert(0, os.path.join("/home", "niche", "pyniche"))
from pyniche.trainer import NicheTrainer
from pyniche.models.detection.yolo import NicheYOLO

# 2. Global Variables ----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda"


def main(args):
    # extract arguments
    model = args.model
    config = args.config

    # 3. Create task folder -------------------------------------------------------------
    DIR_OUT = os.path.join(
        ROOT,
        "out",
        "pretrained",
        "%s_%s" % (model[:-3], config),
    )
    DIR_DATA = os.path.join(
        ROOT,
        "data",
        config,
    )
    PT_NAME = os.path.join(
        ROOT,
        "%s_%s.pt" % (model[:-3], config),
    )

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT, exist_ok=True)

    # 4. Modeling -------------------------------------------------------------
    trainer = NicheTrainer(device=DEVICE)
    trainer.set_model(
        modelclass=NicheYOLO,
        checkpoint=model,
    )
    trainer.set_data(
        dataclass=DIR_DATA,
        batch=16,
        merge_train_test=True,
    )
    trainer.set_out(DIR_OUT)
    trainer.fit(
        epochs=300,
        rm_threshold=0,
        # yolov9 augmentation
        copy_paste=0.3,
        mixup=0.15,
    )

    # 5. Move best.pt -------------------------------------------------------------
    os.rename(
        os.path.join(DIR_OUT, "weights", "best.pt"),
        os.path.join(DIR_OUT, PT_NAME),
    )
    os.remove(os.path.join(DIR_OUT, "weights", "last.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="yolo checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="options: 0_all, 1_top, 2_side",
    )
    args = parser.parse_args()
    main(args)
