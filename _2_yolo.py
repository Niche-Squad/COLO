"""
Training YOLOv8

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
    n = int(args.n)
    i = int(args.i)

    # create result file
    DIR_OUT = os.path.join(
        ROOT,
        "out",
        "yolo",
        config,
        "%s_%d_%d" % (model[:-3], n, i),
    )
    DIR_DATA = os.path.join(
        ROOT,
        "data",
        config,
    )
    FILE_OUT = os.path.join(
        DIR_OUT,
        "results.csv",
    )

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT, exist_ok=True)

    # 3. Initialize outputs -------------------------------------------------------------
    if not os.path.exists(FILE_OUT):
        os.makedirs(os.path.dirname(FILE_OUT), exist_ok=True)
        with open(FILE_OUT, "w") as file:
            file.write(
                "map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,config,model,n\n"
            )

    # 4. Modeling -------------------------------------------------------------
    trainer = NicheTrainer(device=DEVICE)
    trainer.set_model(
        modelclass=NicheYOLO,
        checkpoint=model,
    )
    trainer.set_data(
        dataclass=DIR_DATA,
        batch=16,
        n=n,
    )
    trainer.set_out(DIR_OUT)
    trainer.fit(
        epochs=100,
        rm_threshold=0,
        copy_paste=0.3,
        mixup=0.15,
    )

    # 5. Evaluation -------------------------------------------------------------
    metrics = trainer.evaluate_on_test()
    metrics["config"] = config
    metrics["model"] = model.split(".")[0]  # remove .pt
    metrics["n"] = n
    line = ",".join([str(value) for value in metrics.values()])
    with open(FILE_OUT, "a") as file:
        file.write(line + "\n")


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
        help="options: a1_t2s, a2_s2t, b_light, c_external, 0_all, 1_top, 2_side, 3_external",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="number of images in training set",
    )
    parser.add_argument(
        "--i",
        type=int,
        help="iteration number",
    )
    args = parser.parse_args()
    main(args)
