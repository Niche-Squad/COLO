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
DIR_OUT = os.path.join(ROOT, "out", "yolov8")
DATASET = "Niche-Squad/cowsformer"
DEVICE = "cuda"

def main(args):
    # extract arguments
    thread = "run_%d" % int(args.thread)
    model = args.model
    config = args.config
    n = int(args.n)

    # create result file
    DIR_OUT = os.path.join(ROOT, "out", "yolov8", thread)
    FILE_OUT = os.path.join(DIR_OUT, "results.csv")
    DIR_DATA = os.path.join(ROOT, "data", "yolo", config, thread)

    # create task folder
    i = 0
    while True:
        path_task = os.path.join(DIR_OUT, "%s_%s_%d_%d" % (model[:-3], config, n, i))
        if not os.path.exists(path_task):
            break
        i += 1  
 
    # 3. Initialize outputs -------------------------------------------------------------
    if not os.path.exists(FILE_OUT):
        os.makedirs(os.path.dirname(FILE_OUT), exist_ok=True)
        with open(FILE_OUT, "w") as file:
            file.write("map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,config,model,n\n")

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
    trainer.set_out(os.path.join(DIR_OUT, path_task))
    trainer.fit(epochs=100, rm_threshold=0)

    # 5. Evaluation -------------------------------------------------------------
    metrics = trainer.evaluate_on_test()
    metrics["config"] = config
    metrics["model"] = model.split(".")[0] # remove .pt
    metrics["n"] = n
    line = ",".join([str(value) for value in metrics.values()])
    with open(FILE_OUT, "a") as file:
        file.write(line + "\n")
                
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", type=str, help="thread number")
    parser.add_argument("--model", type=str, help="options: yolov8n, yolov8m, yolov8x")
    parser.add_argument("--config", type=str, help="options: 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all")
    parser.add_argument("--n", type=int, help="number of images in training set")
    args = parser.parse_args()
    main(args)