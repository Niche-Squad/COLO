"""
Training YOLOv8
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
MODELS = ["yolov8n.pt", "yolov8m.pt", "yolov8x.pt"]
CONFIGS = [
    "1a_angle_t2s",
    "1b_angle_s2t",
    "2_light",
    "3_breed",
    "4_all",
]
LS_N = [20, 50, 100, 200, 500]
DEVICE = "cuda"

def main(args):
    run = "run_%d" % int(args.thread)
    DIR_OUT = os.path.join(ROOT, "out", "yolov8", run)
    FILE_OUT = os.path.join(DIR_OUT, "results.csv")

    # 3. Initialize outputs -------------------------------------------------------------
    if not os.path.exists(FILE_OUT):
        os.makedirs(os.path.dirname(FILE_OUT), exist_ok=True)
        with open(FILE_OUT, "w") as file:
            file.write("map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,config,model,n\n")

    # 4. Modeling -------------------------------------------------------------
    detr_trainer = NicheTrainer(device=DEVICE)

    # iterate over each model
    for model in MODELS:

        # iterate over each configuration
        for config in CONFIGS:
            ls_n = LS_N if config != "3_breed" else [20, 50, 100, 200, 250]
            DIR_DATA = os.path.join(ROOT, "data", "yolo", config, run)

            # iterate over each training n
            for n in ls_n:
                # define task folders
                i = 0
                while True:
                    path_task = os.path.join(DIR_OUT, "%s_%s_%d_%d" % (model[:-3], config, n, i))
                    if not os.path.exists(path_task):
                        break
                    i += 1      
                # training
                detr_trainer.set_model(
                    modelclass=NicheYOLO,
                    checkpoint=model,
                )
                detr_trainer.set_data(
                    dataclass=DIR_DATA,
                    batch=16,
                    n=n,
                )
                detr_trainer.set_out(os.path.join(DIR_OUT, path_task))
                detr_trainer.fit(epochs=50, rm_threshold=0)

                # evaluation
                metrics = detr_trainer.evaluate_on_test()
                metrics["config"] = config
                metrics["model"] = model.split(".")[0] # remove .pt
                metrics["n"] = n
                line = ",".join([str(value) for value in metrics.values()])
                with open(FILE_OUT, "a") as file:
                    file.write(line + "\n")
                
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread", type=str, help="thread number")
    args = parser.parse_args()
    main(args)