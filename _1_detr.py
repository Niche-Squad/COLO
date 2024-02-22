"""
Training DETR
"""

# 1. Imports -------------------------------------------------------------------
# native imports
import os

# custom imports
from pyniche.trainer import NicheTrainer
from pyniche.models.detection.detr import NicheDETR
from pyniche.data.coco.detr import DetectDataModule

# huggingface imports
import datasets

# 2. Global Variables ----------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_OUT = os.path.join(ROOT, "out", "detr")
FILE_OUT = os.path.join(DIR_OUT, "results.csv")
DATASET = "Niche-Squad/cowsformer"
CONFIGS = [
    # "1a_angle_t2s",
    # "1b_angle_s2t",
    # "2_light",
    # "3_breed",
    "4_all",
]
LS_N = [20, 50, 100, 200, 500]
DEVICE = "cuda"

# 3. Initialize outputs -------------------------------------------------------------
if not os.path.exists(FILE_OUT):
    os.makedirs(os.path.dirname(FILE_OUT), exist_ok=True)
    with open(FILE_OUT, "w") as file:
        file.write("map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,config,n\n")

# 4. Modeling -------------------------------------------------------------
detr_trainer = NicheTrainer(device=DEVICE)

for config in CONFIGS:
    for n in LS_N:
        # define task folders
        i = 0
        while True:
            path_task = os.path.join(DIR_OUT, "%s_%d_%d" % (config, n, i))
            if not os.path.exists(path_task):
                break
            i += 1      
        # training
        detr_trainer.set_model(
            modelclass=NicheDETR,
            pretrained="facebook/detr-resnet-50",
        )
        detr_trainer.set_data(
            dataclass=DetectDataModule,
            dataname=DATASET,
            configname=config,
            batch=16,
            n=n,
        )
        detr_trainer.set_out(os.path.join(DIR_OUT, path_task))
        detr_trainer.fit(epochs=50, rm_threshold=0)

        # evaluation
        metrics = detr_trainer.evaluate_on_test()
        metrics["config"] = config
        metrics["n"] = n
        print(metrics)
        line = ",".join([str(value) for value in metrics.values()])
        with open(FILE_OUT, "a") as file:
            file.write(line + "\n")
         