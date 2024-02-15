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
DATASET = "Niche-Sqaud/cowsformer"
CONFIGS = [
    # "1a_angle_t2s",
    # "1b_angle_s2t",
    # "2_light",
    # "3_breed",
    "4_all",
]
LS_N = [20, 50, 100, 200, 500]

# 3. Initialize outputs -------------------------------------------------------------
with open(FILE_OUT, "w") as file:
    file.write("config,n,map5095,map50,precision,recall,f1,n_all,n_fn,n_fp\n")


# 4. Modeling -------------------------------------------------------------
detr_trainer = NicheTrainer(device="cuda")
detr_trainer.set_out(DIR_OUT)

for config in CONFIGS:
    for n in LS_N:
        # training
        detr_trainer.set_model(
            modelclass=NicheDETR,
            pretrained="facebook/detr-resnet-50",
        )
        detr_trainer.set_data(
            dataclass=DetectDataModule,
            dataname="cowsformer",
            configname=config,
            batch=32,
            n=n,
        )
        detr_trainer.fit(epochs=10)

        # evaluation
        metrics = detr_trainer.evaluate_on_test()
        metrics["n"] = n
        metrics["config"] = config
        line = ",".join([str(value) for value in metrics.values()])
        with open(FILE_OUT, "a") as file:
            file.write(line + "\n")
