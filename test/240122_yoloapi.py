import os
from pyniche.data.YOLO.API import YOLO_API

ROOT = os.path.join("/Users", "niche", "_03_Papers", "2024", "cowsformer", "data")


api = YOLO_API(os.path.join(ROOT, "1a_angle_t2s"))
api.shuffle_train_val(n=20)
api.save_yaml(["None", "cow"])
