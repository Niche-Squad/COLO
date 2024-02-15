import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))

dir_imgs = os.path.join(ROOT, "data", "cow200", "yolov5", "test", "images")
ls_files = os.listdir(dir_imgs)
path_txt = os.path.join(ROOT, "data", "cow200", "yolov5", "test.txt")
# write filename into txt
with open(path_txt, "w") as f:
    for file in ls_files:
        path = os.path.join("images", file)
        f.write(path + "\n")