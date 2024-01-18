"""
https://huggingface.co/docs/datasets/upload_dataset
https://huggingface.co/settings/tokens
"""

import os
from datasets import load_dataset

PATH = os.path.join("/Users", "niche", "_03_Papers", "2024", "cowsformer", "data")
os.chdir(PATH)
setnames = ["1a_angle_t2s", "1b_angle_s2t", "2_light", "3_breed", "4_all"]
setname = setnames[0]
dataset = load_dataset("config.py", setname)
dataset.push_to_hub("Niche-Squad/cowsformer", setname)
