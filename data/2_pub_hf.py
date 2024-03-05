"""
https://huggingface.co/docs/datasets/upload_dataset
https://huggingface.co/settings/tokens
"""

import os
from datasets import load_dataset

ROOT = os.path.join(
    "/Users",
    "niche",
    "_03_Papers",
    "2024",
    "COLO",
    "data",
    "huggingface",
)

os.chdir(ROOT)
setnames = [
    "0_all",
    "1a_angle_t2s",
    "1b_angle_s2t",
    "2_light",
    "3_external",
]
for setname in setnames:
    dataset = load_dataset(
        "config.py",
        setname,
        trust_remote_code=True,
        cache_dir=".huggingface",
    )
    dataset.push_to_hub("Niche-Squad/COLO", setname)


# # test
# setname = "4_all"
# dataset = load_dataset("config.py", setname, trust_remote_code=True, cache_dir=".hf")
# dataset["test"]["annotations"][0]
