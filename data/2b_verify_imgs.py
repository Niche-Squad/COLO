import os
import sys

sys.path.insert(
    0,
    os.path.join(
        "/Users",
        "niche",
        "_04_Software",
        "pyniche",
    ),
)

from pyniche.data.coco.API import COCO_API


ROOT = os.path.join(
    "/Users",
    "niche",
    "_03_Papers",
    "2024",
    "COLO",
    "data",
    "huggingface_cropped",
)


def get_splits(setname):
    splits = [s for s in os.listdir(os.path.join(ROOT, setname)) if "DS_Store" not in s]
    return splits

N_IMGS = 3
LS_SETS = [
    # "0_all",
    "1_top",
    "2_side",
    "3_external",
    # "a1_t2s",
    # "a2_s2t",
    # "b_light",
    # "c_external",
]

for s in LS_SETS:
    splits = get_splits(s)
    for split in splits:
        dir_src = os.path.join(ROOT, s, split)
        api = COCO_API(os.path.join(dir_src, "coco.json"))
        api.verify(n=N_IMGS)

# rm all verified files
for s in LS_SETS:
    splits = get_splits(s)
    for split in splits:
        # rm all _verify_*.png
        dir_src = os.path.join(ROOT, s, split)
        ls_verify = [f for f in os.listdir(dir_src) if "_verify_" in f]
        for f in ls_verify:
            os.remove(os.path.join(dir_src, f))

