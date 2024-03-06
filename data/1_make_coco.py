import copy
import os
import shutil
from webbrowser import get
from pyniche.data.coco.API import COCO_API


ROOT = os.path.join(
    "/Users",
    "niche",
    "_03_Papers",
    "2024",
    "COLO",
    "data",
    "huggingface",
)


def get_json(setname):
    """
    arguments
    ---
    setname: side, top, holstein

    prerequisite
    ---
    PATH is the root path of the data
    """
    return os.path.join(ROOT, "coco_{}.json".format(setname))


def get_splits(setname):
    splits = [s for s in os.listdir(os.path.join(ROOT, setname)) if "DS_Store" not in s]
    return splits


def copy_substract_from_test_to_train(setname):
    """
    Obtain the substraction of test from all, and copy the images to train
    """
    dir_all = os.path.join(ROOT, setname, "all")
    dir_test = os.path.join(ROOT, setname, "test")
    dir_train = os.path.join(ROOT, setname, "train")
    ls_all = os.listdir(dir_all)
    ls_test = os.listdir(dir_test)
    ls_train = list(set(ls_all) - set(ls_test))
    # copy the images
    for img in ls_train:
        img_train = os.path.join(dir_train, img)
        img_all = os.path.join(dir_all, img)
        if not os.path.exists(img_train):
            shutil.copy(img_all, img_train)


def make_coco(split, ls_coco=[]):
    """
    generate a coco file from a list of coco files and images
    """

    if len(ls_coco) == 0:
        print("No COCO files to concatenate")

    else:
        dir_tgt = os.path.join(ROOT, setname, split)
        coco_base = ls_coco[0].subset_by_dir(dir_tgt)

        if len(ls_coco) > 1:
            for i in range(1, len(ls_coco)):
                coco_base = coco_base.concatenate(ls_coco[i].subset_by_dir(dir_tgt))

        coco_base.save(os.path.join(dir_tgt, "coco.json"))


# load json files
coco_side = COCO_API(path_json=get_json("side"))
coco_top = COCO_API(path_json=get_json("top"))
coco_ext = COCO_API(path_json=get_json("external"))

# 0_all --------------------------------------------------------------
## A. copy images to train from the substraction of test from all
setname = "0_all"
copy_substract_from_test_to_train(setname)

## B. generate coco file
for split in get_splits(setname):
    make_coco(split, [coco_side, coco_top])

# 1_top --------------------------------------------------------------
## A. copy images to train from the substraction of test from all
setname = "1_top"
copy_substract_from_test_to_train(setname)

## B. generate coco file
for split in get_splits(setname):
    make_coco(split, [coco_top])

# 2_side --------------------------------------------------------------
## A. copy images to train from the substraction of test from all
setname = "2_side"
copy_substract_from_test_to_train(setname)

## B. generate coco file
for split in get_splits(setname):
    make_coco(split, [coco_side])

# 3_external --------------------------------------------------------------
setname = "3_external"
## A. generate coco file
for split in get_splits(setname):
    make_coco(split, [coco_ext])

# a1_t2s --------------------------------------------------------------
setname = "a1_t2s"
make_coco("train", [coco_top])
make_coco("test", [coco_side])

# a2_s2t --------------------------------------------------------------
setname = "a2_s2t"
make_coco("train", [coco_side])
make_coco("test", [coco_top])

# b_light --------------------------------------------------------------
setname = "b_light"
for split in get_splits(setname):
    make_coco(split, [coco_side, coco_top])

# c_external --------------------------------------------------------------
setname = "c_external"
make_coco("train", [coco_side, coco_top])
make_coco("test", [coco_ext])
