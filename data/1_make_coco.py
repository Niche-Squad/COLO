import os
from pyniche.data.coco.API import COCO_API


ROOT = os.path.join(
    "/Users",
    "niche",
    "_03_Papers",
    "2024",
    "cowsformer",
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


def get_dir(setname, task):
    """
    arguments
    ---
    setname: 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all
    task: train, test

    prerequisite
    ---
    PATH is the root path of the data
    """
    return os.path.join(ROOT, setname, task)


coco_side = COCO_API(path_json=get_json("side"))
coco_top = COCO_API(path_json=get_json("top"))
coco_holstein = COCO_API(path_json=get_json("holstein"))

# 1a_angle_t2s ---
# train
dir_tgt = get_dir("1a_angle_t2s", "train")
coco_top.subset_by_dir(dir_tgt).save(os.path.join(dir_tgt, "coco.json"))
# test
dir_tgt = get_dir("1a_angle_t2s", "test")
coco_side.subset_by_dir(dir_tgt).save(os.path.join(dir_tgt, "coco.json"))

# 1b_angle_s2t ---
# train
dir_tgt = get_dir("1b_angle_s2t", "train")
coco_side.subset_by_dir(dir_tgt).save(os.path.join(dir_tgt, "coco.json"))
# test
dir_tgt = get_dir("1b_angle_s2t", "test")
coco_top.subset_by_dir(dir_tgt).save(os.path.join(dir_tgt, "coco.json"))

# 2_light ---
# train
dir_tgt = get_dir("2_light", "train")
coco_side_sub = coco_side.subset_by_dir(dir_tgt)
coco_top_sub = coco_top.subset_by_dir(dir_tgt)
coco_side_sub.concatenate(coco_top_sub).save(os.path.join(dir_tgt, "coco.json"))
# test
dir_tgt = get_dir("2_light", "test")
coco_side_sub = coco_side.subset_by_dir(dir_tgt)
coco_top_sub = coco_top.subset_by_dir(dir_tgt)
coco_side_sub.concatenate(coco_top_sub).save(os.path.join(dir_tgt, "coco.json"))

# 3_breed ---
# train
dir_tgt = get_dir("3_breed", "train")
coco_holstein.subset_by_dir(dir_tgt).save(os.path.join(dir_tgt, "coco.json"))
# test
dir_tgt = get_dir("3_breed", "test")
coco_side_sub = coco_side.subset_by_dir(dir_tgt)
coco_top_sub = coco_top.subset_by_dir(dir_tgt)
coco_side_sub.concatenate(coco_top_sub).save(os.path.join(dir_tgt, "coco.json"))

# 4_all ---
# train
dir_tgt = get_dir("4_all", "train")
coco_side_sub = coco_side.subset_by_dir(dir_tgt)
coco_top_sub = coco_top.subset_by_dir(dir_tgt)
coco_side_sub.concatenate(coco_top_sub).save(os.path.join(dir_tgt, "coco.json"))
# test
dir_tgt = get_dir("4_all", "test")
coco_side_sub = coco_side.subset_by_dir(dir_tgt)
coco_top_sub = coco_top.subset_by_dir(dir_tgt)
coco_side_sub.concatenate(coco_top_sub).save(os.path.join(dir_tgt, "coco.json"))
