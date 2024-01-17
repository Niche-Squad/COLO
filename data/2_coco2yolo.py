import os
import supervision as sv

PATH = os.path.join("/Users", "niche", "_03_Papers", "2024", "cowsformer", "data")

setnames = [
    "1a_angle_t2s",
    "1b_angle_s2t",
    "2_light",
    "3_breed",
    "4_all",
]

for setname in setnames:
    for split in ["train", "test"]:
        path_json = os.path.join(PATH, setname, split, "coco.json")
        dir_imgs = os.path.join(PATH, setname, split)
        data = sv.DetectionDataset.from_coco(
            images_directory_path=dir_imgs,
            annotations_path=path_json,
        )
        data.as_yolo(annotations_directory_path=dir_imgs)
