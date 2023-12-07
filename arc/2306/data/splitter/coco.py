"""
Folder structure:

root/
    _annotations.coco.json
    _train_annotations.coco.json
    _val_annotations.coco.json
    _test_annotations.coco.json
    img_1_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
    img_2_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
    ...
"""

import os
import json
import random

from splitter import Splitter


class COCO_Splitter(Splitter):
    def __init__(
        self,
        path_root: str,
        ratio_train: float = 0.64,
        ratio_val: float = 0.16,
        ratio_test: float = 0.2,
    ):
        super().__init__(path_root, ratio_train, ratio_val, ratio_test)

    def read_dataset(self):
        path_json = os.path.join(self.path_root, "_annotations.coco.json")
        with open(path_json, "r") as f:
            coco_annot = json.load(f)
        return coco_annot

    def get_ids(self):
        return [img["id"] for img in self.config["images"]]

    def shuffle_train_test(self):
        random.shuffle(self.ids)
        n = len(self.ids)
        n_train = int(self.ratio_train * n)
        n_val = int(self.ratio_val * n)
        # assignment
        self.id_train = self.ids[:n_train]
        self.id_val = self.ids[n_train : n_train + n_val]
        self.id_test = self.ids[n_train + n_val :]

    def write_dataset(self):
        dict_id = dict(
            train=self.id_train,
            val=self.id_val,
            test=self.id_test,
        )
        for split in dict_id:
            json_out = make_json(image_ids=dict_id[split], coco_annot=self.config)
            path_out = os.path.join(self.path_root, "_%s_annotations.coco.json" % split)
            with open(path_out, "w") as f:
                json.dump(json_out, f)


def make_json(image_ids: list, coco_annot: dict) -> dict:
    json_out = dict(
        info=coco_annot["info"],
        licenses=coco_annot["licenses"],
        images=[],
        annotations=[],
        categories=coco_annot["categories"],
    )
    for img in coco_annot["images"]:
        if img["id"] in image_ids:
            json_out["images"].append(img)
    for ann in coco_annot["annotations"]:
        if ann["image_id"] in image_ids:
            json_out["annotations"].append(ann)
    return json_out
