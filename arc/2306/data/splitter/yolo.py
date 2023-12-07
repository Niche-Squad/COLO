"""
Folder structure:

root/
    test.txt
    images/
        img_1_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
        img_2_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.jpg
        ...
    labels/
        img_1_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.txt
        img_2_13_jpg.rf.d69528304f2d10b633c6d94982185cb2.txt
        ...
    <suffix>/
        images/ # copy from root/images/
        labels/ # copy from root/labels/
        train.txt
        val.txt
        test.txt
        data.yaml

"""
## TODO: Add multi-threading vs. single-threading option

import os
import shutil
import random
import pathlib

from .splitter import Splitter


class YOLO_Splitter(Splitter):
    def __init__(
        self,
        path_root: str,
        classes: list,
        suffix: str = "",
        ratio_train: float = 0.64,
        ratio_val: float = 0.16,
        ratio_test: float = 0.2,
    ):
        self.suffix = suffix
        self.classes = classes
        self.path_yaml = None
        # copy images and labels
        if not os.path.exists(os.path.join(path_root, suffix, "images")):
            pathlib.Path(os.path.join(path_root, suffix)).mkdir(
                parents=True, exist_ok=True
            )
            shutil.copytree(
                os.path.join(path_root, "images"),
                os.path.join(path_root, suffix, "images"),
            )
            shutil.copytree(
                os.path.join(path_root, "labels"),
                os.path.join(path_root, suffix, "labels"),
            )
            shutil.copy(
                os.path.join(path_root, "test.txt"),
                os.path.join(path_root, suffix, "test.txt"),
            )
        self.path_root = os.path.join(path_root, suffix)
        super().__init__(self.path_root, ratio_train, ratio_val, ratio_test)

    def read_dataset(self):
        self.path_yaml = os.path.join(self.path_root, "data.yaml")
        f = open(self.path_yaml, "w")
        return f

    def get_ids(self):
        # from img_1_13_jpg.rf.jpg
        # to img_1_13_jpg.rf
        ls_imagefiles = os.listdir(os.path.join(self.path_root, "images"))
        ls_ids = [f[:-4] for f in ls_imagefiles if f.endswith(".jpg")]
        return ls_ids

    def shuffle_train_test(self):
        random.shuffle(self.ids)
        n = len(self.ids)
        n_train = int(self.ratio_train * n)
        n_val = int(self.ratio_val * n)
        # assignment
        self.id_train = self.ids[:n_train]
        self.id_val = self.ids[n_train : n_train + n_val]
        self.id_test = self.ids[n_train + n_val :]

    def shuffle_train_val(self, n_included=0):
        """
        prerequisite
        ------------
            test folder exists
        args
        ----
            n_included: int
                size of the training set and validation set. If 0, then use the entire available data.
        """
        # get test ids from txt
        path_txt = os.path.join(self.path_root, "test.txt")
        with open(path_txt, "r") as f:
            id_test = f.read().splitlines()
        id_test = [i[:-4] for i in id_test]  # 001.jpg -> 001
        # obtain which ids are to be shuffled
        id_remaining = [f for f in self.ids if f not in id_test]
        random.shuffle(id_remaining)
        # obtain each split size
        n_included = len(id_remaining) if n_included == 0 else n_included
        ratio_train = self.ratio_train / (self.ratio_train + self.ratio_val)
        n_train = int(ratio_train * n_included)
        n_val = n_included - n_train
        # assignment
        self.id_train = id_remaining[:n_train]
        self.id_val = id_remaining[n_train : n_train + n_val]
        self.id_test = id_test

    def write_dataset(self):
        self._write_yaml(classes=self.classes)
        for split in ["train", "val", "test"]:
            self._write_txt(split=split, ids=getattr(self, f"id_{split}"))
        return self.path_yaml

    def _write_yaml(self, classes: list):
        self.config.write(
            f"""
                path: {self.path_root}
                train: "train.txt"
                val: "val.txt"
                test: "test.txt"
                nc: {len(classes)}
                names: {classes}
            """
        )
        self.config.close()

    def _write_txt(self, split: str, ids: list):
        path_txt = os.path.join(self.path_root, "%s.txt" % split)
        with open(path_txt, "w") as f:
            for id in ids:
                f.write(os.path.join(self.path_root, "images", f"{id}.jpg") + "\n")

    # def _handle_folders(self):
    #     for s in ["train", "val", "test"]:
    #         dir_out = os.path.join(self.path_root, s)
    #         if os.path.exists(dir_out):
    #             shutil.rmtree(dir_out)
    #         os.mkdir(dir_out)
    #         os.mkdir(os.path.join(dir_out, "images"))
    #         os.mkdir(os.path.join(dir_out, "labels"))

    # def _copy_images_labels(self):
    #     for s in ["train", "val", "test"]:
    #         for id in getattr(self, f"id_{s}"):
    #             # copy images
    #             path_img = os.path.join(self.path_root, "images", f"{id}.jpg")
    #             path_img_out = os.path.join(self.path_root, s, "images", f"{id}.jpg")
    #             shutil.copy(path_img, path_img_out)
    #             # copy labels
    #             path_label = os.path.join(self.path_root, "labels", f"{id}.txt")
    #             path_label_out = os.path.join(self.path_root, s, "labels", f"{id}.txt")
    #             shutil.copy(path_label, path_label_out)
