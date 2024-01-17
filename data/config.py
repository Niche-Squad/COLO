# coding=utf-8
# Copyright 2023 The NicheSquad Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"https://huggingface.co/datasets/Howuhh/nle_hf_dataset/blob/main/nle_hf_dataset.py"
"https://huggingface.co/datasets/vivos/blob/main/vivos.py"  # audio
"""
Documentation:
- datasets in lightining
https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/text-transformers.html
- how to use datasets

"""

import pandas as pd
import os
import json
import datasets
from PIL import Image
from collections import defaultdict

CITATION = """Niche-Squad Cowsformer Dataset"""
DESCRIPTION = """\
"""
VERSION = datasets.Version("1.0.0")

PATHS = {
    "image": "data/images.tar.gz",
    "train": "data/train.json",
    "val": "data/val.json",
    "test": "data/test.json",
}


def get_paths(setname, split):
    """
    setname: 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all
    split: train, test
    """
    path = os.path.join(setname, split)


class BalloonDatasets(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="1a_angle_t2s",
            version=VERSION,
            description="Traing on top-down images, testing on side-view images",
        ),
        datasets.BuilderConfig(
            name="1b_angle_s2t",
            version=VERSION,
            description="Traing on side-view images, testing on top-down images",
        ),
        datasets.BuilderConfig(
            name="2_light",
            version=VERSION,
            description="Traing on images with light, testing on images without light",
        ),
        datasets.BuilderConfig(
            name="3_breed",
            version=VERSION,
            description="Traing on images with Holstein cows, testing on images with all cows",
        ),
        datasets.BuilderConfig(
            name="4_all",
            version=VERSION,
            description="Traing on all images, testing on all images",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "image_id": datasets.Value("int64"),
                    "filename": datasets.Value("string"),
                    "annotations": datasets.Sequence(
                        {
                            "id": datasets.Value("int64"),
                            "image_id": datasets.Value("int64"),
                            "category_id": datasets.Value("int64"),
                            "iscrowd": datasets.Value("int64"),
                            "area": datasets.Value("float64"),
                            "bbox": datasets.Sequence(
                                datasets.Value("float64"), length=4
                            ),
                            "segmentation": datasets.Sequence(
                                datasets.Sequence(datasets.Value("int64"))
                            ),
                        }
                    ),
                }
            ),
            homepage="github.com/niche-squad",
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        # extract the images
        path_archive = dl_manager.download(PATHS["image"])

        # check config types
        path_train = PATHS["train"]
        path_val = PATHS["val"]
        path_test = PATHS["test"]

        # return the splits
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_label": dl_manager.download(path_train),
                    "images": dl_manager.iter_archive(path_archive),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path_label": dl_manager.download(path_val),
                    "images": dl_manager.iter_archive(path_archive),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path_label": dl_manager.download(path_test),
                    "images": dl_manager.iter_archive(path_archive),
                },
            ),
        ]

    def _generate_examples(self, path_label, images):
        labels = COCO(path_label)
        for file_path, file_obj in images:
            filename = file_path.split("/")[-1]  # it's "images/xxx.jpg"
            img_id = labels.get_img_id(filename)
            if img_id is None:
                continue
            bytes_img = file_obj.read()
            record = {
                "image": {"path": file_path, "bytes": bytes_img},
                "image_id": img_id,
                "filename": filename,
                "annotations": [
                    {
                        "id": ann["id"],
                        "image_id": ann["image_id"],
                        "category_id": ann["category_id"],
                        "iscrowd": ann["iscrowd"],
                        "area": ann["area"],
                        "bbox": ann["bbox"],
                        "segmentation": ann["segmentation"],
                    }
                    for ann in labels.load_ann_by_id(img_id)
                ],
            }
            yield filename, record


class COCO:
    def __init__(self, filename):
        with open(filename) as f:
            data = json.load(f)
        self.imgs = data["images"]
        self.anns = data["annotations"]
        self.cats = data["categories"]
        self.licenses = data["licenses"]

    def load_ann_by_id(self, img_id):
        ls = [ann for ann in self.anns if ann["image_id"] == img_id]
        print("---")
        print([i["id"] for i in ls])
        return ls

    def get_img_id(self, filename):
        for img in self.imgs:
            if img["file_name"] == filename:
                return img["id"]
        return None
