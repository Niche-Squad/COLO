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

import os
import json
import datasets

CITATION = """Das, M., G. Ferreira, and C.P.J. Chen. (2024)"""
VERSION = datasets.Version("1.0.0")


def get_coco(setname, split):
    """
    setname: 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all
    split: train, test
    """
    path = os.path.join(setname, split, "coco.json")
    return path


def get_imgdir(setname, split):
    """
    setname: 1a_angle_t2s, 1b_angle_s2t, 2_light, 3_breed, 4_all
    split: train, test
    """
    path = os.path.join(setname, split)
    return path


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
        setname = self.config.name

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path_label": dl_manager.download(get_coco(setname, "train")),
                    "images": dl_manager.iter_files(get_imgdir(setname, "train")),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path_label": dl_manager.download(get_coco(setname, "test")),
                    "images": dl_manager.iter_files(get_imgdir(setname, "test")),
                },
            ),
        ]

    def _generate_examples(self, path_label, images):
        labels = COCO(path_label)
        for filename in images:
            # if the file is coco.json, skip
            if not filename.endswith(".jpg"):
                continue
            # if the file is not in the json, skip
            img_id = labels.get_img_id(filename)
            if img_id is None:
                continue
            # read the image into bytes by the filenamt
            bytes_img = open(filename, "rb").read()
            record = {
                "image": {"path": filename, "bytes": bytes_img},
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
        # cut the filename from the path
        imagename = os.path.basename(filename)
        for img in self.imgs:
            if img["file_name"] == imagename:
                return img["id"]
        return None
