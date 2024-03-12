"""
This script intend to:

1. Cut right side (2t%) of the image if "side" prefixed
2. Cut both left and right side (t%) of the image if "img_" or "top" prefixed
"""

from PIL import Image
import os
import sys
import shutil
import json
import random

sys.path.insert(
    0,
    os.path.join(
        "/Users",
        "niche",
        "_04_Software",
        "pyniche",
    ),
)

from pyniche.visualization.supervision import vis_detections
from pyniche.data.coco.API import COCO_API

ROOT = os.path.join(
    "/Users",
    "niche",
    "_03_Papers",
    "2024",
    "COLO",
    "data",
)
DIR_SRC = os.path.join(ROOT, "huggingface")
DIR_DST = os.path.join(ROOT, "huggingface_cropped")
CROP_RATE = 0.2

ls_sets = [
    "0_all",
    "1_top",
    "2_side",
    "3_external",
    "a1_t2s",
    "a2_s2t",
    "b_light",
    "c_external",
]


for setname in ls_sets:
    splits = [
        s for s in os.listdir(os.path.join(DIR_SRC, setname)) if "DS_Store" not in s
    ]
    for split in splits:
        dir_src = os.path.join(DIR_SRC, setname, split)
        dir_dst = os.path.join(DIR_DST, setname, split)
        coco_src = os.path.join(dir_src, "coco.json")
        coco_dst = os.path.join(dir_dst, "coco.json")
        print(dir_src)
        # create folder
        if not os.path.exists(dir_dst):
            os.makedirs(dir_dst)

        api = COCO_API(coco_src)
        imgs = api.images()
        anns = api.annotations()
        new_anns = []
        for img in imgs:
            ann = [a for a in anns if a["image_id"] == img["id"]]

            filename = img["file_name"]
            crop_left = True if filename.startswith("side") else False

            img_pil = Image.open(os.path.join(dir_src, filename))
            ori_w, ori_h = img_pil.size
            new_w, new_h = int(ori_w * (1 - 2 * CROP_RATE)), ori_h

            if crop_left:
                # crop left 60%
                new_left = 0
                new_top = 0
                new_right = new_w
                new_bottom = new_h
                img_pil_c = img_pil.crop((new_left, new_top, new_right, new_bottom))
            else:
                # crop central 60%
                new_left = (ori_w - new_w) // 2
                new_top = 0
                new_right = (ori_w + new_w) // 2
                new_bottom = new_h
                img_pil_c = img_pil.crop((new_left, new_top, new_right, new_bottom))

            # adjust annotations
            # Assuming COCO format, each annotation has 'bbox' = [x, y, width, height]
            ls_rm = []
            for a in ann:
                x, y, w, h = a["bbox"]
                # Adjust the coordinates to the new cropped image
                x -= new_left
                # Make sure the bounding box is still within the image
                if x > new_w:
                    # bbox falls outside the cropped image
                    ls_rm.append(a)
                    continue
                elif x < 0:
                    # bbox could either be partially inside or outside the cropped image
                    w = x + w
                    if w < 0:
                        ls_rm.append(a)
                        continue
                    x = 0
                else:
                    # bbox is completely inside the cropped image, still need to check the right side
                    if x + w > new_w:
                        w = new_w - x

                # Update the annotation
                a["bbox"] = [x, y, w, h]
                a["area"] = w * h
            # rm the annotations
            for a in ls_rm:
                ann.remove(a)
            new_anns.extend(ann)
            img["width"], img["height"] = new_w, new_h

            # save the cropped image
            img_pil_c.save(os.path.join(dir_dst, filename))

        # save the updated annotations
        api.data["annotations"] = new_anns
        api.save(coco_dst)

        # verify
        new_api = COCO_API(coco_dst)
        detections = new_api.get_detections()
        n_detect = len(detections)
        for _ in range(20):
            i = random.randint(0, n_detect - 1)
            image = new_api.get_PIL(i)
            vis_detections(
                image,
                detections[i],
                text="cow",
                thickness=1,
                save=os.path.join(dir_dst, f"_verify_{i}.png"),
            )
