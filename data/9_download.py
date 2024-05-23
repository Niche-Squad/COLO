LOAD_LOCAL = True

# imports
import os
import sys

if LOAD_LOCAL:
    LOCAL_PYNICHE = os.path.join(
        "/Users",
        "niche",
        "_04_Software",
        "pyniche",
    )
    sys.path.insert(0, LOCAL_PYNICHE)
from pyniche.data.download import COLO

COLO(
    root="download/yolo",
    format="yolo",
)


COLO(
    root="download/coco",
    format="coco",
)
