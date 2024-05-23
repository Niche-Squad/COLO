# COw LOcalization (COLO) Dataset

## Download the dataset

To download the dataset, it is required to have the python dependencies installed:

```sh
python -m pip install pyniche
```

or

```sh
pip install pyniche
```

In a Python console provide the download destination folder in the parameter `root` and specify the export data format in `format`:

```python
from pyniche.data.download import COLO

# example: download COLO in the YOLO format
COLO(
    root="download/yolo",
    format="yolo",
)

# example: download COLO in the COCO format
COLO(
    root="download/coco",
    format="coco",
)
```
