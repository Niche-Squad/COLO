import os
import json
from ultralytics import YOLO

# local imports
from .device import get_device

# constants
ROOT = os.path.dirname(os.path.dirname(__file__))
# mAP50 is the average precision that considers both classification and bounding box localization


class Niche_YOLO:
    def __init__(self, path_model, dir_out, name_task=""):
        """
        Folder structure
        models/
            <path_model>.pt
        out/
            train/
                <name_task>/
                    weights/
                        best.pt
                    ...
            val/
                <name_task>/
                    results.json
                    ...

        """
        # attributes
        self.model = None
        self.dir_train = os.path.join(dir_out, "train")
        self.dir_val = os.path.join(dir_out, "val")
        self.name_task = name_task  # task folder name

        # init
        self.load(path_model)

    def load(self, path_model):
        self.model = YOLO(path_model)
        print("model %s loaded" % path_model)

    def train(self, path_data, batch=16, epochs=100):
        """
        args
        ----
            path_data: str
                path to data.yaml
            batch: int
                batch size
            epochs: int
                number of epochs
        """
        self.model.train(
            data=path_data,
            batch=batch,
            epochs=epochs,
            device=get_device(),
            project=self.dir_train,
            name=self.name_task,
            exist_ok=True,
        )
        best_model = os.path.join(self.dir_train, self.name_task, "weights", "best.pt")
        self.load(best_model)

    def evaluate(self, split="test"):
        metrics = self.model.val(
            split=split,
            device=get_device(),
            project=self.dir_val,
            name=self.name_task,
            exist_ok=True,
        )
        # save metrics
        path_out = os.path.join(self.dir_val, self.name_task, "results.json")
        with open(path_out, "w") as f:
            json_metrics = ext_metrics(metrics)
            json.dump(json_metrics, f)
        # return mAP50
        return json_metrics["map50"]


def ext_metrics(metrics):
    """
    args
    ----
        metrics: dict
            metrics from model.val()

    return
    ------
        json: dict
    """
    # metrics
    map5095 = metrics.box.map.round(4)
    map50 = metrics.box.map50.round(4)
    precision = metrics.box.p[0].round(4)
    recall = metrics.box.r[0].round(4)
    f1 = metrics.box.f1[0].round(4)
    # confusion matrix
    conf_mat = metrics.confusion_matrix.matrix  # conf=0.25, iou_thres=0.45
    n_all = conf_mat[:, 0].sum()
    n_missed = conf_mat[1, 0].sum()
    n_false = conf_mat[0, 1].sum()
    # write json
    json_out = dict(
        map5095=map5095,
        map50=map50,
        precision=precision,
        recall=recall,
        f1=f1,
        n_all=int(n_all),
        n_missed=int(n_missed),
        n_false=int(n_false),
    )
    return json_out
