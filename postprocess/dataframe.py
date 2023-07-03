"""
This script is used to create a dataframe from the results of the validation
"""

import pandas as pd
import os
import json

ROOT = os.path.dirname(os.path.dirname(__file__))
DIR_VAL = os.path.join(ROOT, "out", "val")
PATH_OUT = os.path.join(ROOT, "out", "result.csv")


def main():
    ls_trials = os.listdir(DIR_VAL)
    df_out = pd.DataFrame(
        columns=[
            "map5095",
            "map50",
            "precision",
            "recall",
            "f1",
            "n_all",
            "n_missed",
            "n_false",
            "size",
            "model",
            "iter",
        ]
    )
    for t in ls_trials:
        dir_trial = os.path.join(DIR_VAL, t)
        # folder name: n20_yolov8n_i0_run0
        size, model, iter, _ = t.split("_")
        size = int(size[1:])
        iter = int(iter[1:])
        # load json
        with open(os.path.join(dir_trial, "results.json")) as f:
            metrics = json.load(f)
        # make dataframe
        df_t = pd.DataFrame.from_dict(metrics, orient="index").T
        df_t["size"] = size
        df_t["model"] = model
        df_t["iter"] = iter
        # append
        df_out = pd.concat([df_out, df_t], axis=0)
    # sort and save
    df_out.sort_values(["size", "model", "iter"], inplace=True)
    df_out.to_csv(PATH_OUT, index=False)


if __name__ == "__main__":
    main()
