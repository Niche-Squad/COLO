import numpy as np
import pandas as pd
import re
import os
import seaborn as sns


FILE_OUT = os.path.join("out", "b0313", "fig5_inference.png")


def find_init(lines):
    s = []
    for line in lines:
        s += [re.search(r"Ultralytics YOLOv8", line)]
    return np.where(s)[0]


def find_model(s):
    find = re.findall(r"YOLOv\d.", s)[0]
    return str(find)


def find_device(s):
    return "CPU" if "CPU" in s else "GPU"


def find_ms(s):
    ms_str = re.findall(r"\d+\.\d+ms", s)[0]
    return float(ms_str[:-2])


lines = open("data/mock/log.txt").readlines()

idx_init = find_init(lines)
idx_model = idx_init + 1
idx_iter_st = idx_init + 3
idx_iter_ed = idx_init + 3 + 64


data = pd.DataFrame(data=None)
for i in range(len(idx_init)):
    str_model = lines[idx_model[i]]
    str_device = lines[idx_init[i]]
    idx_st = idx_iter_st[i]
    idx_ed = idx_iter_ed[i]
    model = find_model(str_model)
    device = find_device(str_device)
    ms = []
    for j in range(idx_st, idx_ed):
        ms += [find_ms(lines[j])]
    data_tmp = pd.DataFrame(data=dict({"model": model, "device": device, "ms": ms}))
    data = pd.concat([data, data_tmp])

# turn model to string
data["fps"] = 1000 / data["ms"]
data["yolo"] = data["model"].apply(lambda x: "v8" if "8" in x else "v9")

df_param = pd.DataFrame(
    data={
        "model": ["YOLOv8n", "YOLOv9c", "YOLOv8m", "YOLOv9e", "YOLOv8x"],
        "size": [3.2, 25.3, 25.9, 57.3, 68.2],
    }
)
data = pd.merge(data, df_param, how="left")


sns.set_style("whitegrid")
g = sns.FacetGrid(
    data,
    col="device",
    col_order=["CPU", "GPU"],
    col_wrap=2,
    margin_titles=True,
    sharey=False,
)
g.map_dataframe(
    sns.lineplot,
    x="size",
    y="fps",
    hue="yolo",
    style="yolo",
    hue_order=["v8", "v9"],
    style_order=["v8", "v9"],
    err_style="band",
    errorbar=("se", 2),
    markers=True,
    palette=["Grey", "#FF1F5B"],
)
g.set(
    ylabel="FPS",
    xlabel="Number of Model Parameters (M)",
)

g.figure.subplots_adjust(right=1.2)
g.add_legend()
g.figure.set_size_inches(8, 4)
g.figure.savefig(FILE_OUT, dpi=300)
