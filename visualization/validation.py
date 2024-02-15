import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
PATH_FILE = os.path.join(ROOT, "out", "result.csv")
PATH_PNG1 =  os.path.join(ROOT, "out", "result_mAP.png")
PATH_PNG2 =  os.path.join(ROOT, "out", "result_n.png")

def main():
    data = pd.read_csv(PATH_FILE)
    data = data.melt(id_vars=["size", "model", "iter"],
            var_name="metrics").query("metrics != 'n_all'")

    # boxplot
    sns.set_theme(palette="Set2",)
    # figure 1
    plt.figure(figsize=(10, 10))
    data_1 = data.query("metrics in ['precision', 'recall', 'map50', 'map5095']")
    sns.relplot(x="size", 
                y="value", 
                kind="line",
                hue="model",
                hue_order=["yolov8n", "yolov8m", "yolov8x"],
                col="metrics",
                col_order=["precision", "recall", "map50", "map5095"],
                col_wrap=2,
                marker="o",
                errorbar=("se", 2),
                # facet_kws=dict(sharey=False),
                data=data_1)

    # save figure
    plt.savefig(PATH_PNG1, dpi=300)

    # figure 2
    data_2 = data.query("metrics in ['n_missed', 'n_false']")
    sns.boxplot(
            x="metrics", 
            y="value", 
            hue="model",
            hue_order=["yolov8n", "yolov8m", "yolov8x"],
            data=data_2.query("size == 200"))
    # save figure
    plt.savefig(PATH_PNG2, dpi=300)

    # summary
    data.groupby(["size", "model", "metrics"]).\
        median().\
            reset_index().\
                sort_values(["metrics", "size", "model"]).\
                    to_csv(os.path.join(ROOT, "out", "result_median.csv"), index=False)




if __name__ == "__main__":
    main()