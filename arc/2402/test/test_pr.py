import numpy as np
from pathlib import Path

tp = np.array(
    [True, False, True, False, True, False, True, False, True, False],
    [True, False, True, False, True, False, True, False, True, False],
)
conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.8, 0.7, 0.6])
pred_cls = np.array([0, 0, 0, 1, 0, 0, 1, 0, 1, 1])
target_cls = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0])
plot = False
on_plot = None
save_dir = Path()
names = ()
eps = 1e-16
prefix = ""

# def ap_per_class(
#     tp,
#     conf,
#     pred_cls,
#     target_cls,
#     plot=False,
#     on_plot=None,
#     save_dir=Path(),
#     names=(),
#     eps=1e-16,
#     prefix="",
# ):
"""
Computes the average precision per class for object detection evaluation.

Args:
    tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
    conf (np.ndarray): Array of confidence scores of the detections.
    pred_cls (np.ndarray): Array of predicted classes of the detections.
    target_cls (np.ndarray): Array of true classes of the detections.
    plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
    on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
    save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
    names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
    eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
    prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

Returns:
    (tuple): A tuple of six arrays and one array of unique classes, where:
        tp (np.ndarray): True positive counts for each class.
        fp (np.ndarray): False positive counts for each class.
        p (np.ndarray): Precision values at each confidence threshold.
        r (np.ndarray): Recall values at each confidence threshold.
        f1 (np.ndarray): F1-score values at each confidence threshold.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.

"""

# Sort by objectness
i = np.argsort(-conf)
tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

# Find unique classes
unique_classes, nt = np.unique(target_cls, return_counts=True)
nc = unique_classes.shape[0]  # number of classes, number of detections

# Create Precision-Recall curve and compute AP for each class
px, py = np.linspace(0, 1, 1000), []  # for plotting
ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
for ci, c in enumerate(unique_classes):
    i = pred_cls == c
    n_l = nt[ci]  # number of labels
    n_p = i.sum()  # number of predictions
    if n_p == 0 or n_l == 0:
        continue

    # Accumulate FPs and TPs
    fpc = (1 - tp[i]).cumsum(0)
    tpc = tp[i].cumsum(0)

    # Recall
    recall = tpc / (n_l + eps)  # recall curve
    r[ci] = np.interp(
        -px, -conf[i], recall[:, 0], left=0
    )  # negative x, xp because xp decreases

    # Precision
    precision = tpc / (tpc + fpc)  # precision curve
    p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

    # AP from recall-precision curve
    for j in range(tp.shape[1]):
        ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if plot and j == 0:
            py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

# Compute F1 (harmonic mean of precision and recall)
f1 = 2 * p * r / (p + r + eps)
names = [
    v for k, v in names.items() if k in unique_classes
]  # list: only classes that have data
names = dict(enumerate(names))  # to dict
if plot:
    plot_pr_curve(
        px, py, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot
    )
    plot_mc_curve(
        px,
        f1,
        save_dir / f"{prefix}F1_curve.png",
        names,
        ylabel="F1",
        on_plot=on_plot,
    )
    plot_mc_curve(
        px,
        p,
        save_dir / f"{prefix}P_curve.png",
        names,
        ylabel="Precision",
        on_plot=on_plot,
    )
    plot_mc_curve(
        px,
        r,
        save_dir / f"{prefix}R_curve.png",
        names,
        ylabel="Recall",
        on_plot=on_plot,
    )

i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
p, r, f1 = p[:, i], r[:, i], f1[:, i]
tp = (r * nt).round()  # true positives
fp = (tp / (p + eps) - tp).round()  # false positives
