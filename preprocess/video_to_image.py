"""
To use this file:
(1) change DIR_SRC to the folder containing mp4 files
(2) chang the prefix of the output file name
"""

import os

import numpy as np
import cv2
import time

# constants
ROOT = os.path.dirname(os.path.dirname(__file__))

# variables
DIR_SRC = os.path.join(ROOT, "modules", "ring_capture", "out", "batch_top")
TGT_PROGRESS = np.arange(0, 1, 0.2)  # 0, 0.2, 0.4, 0.6, 0.8 progress of all frames
PREFIX = "top"


# main
def main():
    # folder setup and check
    ls_mp4 = ls_all_mp4(DIR_SRC)
    if len(ls_mp4) == 0:
        print("No mp4 files in %s" % DIR_SRC)
        return
    DIR_OUT = make_dir(DIR_SRC)

    # iterate over mp4 files
    for i, name_mp4 in enumerate(ls_mp4):
        path_mp4 = os.path.join(DIR_SRC, name_mp4)
        frames = get_frames(path_mp4, TGT_PROGRESS)
        # iterate over frames and save
        for j, frame in enumerate(frames):
            path_jpg = os.path.join(DIR_OUT, "%s_%d_%d.jpg" % (PREFIX, i, j))
            cv2.imwrite(path_jpg, frame)
            print("Saved %s" % path_jpg)


def ls_all_mp4(dir):
    return [f for f in os.listdir(dir) if f.endswith("_0.mp4")]


def make_dir(path):
    # make output folder named by time yymmdd_hhmmss
    YY = time.strftime("%y")
    MM = time.strftime("%m")
    DD = time.strftime("%d")
    TIME = time.strftime("%H%M%S")
    dir_out = os.path.join(path, "%s%s%s_%s" % (YY, MM, DD, TIME))
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    return dir_out


def get_frames(path_mp4, tgt_progress):
    """
    tgt_progress is a float between 0 and 1
    For example, tgt_progress = 0.5 means the middle frame
    """
    cap = cv2.VideoCapture(path_mp4)
    # get total frames
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get target frame
    frames = []
    for tgt in tgt_progress:
        n_frame = int(n_frames * tgt)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


if __name__ == "__main__":
    main()
