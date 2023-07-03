import os

from matplotlib import pyplot as plt
import cv2
import time
import re

# variables
YY = time.strftime("%Y")[-2:]
MM = time.strftime("%m")
DD = time.strftime("%d")
NAME_BATCH = "batch_%s%s%s" % (YY, MM, DD)
N_FRAME = 30

# constants
ROOT = os.path.dirname(os.path.dirname(__file__))
DIR_SRC = os.path.join(ROOT, "modules", "ring_capture", "out")
DIR_OUT = os.path.join(ROOT, "data", "cow200", "_images")
DIR_BATCH = os.path.join(DIR_SRC, NAME_BATCH)


# main
def main():
    result = make_dir(DIR_BATCH)
    if not result:
        print("No new videos to process")
        exit()
    rm_incomplete_videos(DIR_SRC)
    mv_videos_to_batch(DIR_SRC, DIR_BATCH)
    n_batch = get_n_batch(DIR_OUT)
    ls_mp4 = ls_all_mp4(DIR_BATCH)
    for i, name_mp4 in enumerate(ls_mp4):
        # define path
        name_img = "img_%d_%d.jpg" % (n_batch, i)
        path_mp4 = os.path.join(DIR_BATCH, name_mp4)
        path_img = os.path.join(DIR_OUT, name_img)
        # read and write video
        frame = get_frame(path_mp4, N_FRAME)
        if frame is None:
            frame = get_frame(path_mp4, int(N_FRAME / 2))
        elif frame is None:
            frame = get_frame(path_mp4, 1)
        try:
            cv2.imwrite(path_img, frame)
        except:
            print("Failed to write %s" % name_mp4)
    print("Wrote %d images to %s" % (len(ls_mp4), DIR_OUT))


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False


def rm_incomplete_videos(dir_src):
    # remove videos that are not complete
    ls_rm = [f for f in os.listdir(dir_src) if f.endswith("_1.mp4")]
    for rm in ls_rm:
        file = os.path.join(dir_src, rm)
        os.remove(file)
    print("Removed %d incomplete videos" % len(ls_rm))


def mv_videos_to_batch(dir_src, dir_batch):
    # move videos to batch folder
    ls_mv = [f for f in os.listdir(dir_src) if f.endswith(".mp4")]
    for mv in ls_mv:
        file_old = os.path.join(dir_src, mv)
        file_new = os.path.join(dir_batch, mv)
        os.rename(file_old, file_new)
    print("Moved %d videos to %s" % (len(ls_mv), dir_batch))


def get_n_batch(dir_out):
    ls_jpgs = [f for f in os.listdir(dir_out) if f.endswith(".jpg")]
    max_batch = max([int(re.findall(r"\d+", f)[0]) for f in ls_jpgs])
    return max_batch + 1


def ls_all_mp4(dir):
    return [f for f in os.listdir(dir) if f.endswith(".mp4")]


def get_frame(path_mp4, n_frame):
    cap = cv2.VideoCapture(path_mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ret, frame = cap.read()
    cap.release()
    return frame


def get_mp4_id(dir):
    """
    Get all mp4 id in a directory
    Example:
        input:  Niche Camera Black_1688414079227_0.mp4
        output: Niche Camera Black_1688414079227
    """
    ls_id = [f[:-6] for f in os.listdir(dir) if f.endswith(".mp4")]
    # drop duplicates
    ls_id = list(set(ls_id))
    # return
    return ls_id


if __name__ == "__main__":
    main()
