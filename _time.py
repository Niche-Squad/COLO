import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
import os

# LS_DIR = []
# ROOTS = [
#     os.path.join("/home", "niche", "COLO"),
#     os.path.join("/projects", "niche_squad", "COLO"),]
# for root in ROOTS:
#     DIR_OUT = os.path.join(root, "out", "b0313")
#     LS_DIR += [os.path.join(DIR_OUT, f) for f in os.listdir(DIR_OUT) if "csv" not in f]
# gather all event files

# Path to the TensorBoard log directory
# # took 6 minutes to run
# files = []
# for dir_task in LS_DIR:
#     dir_runs = [os.path.join(dir_task, f) for f in os.listdir(dir_task) if "yolo" in f]
#     files += [os.path.join(dir_run, f) for dir_run in dir_runs for f in os.listdir(dir_run) if "tfevents" in f]
# print(len(files))

# # write paths to csv
# df = pd.DataFrame(files, columns=["path"])
# df.to_csv("event_paths.csv", index=False)


# load
def get_time_from_tf(filename):
    # Initialize an accumulator to load the event file
    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()
    # Assuming that the first and last events correspond to the start and end of training
    first_event = ea.scalars.Keys()[0]
    last_event = ea.scalars.Keys()[-1]

    start_time = ea.Scalars(first_event)[0].wall_time
    end_time = ea.Scalars(last_event)[-1].wall_time

    # Calculate the training time in seconds
    training_time_seconds = end_time - start_time
    return training_time_seconds


def main():
    FILE_OUT = "time_table.csv"
    with open(FILE_OUT, "w") as f:
        f.write("model,n,time,params,location\n")
    files = pd.read_csv("event_paths.csv")

    sizes = dict(
        {
            "yolov8n": 3.2,
            "yolov8m": 25.9,
            "yolov8x": 68.2,
            "yolov9c": 25.3,
            "yolov9e": 57.3,
        }
    )
    for row in files.iterrows():
        filename = row[1][0]
        items = filename.split("/")[-2].split("_")
        # extract
        cate = "home" if "home" in filename else "projects"
        time = get_time_from_tf(filename)
        model = items[0]
        n = items[-2]
        params = sizes[model]
        with open(FILE_OUT, "a") as f:
            f.write(f"{model},{n},{time},{params},{cate}\n")


if __name__ == "__main__":
    main()
