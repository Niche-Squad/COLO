import argparse
from pyniche.data.yolo.API import YOLO_API
from pyniche.evaluate import from_sv


def main(args):
    dir_data = args.dir_data
    dir_preds = args.dir_preds
    file_out = args.file_out

    api = YOLO_API(dir_data)
    lbs = api.get_detections("test")
    pre = api.get_detections("test", path_preds=dir_preds)
    out = from_sv(pre, lbs)
    out["config"] = args.config
    out["model"] = args.model
    out["n"] = args.n
    line = ",".join([str(value) for value in out.values()])
    with open(file_out, "a") as file:
        file.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data")
    parser.add_argument("--dir_preds")
    parser.add_argument("--file_out")
    parser.add_argument("--config")
    parser.add_argument("--model")
    parser.add_argument("--n")
    args = parser.parse_args()
    main(args)
