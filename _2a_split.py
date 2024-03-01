"""
This script is to shuffle the YOLO-formatted dataset given the sample size n
"""
import argparse

from pyniche.trainer import NicheTrainer

def main(args):
    dir_data = args.dir_data
    n = int(args.n)
    
    # trainer will shuffle the data once it is set
    trainer = NicheTrainer()
    trainer.type = "yolo"
    trainer.set_data(
        dataclass=dir_data,
        batch=16,
        n=n,
    )    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_data", help="Directory of the YOLO-formatted dataset")
    parser.add_argument("--n", help="training sample size")
    args = parser.parse_args()
    main(args)
