#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=48:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G


export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

for i in {1..300}
do
    for n_train in 10 25 50 100 200
    do
        for yolo_base in "yolov8n.pt" "yolov8m.pt" "yolov8x.pt"
        do
            echo "Iteration $i, n_train $n_train, model $yolo_base, suffix $suffix"
            python3.9 trial_0.py \
                --iter $i\
                --n_train $n_train\
                --yolo_base $yolo_base\
                --suffix $suffix
            sleep 1
        done
    done
done