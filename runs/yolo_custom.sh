#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

# home
dir_out=/home/niche/COLO/out/b0313/
dir_data=/home/niche/COLO/data/
# projects
# dir_out=/projects/niche_squad/COLO/out/b0313/
# dir_data=/projects/niche_squad/COLO/data/

for i in {1..300}
do
    if [ "$1" == "1_top" ]; then
        suffix="2_side"
        ls_n=(16 32 64 128 256 500)
    elif [ "$1" == "2_side" ]; then
        suffix="1_top"
        ls_n=(16 32 64 128 256 500)
    elif [ "$1" == "3_external" ]; then
        suffix="0_all"
        ls_n=(16 32 64 128 200)
    fi

    for model in "yolov9c" "yolov9e" "yolov8n" "yolov8m" "yolov8x"
    do
        for n in ${ls_n[@]}
        do
            python3.9 _2_yolo.py\
                --model ${model}.pt\
                --config $1\
                --n $n\
                --dir_out $dir_out\
                --dir_data $dir_data

            python3.9 _2_yolo.py\
                --model ${model}_${suffix}.pt\
                --config $1\
                --n $n\
                --dir_out $dir_out\
                --dir_data $dir_data
        done
    done

done
