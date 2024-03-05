#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

for i in {1..300}
do
    for model in "yolov9c.pt" "yolov9e.pt" "yolov8n.pt" "yolov8m.pt" "yolov8x.pt" 
    do
        for config in "0_all" "1a_angle_t2s" "1b_angle_s2t" "2_light" "3_external"
        do
            for n in 16 32 64 128 256 500
            do 
                python3.9 _1_yolov8.py\
                    --thread $1\
                    --model $model\
                    --config $config\
                    --n $n
            done
        done
    done

done

