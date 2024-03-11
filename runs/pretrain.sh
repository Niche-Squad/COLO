

#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

for model in "yolov9c.pt" "yolov9e.pt" "yolov8n.pt" "yolov8m.pt" "yolov8x.pt"
do
    python3.9 _1_pretrain.py\
        --model $model\
        --config $1
done

