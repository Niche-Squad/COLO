#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=48:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=out.log
#SBATCH --error=out.err

# module load site/tinkercliffs-rome_a100/easybuild/setup
# module load Anaconda3/2020.11
# source activate tf

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

for i in {1..500}
do
    echo "Iteration $i"
    python3.9 /home/niche/cowsformer/test.py --iter $i
    sleep 3
done