#!/bin/bash
#SBATCH -p dgx_normal_q
#SBATCH --account=niche_squad
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128

cd /home/niche/COLO/yolov9

run=run_$1
file_out=out/$run/results.csv
header="map5095,map50,precision,recall,f1,n_all,n_fn,n_fp,config,model,n"
# check file
if [ ! -f $file_out ]; then
    echo $header > $file_out
fi  

for i in {1..300}
do
    for model in "yolov9-c" "yolov9-e"
    do
        for config in "1a_angle_t2s" "1b_angle_s2t" "2_light" "3_breed" "4_all"
        do
            if [ $config == "3_breed" ]; then
                n_values=(16 32 64 128 250)
            else
                n_values=(16 32 64 128 256 500)
            fi

            for n in "${n_values[@]}"
            do 
                task="${model}_${config}_${n}_${i}"
                dir_out="out/${run}/${task}"
                dir_data="data/${config}/${run}"
                best_model="${dir_out}/weights/best.pt"

                # shuffle data
                python3.9 ../_2a_split.py \
                    --dir_data $dir_data\
                    --n $n
                # training
                python3.9 train_dual.py \
                    --batch 16\
                    --epochs 100\
                    --img 640\
                    --device 0\
                    --min-items 0\
                    --close-mosaic 15\
                    --data "${dir_data}/data.yaml"\
                    --weights "${model}.pt"\
                    --cfg "models/detect/${model}.yaml"\
                    --hyp data/hyps/hyp.scratch-high.yaml\
                    --project .\
                    --name ${dir_out}
                # eval
                python3.9 val_dual.py \
                    --weights ${best_model}\
                    --device 0\
                    --data "${dir_data}/data.yaml"\
                    --task test\
                    --save-txt \
                    --save-conf \
                    --single-cls \
                    --exist-ok \
                    --project .\
                    --name ${dir_out}
                # save results
                python3.9 ../_2b_save_results.py \
                    --dir_data  "${dir_data}"\
                    --dir_preds "${dir_out}/labels"\
                    --file_out $file_out\
                    --config $config\
                    --model $model\
                    --n $n
                # remove weights
                rm -rf "${dir_out}/weights" 
            done
        done
    done

done