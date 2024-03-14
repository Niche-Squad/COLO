# this file is just a note for demonstrating the pipeline

# download data
sbatch runs/init_data.sh data

# pre-train models
for config in "0_all" "1_top" "2_side"
do
    # sbatch --job-name=pre-${config} runs/pretrain.sh $config
    sh runs/pretrain.sh $config
done

# generalization test - COCO pre-trained models
for config in "0_all" "a1_t2s" "a2_s2t" "b_light" "c_external"
do
    sbatch\
        --job-name=${config}\
        --output=logs/${config}.out\
        --error=logs/${config}.err\
        runs/yolo_coco.sh $config
done

# generalization test - custom pre-trained models
for config in "1_top" "2_side" "3_external"
do
    sbatch\
        --job-name=${config}\
        --output=logs/${config}.out\
        --error=logs/${config}.err\
        runs/yolo_custom.sh $config
done

