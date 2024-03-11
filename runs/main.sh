# this file is just a note for demonstrating the pipeline

# download data
sbatch runs/init_data.sh data

# pre-train models
sbatch runs/pretrain.sh 0_all
sbatch runs/pretrain.sh 1_top
sbatch runs/pretrain.sh 2_side

# generalization test - COCO pre-trained models
for config in "0_all" "a1_t2s" "a2_s2t" "b_light" "c_external"
do
    sbatch runs/yolo_coco.sh $config
done

# generalization test - custom pre-trained models
for config in "1_top" "2_side" "3_external"
do
    sbatch runs/yolo_custom.sh $config
done


