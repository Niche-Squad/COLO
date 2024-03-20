
ls_config=("0_all" "1_top" "2_side" "3_external" "a1_t2s" "a2_s2t" "b_light" "c_external")

# rm datasets
# mv data to projects
for config in ${ls_config[@]}
do
    rm -rf data/${config} &
done


# mv data to projects
for config in ${ls_config[@]}
do
    dst_folder=/projects/niche_squad/COLO/data/${config}
    src_folder=/home/niche/COLO/data/${config}
    mkdir -p $dst_folder
    cp -r $src_folder $dst_folder &
done

# mv output to projects
for config in ${ls_config[@]}
do
    dst_folder=/projects/niche_squad/COLO/out/batch_0312/${config}
    mkdir -p $dst_folder
    src_folder=/home/niche/COLO/out/yolo/${config}
    mv $src_folder $dst_folder &
done