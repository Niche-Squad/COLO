#!/bin/bash

# Define the models and devices
models=("yolov8n.pt" "yolov8m.pt" "yolov8x.pt" "yolov9c.pt" "yolov9e.pt")
devices=("cpu" "mps")

# Source directory for input data
source_dir="."

# Function to measure inference time
measure_inference_time() {
  model=$1
  device=$2
  yolo detect predict model=$model device=$device source=$source_dir
}

for model in "${models[@]}"; do
    for device in "${devices[@]}"; do
        measure_inference_time $model $device
    done
done

