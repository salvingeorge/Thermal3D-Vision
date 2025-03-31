#!/bin/bash
# test_preprocessing.sh

MODEL_PATH="checkpoints/final/thermal_dustr_model_best.pth"
IMAGE_PATH="ThermalImages/siyi_z6t/t_02260.png"
# SECOND_IMAGE_PATH="ThermalImages/flir_boson/s0/s0_t_1.jpg"
OUTPUT_DIR="submission/final/siyi_z6t"

# Base configuration with all enhancements
# python thermal_dustr_inference_edits.py \
#   --checkpoint $MODEL_PATH \
#   --input $IMAGE_PATH \
#   --output "${OUTPUT_DIR}/all_enhancements" \
#   --img_size 224 224 \
#   --preprocess_mode fire

mkdir -p $OUTPUT_DIR

# # Try original preprocessing for comparison
python thermal_dustr_inference_for_experiment.py \
  --checkpoint $MODEL_PATH \
  --input $IMAGE_PATH \
  --output "${OUTPUT_DIR}" \
  --img_size 224 224