#!/bin/bash
# run_train_thermal_dustr.sh
# Script to fine-tune DUSt3R on thermal images with pseudo-GT

# Default paths - adjust these according to your setup
DATASET_DIR="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg"
PSEUDO_GT_DIR="./pseudo_gt_data"
WEIGHTS_PATH="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
OUTPUT_MODEL="./checkpoints/thermal_dustr_model.pth"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --pseudo_gt_dir)
      PSEUDO_GT_DIR="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    --output_model)
      OUTPUT_MODEL="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --lr)
      LEARNING_RATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Set default values if not provided
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-0.0001}

# Create checkpoint directory if it doesn't exist
mkdir -p $(dirname "$OUTPUT_MODEL")

# Run the training script
python train_thermal_dustr.py \
  --dataset_dir "$DATASET_DIR" \
  --pseudo_gt_dir "$PSEUDO_GT_DIR" \
  --weights "$WEIGHTS_PATH" \
  --output_model "$OUTPUT_MODEL" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LEARNING_RATE \
  --img_size 224 224 \
  --frame_skip 3 \
  --device "cuda" \
  --log_interval 100

echo "Training complete. Model saved to $OUTPUT_MODEL"