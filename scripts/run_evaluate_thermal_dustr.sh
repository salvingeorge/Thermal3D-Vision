#!/bin/bash
# run_evaluate_thermal_dustr.sh

# Default values
TEST_DIR="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/test"
MODEL_PATH="thermal_dustr_model.pth"
OUTPUT_DIR="thermal_evaluation_results"
BATCH_SIZE=1
IMG_SIZE="224 224"
FRAME_SKIP=3
DEVICE="cuda"
VISUALIZE=false
CREATE_POINTCLOUD=false
NUM_SAMPLES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --test_dir)
      TEST_DIR="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --img_size)
      IMG_SIZE="$2 $3"
      shift 3
      ;;
    --frame_skip)
      FRAME_SKIP="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --visualize)
      VISUALIZE=true
      shift
      ;;
    --create_pointcloud)
      CREATE_POINTCLOUD=true
      shift
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the command
CMD="python evaluate_thermal_dustr.py --test_dir ${TEST_DIR} --model_path ${MODEL_PATH} --output_dir ${OUTPUT_DIR} --batch_size ${BATCH_SIZE} --img_size ${IMG_SIZE} --frame_skip ${FRAME_SKIP} --device ${DEVICE}"

# Add optional args
if $VISUALIZE; then
  CMD="${CMD} --visualize"
fi

if $CREATE_POINTCLOUD; then
  CMD="${CMD} --create_pointcloud"
fi

if [ ! -z "$NUM_SAMPLES" ]; then
  CMD="${CMD} --num_samples ${NUM_SAMPLES}"
fi

# Print the command
echo "Running: ${CMD}"

# Execute
eval ${CMD}