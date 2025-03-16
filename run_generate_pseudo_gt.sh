#!/bin/bash
# run_generate_pseudo_gt.sh
# Script to generate pseudo-ground truth data using MASt3R

# Default paths - adjust these according to your setup
DATASET_DIR="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg"
OUTPUT_DIR="./pseudo_gt_data"
WEIGHTS_PATH="./checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
CALIB_FILE="./calibrations/t_calib.json"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --weights)
      WEIGHTS_PATH="$2"
      shift 2
      ;;
    --calib_file)
      CALIB_FILE="$2"
      shift 2
      ;;
    --visualize)
      VISUALIZE="--visualize"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Create calibration directory if it doesn't exist
CALIB_DIR=$(dirname "$CALIB_FILE")
mkdir -p "$CALIB_DIR"

# Check if calibration file exists, if not, download it
if [ ! -f "$CALIB_FILE" ]; then
  echo "Calibration file not found, downloading..."
  # If this is t_calib.json
  if [[ "$CALIB_FILE" == *"t_calib.json" ]]; then
    curl -o "$CALIB_FILE" https://raw.githubusercontent.com/jzuern/heatnet-pub/main/data/calibrations/t_calib.json
  # If this is thermal_stereo_calib.yaml
  elif [[ "$CALIB_FILE" == *"thermal_stereo_calib.yaml" ]]; then
    curl -o "$CALIB_FILE" https://raw.githubusercontent.com/jzuern/heatnet-pub/main/data/calibrations/thermal_29_07_19/thermal_stereo_calib.yaml
  fi
fi

# Run the script with the specified parameters
python scripts/pseudo_gt.py \
  --dataset_dir "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --weights "$WEIGHTS_PATH" \
  --calib_file "$CALIB_FILE" \
  --batch_size 1 \
  --img_size 512 512 \
  --frame_skip 5 \
  ${VISUALIZE}

echo "Pseudo-GT generation complete. Results saved to $OUTPUT_DIR"