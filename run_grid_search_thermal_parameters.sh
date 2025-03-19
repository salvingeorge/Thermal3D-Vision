#!/bin/bash
# grid_search_thermal_parameters.sh - Modified to capture output directly
# Script to perform grid search for thermal-aware loss parameters

# Default paths - adjust these according to your setup
DATASET_DIR="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg"
PSEUDO_GT_DIR="./pseudo_gt_data"
WEIGHTS_PATH="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_224_linear.pth"
OUTPUT_DIR="./checkpoints/grid_search"

# Grid search parameters
EDGE_WEIGHTS=(0.3 0.5 0.7)
SMOOTHNESS_WEIGHTS=(0.1 0.3 0.5)

# Fixed training parameters for grid search
EPOCHS=2  # Short training for grid search
BATCH_SIZE=4
LEARNING_RATE=0.0001

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
    --output_dir)
      OUTPUT_DIR="$2"
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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create a results log file
RESULTS_FILE="$OUTPUT_DIR/grid_search_results.txt"
echo "Edge Weight,Smoothness Weight,Val Loss" > "$RESULTS_FILE"

# Create a summary file for the best parameters
SUMMARY_FILE="$OUTPUT_DIR/best_parameters.txt"

# Run grid search
BEST_VAL_LOSS=9999999
BEST_EDGE_WEIGHT=0
BEST_SMOOTHNESS_WEIGHT=0

echo "Starting grid search for thermal-aware loss parameters..."
echo "Epochs per configuration: $EPOCHS"
echo "Results will be saved to $RESULTS_FILE"

for EDGE_WEIGHT in "${EDGE_WEIGHTS[@]}"; do
  for SMOOTHNESS_WEIGHT in "${SMOOTHNESS_WEIGHTS[@]}"; do
    CONFIG_NAME="edge${EDGE_WEIGHT}_smooth${SMOOTHNESS_WEIGHT}"
    OUTPUT_MODEL="$OUTPUT_DIR/thermal_dustr_${CONFIG_NAME}.pth"
    LOG_FILE="$OUTPUT_DIR/log_${CONFIG_NAME}.txt"
    
    echo "====================================================="
    echo "Running configuration: $CONFIG_NAME"
    echo "Edge weight: $EDGE_WEIGHT, Smoothness weight: $SMOOTHNESS_WEIGHT"
    echo "====================================================="
    
    # Run training with the current parameter configuration and capture the output
    ./run_train_thermal_dustr.sh \
      --pseudo_gt_dir "$PSEUDO_GT_DIR" \
      --weights "$WEIGHTS_PATH" \
      --output_model "$OUTPUT_MODEL" \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LEARNING_RATE \
      --use_thermal_aware_loss \
      --edge_weight $EDGE_WEIGHT \
      --smoothness_weight $SMOOTHNESS_WEIGHT \
      | tee "$LOG_FILE"
    
    # Extract validation loss from the captured output
    VAL_LOSS=$(grep -o 'Average Validation Loss: [0-9.]*' "$LOG_FILE" | tail -1 | awk '{print $4}')
    
    # If validation loss extraction doesn't work, check for alternative format
    if [ -z "$VAL_LOSS" ]; then
      VAL_LOSS=$(grep -o 'val_loss: [0-9.]*' "$LOG_FILE" | tail -1 | awk '{print $2}')
    fi
    
    # If still no match, use default value
    if [ -z "$VAL_LOSS" ]; then
      echo "Couldn't automatically extract validation loss. Please check $LOG_FILE manually."
      VAL_LOSS=9999  # Default high value if extraction fails
    fi
    
    # Save results to the log file
    echo "$EDGE_WEIGHT,$SMOOTHNESS_WEIGHT,$VAL_LOSS" >> "$RESULTS_FILE"
    
    # Update best parameters if this configuration has lower validation loss
    if (( $(echo "$VAL_LOSS < $BEST_VAL_LOSS" | bc -l) )); then
      BEST_VAL_LOSS=$VAL_LOSS
      BEST_EDGE_WEIGHT=$EDGE_WEIGHT
      BEST_SMOOTHNESS_WEIGHT=$SMOOTHNESS_WEIGHT
    fi
    
    echo "Completed configuration: $CONFIG_NAME"
    echo "Validation loss: $VAL_LOSS"
    echo "---------------------------------------------------"
  done
done

# Save the best parameters
echo "Grid search complete!"
echo "Best parameters:" > "$SUMMARY_FILE"
echo "Edge weight: $BEST_EDGE_WEIGHT" >> "$SUMMARY_FILE"
echo "Smoothness weight: $BEST_SMOOTHNESS_WEIGHT" >> "$SUMMARY_FILE"
echo "Validation loss: $BEST_VAL_LOSS" >> "$SUMMARY_FILE"

echo "====================================================="
echo "Grid search complete!"
echo "Best parameters:"
echo "Edge weight: $BEST_EDGE_WEIGHT"
echo "Smoothness weight: $BEST_SMOOTHNESS_WEIGHT"
echo "Validation loss: $BEST_VAL_LOSS"
echo "====================================================="

# Create a final training script with the best parameters
FINAL_SCRIPT="$OUTPUT_DIR/train_with_best_params.sh"
cat > "$FINAL_SCRIPT" << EOF
#!/bin/bash
# Script to train with the best parameters from grid search

./run_train_thermal_dustr.sh \\
  --pseudo_gt_dir "$PSEUDO_GT_DIR" \\
  --weights "$WEIGHTS_PATH" \\
  --output_model "./checkpoints/thermal_dustr_best.pth" \\
  --epochs 20 \\
  --batch_size $BATCH_SIZE \\
  --lr $LEARNING_RATE \\
  --use_thermal_aware_loss \\
  --edge_weight $BEST_EDGE_WEIGHT \\
  --smoothness_weight $BEST_SMOOTHNESS_WEIGHT
EOF

chmod +x "$FINAL_SCRIPT"
echo "Created script with best parameters: $FINAL_SCRIPT"