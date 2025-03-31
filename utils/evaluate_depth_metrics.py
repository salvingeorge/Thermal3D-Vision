#!/usr/bin/env python3
# evaluate_test_thermal_depth.py

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import glob
from pathlib import Path

# Import functions from your training pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import enhance_thermal_contrast
from thermal_dustr_model import load_dustr_model, ThermalDUSt3R

def compute_depth_metrics(pred_depth, gt_depth, mask=None, median_scaling=True):
    """
    Compute standard metrics for depth estimation evaluation.
    
    Args:
        pred_depth: Predicted depth map [H, W]
        gt_depth: Ground truth depth map [H, W]
        mask: Optional mask for valid depth values [H, W]
        median_scaling: Whether to scale prediction to match GT median
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy for processing
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create mask for valid depth values (non-zero, non-inf, non-nan)
    if mask is None:
        mask = (gt_depth > 0) & np.isfinite(gt_depth) & ~np.isnan(gt_depth)
    
    # Apply mask
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    
    # Skip if no valid pixels
    if pred.size == 0:
        return {
            'rmse': np.nan,
            'acc_1.25': 0.0,
            'acc_1.25^2': 0.0
        }
    
    # Apply median scaling if requested
    if median_scaling:
        scale = np.median(gt) / np.median(pred)
        pred *= scale
    
    # Calculate metrics
    # 1. RMSE
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    
    # 2. Accuracy metrics
    # Calculate max(pred/gt, gt/pred) for each pixel
    thresh = np.maximum((gt / pred), (pred / gt))
    
    # Acc[<1.25]
    acc_1 = (thresh < 1.25).mean()
    
    # Acc[<1.25²]
    acc_2 = (thresh < 1.25 ** 2).mean()
    
    return {
        'rmse': rmse,
        'acc_1.25': acc_1,
        'acc_1.25^2': acc_2
    }

def run_inference(model, img_path, img_size=(224, 224), monocular=True):
    """Run inference on a thermal image to predict depth."""
    device = next(model.parameters()).device
    
    # Load and preprocess image
    img = load_and_preprocess_thermal_image(img_path, img_size)
    if img is None:
        return None
    
    # Prepare input in the same format as in training
    view1 = {"img": img.unsqueeze(0).to(device), "instance": []}
    
    if monocular:
        view2 = view1  # Use same image for both views
    else:
        view2 = view1
    
    # Run inference
    with torch.no_grad():
        output = model(view1, view2)
    
    # Extract predictions
    if isinstance(output, tuple):
        pred1, pred2 = output
    else:
        pred1 = output.get("pred1", {})
        pred2 = output.get("pred2", {})
    
    # Extract pointmaps
    if isinstance(pred1, dict):
        pointmap1 = pred1.get("pts3d")
        if "pts3d_in_other_view" in pred2:
            pointmap2 = pred2.get("pts3d_in_other_view")
        else:
            pointmap2 = pred2.get("pts3d")
        
        confidence1 = pred1.get("conf", None)
        confidence2 = pred2.get("conf", None)
    else:
        pointmap1, pointmap2 = pred1, pred2
        confidence1, confidence2 = None, None
    
    # Move tensors to CPU and handle batch dimension correctly
    pointmap1 = pointmap1.detach().cpu()
    pointmap2 = pointmap2.detach().cpu()
    
    if len(pointmap1.shape) == 4:  # [B, H, W, 3]
        pointmap1 = pointmap1[0]  # [H, W, 3]
    if len(pointmap2.shape) == 4:  # [B, H, W, 3]
        pointmap2 = pointmap2[0]  # [H, W, 3]
    
    # Extract depths (Z coordinate)
    depth1 = pointmap1[:, :, 2]
    depth2 = pointmap2[:, :, 2]
    
    # Process confidence maps if available
    if confidence1 is not None:
        confidence1 = confidence1.detach().cpu()
        if len(confidence1.shape) == 3:  # [B, H, W]
            confidence1 = confidence1[0]
    else:
        confidence1 = torch.ones_like(depth1)
        
    if confidence2 is not None:
        confidence2 = confidence2.detach().cpu()
        if len(confidence2.shape) == 3:  # [B, H, W]
            confidence2 = confidence2[0]
    else:
        confidence2 = torch.ones_like(depth2)
    
    # Return results
    return {
        "pointmap1": pointmap1.numpy(),
        "pointmap2": pointmap2.numpy(),
        "depth1": depth1.numpy(),
        "depth2": depth2.numpy(),
        "confidence1": confidence1.numpy(),
        "confidence2": confidence2.numpy()
    }

def load_and_preprocess_thermal_image(path, img_size=(224, 224)):
    """Load and preprocess a thermal image for inference using same steps as training."""
    if not os.path.exists(path):
        print(f"Error: Image file {path} does not exist")
        return None
    
    # Load thermal image with support for different formats
    thermal_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    if thermal_img is None:
        thermal_img = cv2.imread(path)
        if thermal_img is None:
            print(f"Error: Could not read image {path}")
            return None
        thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
    
    # Normalize based on bit depth
    if thermal_img.dtype == np.uint16:
        thermal_img = thermal_img.astype(np.float32) / 65535.0
    else:
        thermal_img = thermal_img.astype(np.float32) / 255.0
    
    # Convert to 3 channels if grayscale
    if len(thermal_img.shape) == 2:
        thermal_img = np.stack([thermal_img] * 3, axis=-1)
    
    # Resize to target size
    thermal_img = cv2.resize(thermal_img, img_size)
    
    # Convert to torch tensor [C, H, W]
    thermal_img = torch.from_numpy(thermal_img.transpose(2, 0, 1)).float()
    
    # Apply thermal contrast enhancement as in training
    thermal_img = enhance_thermal_contrast(thermal_img)
    
    return thermal_img

def find_matching_depth_file(thermal_path, depth_dir):
    """Find the corresponding depth file for a thermal image path."""
    thermal_basename = os.path.basename(thermal_path)
    thermal_name = os.path.splitext(thermal_basename)[0]  # Remove .png
    
    # Extract the identifier part (timestamp) from thermal filename
    # E.g., from fl_ir_aligned_1570730891_191987444_ir.png get 1570730891_191987444
    parts = thermal_name.split('_')
    if len(parts) < 3:
        return None
        
    timestamp = '_'.join(parts[2:-1])  # Extract parts between prefix and _ir
    
    # Construct potential RGB name - add '0' to the last numeric part
    last_numeric = parts[-2]
    rgb_numeric = last_numeric + '0'
    rgb_basename = f"fl_ir_aligned_{timestamp}_{rgb_numeric}_rgb"
    
    # First try direct matching
    depth_file = os.path.join(depth_dir, f"{rgb_basename}_depth.npy")
    if os.path.exists(depth_file):
        return depth_file
    
    # Try glob pattern matching
    pattern = os.path.join(depth_dir, f"*{timestamp}*_depth.npy")
    matching_files = glob.glob(pattern)
    
    if matching_files:
        return matching_files[0]
    
    # Try more flexible matching - just check every depth file
    for filename in os.listdir(depth_dir):
        if not filename.endswith('_depth.npy'):
            continue
            
        # Extract timestamp from depth file
        parts = filename.split('_')
        if len(parts) < 3:
            continue
            
        file_timestamp = '_'.join(parts[2:4])  # Get the timestamp portions
        
        # Check if timestamps are similar
        if timestamp in file_timestamp or file_timestamp in timestamp:
            return os.path.join(depth_dir, filename)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate Thermal DUSt3R Model on Test Dataset")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to your fine-tuned thermal DUSt3R model")
    parser.add_argument("--thermal_dir", type=str, default="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg/test/night/ImagesIR",
                        help="Directory with test thermal images")
    parser.add_argument("--pseudo_gt_dir", type=str, default="pseudo_gt_test_set/depth",
                        help="Directory containing pseudo-GT depth maps")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (width height)")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of sample images to evaluate (0=all)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dustr_model(args.model, device)
    model.eval()
    
    # Check if pseudo-GT directory exists
    if not os.path.exists(args.pseudo_gt_dir):
        print(f"ERROR: Pseudo-GT directory {args.pseudo_gt_dir} does not exist!")
        return
        
    depth_files = [f for f in os.listdir(args.pseudo_gt_dir) if f.endswith('_depth.npy')]
    print(f"Found {len(depth_files)} depth files in pseudo-GT directory")
    if depth_files:
        print(f"Sample depth files: {depth_files[:3]}")
    
    # Get all thermal images in the test dir
    thermal_files = sorted(glob.glob(os.path.join(args.thermal_dir, '*.png')))
    print(f"Found {len(thermal_files)} thermal images in test directory")
    
    # Limit samples if requested
    if args.num_samples > 0 and args.num_samples < len(thermal_files):
        np.random.seed(42)  # For reproducibility
        thermal_files = np.random.choice(thermal_files, args.num_samples, replace=False).tolist()
    
    # Initialize metrics collection
    all_metrics = []
    
    # Process each thermal image
    for thermal_path in tqdm(thermal_files, desc="Evaluating"):
        # Find matching depth file
        gt_depth_path = find_matching_depth_file(thermal_path, args.pseudo_gt_dir)
        
        if not gt_depth_path:
            print(f"No matching depth file found for {os.path.basename(thermal_path)}")
            continue
        
        # Run inference with model
        results = run_inference(
            model=model,
            img_path=thermal_path,
            img_size=tuple(args.img_size),
            monocular=True
        )
        
        if results is None:
            print(f"Inference failed for {thermal_path}")
            continue
            
        # Load GT depth
        try:
            gt_depth = np.load(gt_depth_path)
            pred_depth = results["depth1"]
            
            # Ensure GT and prediction have the same shape
            if gt_depth.shape != pred_depth.shape:
                gt_depth = cv2.resize(gt_depth, 
                                     (pred_depth.shape[1], pred_depth.shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Compute metrics
            metrics = compute_depth_metrics(pred_depth, gt_depth, median_scaling=True)
            all_metrics.append(metrics)
            
            # Create visualization
            thermal_basename = os.path.splitext(os.path.basename(thermal_path))[0]
            vis_path = os.path.join(args.output_dir, f"{thermal_basename}_comparison.png")
            
            plt.figure(figsize=(15, 5))
            
            # Load and show thermal image
            thermal_img = cv2.imread(thermal_path)
            if thermal_img is not None:
                if len(thermal_img.shape) == 2:  # Grayscale
                    plt.subplot(1, 3, 1)
                    plt.imshow(thermal_img, cmap='hot')
                else:  # RGB
                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB))
            else:
                plt.subplot(1, 3, 1)
                plt.text(0.5, 0.5, "Image not available", 
                         horizontalalignment='center', verticalalignment='center')
            plt.title("Thermal Input")
            plt.axis('off')
            
            # Show predicted depth
            plt.subplot(1, 3, 2)
            plt.imshow(pred_depth, cmap='plasma')
            plt.title(f"Predicted Depth\nRMSE: {metrics['rmse']:.4f}")
            plt.axis('off')
            
            # Show GT depth
            plt.subplot(1, 3, 3)
            plt.imshow(gt_depth, cmap='plasma')
            plt.title("Pseudo-GT Depth")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(vis_path, dpi=200)
            plt.close()
            
            # Write metrics to individual file
            metrics_path = os.path.join(args.output_dir, f"{thermal_basename}_metrics.txt")
            with open(metrics_path, 'w') as f:
                f.write(f"RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"Acc[<1.25]: {metrics['acc_1.25']:.4f}\n")
                f.write(f"Acc[<1.25^2]: {metrics['acc_1.25^2']:.4f}\n")
                
        except Exception as e:
            print(f"Error processing {thermal_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate and print average metrics
    if all_metrics:
        avg_rmse = np.mean([m['rmse'] for m in all_metrics if not np.isnan(m['rmse'])])
        avg_acc_1 = np.mean([m['acc_1.25'] for m in all_metrics])
        avg_acc_2 = np.mean([m['acc_1.25^2'] for m in all_metrics])
        
        print("\nAverage metrics:")
        print(f"RMSE: {avg_rmse:.4f}")
        print(f"Acc[<1.25]: {avg_acc_1:.4f}")
        print(f"Acc[<1.25^2]: {avg_acc_2:.4f}")
        
        # Save summary metrics
        summary_path = os.path.join(args.output_dir, "metrics_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Number of images evaluated: {len(all_metrics)}\n")
            f.write(f"Average RMSE: {avg_rmse:.4f}\n")
            f.write(f"Average Acc[<1.25]: {avg_acc_1:.4f}\n")
            f.write(f"Average Acc[<1.25^2]: {avg_acc_2:.4f}\n")
    else:
        print("No valid metrics computed")

if __name__ == "__main__":
    main()