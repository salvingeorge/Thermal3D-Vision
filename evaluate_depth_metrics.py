#!/usr/bin/env python3
# evaluate_thermal_depth.py

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
    
    # Load and preprocess image - use the exact same preprocessing as in thermal_dustr_inference.py
    img = load_and_preprocess_thermal_image(img_path, img_size)
    if img is None:
        return None
    
    # Prepare input in the same format as in thermal_dustr_inference.py
    view1 = {"img": img.unsqueeze(0).to(device), "instance": []}
    
    if monocular:
        view2 = view1  # Use same image for both views
    else:
        view2 = view1
    
    # Run inference
    with torch.no_grad():
        output = model(view1, view2)
    
    # Extract predictions - using same logic as thermal_dustr_inference.py
    if isinstance(output, tuple):
        pred1, pred2 = output
    else:
        pred1 = output.get("pred1", {})
        pred2 = output.get("pred2", {})
    
    # Extract pointmaps using same extraction logic
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

# Add this function to load and preprocess the thermal image in exactly the same way
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
    
    # Normalize based on bit depth - same as training
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
    
def check_pseudo_gt_directories(base_dir):
    """Print information about what files exist in the pseudo-GT directories."""
    print(f"\nExamining contents of pseudo-GT directory: {base_dir}")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory {base_dir} does not exist!")
        return
    
    # List all subdirectories
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found subdirectories: {subdirs}")
    
    # Examine depth directories
    for depth_dir in ['depth1', 'depth2']:
        dir_path = os.path.join(base_dir, depth_dir)
        if not os.path.exists(dir_path):
            print(f"- {depth_dir}: Directory not found!")
            continue
            
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        print(f"- {depth_dir}: Contains {len(files)} .npy files")
        
        if len(files) > 0:
            print(f"  Sample filenames: {files[:3]}")
    
    # Check if we have pair files
    pair_dirs = ['pointmap1', 'pointmap2']
    for pair_dir in pair_dirs:
        dir_path = os.path.join(base_dir, pair_dir)
        if os.path.exists(dir_path):
            pair_files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
            if len(pair_files) > 0:
                print(f"- {pair_dir}: Sample pair files: {pair_files[:3]}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Thermal DUSt3R Model")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to your fine-tuned thermal DUSt3R model")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory of the Freiburg dataset")
    parser.add_argument("--pseudo_gt_dir", type=str, required=True,
                        help="Directory containing pseudo-GT depth maps")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (width height)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of sample images to evaluate")
    parser.add_argument("--sequences", type=str, nargs='+', default=None,
                        help="Specific sequences to evaluate")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dustr_model(args.model, device)
    model.eval()
    
    check_pseudo_gt_directories(args.pseudo_gt_dir)
    
    # Initialize metrics collection
    all_metrics = []
    
    # Find sequences to process
    train_dir = os.path.join(args.dataset_dir, 'train')
    if not os.path.exists(train_dir):
        train_dir = args.dataset_dir  # Use dataset_dir if 'train' doesn't exist
    
    if args.sequences is None:
        sequences = [seq for seq in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, seq))]
    else:
        sequences = args.sequences
    
    print(f"Found {len(sequences)} sequences: {sequences}")
    
    # Collect thermal paths and corresponding RGB paths
    thermal_rgb_pairs = []
    
    for seq_name in sequences:
        seq_dir = os.path.join(train_dir, seq_name)
        if not os.path.isdir(seq_dir):
            continue
            
        # Find all numbered subdirectories
        drive_dirs = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
        
        for drive in drive_dirs:
            drive_path = os.path.join(seq_dir, drive)
            
            # Find thermal images in this drive
            thermal_dir = os.path.join(drive_path, 'fl_ir_aligned')
            if not os.path.isdir(thermal_dir):
                continue
                
            thermal_files = sorted(glob.glob(os.path.join(thermal_dir, '*.png')))
            
            # For each thermal image, get RGB path using the same replacement logic as in training
            for thermal_path in thermal_files:
                rgb_path = thermal_path.replace('fl_ir_aligned', 'fl_rgb').replace('fl_ir_aligned_', 'fl_rgb_')
                
                if os.path.exists(rgb_path):
                    thermal_rgb_pairs.append({
                        'thermal': thermal_path,
                        'rgb': rgb_path,
                        'sequence': seq_name,
                        'drive': drive
                    })
    
    print(f"Found {len(thermal_rgb_pairs)} thermal images with matching RGB images")
    
    # Randomly sample pairs for evaluation
    if len(thermal_rgb_pairs) > args.num_samples:
        np.random.seed(42)  # For reproducibility
        thermal_rgb_pairs = np.random.choice(thermal_rgb_pairs, args.num_samples, replace=False).tolist()
    
    # Process each image pair
    for pair in tqdm(thermal_rgb_pairs, desc="Evaluating"):
        thermal_path = pair['thermal']
        rgb_path = pair['rgb']
        
        # Get base names using EXACTLY the same logic as pseudo_gt.py
        rgb_basename = os.path.splitext(os.path.basename(rgb_path))[0]
        
        gt_depth_found = False
        gt_depth = None
        # First try in depth1 directory
        gt_depth_path1 = os.path.join(args.pseudo_gt_dir, 'depth1', f"{rgb_basename}.npy")
        if os.path.exists(gt_depth_path1):
            gt_depth_path = gt_depth_path1
            gt_depth_found = True
            print(f"Found GT depth in depth1: {gt_depth_path1}")
        else:
            # If not found in depth1, try depth2
            gt_depth_path2 = os.path.join(args.pseudo_gt_dir, 'depth2', f"{rgb_basename}.npy")
            if os.path.exists(gt_depth_path2):
                gt_depth_path = gt_depth_path2
                gt_depth_found = True
                print(f"Found GT depth in depth2: {gt_depth_path2}")
            else:
                # Check if we can find a file with similar name pattern
                # Try more flexible matching by looking for files containing the timestamp part
                # First extract the timestamp part (assuming format like fl_rgb_1579630825_2079809260)
                parts = rgb_basename.split('_')
                if len(parts) >= 3:  # At least has format like fl_rgb_timestamp
                    timestamp_part = '_'.join(parts[2:])  # Get everything after prefix
                    
                    # Search with flexible pattern in both directories
                    for depth_dir in ['depth1', 'depth2']:
                        search_dir = os.path.join(args.pseudo_gt_dir, depth_dir)
                        if not os.path.exists(search_dir):
                            continue
                            
                        for filename in os.listdir(search_dir):
                            if filename.endswith('.npy') and timestamp_part in filename:
                                gt_depth_path = os.path.join(search_dir, filename)
                                gt_depth_found = True
                                print(f"Found GT depth with flexible matching in {depth_dir}: {filename}")
                                break
                        
                        if gt_depth_found:
                            break

        if not gt_depth_found:
            print(f"No GT depth found for {thermal_path} after searching both depth1 and depth2 directories")
            continue

        # Run inference with your model
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
            # (visualization code from previous example)
            vis_path = os.path.join(args.output_dir, f"{rgb_basename}_comparison.png")
            
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
            metrics_path = os.path.join(args.output_dir, f"{rgb_basename}_metrics.txt")
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
        # (same summary code as before)
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