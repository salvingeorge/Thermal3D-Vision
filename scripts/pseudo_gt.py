#!/usr/bin/env python3
# Copyright (C) 2024-present
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
import glob

def load_mast3r_model(weights_path="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", device=None):
    """Load MASt3R model for generating pseudo-annotations."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        mast3r_path = os.path.abspath("mast3r")
        if mast3r_path not in sys.path:
            sys.path.append(mast3r_path)
        from mast3r.model import AsymmetricMASt3R
        dust3r_path = os.path.join(mast3r_path, "dust3r")
        if dust3r_path not in sys.path:
            sys.path.append(dust3r_path)
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        model.eval()
        return model
    except ImportError as e:
        print(f"Could not import MASt3R: {e}")
        sys.exit(1)

def skip_none_collate(batch):
    """Custom collate function that filters out any None samples."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None  # or you can raise an exception
    return default_collate(batch)

def flatten_tensor(x):
    """
    Recursively extract the first element until x is not a tuple/list,
    then ensure it's a torch.Tensor.
    """
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            break
        x = x[0]
    if not isinstance(x, torch.Tensor):
        try:
            x = torch.tensor(x)
        except Exception as e:
            raise ValueError("Cannot convert value to tensor: " + str(e))
    return x

def extract_pointmaps_from_output(output):
    """
    Extract pointmaps, depths, and confidences from MASt3R output.
    """
    # Extract predictions
    if isinstance(output, tuple):
        pred1, pred2 = output[0], output[1]
    else:
        pred1 = output.get("pred1", {})
        pred2 = output.get("pred2", {})
    
    # Check keys
    print("Pred1 keys:", list(pred1.keys()) if isinstance(pred1, dict) else "Not a dict")
    print("Pred2 keys:", list(pred2.keys()) if isinstance(pred2, dict) else "Not a dict")
    
    # Extract pointmaps
    if isinstance(pred1, dict):
        # Get pointmap1
        if "pts3d" in pred1:
            pointmap1 = pred1["pts3d"]
        else:
            raise KeyError(f"No 'pts3d' found in pred1: {list(pred1.keys())}")
        
        # Get pointmap2 - use 'pts3d_in_other_view' if available
        if "pts3d_in_other_view" in pred2:
            pointmap2 = pred2["pts3d_in_other_view"]
        elif "pts3d" in pred2:
            pointmap2 = pred2["pts3d"]
        else:
            raise KeyError(f"No pointmap key found in pred2: {list(pred2.keys())}")
        
        # Extract confidences
        confidence1 = pred1.get("conf", None)
        confidence2 = pred2.get("conf", None)
    else:
        # Fallback for non-dict predictions
        pointmap1 = pred1
        pointmap2 = pred2
        confidence1 = None
        confidence2 = None
    
    # Ensure proper tensor format
    pointmap1 = flatten_tensor(pointmap1)
    pointmap2 = flatten_tensor(pointmap2)
    
    # If batch dimension is present, take the first item
    if pointmap1.dim() == 4:  # [B, H, W, 3]
        pointmap1 = pointmap1[0]
    if pointmap2.dim() == 4:
        pointmap2 = pointmap2[0]
    
    # Convert to numpy
    pointmap1_np = pointmap1.detach().cpu().numpy()  # [H, W, 3]
    pointmap2_np = pointmap2.detach().cpu().numpy()  # [H, W, 3]
    
    # Extract depths (Z coordinates)
    depth1 = pointmap1_np[:, :, 2]
    depth2 = pointmap2_np[:, :, 2]
    
    # Process confidences if available
    if confidence1 is not None:
        confidence1 = flatten_tensor(confidence1)
        if confidence1.dim() == 3:  # [B, H, W]
            confidence1 = confidence1[0]
        confidence1_np = confidence1.detach().cpu().numpy()
    else:
        confidence1_np = np.ones_like(depth1)
        
    if confidence2 is not None:
        confidence2 = flatten_tensor(confidence2)
        if confidence2.dim() == 3:  # [B, H, W]
            confidence2 = confidence2[0]
        confidence2_np = confidence2.detach().cpu().numpy()
    else:
        confidence2_np = np.ones_like(depth2)
    
    return pointmap1_np, pointmap2_np, confidence1_np, confidence2_np, depth1, depth2

def estimate_camera_intrinsics(pointmap, depth, calib_path=None):
    """
    Estimate camera intrinsics from pointmap and depth or load from calibration file.
    """
    if calib_path and os.path.exists(calib_path):
        try:
            # Try to load calibration from file
            K, _, _ = load_thermal_calibration(calib_path)
            print(f"Loaded camera intrinsics from {calib_path}")
            return K
        except Exception as e:
            print(f"Error loading calibration: {e}, falling back to estimation")
    
    # Fall back to estimation if loading fails or no path provided
    H, W = depth.shape
    
    # Generate pixel coordinates
    v, u = np.indices((H, W))
    
    # Normalize 3D coordinates by depth
    X = pointmap[:, :, 0]
    Y = pointmap[:, :, 1]
    Z = depth
    
    # Filter valid points (non-zero depth)
    mask = Z > 0
    u_valid = u[mask]
    v_valid = v[mask]
    X_valid = X[mask]
    Y_valid = Y[mask]
    Z_valid = Z[mask]
    
    # Compute normalized coordinates
    X_norm = X_valid / Z_valid
    Y_norm = Y_valid / Z_valid
    
    # Estimate focal lengths using least squares
    fx_est = np.median((u_valid - W/2) / X_norm)
    fy_est = np.median((v_valid - H/2) / Y_norm)
    
    # Create intrinsics matrix
    K = np.array([
        [fx_est, 0,      W/2],
        [0,      fy_est, H/2],
        [0,      0,      1]
    ])
    
    return K

def extract_relative_pose(pointmap1, pointmap2):
    """
    Extract relative pose between two pointmaps using Umeyama alignment.
    """
    # Select valid points from both pointmaps
    mask1 = pointmap1[:,:,2] > 0
    mask2 = pointmap2[:,:,2] > 0
    mask = mask1 & mask2
    
    # If not enough valid points, return identity
    if np.sum(mask) < 10:
        print("Warning: Not enough valid points for pose estimation")
        return np.eye(4)
    
    # Extract point sets
    pts1 = pointmap1[mask]  # Source points
    pts2 = pointmap2[mask]  # Target points
    
    # Randomly sample points if too many (for efficiency)
    max_points = 1000
    if pts1.shape[0] > max_points:
        indices = np.random.choice(pts1.shape[0], max_points, replace=False)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
    
    # Reshape to format expected by umeyama_alignment
    # umeyama expects mxn where m=dimension (3) and n=nr of points
    pts1 = pts1.T  # Now 3 x n
    pts2 = pts2.T  # Now 3 x n
    
    try:
        # Estimate transformation using Umeyama
        r, t, scale = umeyama_alignment(pts1, pts2, with_scale=False)
        
        # Build 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = r
        transform[:3, 3] = t
        
        print(f"Computed relative pose with rotation: {r} and translation: {t}")
        return transform
    except GeometryException as e:
        print(f"Error in pose estimation: {e}, returning identity")
        return np.eye(4)


def load_thermal_calibration(calib_path):
    """Load thermal camera calibration parameters."""
    if calib_path.endswith('.json'):
        import json
        with open(calib_path, 'r') as f:
            calib = json.load(f)
            # Parse JSON calibration format
            intrinsics = calib["intrinsic"]
            fx, fy, cx, cy = intrinsics
            
            # Create intrinsics matrix
            K = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ])
            
            # Get rotation and translation if needed
            R = np.array(calib["rotation"])
            t = np.array(calib["translation"])
            
            return K, R, t
            
    elif calib_path.endswith('.yaml'):
        import yaml
        with open(calib_path, 'r') as f:
            calib = yaml.safe_load(f)
            
            # Parse YAML calibration format
            left_intrinsics = calib["left"]["intrinsics"]
            fx, fy, cx, cy = left_intrinsics
            
            # Create intrinsics matrix
            K_left = np.array([
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,  1]
            ])
            
            # Get right camera parameters if needed
            if "right" in calib:
                right_intrinsics = calib["right"]["intrinsics"]
                fx_r, fy_r, cx_r, cy_r = right_intrinsics
                
                K_right = np.array([
                    [fx_r, 0,    cx_r],
                    [0,    fy_r, cy_r],
                    [0,    0,    1]
                ])
                
                # Transform from right to left camera
                T_right_left = np.array(calib["right"]["T_cn_cnm1"])
                
                return K_left, K_right, T_right_left
            
            return K_left, None, None
    else:
        raise ValueError(f"Unsupported calibration file format: {calib_path}")
    
class GeometryException(Exception):
    """Exception for geometry-related errors."""
    pass

def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: False)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise GeometryException("Data matrices must have the same shape")
    
    # m = dimension, n = nr. of data points
    m, n = x.shape
    
    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)
    
    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)
    
    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)
    
    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < m - 1:
        raise GeometryException("Degenerate covariance rank, Umeyama alignment is not possible")
    
    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1
    
    # rotation, eq. 40
    r = u.dot(s).dot(v)
    
    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))
    
    return r, t, c

def visualize_data(rgb_image1, rgb_image2, depth1, depth2, save_path=None):
    """Visualize RGB images and their corresponding depth maps."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image1)
    plt.title('RGB Image 1')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image2)
    plt.title('RGB Image 2')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.imshow(depth1, cmap='plasma')
    plt.title('Depth Map 1')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(depth2, cmap='plasma')
    plt.title('Depth Map 2')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

class RGBPairDataset(torch.utils.data.Dataset):
    """Dataset for loading pairs of RGB images from a video sequence."""
    
    def __init__(self, root_dir, sequences=None, img_size=(512, 512), frame_skip=5):
        """
        Args:
            root_dir: Root directory of the Freiburg dataset
            sequences: List of sequence names to include (e.g., ['seq_00_day'])
                      If None, all available sequences will be used
            img_size: Target image size (width, height)
            frame_skip: Number of frames to skip between pairs (controls baseline width)
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.frame_skip = frame_skip
        
        self.pairs = []
        
        # Check if 'train' directory exists, otherwise use root_dir directly
        train_dir = os.path.join(root_dir, 'train')
        if not os.path.isdir(train_dir):
            train_dir = root_dir  # Use root_dir if 'train' doesn't exist
        
        # Find all sequences if not specified
        if sequences is None:
            sequences = []
            for item in os.listdir(train_dir):
                item_path = os.path.join(train_dir, item)
                if os.path.isdir(item_path) and (
                    "seq" in item.lower() or  # Sequence directory
                    os.path.exists(os.path.join(item_path, "fl_rgb")) or  # Contains RGB directory
                    any("rgb" in f.lower() for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f)))  # Contains RGB files
                ):
                    sequences.append(item)
        
        print(f"Found {len(sequences)} sequences: {sequences}")
        
        # For each sequence, find RGB frame pairs
        for seq_name in sequences:
            seq_dir = os.path.join(train_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
                
            # Find all numbered subdirectories
            drive_dirs = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
            
            for drive in drive_dirs:
                drive_path = os.path.join(seq_dir, drive)
                
                # Find RGB images in this drive
                rgb_dir = os.path.join(drive_path, 'fl_rgb')
                if os.path.isdir(rgb_dir):
                    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
                else:
                    # Try alternative path patterns if needed
                    rgb_files = []
                    for subdir in os.listdir(drive_path):
                        subdir_path = os.path.join(drive_path, subdir)
                        if os.path.isdir(subdir_path):
                            rgb_files.extend(sorted(glob.glob(os.path.join(subdir_path, '*rgb*.png'))))
                
                # Create pairs with frame_skip
                for i in range(len(rgb_files) - frame_skip):
                    rgb_path1 = rgb_files[i]
                    rgb_path2 = rgb_files[i + frame_skip]
                    
                    # Also find corresponding thermal paths for later use
                    thermal_path1 = rgb_path1.replace('fl_rgb', 'fl_ir_aligned').replace('rgb', 'ir')
                    thermal_path2 = rgb_path2.replace('fl_rgb', 'fl_ir_aligned').replace('rgb', 'ir')
                    
                    if os.path.exists(thermal_path1) and os.path.exists(thermal_path2):
                        self.pairs.append({
                            'rgb_path1': rgb_path1,
                            'rgb_path2': rgb_path2,
                            'thermal_path1': thermal_path1,
                            'thermal_path2': thermal_path2,
                            'sequence': seq_name,
                            'drive': drive,
                            'frame_idx1': i,
                            'frame_idx2': i + frame_skip
                        })
        
        print(f"Created {len(self.pairs)} RGB image pairs across {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load RGB images
        rgb_path1 = pair['rgb_path1']
        rgb_path2 = pair['rgb_path2']
        
        rgb_img1 = cv2.imread(rgb_path1)
        rgb_img2 = cv2.imread(rgb_path2)
        
        if rgb_img1 is None or rgb_img2 is None:
            print(f"Warning: Could not read RGB files: {rgb_path1} or {rgb_path2}, skipping.")
            return None
        
        # Convert BGR to RGB
        rgb_img1 = cv2.cvtColor(rgb_img1, cv2.COLOR_BGR2RGB)
        rgb_img2 = cv2.cvtColor(rgb_img2, cv2.COLOR_BGR2RGB)
        
        # Resize
        rgb_img1 = cv2.resize(rgb_img1, self.img_size)
        rgb_img2 = cv2.resize(rgb_img2, self.img_size)
        
        # Normalize
        rgb_img1 = rgb_img1.astype(np.float32) / 255.0
        rgb_img2 = rgb_img2.astype(np.float32) / 255.0
        
        # Convert to torch tensors [C, H, W]
        rgb_img1 = torch.from_numpy(rgb_img1.transpose(2, 0, 1)).float()
        rgb_img2 = torch.from_numpy(rgb_img2.transpose(2, 0, 1)).float()
        
        return {
            'rgb1': rgb_img1,
            'rgb2': rgb_img2,
            'rgb_path1': rgb_path1,
            'rgb_path2': rgb_path2,
            'thermal_path1': pair['thermal_path1'],
            'thermal_path2': pair['thermal_path2'],
            'sequence': pair['sequence'],
            'drive': pair['drive']
        }

def generate_pseudo_gt(dataset, output_dir, model, device, batch_size=1, visualize=False, calib_file=None):
    """
    Generate pseudo-GT from pairs of RGB images using MASt3R.
    
    Args:
        dataset: Dataset providing RGB image pairs
        output_dir: Directory to save the pseudo-GT
        model: Loaded MASt3R model
        device: Device to run inference on
        batch_size: Batch size for processing
        visualize: Whether to visualize a few samples
    """
    # Create output directories
    pointmap1_dir = os.path.join(output_dir, 'pointmap1')
    pointmap2_dir = os.path.join(output_dir, 'pointmap2')
    confidence1_dir = os.path.join(output_dir, 'confidence1')
    confidence2_dir = os.path.join(output_dir, 'confidence2')
    depth1_dir = os.path.join(output_dir, 'depth1')
    depth2_dir = os.path.join(output_dir, 'depth2')
    intrinsics_dir = os.path.join(output_dir, 'intrinsics')
    poses_dir = os.path.join(output_dir, 'poses')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    for d in [pointmap1_dir, pointmap2_dir, confidence1_dir, confidence2_dir, 
              depth1_dir, depth2_dir, intrinsics_dir, poses_dir]:
        os.makedirs(d, exist_ok=True)
    
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=skip_none_collate,
        drop_last=False
    )
    
    n_processed = 0
    model.eval()
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating pseudo-GT")):
        if batch is None or len(batch) == 0:
            continue
        
        rgb1 = batch['rgb1'].to(device)
        rgb2 = batch['rgb2'].to(device)
        rgb_path1 = batch['rgb_path1']
        rgb_path2 = batch['rgb_path2']
        
        try:
            # Process each pair individually for stability
            for i in range(rgb1.size(0)):
                # Create input views
                view1 = {"img": rgb1[i].unsqueeze(0), "instance": []}
                view2 = {"img": rgb2[i].unsqueeze(0), "instance": []}
                
                # Forward pass through MASt3R
                with torch.no_grad():
                    output = model(view1, view2)
                
                # Extract base names for saving files
                base_name1 = os.path.splitext(os.path.basename(rgb_path1[i]))[0]
                base_name2 = os.path.splitext(os.path.basename(rgb_path2[i]))[0]
                pair_name = f"{base_name1}_{base_name2}"
                
                # Extract data from output
                try:
                    pointmap1, pointmap2, confidence1, confidence2, depth1, depth2 = extract_pointmaps_from_output(output)
                
                    # Estimate camera intrinsics
                    K = estimate_camera_intrinsics(pointmap1, depth1, calib_file)
                    
                    # Extract relative pose (from camera2 to camera1)
                    pose = extract_relative_pose(pointmap1, pointmap2)
                    
                    # Save outputs
                    np.save(os.path.join(pointmap1_dir, f"{pair_name}.npy"), pointmap1)
                    np.save(os.path.join(pointmap2_dir, f"{pair_name}.npy"), pointmap2)
                    np.save(os.path.join(confidence1_dir, f"{pair_name}.npy"), confidence1)
                    np.save(os.path.join(confidence2_dir, f"{pair_name}.npy"), confidence2)
                    np.save(os.path.join(depth1_dir, f"{base_name1}.npy"), depth1)
                    np.save(os.path.join(depth2_dir, f"{base_name2}.npy"), depth2)
                    np.save(os.path.join(intrinsics_dir, f"{pair_name}.npy"), K)
                    np.save(os.path.join(poses_dir, f"{pair_name}.npy"), pose)
                    
                    # Visualize a few samples
                    if visualize and n_processed < 10:
                        rgb_img1 = rgb1[i].permute(1, 2, 0).cpu().numpy()
                        rgb_img2 = rgb2[i].permute(1, 2, 0).cpu().numpy()
                        
                        vis_path = os.path.join(vis_dir, f"{pair_name}.png")
                        visualize_data(rgb_img1, rgb_img2, depth1, depth2, save_path=vis_path)
                        n_processed += 1
                        
                except Exception as e:
                    print(f"Error processing output for {pair_name}: {e}")
                    continue
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    print(f"Pseudo-GT generation complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-GT for Freiburg dataset using MASt3R")
    parser.add_argument('--dataset_dir', type=str, default="/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg", help="Path to the Freiburg dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the pseudo-GT")
    parser.add_argument('--weights', type=str, default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                        help="Path to the MASt3R model weights")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing")
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512], help="Image size (width height)")
    parser.add_argument('--frame_skip', type=int, default=5, help="Number of frames to skip between pairs")
    parser.add_argument('--visualize', action='store_true', help="Visualize a few samples for verification")
    parser.add_argument('--calib_file', type=str, default=None, 
                      help="Path to the camera calibration file (JSON or YAML)")
    args = parser.parse_args()
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_mast3r_model(args.weights, device)
    
    print(f"ERROR: No valid image pairs found in the dataset directory: {args.dataset_dir}")
    print("Please check the directory structure and ensure RGB images exist.")

    
    # Need to import glob here (was missing in the imports)

    
    try:
        # Create dataset of RGB pairs
        dataset = RGBPairDataset(
            root_dir=args.dataset_dir,
            sequences=None,
            img_size=tuple(args.img_size),
            frame_skip=args.frame_skip
        )
        
        if len(dataset) == 0:
            print(f"ERROR: No valid image pairs found in the dataset directory: {args.dataset_dir}")
            print("Please check the directory structure and ensure RGB images exist.")
            return
        
        # Generate pseudo-GT
        generate_pseudo_gt(
            dataset=dataset,
            output_dir=args.output_dir,
            model=model,
            device=device,
            calib_file=args.calib_file,
            batch_size=args.batch_size,
            visualize=args.visualize
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()