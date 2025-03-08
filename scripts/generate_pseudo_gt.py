#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
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

# Add the parent directory to the path for our local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.freiburg_dataset import FreiburgDataset

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
        print(f"Current Python path: {sys.path}")
        sys.exit(1)

def flatten_tensor(x):
    """Recursively extract the first element until x is not a tuple/list, then ensure it's a tensor."""
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

def extract_data_from_output(output, device='cuda'):
    """
    Extract depth, pose, and intrinsics from MASt3R output.

    This function looks for the 3D pointmaps in the prediction dictionaries.
    It checks keys "pts3d", "pointmap", "pointmaps", "predicted_pts3d", and "pts3d_in_other_view".
    Then it flattens any nested tuple/list and selects the first element if there is an extra batch dimension.

    Returns:
        depth1: Depth map for first image (z channel of pointmap1).
        depth2: Depth map for second image.
        pose: Relative pose from camera 1 to camera 2 (here, identity).
        intrinsics: Estimated camera intrinsics matrix.
    """
    def get_pointmap(pred):
        for key in ["pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"]:
            if key in pred:
                print(f"Using key '{key}' for pointmap.")
                return pred[key]
        raise KeyError("None of the expected keys for pointmap found in prediction. Found keys: " + str(list(pred.keys())))
    
    pred1 = output.get("pred1", {})
    pred2 = output.get("pred2", {})
    print("pred1 keys:", list(pred1.keys()))
    print("pred2 keys:", list(pred2.keys()))
    
    pts1 = flatten_tensor(get_pointmap(pred1))
    pts2 = flatten_tensor(get_pointmap(pred2))
    print("After flattening, pts1 shape:", pts1.shape)
    print("After flattening, pts2 shape:", pts2.shape)
    
    # Our model returns a tensor with shape [2, 512, 512, 3] (two images in the batch)
    # We want only the first image's prediction for each branch:
    if pts1.shape[0] > 1:
        pts1 = pts1[0]
    if pts2.shape[0] > 1:
        pts2 = pts2[0]
    
    # Now, pts1 and pts2 should have shape [512, 512, 3]
    pointmap1 = pts1.cpu().numpy()
    pointmap2 = pts2.cpu().numpy()
    
    # Use the z channel as the depth map.
    depth1 = pointmap1[:, :, 2]
    depth2 = pointmap2[:, :, 2]
    
    # Estimate intrinsics from a central crop of pointmap1.
    h, w = depth1.shape  # h, w = 512, 512 (for example)
    crop_h, crop_w = h // 4, w // 4
    center_h, center_w = h // 2, w // 2
    y_grid, x_grid = np.mgrid[center_h-crop_h:center_h+crop_h, center_w-crop_w:center_w+crop_w]
    x_grid = x_grid - w / 2
    y_grid = y_grid - h / 2
    x_coords = pointmap1[center_h-crop_h:center_h+crop_h, center_w-crop_w:center_w+crop_w, 0]
    y_coords = pointmap1[center_h-crop_h:center_h+crop_h, center_w-crop_w:center_w+crop_w, 1]
    z_coords = pointmap1[center_h-crop_h:center_h+crop_h, center_w-crop_w:center_w+crop_w, 2]
    valid = z_coords > 0
    if np.sum(valid) > 100:
        f_x_estimates = -x_grid[valid] / (x_coords[valid] / z_coords[valid])
        f_y_estimates = -y_grid[valid] / (y_coords[valid] / z_coords[valid])
        f_x = np.median(f_x_estimates)
        f_y = np.median(f_y_estimates)
    else:
        f_x = f_y = max(h, w) * 0.8
    intrinsics = np.array([[f_x, 0, w / 2],
                           [0, f_y, h / 2],
                           [0, 0, 1]])
    
    # For simplicity, set the relative pose as identity.
    pose = np.eye(4)
    return depth1, depth2, pose, intrinsics

def visualize_data(rgb_image, thermal_image, depth, save_path=None):
    """Visualize RGB, thermal, and depth images side-by-side for verification."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)
    plt.title('RGB Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(thermal_image, cmap='inferno')
    plt.title('Thermal Image')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(depth, cmap='plasma')
    plt.title('Depth Map')
    plt.colorbar(label='Depth')
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def generate_pseudo_gt(dataset, output_dir, model=None, batch_size=1, visualize=False):
    """Generate pseudo-ground truth annotations using MASt3R on a given dataset."""
    depth_dir = os.path.join(output_dir, 'depth')
    intrinsics_dir = os.path.join(output_dir, 'intrinsics')
    poses_dir = os.path.join(output_dir, 'poses')
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    if model is None:
        model = load_mast3r_model()
    device = next(model.parameters()).device
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False
    )
    
    # Debug: print keys in the first batch.
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        break

    from dust3r.inference import inference
    n_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating pseudo-GT")):
        rgb_images = batch['rgb']          # [B, 3, H, W]
        rgb_paths = batch.get('rgb_path', None)
        thermal_paths = batch.get('thermal_path', None)
        
        if len(rgb_images) == 1:
            img1 = rgb_images[0].to(device)
            img2 = rgb_images[0].clone().to(device)
            base_name1 = os.path.splitext(os.path.basename(rgb_paths[0]))[0] if rgb_paths is not None else f"batch{batch_idx}_img0"
            base_name2 = base_name1 + "_dup"
            pair_data = [({"img": img1.unsqueeze(0), "instance": []},
                          {"img": img2.unsqueeze(0), "instance": []})]
        else:
            img1 = rgb_images[0].to(device)
            img2 = rgb_images[1].to(device)
            base_name1 = os.path.splitext(os.path.basename(rgb_paths[0]))[0] if rgb_paths is not None else f"batch{batch_idx}_pair0_img1"
            base_name2 = os.path.splitext(os.path.basename(rgb_paths[1]))[0] if rgb_paths is not None else f"batch{batch_idx}_pair0_img2"
            pair_data = [({"img": img1.unsqueeze(0), "instance": []},
                          {"img": img2.unsqueeze(0), "instance": []})]
        
        with torch.no_grad():
            output = inference(pair_data, model, device, batch_size=1, verbose=False)
        
        try:
            depth1, depth2, pose, intrinsics = extract_data_from_output(output, device)
        except Exception as e:
            print(f"Error extracting data from output: {e}")
            continue
        
        np.save(os.path.join(depth_dir, f"{base_name1}.npy"), depth1)
        np.save(os.path.join(depth_dir, f"{base_name2}.npy"), depth2)
        np.save(os.path.join(intrinsics_dir, f"{base_name1}.npy"), intrinsics)
        np.save(os.path.join(intrinsics_dir, f"{base_name2}.npy"), intrinsics)
        np.save(os.path.join(poses_dir, f"{base_name1}.npy"), np.eye(4))
        np.save(os.path.join(poses_dir, f"{base_name2}.npy"), pose)
        
        if visualize and (n_processed < 10):
            if thermal_paths is not None:
                thermal_img_path = thermal_paths[0]
                thermal_img = cv2.imread(thermal_img_path, cv2.IMREAD_ANYDEPTH)
                if thermal_img is not None:
                    thermal_img = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                else:
                    thermal_img = np.zeros((depth1.shape[0], depth1.shape[1]), dtype=np.uint8)
            else:
                thermal_img = np.zeros((depth1.shape[0], depth1.shape[1]), dtype=np.uint8)
            rgb_img = rgb_images[0].permute(1, 2, 0).cpu().numpy()
            vis_path = os.path.join(vis_dir, f"{base_name1}.png")
            visualize_data(rgb_img, thermal_img, depth1, save_path=vis_path)
            n_processed += 1

    print(f"Pseudo-GT generation complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-GT for Freiburg dataset")
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help="Path to the Freiburg dataset")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Path to save the pseudo-ground truth annotations")
    parser.add_argument('--weights', type=str,
                        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                        help="Path to the MASt3R model weights")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512],
                        help="Image size for processing (width height)")
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize a few samples for verification")
    args = parser.parse_args()
    
    dataset = FreiburgDataset(
        root_dir=args.dataset_dir,
        sequences=None,
        img_size=tuple(args.img_size),
        use_pseudo_gt=False
    )
    
    model = load_mast3r_model(args.weights)
    generate_pseudo_gt(dataset, args.output_dir, model, args.batch_size, args.visualize)
    
    print(f"Pseudo-GT generation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
