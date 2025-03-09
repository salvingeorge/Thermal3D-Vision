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

# Import your dataset
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

def extract_data_from_output(output, device='cuda', which_camera="left"):
    """
    Extract the full 3D pointmaps + depth, and then use official calibration for intrinsics and pose.
    
    Returns:
      pointmap1_xyz: [H, W, 3] full 3D pointmap for image1.
      pointmap2_xyz: [H, W, 3] full 3D pointmap for image2.
      depth1, depth2: 2D depth maps (z channel).
      pose: 4x4 calibration pose (from our hardcoded CALIB).
      K: 3x3 intrinsics matrix.
    """
    def get_pointmap(pred):
        for key in ["pts3d", "pointmap", "pointmaps", "predicted_pts3d", "pts3d_in_other_view"]:
            if key in pred:
                print(f"Using key '{key}' for pointmap.")
                return pred[key]
        raise KeyError("No recognized pointmap key found. Keys: " + str(list(pred.keys())))
    
    # Hardcoded calibration (adjust these values as needed)
    CALIB = {
        "left": {
            "intrinsics": [510.0959, 510.3778, 315.8569, 253.1974],
            "pose": np.eye(4),
        },
        "right": {
            "intrinsics": [510.4250, 510.4975, 300.5257, 244.4115],
            "pose": np.array([
                [ 0.99992344, -0.00321543,  0.01194874, -0.50139442],
                [ 0.00312150,  0.99996415,  0.00787166,  0.00310180],
                [-0.01197362, -0.00783375,  0.99989763,  0.00841094],
                [ 0.0,         0.0,         0.0,         1.0       ]
            ]),
        }
    }
    
    pred1 = output.get("pred1", {})
    pred2 = output.get("pred2", {})
    print("pred1 keys:", list(pred1.keys()))
    print("pred2 keys:", list(pred2.keys()))
    
    pts1 = flatten_tensor(get_pointmap(pred1))
    pts2 = flatten_tensor(get_pointmap(pred2))
    print("After flattening, pts1 shape:", pts1.shape)
    print("After flattening, pts2 shape:", pts2.shape)
    
    if pts1.shape[0] > 1:
        pts1 = pts1[0]
    if pts2.shape[0] > 1:
        pts2 = pts2[0]
    
    pointmap1_xyz = pts1.cpu().numpy()  # full 3D pointmap [H,W,3]
    pointmap2_xyz = pts2.cpu().numpy()
    
    depth1 = pointmap1_xyz[:, :, 2]      # z-channel as depth
    depth2 = pointmap2_xyz[:, :, 2]
    
    # Use official calibration for intrinsics & pose
    if which_camera.lower() == "left":
        fx, fy, cx, cy = CALIB["left"]["intrinsics"]
        pose = CALIB["left"]["pose"]
    else:
        fx, fy, cx, cy = CALIB["right"]["intrinsics"]
        pose = CALIB["right"]["pose"]
    
    # Build intrinsics matrix (K)
    H_val, W_val = depth1.shape
    
    '''
    K = np.array([
        [fx, 0,  W_val / 2],
        [0,  fy, H_val / 2],
        [0,  0,   1 ]
    ])
    
    use this to derive the center from the image dimensions
    '''
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,   1 ]
    ])
    
    return pointmap1_xyz, pointmap2_xyz, depth1, depth2, pose, K

def visualize_data(rgb_image, thermal_image, depth, save_path=None):
    """Visualize side-by-side: RGB, thermal, and depth."""
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
        plt.show()

def generate_pseudo_gt(dataset, output_dir, model=None, batch_size=1, visualize=False):
    """Generate pseudo-ground truth: full pointmaps, depth, intrinsics, pose."""
    # Create subfolders
    pointmap_dir = os.path.join(output_dir, 'pointmaps')
    depth_dir = os.path.join(output_dir, 'depth')
    intrinsics_dir = os.path.join(output_dir, 'intrinsics')
    poses_dir = os.path.join(output_dir, 'poses')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    for d in [pointmap_dir, depth_dir, intrinsics_dir, poses_dir]:
        os.makedirs(d, exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    if model is None:
        model = load_mast3r_model()
    device = next(model.parameters()).device
    
    from dust3r.inference import inference
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=skip_none_collate
    )
    
    # Debug: print keys from the first batch
    for batch in dataloader:
        print("Batch keys:", list(batch.keys()))
        break
    
    def which_camera_from_filename(name):
        # If "fl_rgb" is in the file name, consider it left; otherwise, right.
        if "fl_rgb" in name.lower():
            return "left"
        else:
            return "right"
    
    n_processed = 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating pseudo-GT")):
        if batch is None:
            continue
        
        rgb_images = batch['rgb']          # [B, 3, H, W]
        rgb_paths = batch.get('rgb_path', None)
        thermal_paths = batch.get('thermal_path', None)
        
        # Check thermal paths; if the first thermal path is None or empty, skip this sample.
        if thermal_paths is None or len(thermal_paths) == 0 or not os.path.isfile(thermal_paths[0]):
            print("Warning: Thermal path missing for sample, skipping.")
            continue
        
        if len(rgb_images) == 1:
            img1 = rgb_images[0].to(device)
            img2 = rgb_images[0].clone().to(device)
            base_name1 = os.path.splitext(os.path.basename(rgb_paths[0]))[0]
            base_name2 = base_name1 + "_dup"
            pair_data = [({"img": img1.unsqueeze(0), "instance": []},
                          {"img": img2.unsqueeze(0), "instance": []})]
        else:
            img1 = rgb_images[0].to(device)
            img2 = rgb_images[1].to(device)
            base_name1 = os.path.splitext(os.path.basename(rgb_paths[0]))[0]
            base_name2 = os.path.splitext(os.path.basename(rgb_paths[1]))[0]
            pair_data = [({"img": img1.unsqueeze(0), "instance": []},
                          {"img": img2.unsqueeze(0), "instance": []})]
        
        with torch.no_grad():
            output = inference(pair_data, model, device, batch_size=1, verbose=False)
        
        try:
            (pointmap1_xyz, pointmap2_xyz,
             depth1, depth2, pose, intrinsics) = extract_data_from_output(output, device, which_camera=which_camera_from_filename(base_name1))
        except Exception as e:
            print(f"Error extracting data from output: {e}")
            continue
        
        # Save outputs
        np.save(os.path.join(pointmap_dir, f"{base_name1}.npy"), pointmap1_xyz)
        np.save(os.path.join(pointmap_dir, f"{base_name2}.npy"), pointmap2_xyz)
        np.save(os.path.join(depth_dir, f"{base_name1}.npy"), depth1)
        np.save(os.path.join(depth_dir, f"{base_name2}.npy"), depth2)
        np.save(os.path.join(intrinsics_dir, f"{base_name1}.npy"), intrinsics)
        np.save(os.path.join(intrinsics_dir, f"{base_name2}.npy"), intrinsics)
        np.save(os.path.join(poses_dir, f"{base_name1}.npy"), np.eye(4))  # using identity for left
        np.save(os.path.join(poses_dir, f"{base_name2}.npy"), pose)
        
        # Visualization (only for first few samples)
        if visualize and (n_processed < 10):
            thermal_img_path = thermal_paths[0]
            thermal_img = cv2.imread(thermal_img_path, cv2.IMREAD_ANYDEPTH)
            if thermal_img is not None:
                if thermal_img.dtype == np.uint16:
                    thermal_img = thermal_img.astype(np.float32) / 65535.0
                else:
                    thermal_img = thermal_img.astype(np.float32) / 255.0
            else:
                thermal_img = np.zeros((depth1.shape[0], depth1.shape[1]), dtype=np.float32)
            if len(thermal_img.shape) == 2:
                thermal_img = np.stack([thermal_img]*3, axis=-1)
            
            rgb_img = rgb_images[0].permute(1,2,0).cpu().numpy()
            vis_path = os.path.join(vis_dir, f"{base_name1}.png")
            visualize_data(rgb_img, thermal_img, depth1, save_path=vis_path)
            n_processed += 1

    print(f"Pseudo-GT generation complete. Results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-GT (full 3D pointmaps, depth, intrinsics, pose) for Freiburg dataset using MASt3R")
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to the Freiburg dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to save the pseudo-GT")
    parser.add_argument('--weights', type=str, default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                        help="Path to the MASt3R model weights")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for processing")
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512], help="Image size (width height)")
    parser.add_argument('--visualize', action='store_true', help="Visualize a few samples for verification")
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
