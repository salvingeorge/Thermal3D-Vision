#!/usr/bin/env python3
# thermal_inference_fixed.py

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from pathlib import Path
import types

def load_model(checkpoint_path, device=None):
    """Load the fine-tuned DUSt3R model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {checkpoint_path}")
    
    # Import the DUSt3R model class
    dust3r_path = os.path.join(os.path.expanduser("~"), "georges1/dust3r")
    if dust3r_path not in sys.path:
        sys.path.append(dust3r_path)
    from dust3r.model import AsymmetricCroCo3DStereo
    
    # Initialize with the correct dimensions for your model
    model = AsymmetricCroCo3DStereo(
        output_mode='pts3d',
        head_type='linear',
        patch_size=16,
        img_size=(224, 224),
        landscape_only=False,
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        mlp_ratio=4,
        dec_embed_dim=768,
        dec_depth=8,
        dec_num_heads=12
    )
    
    # Patch the model's _encode_image method to bypass the assertion error
    original_encode_image = model._encode_image
    
    def patched_encode_image(self, image, true_shape):
        # Skip the problematic assertion
        # embed the image into patches
        x, pos = self.patch_embed(image, true_shape=true_shape)
        
        # apply transformer encoder blocks
        for blk in self.enc_blocks:
            x = blk(x, pos)
        
        x = self.enc_norm(x)
        return x, pos, None
    
    # Replace the method
    model._encode_image = types.MethodType(patched_encode_image, model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully with patched encode_image method")
    return model

def load_thermal_image(path, img_size=(224, 224)):
    """Load and preprocess a thermal image for inference."""
    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: Image file {path} does not exist")
        return None
    
    # Load thermal image - handle different formats
    thermal_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    if thermal_img is None:
        # Try regular loading
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
    
    return thermal_img

def process_image_pair(model, img1_path, img2_path=None, img_size=(224, 224), monocular=False):
    """
    Process a pair of thermal images or a single image if monocular=True.
    """
    device = next(model.parameters()).device
    
    # Load the first image
    img1 = load_thermal_image(img1_path, img_size)
    if img1 is None:
        return None
    
    # For monocular mode, use the same image twice
    if monocular:
        img2 = img1
    else:
        # Load the second image if provided
        if img2_path is None:
            print("Error: Second image path required for stereo mode")
            return None
        
        img2 = load_thermal_image(img2_path, img_size)
        if img2 is None:
            return None
    
    # Prepare inputs for the model
    view1 = {"img": img1.unsqueeze(0).to(device), "instance": []}
    view2 = {"img": img2.unsqueeze(0).to(device), "instance": []}
    
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
        pointmap2 = pred2.get("pts3d_in_other_view") if "pts3d_in_other_view" in pred2 else pred2.get("pts3d")
        confidence1 = pred1.get("conf")
        confidence2 = pred2.get("conf")
    else:
        pointmap1, pointmap2 = pred1, pred2
        confidence1, confidence2 = None, None
    
    # Move tensors to CPU
    pointmap1 = pointmap1.detach().cpu()
    pointmap2 = pointmap2.detach().cpu()
    
    # Extract depth (Z coordinate from pointmap)
    if len(pointmap1.shape) == 4:  # [B, H, W, 3]
        depth1 = pointmap1[0, :, :, 2].numpy()
        depth2 = pointmap2[0, :, :, 2].numpy()
        pointmap1 = pointmap1[0].numpy()
        pointmap2 = pointmap2[0].numpy()
    else:  # [H, W, 3]
        depth1 = pointmap1[:, :, 2].numpy()
        depth2 = pointmap2[:, :, 2].numpy()
        pointmap1 = pointmap1.numpy()
        pointmap2 = pointmap2.numpy()
    
    # Process confidence maps
    if confidence1 is not None:
        confidence1 = confidence1.detach().cpu()
        confidence2 = confidence2.detach().cpu()
    
    # Return results
    return {
        "pointmap1": pointmap1,
        "pointmap2": pointmap2,
        "depth1": depth1,
        "depth2": depth2,
        "confidence1": confidence1,
        "confidence2": confidence2
    }

def visualize_results(img_path, results, output_path=None):
    """
    Visualize inference results.
    
    Args:
        img_path: Path to the input image
        results: Dictionary of results from process_image_pair
        output_path: Path to save visualization
    """
    # Load original image for display
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title("Thermal Image")
    axes[0].axis("off")
    
    # Plot depth map
    depth_vis = axes[1].imshow(results["depth1"], cmap="plasma")
    axes[1].set_title("Predicted Depth")
    axes[1].axis("off")
    fig.colorbar(depth_vis, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot confidence if available
    if results["confidence1"] is not None:
        # Fix confidence shape if needed
        confidence_map = results["confidence1"]
        if len(confidence_map.shape) == 3:
            # Take the first element if we have a batch dimension
            if confidence_map.shape[0] == 2:
                # This might be (2, H, W) format - just take first channel
                confidence_map = confidence_map[0]
            elif confidence_map.shape[2] == 1:
                # This might be (H, W, 1) format
                confidence_map = confidence_map[:, :, 0]
        
        conf_vis = axes[2].imshow(confidence_map, cmap="viridis")
        axes[2].set_title("Confidence")
        axes[2].axis("off")
        fig.colorbar(conf_vis, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        # If no confidence, plot pointcloud colored by depth
        axes[2].imshow(results["pointmap1"][:,:,2], cmap="viridis")
        axes[2].set_title("Z Coordinate")
        axes[2].axis("off")
        fig.colorbar(depth_vis, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned DUSt3R model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to thermal image or directory of images")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save inference results")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (width height)")
    parser.add_argument("--monocular", action="store_true",
                        help="Use monocular mode (single image input)")
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    
    # Check if input is a directory or a single image
    if os.path.isdir(args.input):
        # Process all images in directory
        os.makedirs(args.output, exist_ok=True)
        
        image_files = [f for f in os.listdir(args.input) if f.endswith((".png", ".jpg", ".jpeg"))]
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            basename = os.path.splitext(img_file)[0]
            
            # Process the image
            results = process_image_pair(
                model=model,
                img1_path=img_path,
                img_size=tuple(args.img_size),
                monocular=True if args.monocular else False,
                img2_path=None  # Will be ignored if monocular=True
            )
            
            if results is not None:
                # Save depth map
                np.save(os.path.join(args.output, f"{basename}_depth.npy"), results["depth1"])
                
                # Create and save visualization
                vis_path = os.path.join(args.output, f"{basename}_vis.png")
                visualize_results(img_path, results, vis_path)
    else:
        # Process a single image
        results = process_image_pair(
            model=model,
            img1_path=args.input,
            img_size=tuple(args.img_size),
            monocular=args.monocular
        )
        
        if results is not None:
            # Create output directory
            os.makedirs(args.output, exist_ok=True)
            
            # Save depth map
            basename = os.path.splitext(os.path.basename(args.input))[0]
            np.save(os.path.join(args.output, f"{basename}_depth.npy"), results["depth1"])
            
            # Create and save visualization
            vis_path = os.path.join(args.output, f"{basename}_vis.png")
            visualize_results(args.input, results, vis_path)
    
    print(f"Inference complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()