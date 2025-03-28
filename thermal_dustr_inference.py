#!/usr/bin/env python3
# thermal_dustr_inference.py

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
from datetime import datetime

# Import functions from your training pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import enhance_thermal_contrast
from thermal_dustr_model import load_dustr_model, ThermalDUSt3R


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


def run_inference(model, img_path, img_size=(224, 224), monocular=True, use_thermal_model=True):
    """
    Run inference on a thermal image to predict depth.
    
    Args:
        model: Loaded DUSt3R model
        img_path: Path to the thermal image
        img_size: Input image size for the model
        monocular: Whether to use the same image for both inputs
        use_thermal_model: Whether to wrap model with ThermalDUSt3R
    
    Returns:
        Dictionary with results including depth and pointmaps
    """
    device = next(model.parameters()).device
    
    # Load and preprocess image
    img = load_and_preprocess_thermal_image(img_path, img_size)
    if img is None:
        return None
    
    # Prepare input in the same format as during training
    view1 = {"img": img.unsqueeze(0).to(device), "instance": []}
    
    if monocular:
        view2 = view1  # Use same image for both views
    else:
        # If stereo mode is needed, this would be the second image
        # For now, just use same image
        view2 = view1
    
    # Apply ThermalDUSt3R wrapper if used in training
    if use_thermal_model and not isinstance(model, ThermalDUSt3R):
        model = ThermalDUSt3R(model)
    
    # Run inference
    with torch.no_grad():
        output = model(view1, view2)
    
    # Extract predictions - using same logic as training code
    if isinstance(output, tuple):
        pred1, pred2 = output
    else:
        pred1 = output.get("pred1", {})
        pred2 = output.get("pred2", {})
    
    # Extract pointmaps using same extraction logic as training
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


def visualize_depth_result(img_path, results, output_path=None):
    """
    Create high-quality visualization of thermal image and depth prediction.
    
    Args:
        img_path: Path to the input thermal image
        results: Dictionary with inference results
        output_path: Path to save the visualization (if None, display instead)
    """
    # Load original thermal image
    thermal_img_original = cv2.imread(img_path)
    if thermal_img_original is None:
        thermal_img_original = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    
    if thermal_img_original is not None:
        if len(thermal_img_original.shape) == 2:
            # For grayscale/thermal images
            thermal_img = thermal_img_original.astype(np.float32)
            if thermal_img.dtype == np.uint16:
                thermal_img = thermal_img / 65535.0
            else:
                thermal_img = thermal_img / 255.0
        else:
            # For RGB images
            thermal_img = cv2.cvtColor(thermal_img_original, cv2.COLOR_BGR2RGB) / 255.0
    else:
        print(f"Error: Could not load image {img_path}")
        return None
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Normalize thermal image for better contrast
    if len(thermal_img.shape) > 2:
        # Convert RGB to grayscale for visualization
        thermal_gray = 0.299 * thermal_img[:,:,0] + 0.587 * thermal_img[:,:,1] + 0.114 * thermal_img[:,:,2]
    else:
        thermal_gray = thermal_img
    
    # Enhance contrast for visualization
    p2, p98 = np.percentile(thermal_gray, (2, 98))
    thermal_display = np.clip((thermal_gray - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    # Display thermal image
    axes[0].imshow(thermal_display, cmap='inferno')
    axes[0].set_title("Thermal Image", fontsize=14)
    axes[0].axis("off")
    
    # Display depth map with proper processing
    depth = results["depth1"]
    
    # Ensure depth values are positive and non-zero
    depth = np.maximum(depth, 1e-6)
    
    # Visualize depth with plasma colormap for consistency
    depth_vis = axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title("Depth Prediction", fontsize=14)
    axes[1].axis("off")
    
    # Add colorbar
    cbar = fig.colorbar(depth_vis, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth')
    
    plt.tight_layout()
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, 0.01, f"Generated: {timestamp}", ha="center", fontsize=9)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.show()
        return None


def main():
    parser = argparse.ArgumentParser(description="Thermal DUSt3R Inference")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the fine-tuned model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to thermal image or directory of images")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save inference results")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Input image size (width height)")
    parser.add_argument("--use_thermal_model", action="store_true",
                        help="Use ThermalDUSt3R wrapper for enhanced thermal processing")
    parser.add_argument("--monocular", action="store_true", default=True,
                        help="Use monocular mode (single image input)")
    args = parser.parse_args()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dustr_model(args.checkpoint, device)
    print(f"Model loaded successfully on {device}")
    model.eval()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process input (file or directory)
    if os.path.isdir(args.input):
        # Process all images in directory
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"Found {len(image_files)} images to process")
        
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(args.input, img_file)
            base_name = os.path.splitext(img_file)[0]
            
            # Process image
            results = run_inference(
                model=model,
                img_path=img_path,
                img_size=tuple(args.img_size),
                monocular=args.monocular,
                use_thermal_model=args.use_thermal_model
            )
            
            if results is not None:
                # Save depth map as numpy array
                depth_path = os.path.join(args.output, f"{base_name}_depth.npy")
                np.save(depth_path, results["depth1"])
                
                # Save visualization
                vis_path = os.path.join(args.output, f"{base_name}_depth_vis.png")
                visualize_depth_result(img_path, results, vis_path)
    else:
        # Process single image
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        
        # Process image
        results = run_inference(
            model=model,
            img_path=args.input,
            img_size=tuple(args.img_size),
            monocular=args.monocular,
            use_thermal_model=args.use_thermal_model
        )
        
        if results is not None:
            # Save depth map as numpy array
            depth_path = os.path.join(args.output, f"{base_name}_depth.npy")
            np.save(depth_path, results["depth1"])
            
            # Save visualization
            vis_path = os.path.join(args.output, f"{base_name}_depth_vis.png")
            visualize_depth_result(args.input, results, vis_path)
            
            print(f"Results saved to {args.output}")
        else:
            print("Processing failed")

if __name__ == "__main__":
    main()