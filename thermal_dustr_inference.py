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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm  # For colormaps

# Import functions from your training pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preprocessing import enhance_thermal_contrast
from thermal_dustr_model import load_dustr_model, ThermalDUSt3R
from utils.visualize import plot_point_cloud


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


def visualize_depth_result(img_path, results, output_path=None, img_size=(384, 384)):
    """
    Create high-quality visualization of thermal image and depth prediction with enhanced coloring.
    
    Args:
        img_path: Path to the input thermal image
        results: Dictionary with inference results
        output_path: Path to save the visualization (if None, display instead)
        img_size: Size for resizing the image
    """
    # Load thermal image
    thermal_img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    
    if thermal_img is None:
        # Fallback to regular loading
        thermal_img = cv2.imread(img_path)
        if thermal_img is None:
            print(f"Failed to load image from {img_path}")
            return None
    
    # Detect and remove black borders
    if len(thermal_img.shape) == 2:
        # For grayscale/depth images
        threshold = 10  # Adjust as needed for your images
        non_black = thermal_img > threshold
        rows = np.any(non_black, axis=1)
        cols = np.any(non_black, axis=0)
        
        if np.sum(rows) > 0 and np.sum(cols) > 0:
            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]
            if len(row_indices) > 0 and len(col_indices) > 0:
                row_start, row_end = row_indices[[0, -1]]
                col_start, col_end = col_indices[[0, -1]]
                
                # Add a small margin
                row_start = max(0, row_start - 5)
                row_end = min(thermal_img.shape[0] - 1, row_end + 5)
                col_start = max(0, col_start - 5)
                col_end = min(thermal_img.shape[1] - 1, col_end + 5)
                
                # Crop the image to remove black bars
                thermal_img = thermal_img[row_start:row_end+1, col_start:col_end+1]
    
    # Process the thermal image
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
    thermal_tensor = torch.from_numpy(thermal_img.transpose(2, 0, 1)).float()
    
    # Apply contrast enhancement
    enhanced_tensor = enhance_thermal_contrast(thermal_tensor)
    
    # Extract depth and resize to match thermal image size
    depth = results["depth1"]
    depth = cv2.resize(depth, img_size, interpolation=cv2.INTER_NEAREST)
    
    # Create figure with equal-sized subplots and better spacing
    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    
    # Process thermal image for visualization
    if enhanced_tensor.shape[0] == 3:
        viz_img = 0.299 * enhanced_tensor[0] + 0.587 * enhanced_tensor[1] + 0.114 * enhanced_tensor[2]
    else:
        viz_img = enhanced_tensor[0]
    
    viz_img = viz_img.cpu().numpy()
    
    # Enhance contrast
    p2, p98 = np.percentile(viz_img, (2, 98))
    viz_img = np.clip((viz_img - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    # Convert to uint8 for colormap application
    viz_img_uint8 = (viz_img * 255).astype(np.uint8)
    
    # Apply JET colormap (similar to what's in your first example)
    colored_img = cv2.applyColorMap(viz_img_uint8, cv2.COLORMAP_JET)
    colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
    
    # Display enhanced thermal image with jet colormap
    # Thermal image subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(colored_img)
    ax1.set_title("Enhanced Thermal Image", fontsize=16)
    ax1.axis("off")
    
    # Ensure depth values are positive and non-zero
    depth = np.maximum(depth, 1e-6)
    
    # Display depth prediction
    # Depth map subplot
    ax2 = fig.add_subplot(gs[0, 1])
    depth_vis = ax2.imshow(depth, cmap='plasma')
    ax2.set_title("Depth Prediction", fontsize=16)
    ax2.axis("off")
    
    # Add colorbar with better positioning
    cbar = fig.colorbar(depth_vis, ax=ax2, fraction=0.046, pad=0.04, orientation='vertical')
    cbar.set_label('Depth', fontsize=12)
    
     # 3D point cloud subplot
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    plot_point_cloud(ax3, results["pointmap1"], color_mode='depth', point_size=1)
    ax3.set_title("Predicted 3D Point Cloud", fontsize=16)
    
    # Add timestamp at the bottom
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.5, 0.01, f"Generated: {timestamp}", ha="center", fontsize=10)
    
    # Set specific figure size to ensure proper proportions
    fig.set_size_inches(16, 7)
    
    # Save or display
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path
    else:
        plt.tight_layout()
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