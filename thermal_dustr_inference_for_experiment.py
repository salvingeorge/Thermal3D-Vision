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
    thermal_img = preprocess_fire_scene_thermal(thermal_img)
    
    return thermal_img

def preprocess_fire_scene_thermal(thermal_img):
    """
    Specialized preprocessing for fire scenes that explicitly encodes fire as foreground.
    This transforms the thermal image to make the model interpret fire as being closer.
    """
    import cv2
    import numpy as np
    import torch
    
    # Convert to numpy if tensor
    if isinstance(thermal_img, torch.Tensor):
        thermal_np = thermal_img.cpu().numpy()
        if thermal_np.shape[0] == 3:  # [C,H,W] -> [H,W,C]
            thermal_np = thermal_np.transpose(1, 2, 0)
    else:
        thermal_np = np.array(thermal_img)
    
    # Ensure proper data type and range
    if thermal_np.dtype != np.float32:
        thermal_np = thermal_np.astype(np.float32)
        if thermal_np.max() > 1.0:
            thermal_np = thermal_np / 255.0
    
    # Convert to grayscale if needed
    if len(thermal_np.shape) == 3 and thermal_np.shape[2] >= 3:
        thermal_gray = 0.299 * thermal_np[:,:,0] + 0.587 * thermal_np[:,:,1] + 0.114 * thermal_np[:,:,2]
    elif len(thermal_np.shape) == 3 and thermal_np.shape[2] == 1:
        thermal_gray = thermal_np[:,:,0]
    else:
        thermal_gray = thermal_np
    
    # 1. Temperature normalization
    p_low, p_high = np.percentile(thermal_gray, (5, 95))
    thermal_norm = np.clip(thermal_gray, p_low, p_high)
    thermal_norm = (thermal_norm - p_low) / (p_high - p_low + 1e-6)
    
    # 2. Fire detection - high temperature areas
    # Higher threshold identifies definite fire pixels
    fire_mask = thermal_norm > 0.7  # Adjust threshold as needed
    
    # 3. Create an artificial RGB image that encodes fire proximity information
    h, w = thermal_norm.shape[:2]
    result_img = np.zeros((h, w, 3), dtype=np.float32)
    
    # Base image - use a grayscale representation that's darker for distant objects
    base_gray = 1.0 - thermal_norm  # Invert, making hot areas dark/close
    base_gray = np.clip(base_gray * 1.2, 0, 1)  # Enhance contrast
    
    # Apply CLAHE to the base image for better local contrast
    base_uint8 = (base_gray * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    base_clahe = clahe.apply(base_uint8).astype(np.float32) / 255.0
    
    # Fill all channels with the base representation
    for c in range(3):
        result_img[:,:,c] = base_clahe
    
    # 4. Make fire areas appear very close by using strong foreground cues
    # In RGB images, closer objects typically have:
    # - More saturation
    # - Stronger texture
    # - Higher contrast
    
    # Add saturation and color to fire areas
    result_img[fire_mask, 0] = 0.8  # Red channel boosted
    result_img[fire_mask, 1] = 0.3  # Green channel reduced
    result_img[fire_mask, 2] = 0.1  # Blue channel reduced
    
    # 5. Add texture to fire areas to enhance proximity cues
    # Generate noise texture
    noise = np.random.rand(h, w).astype(np.float32) * 0.1
    
    # Apply noise only to fire areas
    for c in range(3):
        result_img[:,:,c] = np.where(fire_mask, result_img[:,:,c] + noise, result_img[:,:,c])
    
    # 6. Extract and enhance edges
    edges = cv2.Canny((thermal_norm * 255).astype(np.uint8), 50, 150).astype(np.float32) / 255.0
    
    # Apply edges with reduced weight to non-fire areas
    edge_weight = np.ones_like(thermal_norm) * 0.15
    edge_weight[fire_mask] = 0.3  # Stronger edges in fire areas
    
    for c in range(3):
        result_img[:,:,c] = result_img[:,:,c] * (1 - edge_weight) + edges * edge_weight
    
    # 7. Final adjustments
    result_img = np.clip(result_img, 0, 1)
    
    # Convert to PyTorch tensor in [C,H,W] format
    result_tensor = torch.from_numpy(result_img.transpose(2, 0, 1)).float()
    
    return result_tensor

def advanced_fire_scene_processing(thermal_img):
    """
    Advanced preprocessing for fire scenes with better structure preservation
    and consistent depth interpretation for fire areas.
    """
    import cv2
    import numpy as np
    import torch
    from scipy import ndimage
    
    # Convert to numpy if tensor
    if isinstance(thermal_img, torch.Tensor):
        thermal_np = thermal_img.cpu().numpy()
        if thermal_np.shape[0] == 3:  # [C,H,W] -> [H,W,C]
            thermal_np = thermal_np.transpose(1, 2, 0)
    else:
        thermal_np = np.array(thermal_img)
    
    # Ensure proper data type and range
    if thermal_np.dtype != np.float32:
        thermal_np = thermal_np.astype(np.float32)
        if thermal_np.max() > 1.0:
            thermal_np = thermal_np / 255.0
    
    # Convert to grayscale if needed
    if len(thermal_np.shape) == 3 and thermal_np.shape[2] >= 3:
        thermal_gray = 0.299 * thermal_np[:,:,0] + 0.587 * thermal_np[:,:,1] + 0.114 * thermal_np[:,:,2]
    elif len(thermal_np.shape) == 3 and thermal_np.shape[2] == 1:
        thermal_gray = thermal_np[:,:,0]
    else:
        thermal_gray = thermal_np
    
    # 1. Use histogram analysis to identify different temperature regions
    hist, bins = np.histogram(thermal_gray.flatten(), bins=100, range=(0, 1))
    
    # Find peaks in the histogram to identify main temperature clusters
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, height=hist.max()*0.3, distance=10)
    peak_values = bins[peaks]
    
    # 2. Multi-level thresholding based on histogram analysis
    if len(peak_values) >= 2:
        # Sort peaks by temperature
        peak_values.sort()
        
        # Define thresholds between peaks
        thresholds = [(peak_values[i] + peak_values[i+1])/2 for i in range(len(peak_values)-1)]
        
        # Create masks for different temperature regions
        masks = []
        for i in range(len(thresholds) + 1):
            if i == 0:
                mask = thermal_gray <= thresholds[0]
            elif i == len(thresholds):
                mask = thermal_gray > thresholds[-1]
            else:
                mask = (thermal_gray > thresholds[i-1]) & (thermal_gray <= thresholds[i])
            masks.append(mask)
    else:
        # Fallback to simple thresholding if histogram analysis fails
        fire_threshold = 0.7
        masks = [thermal_gray <= fire_threshold, thermal_gray > fire_threshold]
    
    # 3. Create processed image
    h, w = thermal_gray.shape[:2]
    result_img = np.zeros((h, w, 3), dtype=np.float32)
    
    # 4. Apply controlled inversion - darker for hot/close areas, brighter for cold/far
    # The inversion needs to maintain structure while making hot areas appear close
    thermal_processed = 1.0 - thermal_gray
    
    # Apply CLAHE for better local contrast
    thermal_uint8 = (thermal_processed * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    thermal_clahe = clahe.apply(thermal_uint8).astype(np.float32) / 255.0
    
    # 5. Enhanced edge detection for better structure
    # Combine multiple edge detection methods
    edges1 = cv2.Canny((thermal_gray * 255).astype(np.uint8), 30, 150).astype(np.float32) / 255.0
    
    # Sobel edges
    sobelx = cv2.Sobel(thermal_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(thermal_gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-6)
    
    # Combine edge maps
    combined_edges = np.maximum(edges1, sobel_mag)
    
    # 6. Create RGB channels with structure preservation
    # Base layer using the inverted and contrast-enhanced thermal image
    for c in range(3):
        result_img[:,:,c] = thermal_clahe
    
    # 7. Adjust color based on temperature regions to enhance depth perception
    for i, mask in enumerate(masks):
        # Higher temperature regions should appear closer (darker) with more texture
        if i == len(masks) - 1:  # Highest temperature region (fire)
            # Make fire areas darker and more saturated
            result_img[mask, 0] = thermal_clahe[mask] * 0.5  # Darker red
            result_img[mask, 1] = thermal_clahe[mask] * 0.3  # Very dark green
            result_img[mask, 2] = thermal_clahe[mask] * 0.2  # Very dark blue
            
            # Add high-frequency texture to fire regions to suggest closeness
            noise = np.random.rand(h, w).astype(np.float32) * 0.15
            for c in range(3):
                result_img[:,:,c] = np.where(mask, result_img[:,:,c] + noise, result_img[:,:,c])
    
    # 8. Apply edge enhancement with varying strength based on temperature
    edge_strength = np.ones_like(thermal_gray) * 0.2
    # Stronger edges for high temperature regions
    if len(masks) > 1:
        edge_strength[masks[-1]] = 0.4  # Strongest edges for fire
    
    for c in range(3):
        result_img[:,:,c] = result_img[:,:,c] * (1 - edge_strength) + combined_edges * edge_strength
    
    # 9. Apply bilateral filter to preserve edges while smoothing
    result_img = cv2.bilateralFilter(result_img, 9, 75, 75)
    
    # 10. Final normalization
    result_img = np.clip(result_img, 0, 1)
    
    # Convert to PyTorch tensor in [C,H,W] format
    result_tensor = torch.from_numpy(result_img.transpose(2, 0, 1)).float()
    
    return result_tensor

def advanced_depth_refinement(depth_map, thermal_img, guided_filter=True, smoothness=0.5):
    """
    Advanced depth map refinement to reduce blockiness and enhance structure
    
    Args:
        depth_map: Raw blocky depth map
        thermal_img: Original thermal image
        guided_filter: Whether to use guided filtering
        smoothness: Smoothness parameter (0-1, higher = smoother)
    
    Returns:
        Refined depth map
    """
    import cv2
    import numpy as np
    import torch
    
    # Convert tensors to numpy
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    if isinstance(thermal_img, torch.Tensor):
        if thermal_img.dim() == 3 and thermal_img.shape[0] == 3:
            thermal_np = thermal_img.detach().cpu().numpy().transpose(1, 2, 0)
        else:
            thermal_np = thermal_img.detach().cpu().numpy()
    else:
        thermal_np = thermal_img
    
    # Resize thermal to match depth map if needed
    if depth_map.shape[:2] != thermal_np.shape[:2]:
        if len(thermal_np.shape) == 3:
            thermal_gray = cv2.cvtColor(
                (thermal_np * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
        else:
            thermal_gray = thermal_np.astype(np.float32)
            if thermal_gray.max() > 1.0:
                thermal_gray = thermal_gray / 255.0
    else:
        if len(thermal_np.shape) == 3:
            thermal_gray = cv2.cvtColor(
                (thermal_np * 255).astype(np.uint8),
                cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
        else:
            thermal_gray = thermal_np
    
    # 1. Initial upsampling if very low resolution
    if depth_map.shape[0] < 100 or depth_map.shape[1] < 100:
        h, w = thermal_gray.shape[:2]
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Apply guided filter for edge-aware smoothing
    if guided_filter:
        radius = int(15 * smoothness)
        eps = 1e-4 + (smoothness * 1e-2)
        refined_depth = cv2.ximgproc.guidedFilter(
            guide=thermal_gray,
            src=depth_map,
            radius=radius,
            eps=eps,
            dDepth=-1
        )
    else:
        refined_depth = depth_map
    
    # 3. Apply additional bilateral filtering for noise reduction while preserving edges
    bilateral_radius = int(5 * smoothness)
    refined_depth = cv2.bilateralFilter(
        refined_depth,
        bilateral_radius,
        50 * smoothness,
        50 * smoothness
    )
    
    return refined_depth

def depth_refinement_with_outlier_removal(depth_map, thermal_img, guided_filter=True):
    """
    Refines depth map and removes outlier spikes
    """
    import cv2
    import numpy as np
    import torch
    
    # Convert tensors to numpy
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    
    if isinstance(thermal_img, torch.Tensor):
        thermal_np = thermal_img.detach().cpu().numpy()
    else:
        thermal_np = thermal_img
        
    # Handle shape correctly - check format of thermal_np
    if len(thermal_np.shape) == 3:
        # Check if it's [C,H,W] (PyTorch format) or [H,W,C] (OpenCV format)
        if thermal_np.shape[0] == 1 or thermal_np.shape[0] == 3:  # [C,H,W]
            # Convert from [C,H,W] to [H,W,C]
            thermal_np = thermal_np.transpose(1, 2, 0)
    
    # Create grayscale image safely regardless of input format
    if len(thermal_np.shape) == 3 and thermal_np.shape[2] == 3:
        # It's [H,W,3] - convert to grayscale
        thermal_gray = cv2.cvtColor((thermal_np * 255).astype(np.uint8), 
                                   cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    elif len(thermal_np.shape) == 3 and thermal_np.shape[2] == 1:
        # It's [H,W,1] - just squeeze
        thermal_gray = thermal_np[:,:,0]
    elif len(thermal_np.shape) == 2:
        # It's already [H,W] - use as is
        thermal_gray = thermal_np
    else:
        # Fallback - try to reshape or average channels
        try:
            if thermal_np.shape[0] == 3:  # [3,H,W]
                thermal_gray = 0.299 * thermal_np[0] + 0.587 * thermal_np[1] + 0.114 * thermal_np[2]
            else:
                thermal_gray = thermal_np.mean(axis=0)  # Average across first dimension
        except:
            # Ultimate fallback - create blank image matching depth map shape
            thermal_gray = np.ones_like(depth_map)
    
    # 1. Remove outliers using statistical filtering
    depth_mean = np.nanmean(depth_map)
    depth_std = np.nanstd(depth_map)
    
    # Remove extreme outliers (more than 3 standard deviations from the mean)
    outlier_mask = np.abs(depth_map - depth_mean) > 3 * depth_std
    cleaned_depth = np.copy(depth_map)
    
    # Replace outliers with local median
    kernel_size = 5
    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            if outlier_mask[i, j]:
                # Define neighborhood bounds
                i_min = max(0, i - kernel_size//2)
                i_max = min(depth_map.shape[0], i + kernel_size//2 + 1)
                j_min = max(0, j - kernel_size//2)
                j_max = min(depth_map.shape[1], j + kernel_size//2 + 1)
                
                # Get neighborhood and calculate median
                neighborhood = depth_map[i_min:i_max, j_min:j_max]
                neighborhood = neighborhood[~outlier_mask[i_min:i_max, j_min:j_max]]
                
                if neighborhood.size > 0:
                    cleaned_depth[i, j] = np.median(neighborhood)
                else:
                    cleaned_depth[i, j] = depth_mean
    
    # 2. Apply guided filter - ensure shapes match
    if guided_filter and thermal_gray.shape == cleaned_depth.shape:
        # Make sure thermal_gray is float32
        thermal_gray_float = thermal_gray.astype(np.float32)
        cleaned_depth_float = cleaned_depth.astype(np.float32)
        
        refined_depth = cv2.ximgproc.guidedFilter(
            guide=thermal_gray_float,
            src=cleaned_depth_float,
            radius=8,
            eps=1e-4,
            dDepth=-1
        )
    else:
        refined_depth = cleaned_depth
    
    # 3. Final bilateral filter to smooth while preserving edges
    refined_depth = cv2.bilateralFilter(refined_depth.astype(np.float32), 5, 50, 50)
    
    return refined_depth

def run_inference(model, img_path1, img_path2 = None, img_size=(224, 224), monocular=True, use_thermal_model=True):
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
    img1 = load_and_preprocess_thermal_image(img_path1, img_size)
    if img1 is None:
        return None
    
    # Prepare input in the same format as during training
    view1 = {"img": img1.unsqueeze(0).to(device), "instance": []}
    
    # Check if in stereo or monocular mode
    if img_path2 is not None:
        # Load and preprocess second image
        img2 = load_and_preprocess_thermal_image(img_path2, img_size)
        if img2 is None:
            print(f"Warning: Could not load second image {img_path2}, falling back to monocular mode")
            view2 = view1  # Fallback to monocular
        else:
            # Create second view from second image
            view2 = {"img": img2.unsqueeze(0).to(device), "instance": []}
    else:
        # Monocular mode - use same image for both views
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
        
    # Apply depth refinement
    depth1_np = depth1.numpy()
    depth2_np = depth2.numpy()
    img_np = img1.cpu().numpy()  # Original thermal image
    
    # Apply depth refinement with outlier removal
    refined_depth1 = depth_refinement_with_outlier_removal(depth1_np, img_np, guided_filter=True)
    refined_depth2 = depth_refinement_with_outlier_removal(depth2_np, img_np, guided_filter=True)
    
    # Update the pointmaps with refined depths
    # We need to create copies to avoid modifying the original tensors
    pointmap1_refined = pointmap1.numpy().copy()
    pointmap2_refined = pointmap2.numpy().copy()
    
    # Update Z coordinates with refined depths
    pointmap1_refined[:, :, 2] = refined_depth1
    pointmap2_refined[:, :, 2] = refined_depth2
    
    # Return results with both original and refined versions
    return {
        "pointmap1": pointmap1_refined,  # Use refined pointmap
        "pointmap2": pointmap2_refined,  # Use refined pointmap
        "depth1": refined_depth1,        # Use refined depth
        "depth2": refined_depth2,        # Use refined depth
        "confidence1": confidence1.numpy(),
        "confidence2": confidence2.numpy(),
        "original_depth1": depth1_np,    # Keep original for comparison
        "original_depth2": depth2_np     # Keep original for comparison
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
    parser.add_argument("--input2", type=str, default=None,
                        help="Path to second thermal image (for stereo mode)")
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
                img_path1=img_path,
                img_path2=args.input2,
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
            img_path1=args.input,
            img_path2=args.input2,
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