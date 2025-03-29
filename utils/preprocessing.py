import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def enhance_thermal_contrast(thermal_tensor):
    """Apply contrast enhancement to thermal images using percentile-based clipping"""
    # Convert to numpy for easier processing
    if thermal_tensor is None:
        print("Warning: Received None instead of a thermal image tensor")
        return None
    thermal_np = thermal_tensor.cpu().numpy()
    if thermal_np.shape[0] == 3:  # CHW format
        # Convert to single channel if all channels are the same
        if np.allclose(thermal_np[0], thermal_np[1]) and np.allclose(thermal_np[0], thermal_np[2]):
            thermal_np = thermal_np[0]
        else:
            # Convert RGB to grayscale
            thermal_np = 0.299 * thermal_np[0] + 0.587 * thermal_np[1] + 0.114 * thermal_np[2]
    
    # Apply contrast enhancement
    p2, p98 = np.percentile(thermal_np, (2, 98))
    thermal_np_enhanced = np.clip((thermal_np - p2) / (p98 - p2), 0, 1)
    
    # Convert back to 3-channel tensor
    enhanced_tensor = torch.from_numpy(thermal_np_enhanced).float()
    if enhanced_tensor.dim() == 2:
        enhanced_tensor = enhanced_tensor.unsqueeze(0).repeat(3, 1, 1)
    
    return enhanced_tensor

def enhance_thermal_fixed_range(thermal_tensor, normalized=True):
    """Apply Freiburg thermal normalization to already normalized [0,1] values"""
    if thermal_tensor is None:
        return None
    
    thermal_np = thermal_tensor.cpu().numpy()
    
    # Handle different channel formats
    if thermal_np.ndim == 3:
        if thermal_np.shape[0] == 3 and np.allclose(thermal_np[0], thermal_np[1]) and np.allclose(thermal_np[0], thermal_np[2]):
            thermal_np = thermal_np[0]
        elif thermal_np.shape[0] == 1:
            thermal_np = thermal_np[0]
    
    # If values are already normalized to [0,1], we need to adjust the approach
    if normalized:
        # Rescale to raw range, then apply Freiburg normalization
        # For uint16 range ~ 65535
        raw_thermal = thermal_np * 65535.0
        
        # Apply Freiburg thermal normalization
        minval = 21800
        maxval = 25000
        raw_thermal = np.clip(raw_thermal, minval, maxval)
        thermal_np = (raw_thermal - minval) / (maxval - minval)
    else:
        # Original Freiburg normalization for raw values
        minval = 21800
        maxval = 25000
        thermal_np = np.clip(thermal_np, minval, maxval)
        thermal_np = (thermal_np - minval) / (maxval - minval)
    
    # Convert back to tensor
    enhanced_tensor = torch.from_numpy(thermal_np.astype(np.float32))
    
    # Ensure output has same format as input
    if enhanced_tensor.dim() == 2 and thermal_tensor.dim() == 3:
        enhanced_tensor = enhanced_tensor.unsqueeze(0)
        if thermal_tensor.shape[0] == 3:
            enhanced_tensor = enhanced_tensor.repeat(3, 1, 1)
    
    return enhanced_tensor