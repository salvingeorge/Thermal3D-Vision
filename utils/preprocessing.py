import torch
import numpy as np

def enhance_thermal_contrast(thermal_tensor):
    """Apply contrast enhancement to thermal images to help model extract features"""
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