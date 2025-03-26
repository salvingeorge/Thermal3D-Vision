# thermal_dustr_model.py
import os
import sys
import torch
import torch.nn as nn
import types

def load_dustr_model(weights_path, device=None, is_thermal=False):
    """
    Load DUSt3R model with support for thermal imagery.
    
    Args:
        weights_path: Path to the DUSt3R checkpoint
        device: Device to load the model on (default: auto-detect)
        is_thermal: Whether to further optimize the model for thermal imagery
        
    Returns:
        Loaded model on the specified device, ready for inference or fine-tuning
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    # Add dust3r path to sys.path if needed
    dust3r_path = os.path.abspath("dust3r")
    if dust3r_path not in sys.path:
        sys.path.append(dust3r_path)
    
    # Import model class
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
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
        # Load state_dict from checkpoint
        checkpoint = torch.load(weights_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # Possibly it's just raw weights
            model.load_state_dict(checkpoint, strict=False)
    except ImportError as e:
        print(f"Error importing model classes: {e}")
        raise ImportError(f"Could not import DUSt3R model classes. Please ensure the repository is properly installed.")
    
    # Move model to device
    model = model.to(device)
    
    # Patch _encode_image method if needed
    original_encode_image = model._encode_image

    def patched_encode_image(self, image, true_shape=None):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        return x, pos, None

    model._encode_image = types.MethodType(patched_encode_image, model)
    
    # Set to training mode
    model.train()
    
    # Make all parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    
    return model


class ThermalDUSt3R(nn.Module):
    """
    Wrapper class for thermal-optimized DUSt3R model with specific
    enhancements for thermal imagery characteristics
    """
    def __init__(self, base_model):
        super(ThermalDUSt3R, self).__init__()
        self.model = base_model
        
        # Define Sobel filters for edge detection in thermal images
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # Register sobel filters as non-trainable buffers
        self.register_buffer('sobel_x', sobel_x_kernel.repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y_kernel.repeat(3, 1, 1, 1))
        
        # Edge enhancement weight - how much to emphasize thermal gradients
        self.edge_weight = nn.Parameter(torch.tensor(0.5))
        
        # Thermal normalization parameters
        self.temp_scale = nn.Parameter(torch.tensor(1.0))
        self.use_local_normalization = True
    
    def preprocess_thermal(self, x):
        """
        Apply thermal-specific preprocessing to enhance features
        that matter for depth estimation in thermal imagery
        """
        # Handle single-channel thermal images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Temperature normalization (crucial for thermal imagery)
        if self.use_local_normalization:
            # Apply per-batch normalization
            batch_min = x.amin(dim=(2, 3), keepdim=True)
            batch_max = x.amax(dim=(2, 3), keepdim=True)
            x = (x - batch_min) / (batch_max - batch_min + 1e-6)
        
        # Make sure the Sobel filters are on the same device as x
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        
        # Extract and enhance thermal edges (crucial since thermal lacks texture)
        edge_x = torch.abs(torch.nn.functional.conv2d(x, sobel_x, padding=1, groups=3))
        edge_y = torch.abs(torch.nn.functional.conv2d(x, sobel_y, padding=1, groups=3))
        edge_magnitude = torch.sqrt(edge_x.pow(2) + edge_y.pow(2))
        
        # Combine original thermal image with enhanced edges
        enhanced = x + self.edge_weight * edge_magnitude
        
        # Final scaling
        enhanced = enhanced * self.temp_scale
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return enhanced
    
    def forward(self, view1, view2):
        # Apply thermal preprocessing depending on input format
        if isinstance(view1, dict) and "img" in view1:
            # DUSt3R dictionary format
            view1_copy = view1.copy()  # Create copy to avoid modifying original
            view2_copy = view2.copy()
            
            # Preprocess thermal images
            view1_copy["img"] = self.preprocess_thermal(view1["img"])
            view2_copy["img"] = self.preprocess_thermal(view2["img"])
            
            # Forward through base model
            return self.model(view1_copy, view2_copy)
        else:
            # Direct tensor input
            enhanced_view1 = self.preprocess_thermal(view1)
            enhanced_view2 = self.preprocess_thermal(view2)
            return self.model(enhanced_view1, enhanced_view2)
    
    @classmethod
    def from_pretrained(cls, weights_path, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the checkpoint
        checkpoint = torch.load(weights_path, map_location=device)
        if 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state = checkpoint['model']
        else:
            state = checkpoint
        # Load the base model (this loads only the DUSt3R part)
        base_model = load_dustr_model(weights_path, device, is_thermal=True)
        # Wrap the base model with ThermalDUSt3R
        model_wrapper = cls(base_model)
        
        new_state = {}
        for key, value in state.items():
            if key.startswith("model."):
                new_state[key[len("model."):]] = value
            else:
                new_state[key] = value
        # Load the complete state dict into the wrapper (non-strict to allow mismatches)
        model_wrapper.load_state_dict(new_state, strict=False)
        return model_wrapper

    
    def save_checkpoint(self, path, optimizer=None, epoch=None, val_loss=None, args=None):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "val_loss": val_loss,
            "args": args
        }
        torch.save(checkpoint, path)

# class ThermalDUSt3R(nn.Module):
#     """
#     Simplified wrapper class for DUSt3R model without thermal-specific modifications
#     """
#     def __init__(self, base_model):
#         super(ThermalDUSt3R, self).__init__()
#         self.model = base_model
    
#     def forward(self, view1, view2):
#         # Simply pass inputs directly to the base model without any preprocessing
#         return self.model(view1, view2)
    
#     @classmethod
#     def from_pretrained(cls, weights_path, device=None):
#         """Load model from pretrained weights"""
#         base_model = load_dustr_model(weights_path, device, is_thermal=False)
#         return cls(base_model)
    
#     def save_checkpoint(self, path, optimizer=None, epoch=None, val_loss=None, args=None):
#         """Save model checkpoint"""
#         checkpoint = {
#             "epoch": epoch,
#             "state_dict": self.state_dict(),
#             "optimizer": optimizer.state_dict() if optimizer is not None else None,
#             "val_loss": val_loss,
#             "args": args
#         }
#         torch.save(checkpoint, path)