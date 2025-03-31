import torch
import sys
import os
import numpy as np
from pathlib import Path
from collections import OrderedDict

def load_model_and_stats(weights_path, model_type="dust3r", verbose=True, dust3r_path=None, mast3r_path=None):
    """
    Load a DUSt3R or MASt3R model and display/return its structure and stats.
    
    Args:
        weights_path: Path to the model weights
        model_type: Type of model ('dust3r' or 'mast3r')
        verbose: Whether to print detailed information
        dust3r_path: Path to dust3r repository (optional)
        mast3r_path: Path to mast3r repository (optional)
    
    Returns:
        model: The loaded model
        stats: Dictionary with model statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up paths
    if dust3r_path is None:
        dust3r_path = os.path.abspath("/home/user/georges1/dust3r")
    if mast3r_path is None:
        mast3r_path = os.path.abspath("/home/user/georges1/mast3r")
    
    # Ensure paths are in sys.path
    if dust3r_path not in sys.path:
        sys.path.insert(0, dust3r_path)
    if mast3r_path not in sys.path:
        sys.path.insert(0, mast3r_path)
    
    if verbose:
        print(f"DUSt3R path: {dust3r_path}")
        print(f"MASt3R path: {mast3r_path}")
        print(f"sys.path: {sys.path}")
    
    # Load checkpoint to get model structure
    checkpoint = torch.load(weights_path, map_location=device)
    
    try:
        if model_type.lower() == "dust3r":
            # Try to directly load model structure from checkpoint
            if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                # Create a simple placeholder model to analyze structure
                model = SimpleDUSt3RModel(checkpoint['model'])
                if verbose:
                    print("Using simplified model from checkpoint structure")
            else:
                try:
                    # Try importing from dust3r module
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
                    # Load weights
                    if 'model' in checkpoint:
                        model.load_state_dict(checkpoint['model'], strict=False)
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                except ImportError as e:
                    if verbose:
                        print(f"Could not import dust3r module: {e}")
                    # Use simplified model
                    model = SimpleDUSt3RModel(checkpoint.get('model', checkpoint))
        
        elif model_type.lower() == "mast3r":
            try:
                from mast3r.model import AsymmetricMASt3R
                model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
            except ImportError as e:
                if verbose:
                    print(f"Could not import mast3r module: {e}")
                # Use simplified model
                model = SimpleDUSt3RModel(checkpoint.get('model', checkpoint))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a simplified model with the state dict structure
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model = SimpleDUSt3RModel(state_dict)
    
    # Count parameters from state dict
    param_sizes = {}
    for name, param in model.state_dict().items():
        if isinstance(param, torch.Tensor):
            param_sizes[name] = param.numel()
    
    total_params = sum(param_sizes.values())
    
    # Group parameters by layer type
    layer_types = {}
    for name in param_sizes:
        layer_type = name.split('.')[0]
        if layer_type not in layer_types:
            layer_types[layer_type] = 0
        layer_types[layer_type] += param_sizes[name]
    
    # Extract model structure information
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    encoder_params = sum(param_sizes.get(name, 0) for name in param_sizes if 'enc_' in name)
    decoder_params = sum(param_sizes.get(name, 0) for name in param_sizes if 'dec_' in name)
    head_params = sum(param_sizes.get(name, 0) for name in param_sizes if 'head' in name)
    
    stats = {
        'total_parameters': total_params,
        'encoder_parameters': encoder_params,
        'decoder_parameters': decoder_params,
        'head_parameters': head_params,
        'device': str(device),
        'layer_types': layer_types,
        'param_sizes': param_sizes
    }
    
    if verbose:
        print(f"\nModel Type: {model_type.upper()}")
        print(f"Device: {device}")
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Encoder parameters: {encoder_params:,} ({100*encoder_params/total_params:.1f}%)")
        print(f"Decoder parameters: {decoder_params:,} ({100*decoder_params/total_params:.1f}%)")
        print(f"Head parameters: {head_params:,} ({100*head_params/total_params:.1f}%)")
        
        print("\nLayer Type Distribution:")
        for layer_type, params in sorted(layer_types.items(), key=lambda x: x[1], reverse=True):
            percent = 100 * params / total_params
            print(f"  {layer_type:<20}: {params:,} parameters ({percent:.2f}%)")
    
    return model, stats
def visualize_model_architecture(model, checkpoint=None, detailed=False):
    """
    Visualize the architecture of a DUSt3R/MASt3R model.
    
    Args:
        model: The model or a state dictionary
        checkpoint: Optional checkpoint for additional information
        detailed: Whether to show detailed layer information
    
    Returns:
        architecture_info: Dictionary with architecture information
    """
    import torch
    import re
    from collections import OrderedDict
    
    # Extract model structure information
    if isinstance(model, dict):
        state_dict = model
    else:
        state_dict = model.state_dict()
    
    # Analyze layer structure from state dict keys
    layer_pattern = re.compile(r'([a-zA-Z_0-9]+)\.(\d+)\.([a-zA-Z_0-9]+)(?:\.(\d+))?(?:\.([a-zA-Z_0-9]+))?')
    
    architecture = OrderedDict()
    
    # First pass: collect all unique main components
    for key in state_dict.keys():
        parts = key.split('.')
        
        if len(parts) >= 1:
            main_component = parts[0]
            if main_component not in architecture:
                architecture[main_component] = {'type': 'Unknown', 'layers': []}
    
    # Second pass: identify structure
    for component in architecture:
        # Check for blocks structure (encoder/decoder blocks)
        block_pattern = re.compile(f'{component}\\.(\\d+)\\..*')
        
        blocks = set()
        for key in state_dict.keys():
            match = block_pattern.match(key)
            if match:
                blocks.add(int(match.group(1)))
        
        if blocks:
            architecture[component]['layers'] = sorted(list(blocks))
            
            # Look at specific components in encoder/decoder blocks
            if component in ['enc_blocks', 'dec_blocks', 'dec_blocks2']:
                sample_idx = min(blocks)
                sub_pattern = re.compile(f'{component}\\.{sample_idx}\\.(.*?)\\.')
                
                sub_components = set()
                for key in state_dict.keys():
                    match = sub_pattern.match(key)
                    if match:
                        sub_component = match.group(1)
                        if '.' not in sub_component:
                            sub_components.add(sub_component)
                
                architecture[component]['sub_components'] = sorted(list(sub_components))
                
                # Identify block type based on sub-components
                if 'attn' in sub_components and 'mlp' in sub_components:
                    architecture[component]['type'] = 'Transformer Block'
                    
                    # Check for attention type
                    if any('cross_attn' in k for k in state_dict.keys() if component in k):
                        architecture[component]['attention_type'] = 'Self + Cross Attention'
                    else:
                        architecture[component]['attention_type'] = 'Self Attention'
    
    # Determine model dimensions
    if 'patch_embed.proj.weight' in state_dict:
        # Extract embedding dimension
        embed_dim = state_dict['patch_embed.proj.weight'].shape[0]
        architecture['patch_embed']['embed_dim'] = embed_dim
        
        # Extract patch size
        kernel_size = state_dict['patch_embed.proj.weight'].shape[2]
        architecture['patch_embed']['patch_size'] = kernel_size
    
    # Extract encoder dimension
    if 'enc_blocks.0.attn.qkv.weight' in state_dict:
        enc_embed_dim = state_dict['enc_blocks.0.attn.qkv.weight'].shape[1] // 3
        architecture['enc_blocks']['embed_dim'] = enc_embed_dim
        
        # Extract number of heads
        qkv_dim = state_dict['enc_blocks.0.attn.qkv.weight'].shape[0]
        head_dim = state_dict.get('enc_blocks.0.attn.head_dim', enc_embed_dim // (qkv_dim // enc_embed_dim // 3))
        architecture['enc_blocks']['num_heads'] = enc_embed_dim // head_dim
    
    # Extract decoder dimension
    if 'dec_blocks.0.attn.qkv.weight' in state_dict:
        dec_embed_dim = state_dict['dec_blocks.0.attn.qkv.weight'].shape[1] // 3
        architecture['dec_blocks']['embed_dim'] = dec_embed_dim
        
        # Extract number of heads
        qkv_dim = state_dict['dec_blocks.0.attn.qkv.weight'].shape[0]
        head_dim = state_dict.get('dec_blocks.0.attn.head_dim', dec_embed_dim // (qkv_dim // dec_embed_dim // 3))
        architecture['dec_blocks']['num_heads'] = dec_embed_dim // head_dim
    
    # Print architecture
    print("DUSt3R Model Architecture")
    
    print("===============================")
    
    # Encoder
    print("\nEncoder:")
    if 'patch_embed' in architecture:
        patch_size = architecture['patch_embed'].get('patch_size', 'unknown')
        embed_dim = architecture['patch_embed'].get('embed_dim', 'unknown')
        print(f"  Patch Embedding: {patch_size}×{patch_size} patches → {embed_dim} dim")
    
    if 'enc_blocks' in architecture:
        num_blocks = len(architecture['enc_blocks']['layers'])
        embed_dim = architecture['enc_blocks'].get('embed_dim', 'unknown')
        num_heads = architecture['enc_blocks'].get('num_heads', 'unknown')
        attn_type = architecture['enc_blocks'].get('attention_type', 'Self Attention')
        print(f"  Transformer Encoder: {num_blocks} blocks with {embed_dim} dim, {num_heads} heads")
        print(f"  Attention Type: {attn_type}")
    
    # Decoder
    print("\nDecoder:")
    if 'decoder_embed' in architecture:
        print(f"  Decoder Embedding: Maps encoder features to decoder space")
    
    if 'dec_blocks' in architecture:
        num_blocks = len(architecture['dec_blocks']['layers'])
        embed_dim = architecture['dec_blocks'].get('embed_dim', 'unknown')
        num_heads = architecture['dec_blocks'].get('num_heads', 'unknown')
        attn_type = architecture['dec_blocks'].get('attention_type', 'Self Attention')
        print(f"  Transformer Decoder: {num_blocks} blocks with {embed_dim} dim, {num_heads} heads")
        print(f"  Attention Type: {attn_type}")
    
    if 'dec_blocks2' in architecture:
        num_blocks = len(architecture['dec_blocks2']['layers'])
        embed_dim = architecture['dec_blocks2'].get('embed_dim', 'unknown')
        print(f"  Second Decoder Branch: {num_blocks} blocks with {embed_dim} dim")
        print(f"  Dual-branch decoder structure (for processing two views)")
    
    # Output Heads
    print("\nOutput Heads:")
    head_components = [comp for comp in architecture if 'head' in comp.lower()]
    for head in head_components:
        print(f"  {head}: Converts features to final outputs")
    
    # Additional Information
    if detailed:
        print("\nDetailed Architecture:")
        for component, info in architecture.items():
            print(f"\n{component}:")
            for key, value in info.items():
                if isinstance(value, list) and len(value) > 10:
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")
    
    return architecture
class SimpleDUSt3RModel(torch.nn.Module):
    """Simple placeholder model that has the same state_dict structure"""
    def __init__(self, state_dict):
        super().__init__()
        self._state_dict = state_dict
        
        # Create dummy parameters for structure analysis
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                setattr(self, name.replace('.', '_'), torch.nn.Parameter(torch.zeros(1)))
    
    def state_dict(self):
        return self._state_dict
    
    def forward(self, x):
        raise NotImplementedError("This is a placeholder model for analysis only")