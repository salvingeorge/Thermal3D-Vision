import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import glob
from pathlib import Path

def load_mast3r_model(weights_path, device=None):
    """Load MASt3R model for generating pseudo-annotations."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Adjust these paths to match your setup
        mast3r_path = os.path.abspath("mast3r")
        sys.path.append(mast3r_path)
        from mast3r.model import AsymmetricMASt3R
        model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
        model.eval()
        return model
    except ImportError as e:
        print(f"Could not import MASt3R: {e}")
        sys.exit(1)

def process_test_images(rgb_dir, thermal_dir, output_dir, model, device):
    """Process test images to generate pseudo-GT."""
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)
    
    # Get all RGB images
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))
    
    for rgb_path in tqdm(rgb_files, desc="Generating pseudo-GT for test images"):
        # Extract base filename
        rgb_basename = os.path.basename(rgb_path)
        
        # Fix the filename pattern: Remove the extra '0' before '.png'
        # Extract the numeric part before '_rgb.png'
        if '_rgb.png' in rgb_basename:
            # Original pattern: 'fl_ir_aligned_1570730891_1919874440_rgb.png'
            # Target pattern:  'fl_ir_aligned_1570730891_191987444_ir.png'
            
            parts = rgb_basename.split('_')
            numeric_part = parts[-2]  # Get the number before '_rgb.png'
            
            # Remove last '0' to match thermal filename pattern
            if numeric_part.endswith('0'):
                fixed_numeric = numeric_part[:-1]
                thermal_basename = rgb_basename.replace(f"_{numeric_part}_rgb.png", f"_{fixed_numeric}_ir.png")
            else:
                # If no trailing 0, try direct replacement
                thermal_basename = rgb_basename.replace("_rgb.png", "_ir.png")
        else:
            # Fallback if naming pattern is different
            thermal_basename = rgb_basename.replace("rgb", "ir")
        
        thermal_path = os.path.join(thermal_dir, thermal_basename)
        
        if not os.path.exists(thermal_path):
            print(f"Warning: No matching thermal image found for {rgb_path}")
            # Try an alternate method - list all thermal files and find closest match
            thermal_files = os.listdir(thermal_dir)
            rgb_id = rgb_basename.split('_')[-2]  # Get numeric ID
            closest_match = None
            
            for thermal_file in thermal_files:
                # Try to find a similar timestamp/ID
                if rgb_id[:-1] in thermal_file:  # Use all but last digit
                    closest_match = thermal_file
                    break
            
            if closest_match:
                thermal_path = os.path.join(thermal_dir, closest_match)
                print(f"Found alternative match: {closest_match}")
            else:
                continue
        
        # Rest of the processing remains the same...
        # Load and preprocess RGB image
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            print(f"Warning: Could not read RGB image {rgb_path}")
            continue
            
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (512, 512))
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_img.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # Create view dict for model
        view = {"img": rgb_tensor, "instance": []}
        
        # Run inference (monocular mode with duplicate input)
        try:
            with torch.no_grad():
                output = model(view, view)
            
            # Extract depth from pointmap
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output.get("pred1", {})
            
            if isinstance(pred, dict):
                pred_pointmap = pred.get("pts3d")
            else:
                pred_pointmap = pred
            
            # Handle batch dimension
            if len(pred_pointmap.shape) == 4:  # [B, H, W, 3]
                pred_pointmap = pred_pointmap[0]  # [H, W, 3]
            
            # Extract depth (Z coordinate)
            depth = pred_pointmap[..., 2].cpu().numpy()
            
            # Save depth
            output_name = os.path.splitext(os.path.basename(rgb_path))[0]
            np.save(os.path.join(depth_dir, f"{output_name}_depth.npy"), depth)
            
            # Also save a reference to the matched thermal image for later evaluation
            with open(os.path.join(depth_dir, f"{output_name}_thermal_path.txt"), 'w') as f:
                f.write(thermal_path)
            
            # Also save visualization
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_img.transpose(2, 0, 1).transpose(1, 2, 0))
            plt.title('RGB Image')
            plt.axis('off')
            
            # Also display the thermal image for verification
            thermal_img = cv2.imread(thermal_path)
            if thermal_img is not None:
                thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)
                plt.subplot(1, 3, 2)
                plt.imshow(thermal_img)
                plt.title('Thermal Image')
                plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(depth, cmap='plasma')
            plt.title('Predicted Depth')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(depth_dir, f"{output_name}_depth_vis.png"))
            plt.close()
            
        except Exception as e:
            print(f"Error processing {rgb_path}: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-GT for test images")
    parser.add_argument('--rgb_dir', type=str, required=True, help="Directory with RGB test images")
    parser.add_argument('--thermal_dir', type=str, required=True, help="Directory with thermal test images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save pseudo-GT")
    parser.add_argument('--weights', type=str, default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                        help="Path to the MASt3R model weights")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_mast3r_model(args.weights, device)
    
    process_test_images(args.rgb_dir, args.thermal_dir, args.output_dir, model, device)
    print(f"Pseudo-GT generation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    main()