#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_dustr_model(weights_path, device=None):
    """
    Load DUSt3R model checkpoint from a local file, forcing it to load 
    from disk instead of from Hugging Face.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(weights_path, weights_only=True).to(device)
    model.eval()
    return model

def preprocess_thermal_image(thermal_path, target_size=(224, 224)):
    """
    Read a thermal image from disk, convert to float32, resize to target_size,
    and replicate channels if needed. Returns a torch.Tensor of shape [3, H, W].
    """
    thermal = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
    if thermal is None:
        raise FileNotFoundError(f"Could not read thermal image: {thermal_path}")
    
    # If 16-bit, scale to [0..1]
    if thermal.dtype == np.uint16:
        thermal = thermal.astype(np.float32) / 65535.0
    else:
        thermal = thermal.astype(np.float32) / 255.0
    
    # If single-channel, replicate 3 times
    if len(thermal.shape) == 2:
        thermal = np.stack([thermal]*3, axis=-1)
    
    # Resize
    thermal_resized = cv2.resize(thermal, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor [3, H, W]
    tensor = torch.from_numpy(thermal_resized.transpose(2,0,1)).float()
    return tensor

def flatten_tensor(x):
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            break
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x

def extract_depth_from_output(output):
    """
    Minimal extraction of depth from DUSt3R output. We'll assume 'pred1' has 'pts3d'.
    Returns the z-channel from pred1.
    """
    pred1 = output.get("pred1", {})
    if "pts3d" not in pred1:
        raise KeyError("Output does not contain 'pts3d' in pred1 keys: " + str(list(pred1.keys())))
    pts3d = flatten_tensor(pred1["pts3d"])
    # If shape is [2, H, W, 3], pick the first
    if pts3d.shape[0] > 1:
        pts3d = pts3d[0]
    
    # [H, W, 3], extract z
    pts3d_np = pts3d.cpu().numpy()
    depth = pts3d_np[:,:,2]
    return depth

def main():
    parser = argparse.ArgumentParser(description="DUSt3R inference on a single thermal image, comparing with pseudo-GT.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DUSt3R checkpoint (e.g. DUSt3R_ViTLarge_BaseDecoder_224_linear.pth)")
    parser.add_argument("--thermal_image", type=str, required=True, help="Path to the thermal image file")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Resize dimension (width height)")
    parser.add_argument("--pseudo_gt_depth_dir", type=str, default="pseudo_gt/depth",
                        help="Folder containing pseudo-GT depth .npy files, named to match thermal base name")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1) Load model
    model = load_dustr_model(args.checkpoint, device)
    
    # 2) Preprocess the thermal image
    thermal_tensor = preprocess_thermal_image(args.thermal_image, target_size=tuple(args.img_size))
    thermal_tensor = thermal_tensor.unsqueeze(0).to(device)  # shape [1, 3, H, W]
    
    # 3) Construct pair data for DUSt3R (same image for both views)
    pair_data = [({"img": thermal_tensor, "instance": []},
                  {"img": thermal_tensor, "instance": []})]
    
    # 4) Inference
    from dust3r.inference import inference
    with torch.no_grad():
        output = inference(pair_data, model, device, batch_size=1, verbose=False)
    
    # 5) Extract predicted depth
    try:
        pred_depth = extract_depth_from_output(output)
    except Exception as e:
        print(f"Error extracting depth: {e}")
        return
    
    # 6) Load the matching pseudo-GT depth if it exists
    base_name = os.path.splitext(os.path.basename(args.thermal_image))[0]
    base_name_for_gt = base_name.replace("fl_ir_aligned", "fl_rgb")
    pseudo_depth_path = os.path.join(args.pseudo_gt_depth_dir, f"{base_name_for_gt}.npy")
    
    pseudo_gt_depth = None
    if os.path.isfile(pseudo_depth_path):
        pseudo_gt_depth = np.load(pseudo_depth_path)
    else:
        print(f"Warning: No pseudo-GT depth found at {pseudo_depth_path}")
        pseudo_gt_depth = None
        
    # # 7) Visualize
    # plt.figure(figsize=(6,6))
    # plt.imshow(pred_depth, cmap="plasma")
    # plt.colorbar(label="Depth")
    # plt.title("DUSt3R Predicted Depth")
    # plt.savefig("predicted_depth.png")
    # print("Saved predicted depth visualization to predicted_depth.png")
    # # plt.show()
    
    # 7) Visualize side by side
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    
    # Left: predicted depth
    im0 = axes[0].imshow(pred_depth, cmap="plasma")
    axes[0].set_title("Predicted Depth")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Right: pseudo-GT depth (if found)
    if pseudo_gt_depth is not None:
        im1 = axes[1].imshow(pseudo_gt_depth, cmap="plasma")
        axes[1].set_title("Pseudo-GT Depth")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].text(0.5, 0.5, "No pseudo-GT found", ha='center', va='center')
        axes[1].axis("off")
    
    plt.suptitle(f"DUSt3R vs. Pseudo-GT\n({base_name})", fontsize=14)
    plt.tight_layout()
    plt.savefig("compare_depths.png")
    print("Saved side-by-side predicted vs. pseudo-GT to compare_depths.png")

if __name__ == "__main__":
    main()

