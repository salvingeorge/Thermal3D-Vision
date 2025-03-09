#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_dustr_model(weights_path, device=None):
    """Load DUSt3R model checkpoint from a local file.
    
    This version checks that the checkpoint file exists locally and passes
    the flag weights_only=True so that the model is loaded from the local file,
    rather than attempting to download from Hugging Face.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    from mast3r.model import AsymmetricMASt3R
    # Pass weights_only=True so that it loads the local checkpoint.
    model = AsymmetricMASt3R.from_pretrained(weights_path, weights_only=True).to(device)
    model.eval()
    return model


def preprocess_thermal_image(thermal_path, target_size=(224, 224)):
    """
    Read a thermal image from disk, convert to float32, resize to target_size,
    and replicate channels to get a 3-channel image if needed.
    Returns a torch.Tensor of shape [3, H, W].
    """
    thermal = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
    if thermal is None:
        raise FileNotFoundError(f"Could not read thermal image: {thermal_path}")
    
    # If 16-bit, scale to 0..1
    if thermal.dtype == np.uint16:
        thermal = thermal.astype(np.float32) / 65535.0
    else:
        # e.g. 8-bit
        thermal = thermal.astype(np.float32) / 255.0
    
    # Possibly replicate channels if it's single-channel
    if len(thermal.shape) == 2:
        thermal = np.stack([thermal]*3, axis=-1)  # [H,W] -> [H,W,3]
    
    # Resize
    thermal_resized = cv2.resize(thermal, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor [3, H, W]
    tensor = torch.from_numpy(thermal_resized.transpose(2,0,1)).float()
    return tensor

def flatten_tensor(x):
    """
    Recursively extract the first element until x is not a tuple/list, then ensure it's a torch.Tensor.
    """
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            break
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x

def extract_depth_from_output(output):
    """
    Minimal extraction of depth from DUSt3R output. We'll assume 'pred1' has 'pts3d' or similar.
    We'll ignore calibration, etc. This just returns the z-channel from pred1.
    """
    pred1 = output.get("pred1", {})
    if "pts3d" not in pred1:
        raise KeyError("Output does not contain 'pts3d' in pred1 keys: " + str(list(pred1.keys())))
    pts3d = flatten_tensor(pred1["pts3d"])
    
    # If shape is [2, H, W, 3], pick the first
    if pts3d.shape[0] > 1:
        pts3d = pts3d[0]
    
    # Now it's [H, W, 3], extract the z channel
    pts3d_np = pts3d.cpu().numpy()  # shape [H,W,3]
    depth = pts3d_np[:,:,2]
    return depth

def main():
    parser = argparse.ArgumentParser(description="DUSt3R inference on a single thermal image.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to DUSt3R checkpoint (e.g. DUSt3R_ViTLarge_BaseDecoder_224_linear.pth)")
    parser.add_argument("--thermal_image", type=str, required=True, help="Path to the thermal image file")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Resize dimension (width height)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1) Load model
    model = load_dustr_model(args.checkpoint, device)
    
    # 2) Preprocess the thermal image
    thermal_tensor = preprocess_thermal_image(args.thermal_image, target_size=tuple(args.img_size))
    thermal_tensor = thermal_tensor.unsqueeze(0).to(device)  # shape [B=1, 3, H, W]
    
    # 3) Build pair data for DUSt3R inference. We'll feed the same image as both views.
    #    Because DUSt3R expects a pair of images in dust3r.inference.
    pair_data = [({"img": thermal_tensor, "instance": []}, {"img": thermal_tensor, "instance": []})]
    
    # 4) Inference
    from dust3r.inference import inference
    with torch.no_grad():
        output = inference(pair_data, model, device, batch_size=1, verbose=False)
    
    # 5) Extract depth from output
    try:
        pred_depth = extract_depth_from_output(output)
    except Exception as e:
        print(f"Error extracting depth: {e}")
        return
    
    # 6) Visualize
    plt.figure(figsize=(6,6))
    plt.imshow(pred_depth, cmap="plasma")
    plt.colorbar(label="Depth")
    plt.title("DUSt3R Predicted Depth")
    plt.savefig("predicted_depth.png")
    print("Saved predicted depth visualization to predicted_depth.png")
    # plt.show()

if __name__ == "__main__":
    main()
