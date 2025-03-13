#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader

# Import the dataset
from data.freiburg_dataset import FreiburgDataset

def flatten_tensor(x):
    """Recursively extract the first element until x is not a tuple/list,
    then ensure it's a torch.Tensor."""
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            break
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x

def extract_depth_from_output(output):
    """
    Extract the depth (z-channel) from the DUSt3R model output.
    Assumes that output["pred1"] contains a key 'pts3d' holding the predicted 3D pointmap.
    """
    pred1 = output.get("pred1", {})
    if "pts3d" not in pred1:
        raise KeyError("Output does not contain 'pts3d'. Keys: " + str(list(pred1.keys())))
    pts3d = flatten_tensor(pred1["pts3d"])
    # If there are multiple predictions, use the first one.
    if pts3d.shape[0] > 1:
        pts3d = pts3d[0]
    # Expecting pts3d shape to be [H, W, 3]; extract z channel.
    pts3d_np = pts3d.cpu().numpy()
    depth = pts3d_np[:, :, 2]
    return depth

def load_dustr_model(weights_path, device=None):
    """Load DUSt3R model checkpoint from a local file, handling missing 'args' if necessary."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    from mast3r.model import AsymmetricMASt3R, load_model
    # Load the checkpoint manually to check for the 'args' key.
    ckpt = torch.load(weights_path, map_location='cpu')
    if 'args' not in ckpt:
        print("Warning: 'args' key not found in checkpoint; inserting dummy args.")
        # Create a dummy object with a model attribute; adjust as needed.
        class DummyArgs:
            pass
        dummy = DummyArgs()
        dummy.model = ""
        ckpt['args'] = dummy
    
    # Now load the model as usual.
    model = AsymmetricMASt3R.from_pretrained(weights_path, weights_only=True).to(device)
    model.eval()
    return model
def compute_depth_metrics(pred, gt, eps=1e-6):
    """
    Compute common monocular depth estimation metrics.
    Both pred and gt should be numpy arrays of shape [H, W].
    Only evaluate on pixels where gt > 0.
    Returns a dictionary of metrics.
    """
    valid = gt > 0
    pred = pred[valid]
    gt = gt[valid]
    
    abs_rel = np.mean(np.abs(pred - gt) / (gt + eps))
    rmse = np.sqrt(np.mean((pred - gt)**2))
    rmse_log = np.sqrt(np.mean((np.log(pred + eps) - np.log(gt + eps))**2))
    
    ratio = np.maximum(pred / (gt + eps), gt / (pred + eps))
    delta1 = np.mean(ratio < 1.25)
    delta2 = np.mean(ratio < 1.25**2)
    delta3 = np.mean(ratio < 1.25**3)
    
    return {
        "abs_rel": abs_rel,
        "rmse": rmse,
        "rmse_log": rmse_log,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3
    }

def save_inference_visualization(thermal_img, pred_depth, vis_dir, sample_idx):
    """
    Save a side-by-side visualization of the thermal image and the predicted depth.
    The output filename includes a timestamp and sample index.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(vis_dir, f"vis_{sample_idx}_{timestamp}.png")
    plt.figure(figsize=(10,5))
    
    # Thermal image (assumed to be a numpy array in [H,W,3])
    plt.subplot(1, 2, 1)
    plt.imshow(thermal_img, cmap='inferno')
    plt.title("Thermal Image")
    plt.axis('off')
    
    # Predicted depth visualization
    plt.subplot(1, 2, 2)
    plt.imshow(pred_depth, cmap='plasma')
    plt.title("Predicted Depth")
    plt.axis('off')
    plt.colorbar(orientation='vertical')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

def evaluate(model, dataloader, device, vis_dir=None, save_vis=False):
    """Evaluate the model on the given dataloader and compute depth metrics.
    Optionally, save visualizations if save_vis is True.
    """
    metrics_sum = {
        "abs_rel": 0.0,
        "rmse": 0.0,
        "rmse_log": 0.0,
        "delta1": 0.0,
        "delta2": 0.0,
        "delta3": 0.0
    }
    num_samples = 0
    
    from dust3r.inference import inference  # Assuming inference function is in dust3r.inference
    
    for batch in dataloader:
        if not batch or "thermal" not in batch or "depth" not in batch:
            continue
        
        thermal_imgs = batch["thermal"].to(device)  # [B, 3, H, W]
        gt_depths = batch["depth"].to(device)         # [B, H, W]
        batch_size = thermal_imgs.size(0)
        
        # Prepare pair data: duplicate the thermal image to form a stereo pair.
        pair_data = []
        for i in range(batch_size):
            img = thermal_imgs[i].unsqueeze(0)  # [1, 3, H, W]
            pair_data.append(({"img": img, "instance": []}, {"img": img, "instance": []}))
        
        with torch.no_grad():
            output = inference(pair_data, model, device, batch_size=1, verbose=False)
        
        # Process each sample in the batch
        for i in range(batch_size):
            try:
                pred_depth = extract_depth_from_output(output[i])
                gt_depth = gt_depths[i].cpu().numpy()
                # Resize predicted depth if necessary to match gt dimensions
                if pred_depth.shape != gt_depth.shape:
                    pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_LINEAR)
                sample_metrics = compute_depth_metrics(pred_depth, gt_depth)
                for key in metrics_sum:
                    metrics_sum[key] += sample_metrics[key]
                num_samples += 1
                
                # If saving visualizations is enabled, generate and save the figure.
                if save_vis and vis_dir is not None:
                    # Convert thermal image tensor to numpy array in [H, W, 3]
                    thermal_np = thermal_imgs[i].permute(1, 2, 0).cpu().numpy()
                    # Optionally convert thermal image to proper range (if normalized between 0 and 1)
                    thermal_np = np.clip(thermal_np, 0, 1)
                    save_inference_visualization(thermal_np, pred_depth, vis_dir, sample_idx=num_samples)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
    
    # Average metrics over all samples
    averaged_metrics = {k: metrics_sum[k] / num_samples for k in metrics_sum}
    return averaged_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DUSt3R for monocular depth estimation on thermal images, with visualization output.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the Freiburg dataset")
    parser.add_argument("--pseudo_gt_dir", type=str, required=True, help="Path to the pseudo-GT annotations directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the fine-tuned DUSt3R checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224], help="Image resize dimensions (width height)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--save_vis", action="store_true", help="If set, save visualization images of thermal input and predicted depth")
    parser.add_argument("--vis_dir", type=str, default="evaluation_vis", help="Directory to save visualization images")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create dataset with pseudo-GT enabled
    dataset = FreiburgDataset(
        root_dir=args.dataset_dir,
        sequences=None,
        img_size=tuple(args.img_size),
        use_pseudo_gt=True,
        pseudo_gt_dir=args.pseudo_gt_dir
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load the fine-tuned DUSt3R model
    model = load_dustr_model(args.checkpoint, device)
    
    # Create visualization directory if saving visuals is enabled
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
    
    print("Starting evaluation...")
    metrics = evaluate(model, dataloader, device, vis_dir=args.vis_dir, save_vis=args.save_vis)
    print("Evaluation Metrics:")
    print(f"Abs Rel Error: {metrics['abs_rel']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"RMSE (log): {metrics['rmse_log']:.4f}")
    print(f"Delta <1.25: {metrics['delta1']*100:.2f}%")
    print(f"Delta <1.25^2: {metrics['delta2']*100:.2f}%")
    print(f"Delta <1.25^3: {metrics['delta3']*100:.2f}%")
    
if __name__ == "__main__":
    main()
