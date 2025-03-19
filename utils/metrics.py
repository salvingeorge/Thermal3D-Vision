import numpy as np
import torch

def compute_depth_metrics(pred_depth, gt_depth, mask=None, median_scaling=True):
    """
    Compute standard metrics for depth estimation evaluation.
    
    Args:
        pred_depth: Predicted depth map [H, W]
        gt_depth: Ground truth depth map [H, W]
        mask: Optional mask for valid depth values [H, W]
        median_scaling: Whether to scale prediction to match GT median
    
    Returns:
        Dictionary of metrics
    """
    # Move to CPU and convert to numpy for easier processing
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    if isinstance(gt_depth, torch.Tensor):
        gt_depth = gt_depth.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Create mask for valid depth values (non-zero, non-inf, non-nan)
    if mask is None:
        mask = (gt_depth > 0) & np.isfinite(gt_depth)
    
    # Apply mask
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    
    # Skip if no valid pixels
    if pred.size == 0:
        return {
            'abs_rel': np.nan,
            'sq_rel': np.nan,
            'rmse': np.nan,
            'rmse_log': np.nan,
            'a1': 0.0,
            'a2': 0.0,
            'a3': 0.0
        }
    
    # Apply median scaling if requested
    if median_scaling:
        scale = np.median(gt) / np.median(pred)
        pred *= scale
    
    # Calculate metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt) - np.log(pred)) ** 2))
    
    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'acc_1': a1,  # Also known as δ < 1.25
        'acc_2': a2,  # δ < 1.25²
        'acc_3': a3   # δ < 1.25³
    }


def evaluate_thermal_depth(model, dataloader, device):
    """
    Evaluate a model on a thermal depth dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        Dict of average metrics
    """
    model.eval()
    metrics_sum = {
        'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 
        'rmse_log': 0, 'acc_1': 0, 'acc_2': 0, 'acc_3': 0
    }
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            thermal1 = batch["thermal1"].to(device)
            
            # If we have ground truth depth
            if "depth1" in batch and batch["depth1"] is not None:
                gt_depth = batch["depth1"].to(device)
                
                # For each sample in batch
                for i in range(thermal1.size(0)):
                    # Forward pass (monocular mode)
                    view = {"img": thermal1[i:i+1], "instance": []}
                    output = model(view, view)
                    
                    # Extract predicted depth
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
                    pred_depth = pred_pointmap[..., 2]
                    curr_gt_depth = gt_depth[i]
                    
                    # Compute metrics
                    sample_metrics = compute_depth_metrics(pred_depth, curr_gt_depth)
                    
                    # Accumulate metrics
                    for key in metrics_sum:
                        if np.isfinite(sample_metrics[key]):
                            metrics_sum[key] += sample_metrics[key]
                    
                    sample_count += 1
    
    # Average metrics
    avg_metrics = {k: v / sample_count if sample_count > 0 else np.nan 
                  for k, v in metrics_sum.items()}
    
    return avg_metrics