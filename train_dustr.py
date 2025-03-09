#!/usr/bin/env python3
import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, default_collate
import torch.nn.functional as F

# Import your dataset class.
from data.freiburg_dataset import FreiburgDataset

def skip_none_collate(batch):
    """Custom collate function to filter out None samples."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

def load_dustr_model(weights_path, device=None):
    """Load DUSt3R model checkpoint from a local file.
    
    We assume that a small DUSt3R variant checkpoint is provided.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Checkpoint not found at {weights_path}")
    
    mast3r_path = os.path.abspath("mast3r")
    if mast3r_path not in sys.path:
        sys.path.append(mast3r_path)
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(weights_path, weights_only=True).to(device)
    model.train()  # Set to training mode
    # Ensure all parameters require gradients (in case some are frozen by default)
    for param in model.parameters():
        param.requires_grad = True
    return model

def flatten_tensor(x):
    """Recursively extract the first element until x is a torch.Tensor."""
    while isinstance(x, (tuple, list)):
        if len(x) == 0:
            break
        x = x[0]
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x

def extract_depth_from_output(output):
    """
    Extract predicted depth from DUSt3R output as a torch tensor.
    The model may return a tuple; in that case, we assume output[0] corresponds to 'pred1'.
    If output is a dict, we try to get 'pts3d' from it.
    Returns the z-channel (depth) as a tensor.
    """
    # If output is a tuple, assume first element is pred1.
    if isinstance(output, tuple):
        pred1 = output[0]
    elif isinstance(output, dict):
        pred1 = output.get("pred1", {})
    else:
        raise ValueError("Unexpected output type from model.")
    
    # If pred1 is a dict, extract pts3d; otherwise assume pred1 is already pts3d.
    if isinstance(pred1, dict):
        if "pts3d" not in pred1:
            raise KeyError("Output does not contain 'pts3d'. Keys: " + str(list(pred1.keys())))
        pts3d = flatten_tensor(pred1["pts3d"])
    else:
        pts3d = flatten_tensor(pred1)
    
    # If pts3d is of shape [2, H, W, 3], select the first element.
    if pts3d.shape[0] > 1:
        pts3d = pts3d[0]
    
    # Slice the z-channel.
    depth_tensor = pts3d[:, :, 2]  # shape [H, W]
    # Ensure the tensor requires grad.
    return depth_tensor

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DUSt3R on pseudo-annotated thermal images")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the Freiburg dataset")
    parser.add_argument("--pseudo_gt_dir", type=str, required=True,
                        help="Path to the pseudo-GT annotations directory")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the small DUSt3R checkpoint (e.g., DUSt3R_small_BaseDecoder_224_linear.pth)")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224,224],
                        help="Resize dimension (width height)")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create the training dataset.
    train_dataset = FreiburgDataset(
        root_dir=args.dataset_dir,
        sequences=None,
        img_size=tuple(args.img_size),
        use_pseudo_gt=True,
        pseudo_gt_dir=args.pseudo_gt_dir
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, collate_fn=skip_none_collate)

    # Load the DUSt3R model (small variant)
    model = load_dustr_model(args.weights, device)

    # Define a simple L1 loss for depth regression.
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting fine-tuning DUSt3R on thermal images with pseudo-GT depth...")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
            if batch is None or "thermal" not in batch or "depth" not in batch:
                continue
            thermal_imgs = batch["thermal"].to(device)  # [B, 3, H, W]
            gt_depths = batch["depth"].to(device)         # [B, H_gt, W_gt]

            optimizer.zero_grad()
            loss_total = torch.tensor(0.0, device=device)
            B = thermal_imgs.size(0)
            for i in range(B):
                # Prepare pair: duplicate the thermal image.
                view1 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                view2 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                # Forward pass: call model directly (do not use inference() wrapper).
                output = model(view1, view2)
                try:
                    pred_depth_tensor = extract_depth_from_output(output)  # [H, W]
                    pred_depth_tensor = pred_depth_tensor.to(device)
                    gt_depth_sample = gt_depths[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H_gt, W_gt]
                    gt_depth_resized = F.interpolate(gt_depth_sample, size=(args.img_size[1], args.img_size[0]),
                                                    mode='bilinear', align_corners=False)
                    gt_depth_resized = gt_depth_resized.squeeze(0).squeeze(0)  # [args.img_size[1], args.img_size[0]]
                    
                    loss = criterion(pred_depth_tensor, gt_depth_resized)
                    loss_total += loss
                except Exception as e:
                    print(f"Error in loss computation: {e}")
                # Delete output to free memory and empty cache
                del output
                torch.cuda.empty_cache()
            if B > 0:
                loss_total = loss_total / B
                loss_total.backward()
                optimizer.step()
                running_loss += loss_total.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), args.output_model)
    print(f"Fine-tuned model saved to {args.output_model}")

if __name__ == "__main__":
    main()
