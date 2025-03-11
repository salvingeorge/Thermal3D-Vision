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
import wandb
from tqdm import tqdm

# Import your dataset class and the helper for splitting
from data.freiburg_dataset import FreiburgDataset, create_freiburg_dataloaders

def skip_none_collate(batch):
    """Custom collate function to filter out None samples."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return {}
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
    if isinstance(output, tuple):
        pred1 = output[0]
    elif isinstance(output, dict):
        pred1 = output.get("pred1", {})
    else:
        raise ValueError("Unexpected output type from model.")
    
    if isinstance(pred1, dict):
        if "pts3d" not in pred1:
            raise KeyError("Output does not contain 'pts3d'. Keys: " + str(list(pred1.keys())))
        pts3d = flatten_tensor(pred1["pts3d"])
    else:
        pts3d = flatten_tensor(pred1)
    
    if pts3d.shape[0] > 1:
        pts3d = pts3d[0]
    
    depth_tensor = pts3d[:, :, 2]  # shape [H, W]
    return depth_tensor

def log_sample_images(wandb_run, rgb_img, thermal_img, pred_depth, gt_depth, sample_name):
    """Log a side-by-side visualization using matplotlib and wandb."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(rgb_img)
    axes[0].set_title("RGB Image")
    axes[0].axis("off")
    
    axes[1].imshow(thermal_img, cmap="inferno")
    axes[1].set_title("Thermal Image")
    axes[1].axis("off")
    
    axes[2].imshow(pred_depth, cmap="plasma")
    axes[2].set_title("Predicted Depth")
    axes[2].axis("off")
    plt.colorbar(axes[2].images[0], ax=axes[2])
    
    axes[3].imshow(gt_depth, cmap="plasma")
    axes[3].set_title("Pseudo-GT Depth")
    axes[3].axis("off")
    plt.colorbar(axes[3].images[0], ax=axes[3])
    
    wandb_run.log({sample_name: wandb.Image(fig, caption=sample_name)})
    plt.close(fig)

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
    parser.add_argument("--log_interval", type=int, default=100, help="Interval (in batches) for logging sample images")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb with a descriptive run name.
    wandb.init(project="thermal-3d-vision",
               config=vars(args),
               name=f"DUSt3R_ft_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}")

    # Split the dataset into training and validation sets using the helper.
    train_loader, val_loader = create_freiburg_dataloaders(
        root_dir=args.dataset_dir,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        pseudo_gt_dir=args.pseudo_gt_dir,
        split=0.8  # 80% training, 20% validation
    )

    # Load the DUSt3R model (small variant)
    model = load_dustr_model(args.weights, device)

    # Define a simple L1 loss for depth regression.
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("Starting fine-tuning DUSt3R on thermal images with pseudo-GT depth...")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in pbar:
            if not batch or "thermal" not in batch or "depth" not in batch:
                continue
            thermal_imgs = batch["thermal"].to(device)  # [B, 3, H, W]
            gt_depths = batch["depth"].to(device)         # [B, H_gt, W_gt]

            optimizer.zero_grad()
            loss_total = torch.tensor(0.0, device=device)
            B = thermal_imgs.size(0)
            for i in range(B):
                view1 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                view2 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                output = model(view1, view2)
                try:
                    pred_depth_tensor = extract_depth_from_output(output)  # [H, W]
                    pred_depth_tensor = pred_depth_tensor.to(device)
                    gt_depth_sample = gt_depths[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H_gt, W_gt]
                    gt_depth_resized = F.interpolate(gt_depth_sample, size=(args.img_size[1], args.img_size[0]),
                                                    mode='bilinear', align_corners=False)
                    gt_depth_resized = gt_depth_resized.squeeze(0).squeeze(0)  # [H, W]
                    loss = criterion(pred_depth_tensor, gt_depth_resized)
                    loss_total += loss
                except Exception as e:
                    print(f"Error in loss computation: {e}")
                del output
                torch.cuda.empty_cache()
                global_step += 1
                if global_step % args.log_interval == 0:
                    rgb_img = thermal_imgs[i].permute(1,2,0).detach().cpu().numpy()  # Using thermal image as RGB placeholder.
                    thermal_img = rgb_img.copy()
                    pred_depth_np = pred_depth_tensor.detach().cpu().numpy()
                    gt_depth_np = gt_depth_resized.detach().cpu().numpy()
                    log_sample_images(wandb, rgb_img, thermal_img, pred_depth_np, gt_depth_np, 
                                      f"Epoch{epoch+1}_Step{global_step}")
            if B > 0:
                loss_total = loss_total / B
                loss_total.backward()
                optimizer.step()
                running_loss += loss_total.item()
            pbar.set_postfix({"batch_loss": loss_total.item(), "global_step": global_step})
            wandb.log({"batch_loss": loss_total.item(), "global_step": global_step})
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Average Training Loss: {avg_train_loss:.4f}")
        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation Epoch {epoch+1}")
            for batch_idx, batch in val_pbar:
                if batch is None or "thermal" not in batch or "depth" not in batch:
                    continue
                thermal_imgs = batch["thermal"].to(device)
                gt_depths = batch["depth"].to(device)
                loss_total = torch.tensor(0.0, device=device)
                B = thermal_imgs.size(0)
                for i in range(B):
                    view1 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                    view2 = {"img": thermal_imgs[i].unsqueeze(0), "instance": []}
                    output = model(view1, view2)
                    try:
                        pred_depth_tensor = extract_depth_from_output(output)
                        pred_depth_tensor = pred_depth_tensor.to(device)
                        gt_depth_sample = gt_depths[i].unsqueeze(0).unsqueeze(0)
                        gt_depth_resized = F.interpolate(gt_depth_sample, size=(args.img_size[1], args.img_size[0]),
                                                        mode='bilinear', align_corners=False)
                        gt_depth_resized = gt_depth_resized.squeeze(0).squeeze(0)
                        loss = criterion(pred_depth_tensor, gt_depth_resized)
                        loss_total += loss
                    except Exception as e:
                        print(f"Error in validation loss computation: {e}")
                    del output
                    torch.cuda.empty_cache()
                if B > 0:
                    loss_total = loss_total / B
                    val_loss += loss_total.item()
                    val_pbar.set_postfix({"val_batch_loss": loss_total.item()})
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Validation Loss: {avg_val_loss:.4f}")
            wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
    
    torch.save(model.state_dict(), args.output_model)
    print(f"Fine-tuned model saved to {args.output_model}")
    wandb.save(args.output_model)
    wandb.finish()

if __name__ == "__main__":
    main()
