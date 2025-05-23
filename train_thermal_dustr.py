import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, default_collate, random_split
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import types
from types import SimpleNamespace
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


from utils.loss import thermal_aware_loss, enhanced_thermal_aware_loss
from utils.visualize import log_sample_images_with_edges, log_sample_images
from thermal_dustr_model import load_dustr_model, ThermalDUSt3R
from data.dataset_loader import FreiburgDataset
from utils.data_utils import skip_none_collate


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DUSt3R on thermal images with pseudo-GT")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the Freiburg dataset")
    parser.add_argument("--pseudo_gt_dir", type=str, required=True,
                        help="Path to the pseudo-GT annotations directory")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the DUSt3R checkpoint")
    parser.add_argument("--output_model", type=str, required=True,
                        help="Path to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, nargs=2, default=[224, 224],
                        help="Resize dimension (width height)")
    parser.add_argument("--frame_skip", type=int, default=3, 
                        help="Number of frames to skip between pairs")
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="Interval (in steps) for logging sample images")
    parser.add_argument("--use_thermal_aware_loss", action="store_true", 
                    help="Use thermal-aware loss with edge and smoothness terms")
    parser.add_argument("--edge_weight", type=float, default=0.5, 
                        help="Weight for edge-aware term in thermal-aware loss")
    parser.add_argument("--smoothness_weight", type=float, default=0.3, 
                        help="Weight for smoothness term in thermal-aware loss")
    parser.add_argument("--accumulation_steps", type=int, default=1, 
                        help="Number of batches to accumulate gradients over")
    parser.add_argument("--detail_weight", type=float, default=0.4, 
                   help="Weight for detail preservation term in thermal-aware loss")
    parser.add_argument("--multi_scale", action="store_true", 
                    help="Use multi-scale processing in thermal-aware loss")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Maximum number of batches per epoch (for quick tests)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb
    wandb.init(project="thermal-3d-vision",
               config=vars(args),
               name=f"DUSt3R_thermal_ft_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}")

    # Create dataset
    dataset = FreiburgDataset(
        root_dir=args.dataset_dir,
        sequences=None,  # Use all available sequences
        img_size=tuple(args.img_size),
        use_pseudo_gt=True,
        pseudo_gt_dir=args.pseudo_gt_dir,
        frame_skip=args.frame_skip
    )
    
    # Split into train and validation
    split_ratio = 0.8
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, collate_fn=skip_none_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, collate_fn=skip_none_collate)

    # Load the DUSt3R model
    model = load_dustr_model(args.weights, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Calculate total steps (approx.)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs

    # Epoch-based warmup and decay
    warmup_epochs = int(args.epochs * 0.1)  # 10% of total epochs

    warmup_lr_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1,  # Start at 10% of base lr
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=1e-7  # Lower minimum LR for finer convergence
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_lr_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
        

    print("Starting fine-tuning DUSt3R on thermal images...")
    global_step = 0
    best_val_loss = float('inf')
    patience = 10  # Number of epochs to wait before early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                   desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in pbar:
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            # Skip empty batches or batches without essential data
            if not batch or len(batch) == 0 or "thermal1" not in batch or "thermal2" not in batch:
                continue
                
            # Get batch data
            thermal1 = batch["thermal1"].to(device)
            thermal2 = batch["thermal2"].to(device)          
            
            # Check if we have pointmap ground truth
            has_pointmap_gt = "pointmap1" in batch and "pointmap2" in batch
            if not has_pointmap_gt:
                continue
                
            gt_pointmap1 = batch["pointmap1"].to(device)
            gt_pointmap2 = batch["pointmap2"].to(device)
            
            # Check if batch size matches
            actual_batch_size = min(thermal1.size(0), thermal2.size(0), 
                                   gt_pointmap1.size(0), gt_pointmap2.size(0))
            
            if "confidence1" in batch and batch["confidence1"] is not None:
                confidence1 = batch["confidence1"].to(device)
                confidence1 = confidence1[:actual_batch_size] if confidence1.size(0) > actual_batch_size else confidence1
            else:
                confidence1 = None
                
            if "confidence2" in batch and batch["confidence2"] is not None:
                confidence2 = batch["confidence2"].to(device)
                confidence2 = confidence2[:actual_batch_size] if confidence2.size(0) > actual_batch_size else confidence2
            else:
                confidence2 = None
            
            # Ensure all tensors have the same batch dimension
            thermal1 = thermal1[:actual_batch_size]
            thermal2 = thermal2[:actual_batch_size]
            gt_pointmap1 = gt_pointmap1[:actual_batch_size]
            gt_pointmap2 = gt_pointmap2[:actual_batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_samples = 0
            
            # Process each sample in the batch individually to handle errors
            for i in range(actual_batch_size):
                try:
                    # Prepare inputs for the model
                    view1 = {"img": thermal1[i:i+1], "instance": []}
                    view2 = {"img": thermal2[i:i+1], "instance": []}
                    
                    # Forward pass
                    output = model(view1, view2)
                    
                    # Extract predictions
                    if isinstance(output, tuple):
                        pred1, pred2 = output
                    else:
                        pred1 = output.get("pred1", {})
                        pred2 = output.get("pred2", {})
                    
                    # Extract pointmaps from predictions
                    if isinstance(pred1, dict):
                        pred_pointmap1 = pred1.get("pts3d")
                        if "pts3d_in_other_view" in pred2:
                            pred_pointmap2 = pred2.get("pts3d_in_other_view")
                        else:
                            pred_pointmap2 = pred2.get("pts3d")
                            
                        # Extract confidences if available
                        pred_conf1 = pred1.get("conf", None)
                        pred_conf2 = pred2.get("conf", None)
                    else:
                        pred_pointmap1, pred_pointmap2 = pred1, pred2
                        pred_conf1, pred_conf2 = None, None
                    
                    # Handle batch dimension - ensure we have [H, W, 3] tensors
                    if len(pred_pointmap1.shape) == 4:  # [B, H, W, 3]
                        pred_pointmap1 = pred_pointmap1[0]  # [H, W, 3]
                    if len(pred_pointmap2.shape) == 4:  # [B, H, W, 3]
                        pred_pointmap2 = pred_pointmap2[0]  # [H, W, 3]
                        
                    # Also handle confidence maps if present
                    if pred_conf1 is not None and len(pred_conf1.shape) == 3:  # [B, H, W]
                        pred_conf1 = pred_conf1[0]  # [H, W]
                    if pred_conf2 is not None and len(pred_conf2.shape) == 3:  # [B, H, W]
                        pred_conf2 = pred_conf2[0]  # [H, W]
                    
                    # Get current ground truth
                    curr_gt_pointmap1 = gt_pointmap1[i]  # [H, W, 3]
                    curr_gt_pointmap2 = gt_pointmap2[i]  # [H, W, 3]
                    
                    # Get current confidences if available
                    curr_conf1 = confidence1[i] if confidence1 is not None else None
                    curr_conf2 = confidence2[i] if confidence2 is not None else None
                    
                    # Handle dimensionality mismatch between prediction and ground truth
                    if pred_pointmap1.shape[:-1] != curr_gt_pointmap1.shape[:-1]:
                        # Get target dimensions
                        target_h, target_w = pred_pointmap1.shape[0], pred_pointmap1.shape[1]
                        
                        # Reshape ground truth for interpolation [H, W, 3] -> [1, 3, H, W]
                        gt1_temp = curr_gt_pointmap1.permute(2, 0, 1).unsqueeze(0)
                        gt2_temp = curr_gt_pointmap2.permute(2, 0, 1).unsqueeze(0)
                        
                        # Resize to match prediction dimensions
                        gt1_resized = F.interpolate(
                            gt1_temp, size=(target_h, target_w), 
                            mode='bilinear', align_corners=False
                        )
                        gt2_resized = F.interpolate(
                            gt2_temp, size=(target_h, target_w), 
                            mode='bilinear', align_corners=False
                        )
                        
                        # Convert back to [H, W, 3]
                        curr_gt_pointmap1 = gt1_resized.squeeze(0).permute(1, 2, 0)
                        curr_gt_pointmap2 = gt2_resized.squeeze(0).permute(1, 2, 0)
                        
                        # Also resize confidence maps if we have them
                        if curr_conf1 is not None:
                            conf1_temp = curr_conf1.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            conf1_resized = F.interpolate(
                                conf1_temp, size=(target_h, target_w),
                                mode='bilinear', align_corners=False
                            )
                            curr_conf1 = conf1_resized.squeeze(0).squeeze(0)  # [H, W]
                            
                        if curr_conf2 is not None:
                            conf2_temp = curr_conf2.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                            conf2_resized = F.interpolate(
                                conf2_temp, size=(target_h, target_w),
                                mode='bilinear', align_corners=False
                            )
                            curr_conf2 = conf2_resized.squeeze(0).squeeze(0)  # [H, W]
                            
                    # Use predicted confidence if available, else use GT confidence, else use ones
                    conf1 = pred_conf1 if pred_conf1 is not None else curr_conf1 if curr_conf1 is not None else torch.ones_like(pred_pointmap1[..., 0])
                    conf2 = pred_conf2 if pred_conf2 is not None else curr_conf2 if curr_conf2 is not None else torch.ones_like(pred_pointmap2[..., 0])

                    # Ensure confidence is strictly positive (for log stability)
                    conf1 = torch.clamp(conf1, min=1e-5)
                    conf2 = torch.clamp(conf2, min=1e-5)
                    
                    if args.use_thermal_aware_loss:
                        
                        loss, loss_components = enhanced_thermal_aware_loss(
                            pred_pointmap1, pred_pointmap2,
                            curr_gt_pointmap1, curr_gt_pointmap2,
                            confidences1=conf1, confidences2=conf2,
                            thermal_img1=thermal1[i], thermal_img2=thermal2[i],
                            alpha=0.2,
                            edge_weight=args.edge_weight,
                            smoothness_weight=args.smoothness_weight,
                            detail_weight=args.detail_weight,
                            multi_scale=args.multi_scale
                        )
                        
                        # Log individual loss components if desired
                        if valid_samples == 0 and i == 0:  # Log for first sample in batch
                            wandb.log({
                                'basic_loss': loss_components['basic_loss'],
                                'edge_loss': loss_components['edge_loss'] * args.edge_weight,
                                'smoothness_loss': loss_components['smoothness_loss'] * args.smoothness_weight,
                                'detail_loss': loss_components['detail_loss'] * args.detail_weight,
                                'global_step': global_step
                            })
                    else:
                        # Original loss calculation
                        loss1 = torch.abs(pred_pointmap1 - curr_gt_pointmap1).mean(dim=-1)  # [H, W]
                        loss2 = torch.abs(pred_pointmap2 - curr_gt_pointmap2).mean(dim=-1)  # [H, W]
                        
                        # Apply confidence weighting
                        alpha = 0.2  # Confidence regularization parameter
                        
                        # Calculate weighted loss with regularization
                        weighted_loss1 = (conf1 * loss1 - alpha * torch.log(conf1)).mean()
                        weighted_loss2 = (conf2 * loss2 - alpha * torch.log(conf2)).mean()
                        
                        # Total loss
                        loss = weighted_loss1 + weighted_loss2
                    
                    # Check if loss is valid
                    if torch.isfinite(loss) and loss > 0:
                        batch_loss += loss
                        valid_samples += 1
                        
                        # Log sample images occasionally 
                        if global_step % args.log_interval == 0 and i == 0:
                            # Extract depth map from pointmap (Z-coordinate)
                            pred_depth1 = pred_pointmap1[:, :, 2].detach()
                            gt_depth1 = curr_gt_pointmap1[:, :, 2].detach()
                            
                            if args.use_thermal_aware_loss:
                                log_sample_images_with_edges(
                                    wandb, 
                                    thermal1[i], 
                                    thermal2[i],
                                    pred_depth1, 
                                    gt_depth1, 
                                    f"sample_ep{epoch+1}_step{global_step}"
                                )
                            else:
                                log_sample_images(
                                    wandb, 
                                    thermal1[i], 
                                    thermal2[i],
                                    pred_depth1, 
                                    gt_depth1, 
                                    f"sample_ep{epoch+1}_step{global_step}"
                                )
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                global_step += 1
            
            # Only proceed with backward pass if we have valid samples
            if valid_samples > 0:
                # Average loss over valid samples
                batch_loss = batch_loss / valid_samples
                batch_loss.backward()
                # to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += batch_loss.item()
                valid_batches += 1
                
                pbar.set_postfix({
                    "batch_loss": batch_loss.item(), 
                    "valid_samples": valid_samples,
                    "lr": optimizer.param_groups[0]['lr']
                })
                
                wandb.log({
                    "batch_loss": batch_loss.item(), 
                    "global_step": global_step,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })
        
        # End of epoch
        if valid_batches > 0:
            avg_train_loss = running_loss / valid_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Training Loss: {avg_train_loss:.4f}")
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss})
            
        scheduler.step()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        val_valid_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                           desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, batch in val_pbar:
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                # Skip empty batches or batches without essential data
                if not batch or len(batch) == 0 or "thermal1" not in batch or "thermal2" not in batch:
                    continue
                    
                # Check if we have pointmap ground truth
                has_pointmap_gt = "pointmap1" in batch and "pointmap2" in batch
                if not has_pointmap_gt:
                    continue
                
                # Get batch data
                thermal1 = batch["thermal1"].to(device)
                thermal2 = batch["thermal2"].to(device)
                gt_pointmap1 = batch["pointmap1"].to(device)
                gt_pointmap2 = batch["pointmap2"].to(device)
                
                # Check if batch size matches
                actual_batch_size = min(thermal1.size(0), thermal2.size(0), 
                                      gt_pointmap1.size(0), gt_pointmap2.size(0))
                
                # Ensure all tensors have the same batch dimension
                thermal1 = thermal1[:actual_batch_size]
                thermal2 = thermal2[:actual_batch_size]
                gt_pointmap1 = gt_pointmap1[:actual_batch_size]
                gt_pointmap2 = gt_pointmap2[:actual_batch_size]
                
                batch_loss = 0.0
                valid_samples = 0
                
                # Process each sample individually
                for i in range(actual_batch_size):
                    try:
                        # Prepare inputs for the model
                        view1 = {"img": thermal1[i:i+1], "instance": []}
                        view2 = {"img": thermal2[i:i+1], "instance": []}
                        
                        # Forward pass
                        output = model(view1, view2)
                        
                        # Same extraction logic as in training
                        if isinstance(output, tuple):
                            pred1, pred2 = output
                        else:
                            pred1 = output.get("pred1", {})
                            pred2 = output.get("pred2", {})
                        
                        if isinstance(pred1, dict):
                            pred_pointmap1 = pred1.get("pts3d")
                            if "pts3d_in_other_view" in pred2:
                                pred_pointmap2 = pred2.get("pts3d_in_other_view")
                            else:
                                pred_pointmap2 = pred2.get("pts3d")
                        else:
                            pred_pointmap1, pred_pointmap2 = pred1, pred2
                        
                        # Handle batch dimension
                        if len(pred_pointmap1.shape) == 4:
                            pred_pointmap1 = pred_pointmap1[0]
                        if len(pred_pointmap2.shape) == 4:
                            pred_pointmap2 = pred_pointmap2[0]
                        
                        # Get current ground truth
                        curr_gt_pointmap1 = gt_pointmap1[i]
                        curr_gt_pointmap2 = gt_pointmap2[i]
                        
                        # Handle dimensionality mismatch
                        if pred_pointmap1.shape[:-1] != curr_gt_pointmap1.shape[:-1]:
                            target_h, target_w = pred_pointmap1.shape[0], pred_pointmap1.shape[1]
                            
                            gt1_temp = curr_gt_pointmap1.permute(2, 0, 1).unsqueeze(0)
                            gt2_temp = curr_gt_pointmap2.permute(2, 0, 1).unsqueeze(0)
                            
                            gt1_resized = F.interpolate(
                                gt1_temp, size=(target_h, target_w), 
                                mode='bilinear', align_corners=False
                            )
                            gt2_resized = F.interpolate(
                                gt2_temp, size=(target_h, target_w), 
                                mode='bilinear', align_corners=False
                            )
                            
                            curr_gt_pointmap1 = gt1_resized.squeeze(0).permute(1, 2, 0)
                            curr_gt_pointmap2 = gt2_resized.squeeze(0).permute(1, 2, 0)
                        
                        # Calculate loss (same as training)
                        loss1 = torch.abs(pred_pointmap1 - curr_gt_pointmap1).mean(dim=-1)
                        loss2 = torch.abs(pred_pointmap2 - curr_gt_pointmap2).mean(dim=-1)
                        
                        # Use simple L1 loss for validation (no confidence weighting)
                        loss = (loss1.mean() + loss2.mean()) / 2
                        
                        if torch.isfinite(loss) and loss > 0:
                            batch_loss += loss
                            valid_samples += 1
                            
                    except Exception as e:
                        print(f"Error during validation for sample {i}: {e}")
                        continue
                
                # Add batch loss to running total if valid
                if valid_samples > 0:
                    batch_loss = batch_loss / valid_samples
                    val_loss += batch_loss.item()
                    val_valid_batches += 1
                    
                    val_pbar.set_postfix({"val_batch_loss": batch_loss.item()})
        
        # Compute average validation loss
        if val_valid_batches > 0:
            avg_val_loss = val_loss / val_valid_batches
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Validation Loss: {avg_val_loss:.4f}")
            wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
            
            # Save best model with Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint...")
                
                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "args": SimpleNamespace(**{
                        "model": type(model).__name__,
                        "learning_rate": args.lr,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                    })
                }
                
                best_model_path = args.output_model.replace(".pth", "_best.pth")
                torch.save(checkpoint, best_model_path)
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break  # Exit the training loop

    # Save the final model
    checkpoint = {
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "val_loss": best_val_loss if 'best_val_loss' in locals() else None,
        "args": SimpleNamespace(**{
            "model": type(model).__name__,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
        })
    }

    torch.save(checkpoint, args.output_model)
    print(f"Fine-tuned model saved to {args.output_model}")
    # wandb.save(args.output_model)
    wandb.finish()

if __name__ == "__main__":
    main()