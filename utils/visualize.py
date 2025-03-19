import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F


def visualize_predictions(thermal_img, pred_depth, gt_depth, save_path=None):
    """
    Visualize thermal image alongside predicted and ground truth depth maps.
    
    Args:
        thermal_img: Thermal image tensor [C, H, W]
        pred_depth: Predicted depth map tensor [H, W]
        gt_depth: Ground truth depth map tensor [H, W]
        save_path: Path to save the visualization (if None, image is displayed)
    
    Returns:
        None (saves or displays the visualization)
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy arrays for plotting
    if isinstance(thermal_img, torch.Tensor):
        # If image has multiple channels, take the first or average them
        if thermal_img.dim() == 3 and thermal_img.shape[0] > 1:
            thermal_np = thermal_img.detach().cpu().permute(1, 2, 0).numpy()
        else:
            thermal_np = thermal_img.detach().cpu().squeeze().numpy()
            # If 1 channel, repeat to make it a 3-channel grayscale image
            if thermal_np.ndim == 2:
                thermal_np = np.stack([thermal_np] * 3, axis=2)
    else:
        thermal_np = thermal_img
    
    if isinstance(pred_depth, torch.Tensor):
        pred_depth_np = pred_depth.detach().cpu().numpy()
    else:
        pred_depth_np = pred_depth
        
    if isinstance(gt_depth, torch.Tensor):
        gt_depth_np = gt_depth.detach().cpu().numpy()
    else:
        gt_depth_np = gt_depth
    
    # Plot thermal image
    axs[0].imshow(thermal_np)
    axs[0].set_title('Thermal Image')
    axs[0].axis('off')
    
    # Plot predicted depth map
    pred_depth_vis = axs[1].imshow(pred_depth_np, cmap='plasma')
    axs[1].set_title('Predicted Depth')
    axs[1].axis('off')
    fig.colorbar(pred_depth_vis, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Plot ground truth depth map
    gt_depth_vis = axs[2].imshow(gt_depth_np, cmap='plasma')
    axs[2].set_title('Ground Truth Depth')
    axs[2].axis('off')
    fig.colorbar(gt_depth_vis, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save or display the visualization
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def log_sample_images_with_edges(wandb_run, thermal1, thermal2, pred_depth1, gt_depth1, sample_name):
    """Log a side-by-side visualization with edge detection"""
    # Create a figure for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    
    # Convert tensors to numpy arrays for visualization
    thermal1_np = thermal1.detach().cpu().numpy() if isinstance(thermal1, torch.Tensor) else thermal1
    thermal2_np = thermal2.detach().cpu().numpy() if isinstance(thermal2, torch.Tensor) else thermal2
    pred_depth1_np = pred_depth1.detach().cpu().numpy() if isinstance(pred_depth1, torch.Tensor) else pred_depth1
    gt_depth1_np = gt_depth1.detach().cpu().numpy() if isinstance(gt_depth1, torch.Tensor) else gt_depth1
    
    # For image tensors in [C,H,W] format, convert to [H,W,C]
    if isinstance(thermal1, torch.Tensor) and thermal1.dim() == 3 and thermal1.shape[0] == 3:
        thermal1_np = thermal1_np.transpose(1, 2, 0)
    if isinstance(thermal2, torch.Tensor) and thermal2.dim() == 3 and thermal2.shape[0] == 3:
        thermal2_np = thermal2_np.transpose(1, 2, 0)
    
    # Display thermal images
    axes[0, 0].imshow(thermal1_np)
    axes[0, 0].set_title("Thermal Image 1")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(thermal2_np)
    axes[0, 1].set_title("Thermal Image 2")
    axes[0, 1].axis("off")
    
    # Extract thermal edges for visualization
    if isinstance(thermal1, torch.Tensor):
        if thermal1.dim() == 3 and thermal1.shape[0] == 3:
            thermal_gray = 0.299 * thermal1[0] + 0.587 * thermal1[1] + 0.114 * thermal1[2]
        else:
            thermal_gray = thermal1[0]
        
        grad_x = torch.abs(thermal_gray[:, 1:] - thermal_gray[:, :-1])
        grad_y = torch.abs(thermal_gray[1:, :] - thermal_gray[:-1, :])
        
        # Pad to original size
        pad_x = torch.zeros((grad_x.shape[0], 1), device=grad_x.device)
        pad_y = torch.zeros((1, grad_y.shape[1]), device=grad_y.device)
        
        grad_x = torch.cat([grad_x, pad_x], dim=1)
        grad_y = torch.cat([grad_y, pad_y], dim=0)
        
        edge_map = grad_x + grad_y
        edge_map_np = edge_map.detach().cpu().numpy()
    else:
        # Use Sobel operator if not a tensor
        if thermal1_np.ndim == 3:
            thermal_gray = np.mean(thermal1_np, axis=2)
        else:
            thermal_gray = thermal1_np
            
        import cv2
        grad_x = cv2.Sobel(thermal_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(thermal_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_map_np = np.sqrt(grad_x**2 + grad_y**2)
    
    # Display the edge map
    axes[0, 2].imshow(edge_map_np, cmap="hot")
    axes[0, 2].set_title("Thermal Edges")
    axes[0, 2].axis("off")
    
    # Use a consistent colormap for depth visualization
    vmin = min(np.min(pred_depth1_np), np.min(gt_depth1_np))
    vmax = max(np.max(pred_depth1_np), np.max(gt_depth1_np))
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1
    
    im1 = axes[1, 0].imshow(pred_depth1_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Predicted Depth 1")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gt_depth1_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("GT Depth 1")
    axes[1, 1].axis("off")
    
    # Display depth gradient
    depth_grad_x = np.abs(np.gradient(pred_depth1_np, axis=1))
    depth_grad_y = np.abs(np.gradient(pred_depth1_np, axis=0))
    depth_grad = depth_grad_x + depth_grad_y
    
    axes[1, 2].imshow(depth_grad, cmap="hot")
    axes[1, 2].set_title("Depth Gradients")
    axes[1, 2].axis("off")
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('Depth')
    
    wandb_run.log({sample_name: wandb.Image(fig)})
    plt.close(fig)
    
    
def log_sample_images(wandb_run, thermal1, thermal2, pred_depth1, gt_depth1, sample_name):
    """Log a side-by-side visualization using matplotlib and wandb."""
    # Create a larger figure with proper spacing
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    
    # Convert tensors to numpy arrays for visualization
    # Make sure to detach tensors that might require gradients
    thermal1_np = thermal1.detach().cpu().numpy() if isinstance(thermal1, torch.Tensor) else thermal1
    thermal2_np = thermal2.detach().cpu().numpy() if isinstance(thermal2, torch.Tensor) else thermal2
    pred_depth1_np = pred_depth1.detach().cpu().numpy() if isinstance(pred_depth1, torch.Tensor) else pred_depth1
    gt_depth1_np = gt_depth1.detach().cpu().numpy() if isinstance(gt_depth1, torch.Tensor) else gt_depth1
    
    # For image tensors in [C,H,W] format, convert to [H,W,C]
    if isinstance(thermal1, torch.Tensor) and thermal1.dim() == 3 and thermal1.shape[0] == 3:
        thermal1_np = thermal1_np.transpose(1, 2, 0)
    if isinstance(thermal2, torch.Tensor) and thermal2.dim() == 3 and thermal2.shape[0] == 3:
        thermal2_np = thermal2_np.transpose(1, 2, 0)
    
    # Display the images
    axes[0, 0].imshow(thermal1_np)
    axes[0, 0].set_title("Thermal Image 1")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(thermal2_np)
    axes[0, 1].set_title("Thermal Image 2")
    axes[0, 1].axis("off")
    
    # Use a consistent colormap for depth visualization
    # Add small epsilon to avoid division by zero if min=max
    vmin = min(np.min(pred_depth1_np), np.min(gt_depth1_np))
    vmax = max(np.max(pred_depth1_np), np.max(gt_depth1_np))
    if vmin == vmax:
        vmin -= 0.1
        vmax += 0.1
    
    im1 = axes[1, 0].imshow(pred_depth1_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Predicted Depth 1")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(gt_depth1_np, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("GT Depth 1")
    axes[1, 1].axis("off")
    
    # Add colorbar with proper positioning
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
    cbar.set_label('Depth')
    
    # Use constrained_layout instead of tight_layout
    # plt.tight_layout() - removed this line
    
    wandb_run.log({sample_name: wandb.Image(fig)})
    plt.close(fig)