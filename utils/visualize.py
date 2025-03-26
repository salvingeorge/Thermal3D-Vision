import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch.nn.functional as F
import sys
import cv2
import matplotlib.cm as cm
import os
import glob

sys.path.append('/home/user/georges1/Thermal3D-Vision')

def visualize_predictions(thermal_img, pred_depth, gt_depth=None, save_path=None):
    """
    Visualize thermal image alongside predicted depth map (and optionally ground truth if available).
    
    Args:
        thermal_img: Thermal image tensor [C, H, W]
        pred_depth: Predicted depth map tensor [H, W]
        gt_depth: Ground truth depth map tensor [H, W] or None for inference
        save_path: Path to save the visualization (if None, image is displayed)
    
    Returns:
        None (saves or displays the visualization)
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    
    # Determine number of subplots (2 if no ground truth, 3 otherwise)
    n_plots = 3 if gt_depth is not None else 2
    fig, axs = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
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
    
    # Plot thermal image
    axs[0].imshow(thermal_np)
    axs[0].set_title('Thermal Image')
    axs[0].axis('off')
    
    # Plot predicted depth map
    pred_depth_vis = axs[1].imshow(pred_depth_np, cmap='plasma')
    axs[1].set_title('Predicted Depth')
    axs[1].axis('off')
    fig.colorbar(pred_depth_vis, ax=axs[1], fraction=0.046, pad=0.04)
    
    # Plot ground truth depth map if provided
    if gt_depth is not None:
        if isinstance(gt_depth, torch.Tensor):
            gt_depth_np = gt_depth.detach().cpu().numpy()
        else:
            gt_depth_np = gt_depth
        
        gt_depth_vis = axs[2].imshow(gt_depth_np, cmap='plasma')
        axs[2].set_title('Ground Truth Depth')
        axs[2].axis('off')
        fig.colorbar(gt_depth_vis, ax=axs[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save or display the visualization
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
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
    
    

from utils.preprocessing import enhance_thermal_contrast

def visualize_enhanced_thermal(thermal_path, img_size=(224, 224), min_val=None, max_val=None, cmap='jet'):
    """
    Load a thermal image, remove borders, and apply better visualization
    """
    # Load the thermal image
    thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
    
    if thermal_img is None:
        print(f"Failed to load image from {thermal_path}")
        return None
    
    # Store original for comparison
    original_img = thermal_img.copy()
    
    # Detect black borders (find non-zero regions)
    threshold = 100  # Adjust as needed for your images
    non_black = thermal_img > threshold
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)
    
    if np.sum(rows) > 0 and np.sum(cols) > 0:
        # Find the boundaries of non-black regions
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        if len(row_indices) > 0 and len(col_indices) > 0:
            row_start, row_end = row_indices[[0, -1]]
            col_start, col_end = col_indices[[0, -1]]
            
            # Crop the image to remove black bars
            thermal_img = thermal_img[row_start:row_end+1, col_start:col_end+1]
    
    # Process exactly as in dataset_loader's _load_thermal_image method
    if thermal_img.dtype == np.uint16:
        thermal_img = thermal_img.astype(np.float32) / 65535.0
    else:
        thermal_img = thermal_img.astype(np.float32) / 255.0
    
    # Convert to 3 channels if grayscale
    if len(thermal_img.shape) == 2:
        thermal_img = np.stack([thermal_img] * 3, axis=-1)
    
    # Resize to target size
    thermal_img = cv2.resize(thermal_img, img_size)
    
    # Convert to torch tensor [C, H, W]
    thermal_tensor = torch.from_numpy(thermal_img.transpose(2, 0, 1)).float()
    
    # Apply contrast enhancement
    enhanced_tensor = enhance_thermal_contrast(thermal_tensor)
    
    # For visualization purposes only
    if enhanced_tensor.shape[0] == 3:
        # For visualization, convert to grayscale
        viz_img = 0.299 * enhanced_tensor[0] + 0.587 * enhanced_tensor[1] + 0.114 * enhanced_tensor[2]
    else:
        viz_img = enhanced_tensor[0]
    
    viz_img = viz_img.cpu().numpy()
    
    # Improve visualization contrast - if not manually set, use percentiles
    if min_val is None or max_val is None:
        p2, p98 = np.percentile(viz_img, (2, 98))
        min_val = p2
        max_val = p98
    
    # Apply thresholding and normalization
    viz_img = np.clip((viz_img - min_val) / (max_val - min_val + 1e-6), 0, 1)
    
    # Convert to uint8 for colormap application
    viz_img_uint8 = (viz_img * 255).astype(np.uint8)
    
    # Apply colormap
    if cmap == 'jet':
        colored_img = cv2.applyColorMap(viz_img_uint8, cv2.COLORMAP_JET)
        colored_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
    else:
        # Use matplotlib colormaps
        cmap_fn = plt.get_cmap(cmap)
        colored_img = cmap_fn(viz_img)
        if colored_img.shape[-1] == 4:  # If RGBA, convert to RGB
            colored_img = colored_img[..., :3]
    
    # Create figure for visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # # Show original image
    # orig_for_display = original_img.astype(np.float32)
    # if orig_for_display.dtype == np.uint16:
    #     orig_for_display = orig_for_display / 65535.0
    # else:
    #     orig_for_display = orig_for_display / 255.0
    colour_thermal = visualize_ir_standalone(thermal_path)
    axes[0].imshow(colour_thermal, cmap='gray')

    axes[0].set_title('Original Thermal Image')
    axes[0].axis('off')
    
    # Show enhanced image
    if cmap == 'jet':
        axes[1].imshow(colored_img)
    else:
        axes[1].imshow(viz_img, cmap=cmap)
    axes[1].set_title('Enhanced Thermal Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return the tensor - this is what should be fed to the model
    return enhanced_tensor

def visualize_ir_standalone(ir_img, min_val=21800, max_val=23700, viz = False):
    """Visualize IR image with colormap after removing black borders."""
    # If the image is a file path, load it
    if isinstance(ir_img, str):
        ir_img = cv2.imread(ir_img, cv2.IMREAD_ANYDEPTH)
    
    # Store original for comparison
    original_img = ir_img.copy()
    
    ir_img_color = visualize_ir(ir_img)
    ir_img_color = cv2.cvtColor(ir_img_color, cv2.COLOR_BGR2RGB)
    
    
    
    if viz == True:
        # Create figure to show before and after
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # Show original image
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Show processed image
        axes[1].imshow(ir_img_color)
        axes[1].set_title('Processed')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return ir_img_color  # Return the colorized image

def load_rgb_image(path):
    """Load an RGB image with OpenCV, convert BGR->RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_depth(path):
    """Load depth from a .npy file (shape [H, W]) or from the Z channel of a pointmap."""
    depth = np.load(path)
    return depth

def load_pointmap(path):
    """Load a pointmap (shape [H, W, 3]) from .npy."""
    pm = np.load(path)
    return pm

def plot_point_cloud(ax, pointmap, color_mode='depth', point_size=1):
    """
    Scatter plot of a 3D pointmap in the given Axes3D (ax).
    color_mode can be 'depth' to color by Z or 'none' for single color.
    """
    H, W, _ = pointmap.shape
    points = pointmap.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    
    # Color points by depth
    depths = points[:, 2]
    if color_mode == 'depth':
        cmin, cmax = depths.min(), depths.max()
        denom = (cmax - cmin) if (cmax > cmin) else 1.0
        colors = cm.viridis((depths - cmin) / denom)
    else:
        colors = 'blue'
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, c=colors, marker='.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
def plot_cameras(ax, pose1=None, pose2=None, size=0.1):
    """
    Plot camera frustums in 3D space.
    pose1 is assumed to be identity (reference frame)
    pose2 is the relative pose from camera 1 to camera 2
    """
    # Camera 1 (reference frame)
    # Draw camera coordinate system
    origin = np.array([0, 0, 0])
    x_axis = np.array([size, 0, 0])
    y_axis = np.array([0, size, 0])
    z_axis = np.array([0, 0, size])
    
    # Draw camera 1 (reference frame)
    ax.quiver(origin[0], origin[1], origin[2], 
              x_axis[0], x_axis[1], x_axis[2], color='r')
    ax.quiver(origin[0], origin[1], origin[2], 
              y_axis[0], y_axis[1], y_axis[2], color='g')
    ax.quiver(origin[0], origin[1], origin[2], 
              z_axis[0], z_axis[1], z_axis[2], color='b')
    
    # If we have pose2, draw camera 2
    if pose2 is not None:
        # Extract rotation and translation
        R = pose2[:3, :3]
        t = pose2[:3, 3]
        
        # Transform axes by rotation and translation
        x_axis_transformed = R @ np.array([size, 0, 0]) + t
        y_axis_transformed = R @ np.array([0, size, 0]) + t
        z_axis_transformed = R @ np.array([0, 0, size]) + t
        
        # Draw camera 2
        ax.quiver(t[0], t[1], t[2], 
                  x_axis_transformed[0]-t[0], x_axis_transformed[1]-t[1], x_axis_transformed[2]-t[2], color='r')
        ax.quiver(t[0], t[1], t[2], 
                  y_axis_transformed[0]-t[0], y_axis_transformed[1]-t[1], y_axis_transformed[2]-t[2], color='g')
        ax.quiver(t[0], t[1], t[2], 
                  z_axis_transformed[0]-t[0], z_axis_transformed[1]-t[1], z_axis_transformed[2]-t[2], color='b')
        
        # Draw a line connecting the two cameras
        ax.plot([origin[0], t[0]], [origin[1], t[1]], [origin[2], t[2]], 'k--')

def visualize_pair(
    rgb1_path, depth1_path, pm1_path,
    rgb2_path, depth2_path, pm2_path,
    intrinsics_path, pose_path,
    title="Pair Visualization",
    camera_size = 100
):
    """
    Visualize a pair of images (RGB1 & RGB2), their depths, and their 3D pointmaps.
    Also display intrinsics and relative pose in the console or an optional subplot.
    """

    # Load data for view1
    rgb1 = load_rgb_image(rgb1_path)
    depth1 = load_depth(depth1_path)
    pm1 = load_pointmap(pm1_path)

    # Load data for view2
    rgb2 = load_rgb_image(rgb2_path)
    depth2 = load_depth(depth2_path)
    pm2 = load_pointmap(pm2_path)

    # Load intrinsics & pose
    intrinsics = np.load(intrinsics_path)  # shape [3,3]
    pose = np.load(pose_path)             # shape [4,4]

    # Print them to console
    print("Intrinsics:\n", intrinsics)
    print("Relative Pose:\n", pose)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # Subplot 1: RGB Image 1
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb1)
    ax1.set_title("RGB Image 1")
    ax1.axis("off")

    # Subplot 2: Depth 1
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(depth1, cmap="plasma")
    ax2.set_title("Depth 1")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth")

    # Subplot 4: RGB Image 2
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(rgb2)
    ax4.set_title("RGB Image 2")
    ax4.axis("off")

    # Subplot 5: Depth 2
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(depth2, cmap="plasma")
    ax5.set_title("Depth 2")
    ax5.axis("off")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label="Depth")

    # Combined 3D visualization with point clouds and cameras
    ax_combined = fig.add_subplot(2, 3, 3, projection='3d')
    plot_point_cloud(ax_combined, pm1, color_mode='depth', point_size=0.5)
    # Different color for second point cloud
    points = pm2.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    ax_combined.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       s=0.5, c='red', marker='.')
    
    # Add camera poses
    pose1 = np.eye(4)  # Identity for camera 1 (reference frame)
    plot_cameras(ax_combined, pose1=pose1, pose2=pose, size = camera_size)
    
    ax_combined.set_title("3D View with Combined Camera Poses")

    plt.tight_layout()
    plt.show()
    

def find_drive_folders(base_path):
    """Find all drive folders in the Freiburg dataset."""
    train_dir = os.path.join(base_path, 'train')
    sequences = sorted(os.listdir(train_dir))
    
    all_drive_folders = []
    for seq in sequences:
        seq_path = os.path.join(train_dir, seq)
        # Get numbered subfolders (drives)
        subfolders = [f for f in os.listdir(seq_path) if os.path.isdir(os.path.join(seq_path, f))]
        for subfolder in subfolders:
            drive_path = os.path.join(seq_path, subfolder)
            all_drive_folders.append((seq, subfolder, drive_path))
    
    return all_drive_folders

def load_images_from_drive(drive_path):
    """Load RGB and IR images from a drive folder."""
    # Find all image files in the drive folder
    rgb_files = sorted(glob.glob(os.path.join(drive_path, '*rgb*.png')))
    ir_files = sorted(glob.glob(os.path.join(drive_path, '*ir*.png')))
    
    if not rgb_files and not ir_files:
        # Try looking for specific file patterns if the above fails
        rgb_files = sorted(glob.glob(os.path.join(drive_path, '*color*.png')))
        ir_files = sorted(glob.glob(os.path.join(drive_path, '*thermal*.png')))
    
    # If still no files, try searching recursively
    if not rgb_files and not ir_files:
        rgb_files = sorted(glob.glob(os.path.join(drive_path, '**/*rgb*.png'), recursive=True))
        ir_files = sorted(glob.glob(os.path.join(drive_path, '**/*ir*.png'), recursive=True))
    
    return rgb_files, ir_files

def visualize_ir(ir_img, min_val=21800, max_val=23700):
    """Visualize IR image with colormap."""
    # If the image is a file path, load it
    if isinstance(ir_img, str):
        ir_img = cv2.imread(ir_img, cv2.IMREAD_ANYDEPTH)
    
    # Convert to float32 for processing
    ir_img = ir_img.astype(np.float32)
    
    # Apply thresholding
    ir_img[ir_img < min_val] = min_val
    ir_img[ir_img > max_val] = max_val
    
    # Normalize to 0-1
    ir_img = (ir_img - min_val) / (max_val - min_val)
    
    # Apply colormap
    ir_img_color = cv2.applyColorMap((ir_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return ir_img_color

def visualize_rgb(rgb_img):
    """Visualize RGB image."""
    # If the image is a file path, load it
    if isinstance(rgb_img, str):
        rgb_img = cv2.imread(rgb_img)
        # Convert from BGR to RGB
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    
    return rgb_img

def overlay_thermal_rgb(rgb_img, ir_img, alpha=0.4):
    """Overlay thermal image on RGB image."""
    # Make sure both images are of the same size
    if rgb_img.shape[:2] != ir_img.shape[:2]:
        ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # Overlay images
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(rgb_img, alpha, ir_img, beta, 0.0)
    
    return overlay

def visualize_dataset(dataset_path, num_samples=5):
    """Visualize samples from the Freiburg dataset."""
    drive_folders = find_drive_folders(dataset_path)
    
    if not drive_folders:
        print(f"No drive folders found in {dataset_path}")
        return
    
    print(f"Found {len(drive_folders)} drive folders")
    
    # Create output directory
    os.makedirs('freiburg_samples', exist_ok=True)
    
    # Sample from different sequences if possible
    sampled_sequences = set()
    sample_count = 0
    
    for seq, subfolder, drive_path in drive_folders:
        # Skip if we've already sampled from this sequence
        if sample_count >= num_samples and seq in sampled_sequences:
            continue
        
        sampled_sequences.add(seq)
        
        # print(f"\nProcessing {seq}/{subfolder}...")
        rgb_files, ir_files = load_images_from_drive(drive_path)
        
        # print(f"Found {len(rgb_files)} RGB images and {len(ir_files)} IR images")
        
        # If no images found, continue to next drive
        if not rgb_files or not ir_files:
            continue
        
        # If number of RGB and IR images differ, take the minimum
        num_files = min(len(rgb_files), len(ir_files))
        
        # Sample a few images from this drive
        for i in range(min(3, num_files)):
            if sample_count >= num_samples:
                break
                
            rgb_path = rgb_files[i]
            ir_path = ir_files[i]
            
            # Load and visualize images
            rgb_img = visualize_rgb(rgb_path)
            ir_img = visualize_ir(ir_path)
            overlay = overlay_thermal_rgb(rgb_img, ir_img)
            
            # Create figure for visualization
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Plot RGB image
            axes[0].imshow(rgb_img)
            axes[0].set_title(f'RGB Image - {seq}/{subfolder}')
            axes[0].axis('off')
            
            # Plot IR image
            axes[1].imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Thermal Image - {seq}/{subfolder}')
            axes[1].axis('off')
            
            # Plot overlay
            axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2].set_title(f'Overlay - {seq}/{subfolder}')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'freiburg_samples/{seq}_{subfolder}_sample_{i}.png')
            plt.show()
            plt.close()
            
            sample_count += 1
            
    # print(f"\nSaved {sample_count} samples to freiburg_samples/ directory")


def plot_point_cloud_merged(ax, pointmap, color_mode='depth', point_size=1):
    """
    Scatter plot of a 3D pointmap in the given Axes3D (ax).
    color_mode can be 'depth' to color by Z or 'none' for single color.
    """
    H, W, _ = pointmap.shape
    points = pointmap.reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    
    # Color points by depth
    depths = points[:, 2]
    if depths.size == 0:
        # If no valid points, just skip
        return
    if color_mode == 'depth':
        cmin, cmax = depths.min(), depths.max()
        denom = (cmax - cmin) if (cmax > cmin) else 1.0
        colors = cm.viridis((depths - cmin) / denom)
    else:
        colors = 'blue'
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=point_size, c=colors, marker='.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

def visualize_pair_merged(
    rgb1_path, depth1_path, pm1_path,
    rgb2_path, depth2_path, pm2_path,
    intrinsics_path, pose_path,
    out_dir="visualized_pairs",
    pair_name="unknown_pair",
    title="Pair Visualization",
    print_console = False
):
    """
    Visualize a pair of images (RGB1 & RGB2), their depths, and their 3D pointmaps.
    """

    # Load data for view1
    rgb1 = load_rgb_image(rgb1_path)
    depth1 = load_depth(depth1_path)
    pm1 = load_pointmap(pm1_path)

    # Load data for view2
    rgb2 = load_rgb_image(rgb2_path)
    depth2 = load_depth(depth2_path)
    pm2 = load_pointmap(pm2_path)

    # Load intrinsics & pose
    if os.path.exists(intrinsics_path):
        intrinsics = np.load(intrinsics_path)
    else:
        intrinsics = None
    
    if os.path.exists(pose_path):
        pose = np.load(pose_path)
    else:
        pose = None

    # Print them to console
    if print_console:    
        print(f"\n=== Visualizing pair: {pair_name} ===")
        print("RGB1:", rgb1_path)
        print("Depth1:", depth1_path, depth1.shape)
        print("Pointmap1:", pm1_path, pm1.shape)
        print("RGB2:", rgb2_path)
        print("Depth2:", depth2_path, depth2.shape)
        print("Pointmap2:", pm2_path, pm2.shape)
        print("Intrinsics:", intrinsics_path, "(exists? ", os.path.exists(intrinsics_path), ")")
        print("Pose:", pose_path, "(exists? ", os.path.exists(pose_path), ")")

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(title, fontsize=16)

    # Subplot 1: RGB Image 1
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb1)
    ax1.set_title("RGB Image 1")
    ax1.axis("off")

    # Subplot 2: Depth 1
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.imshow(depth1, cmap="plasma")
    ax2.set_title("Depth 1")
    ax2.axis("off")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Depth")

    # Subplot 3: Point Cloud 1
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    plot_point_cloud(ax3, pm1, color_mode='depth')
    ax3.set_title("3D Pointmap 1")

    # Subplot 4: RGB Image 2
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(rgb2)
    ax4.set_title("RGB Image 2")
    ax4.axis("off")

    # Subplot 5: Depth 2
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(depth2, cmap="plasma")
    ax5.set_title("Depth 2")
    ax5.axis("off")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label="Depth")

    # Subplot 6: Point Cloud 2
    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    plot_point_cloud(ax6, pm2, color_mode='depth')
    ax6.set_title("3D Pointmap 2")

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pair_name}.png")
    plt.show()
    plt.savefig(out_path)
    plt.close()
    # print(f"Saved figure to: {out_path}")