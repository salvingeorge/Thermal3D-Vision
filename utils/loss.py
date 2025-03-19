import torch
import torch.nn.functional as F 

def thermal_aware_loss(pred_pts1, pred_pts2, gt_pts1, gt_pts2, 
                     confidences1=None, confidences2=None, 
                     thermal_img1=None, thermal_img2=None,
                     alpha=0.2, edge_weight=0.5, smoothness_weight=0.3):
    """
    Enhanced loss function for thermal imagery with edge-awareness and smoothness.
    """
    # Basic confidence-weighted regression loss
    basic_loss = confidence_weighted_regression_loss(
        pred_pts1, pred_pts2, gt_pts1, gt_pts2, confidences1, confidences2, alpha)
    
    # Edge-aware loss component (preserve depth discontinuities at thermal edges)
    edge_loss = 0
    if thermal_img1 is not None and thermal_img2 is not None:
        # Extract gradients from thermal images
        if isinstance(thermal_img1, torch.Tensor) and thermal_img1.dim() == 3:
            # Convert to grayscale if needed
            if thermal_img1.shape[0] == 3:
                thermal_gray1 = 0.299 * thermal_img1[0] + 0.587 * thermal_img1[1] + 0.114 * thermal_img1[2]
                thermal_gray2 = 0.299 * thermal_img2[0] + 0.587 * thermal_img2[1] + 0.114 * thermal_img2[2]
            else:
                thermal_gray1 = thermal_img1[0]
                thermal_gray2 = thermal_img2[0]
                
            # Calculate gradients
            grad_thermal1_x = torch.abs(thermal_gray1[:, 1:] - thermal_gray1[:, :-1])
            grad_thermal1_y = torch.abs(thermal_gray1[1:, :] - thermal_gray1[:-1, :])
            grad_thermal2_x = torch.abs(thermal_gray2[:, 1:] - thermal_gray2[:, :-1])
            grad_thermal2_y = torch.abs(thermal_gray2[1:, :] - thermal_gray2[:-1, :])
            
            # Calculate depth gradients
            depth1 = pred_pts1[..., 2]
            depth2 = pred_pts2[..., 2]
            grad_depth1_x = torch.abs(depth1[:, 1:] - depth1[:, :-1])
            grad_depth1_y = torch.abs(depth1[1:, :] - depth1[:-1, :])
            grad_depth2_x = torch.abs(depth2[:, 1:] - depth2[:, :-1])
            grad_depth2_y = torch.abs(depth2[1:, :] - depth2[:-1, :])
            
            # Edge-aware loss (encourage depth edges to align with thermal edges)
            edge_loss1_x = torch.mean(grad_depth1_x * torch.exp(-grad_thermal1_x * 10))
            edge_loss1_y = torch.mean(grad_depth1_y * torch.exp(-grad_thermal1_y * 10))
            edge_loss2_x = torch.mean(grad_depth2_x * torch.exp(-grad_thermal2_x * 10))
            edge_loss2_y = torch.mean(grad_depth2_y * torch.exp(-grad_thermal2_y * 10))
            
            edge_loss = edge_loss1_x + edge_loss1_y + edge_loss2_x + edge_loss2_y
    
    # Smoothness loss (encourage smoothness in homogeneous regions)
    smoothness_loss = 0
    if thermal_img1 is not None and thermal_img2 is not None:
        # Use gradients calculated above
        # Encourage smoothness in areas with low thermal gradients
        smoothness_loss1_x = torch.mean(grad_depth1_x * torch.exp(-grad_thermal1_x * 10))
        smoothness_loss1_y = torch.mean(grad_depth1_y * torch.exp(-grad_thermal1_y * 10))
        smoothness_loss2_x = torch.mean(grad_depth2_x * torch.exp(-grad_thermal2_x * 10))
        smoothness_loss2_y = torch.mean(grad_depth2_y * torch.exp(-grad_thermal2_y * 10))
        
        smoothness_loss = smoothness_loss1_x + smoothness_loss1_y + smoothness_loss2_x + smoothness_loss2_y
    
    # Combine all losses
    total_loss = basic_loss + edge_weight * edge_loss + smoothness_weight * smoothness_loss
    
    # Return both the total loss and its components for logging
    loss_components = {
        'basic_loss': basic_loss.item() if isinstance(basic_loss, torch.Tensor) else basic_loss,
        'edge_loss': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss
    }
    
    return total_loss, loss_components
    
    
def confidence_weighted_regression_loss(pred_pts1, pred_pts2, gt_pts1, gt_pts2, 
                                       confidences1=None, confidences2=None, alpha=0.2):
    """
    Compute the confidence-weighted regression loss with robust handling of confidence values.
    """
    # Calculate L1 distance between predicted and ground truth pointmaps
    loss1 = torch.abs(pred_pts1 - gt_pts1).mean(dim=-1)  # [H, W]
    loss2 = torch.abs(pred_pts2 - gt_pts2).mean(dim=-1)  # [H, W]
    
    # Create default confidence values if None provided
    if confidences1 is None:
        confidences1 = torch.ones_like(loss1)
    if confidences2 is None:
        confidences2 = torch.ones_like(loss2)
    
    # Handle potential numerical issues with confidence values
    confidences1 = torch.clamp(confidences1, min=1e-5, max=10.0)
    confidences2 = torch.clamp(confidences2, min=1e-5, max=10.0)
    
    # Weight losses by confidence and add regularization term
    weighted_loss1 = (confidences1 * loss1 - alpha * torch.log(confidences1)).mean()
    weighted_loss2 = (confidences2 * loss2 - alpha * torch.log(confidences2)).mean()
    
    return weighted_loss1 + weighted_loss2

def enhanced_thermal_aware_loss(pred_pts1, pred_pts2, gt_pts1, gt_pts2, 
                         confidences1=None, confidences2=None, 
                         thermal_img1=None, thermal_img2=None,
                         alpha=0.2, edge_weight=0.5, smoothness_weight=0.3,
                         detail_weight=0.4, multi_scale=True):
    """
    Enhanced loss function for thermal imagery with multi-scale processing, 
    edge-awareness, smoothness, and detail preservation.
    """
    # Basic confidence-weighted regression loss
    basic_loss = confidence_weighted_regression_loss(
        pred_pts1, pred_pts2, gt_pts1, gt_pts2, confidences1, confidences2, alpha)
    
    edge_loss = 0
    smoothness_loss = 0
    detail_loss = 0
    
    if thermal_img1 is not None and thermal_img2 is not None:
        # Convert to grayscale if needed
        if isinstance(thermal_img1, torch.Tensor) and thermal_img1.dim() == 3:
            if thermal_img1.shape[0] == 3:
                thermal_gray1 = 0.299 * thermal_img1[0] + 0.587 * thermal_img1[1] + 0.114 * thermal_img1[2]
                thermal_gray2 = 0.299 * thermal_img2[0] + 0.587 * thermal_img2[1] + 0.114 * thermal_img2[2]
            else:
                thermal_gray1 = thermal_img1[0]
                thermal_gray2 = thermal_img2[0]
        
        # Extract depth maps
        depth1 = pred_pts1[..., 2]
        depth2 = pred_pts2[..., 2]
        gt_depth1 = gt_pts1[..., 2]
        gt_depth2 = gt_pts2[..., 2]
        
        # Multi-scale processing
        scales = [1, 2, 4] if multi_scale else [1]
        
        for scale in scales:
            # Downscale if using multi-scale
            if scale > 1:
                # Need to ensure we're working with properly shaped tensors
                # Add batch dimension if needed
                if thermal_gray1.dim() == 2:
                    thermal_gray1_input = thermal_gray1.unsqueeze(0).unsqueeze(0)
                    thermal_gray2_input = thermal_gray2.unsqueeze(0).unsqueeze(0)
                else:
                    thermal_gray1_input = thermal_gray1.unsqueeze(0)
                    thermal_gray2_input = thermal_gray2.unsqueeze(0)
                
                if depth1.dim() == 2:
                    depth1_input = depth1.unsqueeze(0).unsqueeze(0)
                    depth2_input = depth2.unsqueeze(0).unsqueeze(0)
                    gt_depth1_input = gt_depth1.unsqueeze(0).unsqueeze(0)
                    gt_depth2_input = gt_depth2.unsqueeze(0).unsqueeze(0)
                else:
                    depth1_input = depth1.unsqueeze(0)
                    depth2_input = depth2.unsqueeze(0)
                    gt_depth1_input = gt_depth1.unsqueeze(0)
                    gt_depth2_input = gt_depth2.unsqueeze(0)
                
                # Use average pooling for downscaling
                thermal_gray1_scaled = F.avg_pool2d(thermal_gray1_input, scale, scale)
                thermal_gray2_scaled = F.avg_pool2d(thermal_gray2_input, scale, scale)
                
                depth1_scaled = F.avg_pool2d(depth1_input, scale, scale)
                depth2_scaled = F.avg_pool2d(depth2_input, scale, scale)
                
                gt_depth1_scaled = F.avg_pool2d(gt_depth1_input, scale, scale)
                gt_depth2_scaled = F.avg_pool2d(gt_depth2_input, scale, scale)
                
                # Remove batch/channel dimensions if they were added
                thermal_gray1_scaled = thermal_gray1_scaled.squeeze()
                thermal_gray2_scaled = thermal_gray2_scaled.squeeze()
                depth1_scaled = depth1_scaled.squeeze()
                depth2_scaled = depth2_scaled.squeeze()
                gt_depth1_scaled = gt_depth1_scaled.squeeze()
                gt_depth2_scaled = gt_depth2_scaled.squeeze()
            else:
                thermal_gray1_scaled = thermal_gray1
                thermal_gray2_scaled = thermal_gray2
                depth1_scaled = depth1
                depth2_scaled = depth2
                gt_depth1_scaled = gt_depth1
                gt_depth2_scaled = gt_depth2
            
            # Calculate thermal gradients - simpler and safer approach
            grad_thermal1_x = torch.zeros_like(thermal_gray1_scaled)
            grad_thermal1_y = torch.zeros_like(thermal_gray1_scaled)
            grad_thermal2_x = torch.zeros_like(thermal_gray2_scaled)
            grad_thermal2_y = torch.zeros_like(thermal_gray2_scaled)
            
            # X gradients (horizontal)
            if thermal_gray1_scaled.dim() >= 2:
                if thermal_gray1_scaled.shape[1] > 1:
                    grad_thermal1_x[:, :-1] = torch.abs(thermal_gray1_scaled[:, 1:] - thermal_gray1_scaled[:, :-1])
                if thermal_gray2_scaled.shape[1] > 1:
                    grad_thermal2_x[:, :-1] = torch.abs(thermal_gray2_scaled[:, 1:] - thermal_gray2_scaled[:, :-1])
            
            # Y gradients (vertical)
            if thermal_gray1_scaled.dim() >= 2:
                if thermal_gray1_scaled.shape[0] > 1:
                    grad_thermal1_y[:-1, :] = torch.abs(thermal_gray1_scaled[1:, :] - thermal_gray1_scaled[:-1, :])
                if thermal_gray2_scaled.shape[0] > 1:
                    grad_thermal2_y[:-1, :] = torch.abs(thermal_gray2_scaled[1:, :] - thermal_gray2_scaled[:-1, :])
            
            # Calculate depth gradients
            grad_depth1_x = torch.zeros_like(depth1_scaled)
            grad_depth1_y = torch.zeros_like(depth1_scaled)
            grad_depth2_x = torch.zeros_like(depth2_scaled)
            grad_depth2_y = torch.zeros_like(depth2_scaled)
            
            if depth1_scaled.dim() >= 2:
                if depth1_scaled.shape[1] > 1:
                    grad_depth1_x[:, :-1] = torch.abs(depth1_scaled[:, 1:] - depth1_scaled[:, :-1])
                if depth2_scaled.shape[1] > 1:
                    grad_depth2_x[:, :-1] = torch.abs(depth2_scaled[:, 1:] - depth2_scaled[:, :-1])
            
            if depth1_scaled.dim() >= 2:
                if depth1_scaled.shape[0] > 1:
                    grad_depth1_y[:-1, :] = torch.abs(depth1_scaled[1:, :] - depth1_scaled[:-1, :])
                if depth2_scaled.shape[0] > 1:
                    grad_depth2_y[:-1, :] = torch.abs(depth2_scaled[1:, :] - depth2_scaled[:-1, :])
            
            # Calculate ground truth depth gradients
            grad_gt_depth1_x = torch.zeros_like(gt_depth1_scaled)
            grad_gt_depth1_y = torch.zeros_like(gt_depth1_scaled)
            grad_gt_depth2_x = torch.zeros_like(gt_depth2_scaled)
            grad_gt_depth2_y = torch.zeros_like(gt_depth2_scaled)
            
            if gt_depth1_scaled.dim() >= 2:
                if gt_depth1_scaled.shape[1] > 1:
                    grad_gt_depth1_x[:, :-1] = torch.abs(gt_depth1_scaled[:, 1:] - gt_depth1_scaled[:, :-1])
                if gt_depth2_scaled.shape[1] > 1:
                    grad_gt_depth2_x[:, :-1] = torch.abs(gt_depth2_scaled[:, 1:] - gt_depth2_scaled[:, :-1])
            
            if gt_depth1_scaled.dim() >= 2:
                if gt_depth1_scaled.shape[0] > 1:
                    grad_gt_depth1_y[:-1, :] = torch.abs(gt_depth1_scaled[1:, :] - gt_depth1_scaled[:-1, :])
                if gt_depth2_scaled.shape[0] > 1:
                    grad_gt_depth2_y[:-1, :] = torch.abs(gt_depth2_scaled[1:, :] - gt_depth2_scaled[:-1, :])
            
            # Edge-aware weights (encourage depth edges to align with thermal edges)
            # Use a more numerically stable approach with clipping
            thermal_factor = 10  # Controls sensitivity to thermal gradients
            edge_weight1 = torch.exp(-torch.clamp(grad_thermal1_x, 0, 0.5) * thermal_factor) * \
                          torch.exp(-torch.clamp(grad_thermal1_y, 0, 0.5) * thermal_factor)
            edge_weight2 = torch.exp(-torch.clamp(grad_thermal2_x, 0, 0.5) * thermal_factor) * \
                          torch.exp(-torch.clamp(grad_thermal2_y, 0, 0.5) * thermal_factor)
            
            # Edge-aware loss component
            scale_edge_loss1 = torch.mean(grad_depth1_x * (1 - edge_weight1)) + torch.mean(grad_depth1_y * (1 - edge_weight1))
            scale_edge_loss2 = torch.mean(grad_depth2_x * (1 - edge_weight2)) + torch.mean(grad_depth2_y * (1 - edge_weight2))
            
            # Smoothness loss (encourage smoothness in homogeneous regions)
            scale_smoothness_loss1 = torch.mean(grad_depth1_x * edge_weight1) + torch.mean(grad_depth1_y * edge_weight1)
            scale_smoothness_loss2 = torch.mean(grad_depth2_x * edge_weight2) + torch.mean(grad_depth2_y * edge_weight2)
            
            # Detail preservation loss (match the gradient patterns of ground truth)
            scale_detail_loss1 = torch.mean(torch.abs(grad_depth1_x - grad_gt_depth1_x)) + \
                               torch.mean(torch.abs(grad_depth1_y - grad_gt_depth1_y))
            scale_detail_loss2 = torch.mean(torch.abs(grad_depth2_x - grad_gt_depth2_x)) + \
                               torch.mean(torch.abs(grad_depth2_y - grad_gt_depth2_y))
            
            # Weight by scale (give more importance to finer scales)
            scale_weight = 1.0 / scale
            
            edge_loss += scale_weight * (scale_edge_loss1 + scale_edge_loss2)
            smoothness_loss += scale_weight * (scale_smoothness_loss1 + scale_smoothness_loss2)
            detail_loss += scale_weight * (scale_detail_loss1 + scale_detail_loss2)
    
    # Combine all losses
    total_loss = basic_loss + edge_weight * edge_loss + smoothness_weight * smoothness_loss + detail_weight * detail_loss
    
    # Return both the total loss and its components for logging
    loss_components = {
        'basic_loss': basic_loss.item() if isinstance(basic_loss, torch.Tensor) else basic_loss,
        'edge_loss': edge_loss.item() if isinstance(edge_loss, torch.Tensor) else edge_loss,
        'smoothness_loss': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else smoothness_loss,
        'detail_loss': detail_loss.item() if isinstance(detail_loss, torch.Tensor) else detail_loss
    }
    
    return total_loss, loss_components