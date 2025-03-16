# data/data_loader.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class FreiburgDataset(Dataset):
    """Dataset for loading thermal image pairs with pseudo-GT for DUSt3R/MASt3R training."""
    
    def __init__(self, root_dir, sequences=None, transform=None, img_size=(224, 224), 
             use_pseudo_gt=True, pseudo_gt_dir=None, frame_skip=1):
        """
        Args:
            root_dir: Root directory of the Freiburg dataset
            sequences: List of sequence names to include (e.g., ['seq_00_day'])
                    If None, all available sequences will be used
            transform: Optional transforms to apply
            img_size: Target image size
            use_pseudo_gt: Whether to load pseudo-GT depth and camera params
            pseudo_gt_dir: Directory containing pseudo-GT annotations
            frame_skip: Number of frames to skip between pairs (for wider baseline)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.use_pseudo_gt = use_pseudo_gt
        self.pseudo_gt_dir = pseudo_gt_dir
        self.frame_skip = frame_skip
        
        self.pairs = []
        
        # Find all sequences if not specified
        train_dir = os.path.join(root_dir, 'train')
        if sequences is None:
            sequences = [seq for seq in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, seq))]
        
        # print(f"Found sequences: {sequences}")
        
        # For each sequence, create pairs of images from consecutive frames
        for seq_name in sequences:
            seq_dir = os.path.join(train_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
                
            # Find all numbered subdirectories
            drive_dirs = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
            
            for drive in drive_dirs:
                drive_path = os.path.join(seq_dir, drive)
                
                # Find thermal images in this drive
                thermal_dir = os.path.join(drive_path, 'fl_ir_aligned')
                if not os.path.isdir(thermal_dir):
                    continue
                    
                thermal_files = sorted(glob.glob(os.path.join(thermal_dir, '*.png')))
                
                # Create pairs of consecutive frames (with frame_skip)
                for i in range(len(thermal_files) - frame_skip):
                    # Get the thermal paths
                    thermal_path1 = thermal_files[i]
                    thermal_path2 = thermal_files[i + frame_skip]
                    
                    # Get the corresponding RGB paths
                    rgb_path1 = thermal_path1.replace('fl_ir_aligned', 'fl_rgb').replace('fl_ir_aligned_', 'fl_rgb_')
                    rgb_path2 = thermal_path2.replace('fl_ir_aligned', 'fl_rgb').replace('fl_ir_aligned_', 'fl_rgb_')
                    
                    # Check if both RGB files exist
                    if os.path.exists(rgb_path1) and os.path.exists(rgb_path2):
                        # Explicitly including both thermal and RGB paths in each pair
                        self.pairs.append({
                            'thermal1': thermal_path1,
                            'thermal2': thermal_path2,
                            'rgb1': rgb_path1,
                            'rgb2': rgb_path2,
                            'sequence': seq_name,
                            'drive': drive,
                        })
        
        print(f"Created {len(self.pairs)} thermal image pairs across {len(sequences)} sequences")
        
        # Print example pair for debugging
        if self.pairs:
            example = self.pairs[0]
            print(f"Example pair:")
            for k, v in example.items():
                print(f"  {k}: {v}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load thermal images
        thermal_path1 = pair['thermal1']
        thermal_path2 = pair['thermal2']
        
        thermal_img1 = self._load_thermal_image(thermal_path1)
        thermal_img2 = self._load_thermal_image(thermal_path2)
        
        if thermal_img1 is None or thermal_img2 is None:
            print(f"Warning: Could not read thermal files: {thermal_path1} or {thermal_path2}, skipping.")
            return None
        
        # Get RGB paths
        rgb_path1 = pair['rgb1']
        rgb_path2 = pair['rgb2']
        
        sample = {
            'thermal1': thermal_img1,
            'thermal2': thermal_img2,
            'thermal_path1': thermal_path1,
            'thermal_path2': thermal_path2,
            'sequence': pair['sequence'],
            'drive': pair['drive']
        }
        
        # Load pseudo-GT if available
        if self.use_pseudo_gt and self.pseudo_gt_dir:
            # Get base names for searching
            base_name1 = os.path.splitext(os.path.basename(rgb_path1))[0]
            
            # FLEXIBLE MATCHING: Search for any pointmap file that starts with this base name
            pattern = os.path.join(self.pseudo_gt_dir, 'pointmap1', f"{base_name1}_*.npy")
            matching_files = glob.glob(pattern)
            
            if matching_files:
                # Take the first matching file
                pointmap1_path = matching_files[0]
                # Extract the full pair name (without .npy)
                pair_name = os.path.splitext(os.path.basename(pointmap1_path))[0]
                
                # Now we can load all files using this pair_name
                try:
                    pointmap1 = np.load(pointmap1_path)
                    sample['pointmap1'] = torch.from_numpy(pointmap1).float()
                    
                    # Extract the second image name from the pair name
                    parts = pair_name.split('_')
                    # The second base name starts after the first base name
                    second_part_idx = pair_name.find('_', pair_name.find(base_name1) + len(base_name1))
                    second_base_name = pair_name[second_part_idx+1:]
                    
                    # Find the corresponding pointmap2
                    pointmap2_path = os.path.join(self.pseudo_gt_dir, 'pointmap2', f"{pair_name}.npy")
                    if os.path.exists(pointmap2_path):
                        pointmap2 = np.load(pointmap2_path)
                        sample['pointmap2'] = torch.from_numpy(pointmap2).float()
                    
                    # Find the corresponding confidence maps
                    conf1_path = os.path.join(self.pseudo_gt_dir, 'confidence1', f"{pair_name}.npy")
                    if os.path.exists(conf1_path):
                        conf1 = np.load(conf1_path)
                        sample['confidence1'] = torch.from_numpy(conf1).float()
                    
                    conf2_path = os.path.join(self.pseudo_gt_dir, 'confidence2', f"{pair_name}.npy")
                    if os.path.exists(conf2_path):
                        conf2 = np.load(conf2_path)
                        sample['confidence2'] = torch.from_numpy(conf2).float()
                    
                    # Load corresponding depths
                    depth1_path = os.path.join(self.pseudo_gt_dir, 'depth1', f"{base_name1}.npy")
                    if os.path.exists(depth1_path):
                        depth1 = np.load(depth1_path)
                        sample['depth1'] = torch.from_numpy(depth1).float()
                    
                    # For depth2, use the second base name from the pair name
                    depth2_path = os.path.join(self.pseudo_gt_dir, 'depth2', f"{second_base_name}.npy")
                    if os.path.exists(depth2_path):
                        depth2 = np.load(depth2_path)
                        sample['depth2'] = torch.from_numpy(depth2).float()
                    
                    # Load pose
                    pose_path = os.path.join(self.pseudo_gt_dir, 'poses', f"{pair_name}.npy")
                    if os.path.exists(pose_path):
                        pose = np.load(pose_path)
                        sample['pose'] = torch.from_numpy(pose).float()
                except Exception as e:
                    print(f"Error loading files for {pair_name}: {e}")
            else:
                # No matching pointmap files found - we can still try to load the depth
                depth1_path = os.path.join(self.pseudo_gt_dir, 'depth1', f"{base_name1}.npy")
                if os.path.exists(depth1_path):
                    depth1 = np.load(depth1_path)
                    sample['depth1'] = torch.from_numpy(depth1).float()
                
                base_name2 = os.path.splitext(os.path.basename(rgb_path2))[0]
                depth2_path = os.path.join(self.pseudo_gt_dir, 'depth2', f"{base_name2}.npy")
                if os.path.exists(depth2_path):
                    depth2 = np.load(depth2_path)
                    sample['depth2'] = torch.from_numpy(depth2).float()
        
        return sample
    
    def debug_loading(self, idx=0):
        """Debug function to check if a sample loads correctly."""
        pair = self.pairs[idx]
        print(f"Loading sample {idx}:")
        print(f"  RGB paths: {pair['rgb_path1']}, {pair['rgb_path2']}")
        print(f"  Thermal paths: {pair['thermal_path1']}, {pair['thermal_path2']}")
        
        # Try to load pointmaps
        if self.use_pseudo_gt and self.pseudo_gt_dir:
            rgb_path1 = pair['rgb1']
            rgb_path2 = pair['rgb2']
            base_name1 = os.path.splitext(os.path.basename(rgb_path1))[0]
            base_name2 = os.path.splitext(os.path.basename(rgb_path2))[0]
            
            pointmap1_path = os.path.join(self.pseudo_gt_dir, 'pointmap1', f"{base_name1}_{base_name2}.npy")
            print(f"  Looking for pointmap1 at: {pointmap1_path}")
            print(f"  File exists: {os.path.exists(pointmap1_path)}")
    
    def _load_thermal_image(self, path):
        """Helper to load and preprocess thermal images."""
        thermal_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if thermal_img is None:
            return None

        if thermal_img.dtype == np.uint16:
            thermal_img = thermal_img.astype(np.float32) / 65535.0
        else:
            thermal_img = thermal_img.astype(np.float32) / 255.0
        
        # Convert to 3 channels if grayscale
        if len(thermal_img.shape) == 2:
            thermal_img = np.stack([thermal_img] * 3, axis=-1)
        
        thermal_img = cv2.resize(thermal_img, self.img_size)
        
        # Convert to torch tensor [C, H, W]
        thermal_img = torch.from_numpy(thermal_img.transpose(2, 0, 1)).float()
        return thermal_img
    
    def _load_rgb_image(self, path):
        """Helper to load and preprocess RGB images."""
        if not os.path.exists(path):
            return None
            
        rgb_img = cv2.imread(path)
        if rgb_img is None:
            return None
            
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        rgb_img = rgb_img.astype(np.float32) / 255.0
        
        # Convert to torch tensor [C, H, W]
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float()
        return rgb_img