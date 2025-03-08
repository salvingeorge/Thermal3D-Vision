Freiburg_dataset_dir = '/home/nfs/inf6/data/datasets/ThermalDBs/Freiburg'
flir_adas_dataset_dir = '/home/nfs/inf6/data/datasets/ThermalDBs/ADAS/'


# data/freiburg_dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image

class FreiburgDataset(Dataset):
    """Dataset for loading paired RGB and thermal images from Freiburg dataset."""
    
    def __init__(self, root_dir, sequences=None, transform=None, img_size=(224, 224), 
                 use_pseudo_gt=False, pseudo_gt_dir=None):
        """
        Args:
            root_dir: Root directory of the Freiburg dataset
            sequences: List of sequence names to include (e.g., ['seq_00_day'])
                      If None, all available sequences will be used
            transform: Optional transforms to apply
            img_size: Target image size
            use_pseudo_gt: Whether to load pseudo-GT depth and camera params
            pseudo_gt_dir: Directory containing pseudo-GT annotations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.use_pseudo_gt = use_pseudo_gt
        self.pseudo_gt_dir = pseudo_gt_dir
        
        self.pairs = []
        
        # Find all sequences if not specified
        train_dir = os.path.join(root_dir, 'train')
        if sequences is None:
            sequences = [seq for seq in os.listdir(train_dir)]
        
        # For each sequence, find RGB and thermal image pairs
        for seq_name in sequences:
            seq_dir = os.path.join(train_dir, seq_name)
            if not os.path.isdir(seq_dir):
                continue
                
            # Find all numbered subdirectories
            drive_dirs = [d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))]
            
            for drive in drive_dirs:
                drive_path = os.path.join(seq_dir, drive)
                
                # Find RGB and thermal images in this drive
                rgb_images = []
                thermal_images = []
                
                # Look for images directly in the drive directory
                rgb_files = sorted(glob.glob(os.path.join(drive_path, '*rgb*.png')))
                thermal_files = sorted(glob.glob(os.path.join(drive_path, '*ir*.png')))
                
                # If not found, try looking in subdirectories
                if not rgb_files or not thermal_files:
                    for subdir in os.listdir(drive_path):
                        subdir_path = os.path.join(drive_path, subdir)
                        if os.path.isdir(subdir_path):
                            rgb_files.extend(sorted(glob.glob(os.path.join(subdir_path, '*rgb*.png'))))
                            thermal_files.extend(sorted(glob.glob(os.path.join(subdir_path, '*ir*.png'))))
                
                # Match RGB and thermal images by index if same count
                if len(rgb_files) == len(thermal_files):
                    for i in range(len(rgb_files)):
                        self.pairs.append({
                            'rgb': rgb_files[i],
                            'thermal': thermal_files[i],
                            'sequence': seq_name,
                            'drive': drive
                        })
                else:
                    # Match by filename or timestamp if possible
                    # This is a simplified approach - may need to be adjusted based on filename patterns
                    rgb_base = {os.path.splitext(os.path.basename(f))[0].split('_')[0]: f for f in rgb_files}
                    thermal_base = {os.path.splitext(os.path.basename(f))[0].split('_')[0]: f for f in thermal_files}
                    
                    common_keys = set(rgb_base.keys()) & set(thermal_base.keys())
                    for key in common_keys:
                        self.pairs.append({
                            'rgb': rgb_base[key],
                            'thermal': thermal_base[key],
                            'sequence': seq_name,
                            'drive': drive
                        })
        
        print(f"Found {len(self.pairs)} RGB-thermal pairs across {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load RGB image
        rgb_path = pair['rgb']
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.img_size)
        print(f"Loaded RGB from {rgb_path}, shape={rgb_img.shape}, min={rgb_img.min()}, max={rgb_img.max()}")
        rgb_img = rgb_img.astype(np.float32) / 255.0
        print(f"Loaded RGB from {rgb_path}, shape={rgb_img.shape}, min={rgb_img.min()}, max={rgb_img.max()}")

        
        # Load thermal image - handling 16-bit if necessary
        thermal_path = pair['thermal']
        thermal_img = cv2.imread(thermal_path, cv2.IMREAD_ANYDEPTH)
        
        if thermal_img.dtype == np.uint16:
            thermal_img = thermal_img.astype(np.float32) / 65535.0
        else:
            thermal_img = thermal_img.astype(np.float32) / 255.0
        
        # Convert thermal to 3 channels (required by DUSt3R model)
        if len(thermal_img.shape) == 2:
            thermal_img = np.stack([thermal_img] * 3, axis=-1)
        
        thermal_img = cv2.resize(thermal_img, self.img_size)
        
        # Apply transforms if available
        if self.transform:
            rgb_img = self.transform(rgb_img)
            thermal_img = self.transform(thermal_img)
        else:
            # Convert to PyTorch tensors (CxHxW)
            rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float()
            thermal_img = torch.from_numpy(thermal_img.transpose(2, 0, 1)).float()
        
        sample = {
            'rgb': rgb_img,
            'thermal': thermal_img,
            'rgb_path': rgb_path,
            'thermal_path': thermal_path,
            'sequence': pair['sequence'],
            'drive': pair['drive']
        }
        
        # Load pseudo-GT if available
        if self.use_pseudo_gt and self.pseudo_gt_dir:
            # Construct pseudo-GT path based on RGB image path
            relative_path = os.path.relpath(rgb_path, self.root_dir)
            base_name = os.path.splitext(os.path.basename(rgb_path))[0]
            
            # Load depth
            depth_path = os.path.join(self.pseudo_gt_dir, 'depth', f"{base_name}.npy")
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
                sample['depth'] = torch.from_numpy(depth).float()
            
            # Load camera intrinsics
            intrinsics_path = os.path.join(self.pseudo_gt_dir, 'intrinsics', f"{base_name}.npy")
            if os.path.exists(intrinsics_path):
                intrinsics = np.load(intrinsics_path)
                sample['intrinsics'] = torch.from_numpy(intrinsics).float()
            
            # Load relative pose
            pose_path = os.path.join(self.pseudo_gt_dir, 'poses', f"{base_name}.npy")
            if os.path.exists(pose_path):
                pose = np.load(pose_path)
                sample['pose'] = torch.from_numpy(pose).float()
        
        return sample

def create_freiburg_dataloaders(root_dir, batch_size=8, img_size=(224, 224), split=0.8, 
                               pseudo_gt_dir=None, day_only=False, night_only=False):
    """Create training and validation dataloaders for the Freiburg dataset."""
    
    # Find sequences
    train_dir = os.path.join(root_dir, 'train')
    all_sequences = [seq for seq in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, seq))]
    
    if day_only:
        sequences = [seq for seq in all_sequences if 'day' in seq]
    elif night_only:
        sequences = [seq for seq in all_sequences if 'night' in seq]
    else:
        sequences = all_sequences
    
    # Create dataset
    dataset = FreiburgDataset(
        root_dir=root_dir,
        sequences=sequences,
        img_size=img_size,
        use_pseudo_gt=(pseudo_gt_dir is not None),
        pseudo_gt_dir=pseudo_gt_dir
    )
    
    # Split into train and validation
    train_size = int(len(dataset) * split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader