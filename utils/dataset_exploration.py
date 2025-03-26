# dataset_exploration.py
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def explore_dataset(root_dir, dataset_name):
    """Explore the structure and statistics of a thermal dataset."""
    print(f"\n--- Exploring {dataset_name} Dataset at {root_dir} ---\n")
    
    # List all subdirectories
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Subdirectories: {subdirs}")
    
    # Count image files by type
    img_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    img_counts = defaultdict(int)
    img_resolutions = defaultdict(list)
    
    for ext in img_extensions:
        for filepath in glob.glob(os.path.join(root_dir, '**', f'*{ext}'), recursive=True):
            img_counts[ext] += 1
            
            # Sample some resolutions
            if img_counts[ext] <= 10:  # Just sample a few images
                try:
                    with Image.open(filepath) as img:
                        img_resolutions[ext].append(img.size)
                except Exception as e:
                    print(f"Error opening {filepath}: {e}")
    
    print("\nImage counts by extension:")
    for ext, count in img_counts.items():
        print(f"  {ext}: {count} files")
    
    print("\nSample image resolutions:")
    for ext, resolutions in img_resolutions.items():
        print(f"  {ext}: {resolutions}")
    
    # Try to identify RGB and thermal directories
    rgb_dirs = []
    thermal_dirs = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        # Look at filenames to guess content
        sample_files = glob.glob(os.path.join(subdir_path, '**', '*.*'), recursive=True)[:10]
        
        if any('rgb' in f.lower() or 'color' in f.lower() for f in sample_files):
            rgb_dirs.append(subdir)
        if any('thermal' in f.lower() or 'ir' in f.lower() or 'flir' in f.lower() for f in sample_files):
            thermal_dirs.append(subdir)
    
    print("\nPotential RGB directories:", rgb_dirs)
    print("Potential thermal directories:", thermal_dirs)
    
    return {
        'subdirs': subdirs,
        'img_counts': img_counts,
        'rgb_dirs': rgb_dirs, 
        'thermal_dirs': thermal_dirs
    }
