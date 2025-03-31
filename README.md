# 3D Vision in Thermal Images
## Submitted by,
- Pallavi Aithal Narayan
- Salvin George

## Installation
### Install DUSt3R and MASt3R Models
[DUSt3R](https://github.com/naver/dust3r) <p>
[MASt3R](https://github.com/naver/mast3r)

### Clone this repository
```bash
git clone https://github.com/salvingeorge/Thermal3D-Vision.git
cd thermal-3d-vision
```
### Create a conda environment
```bash
conda create -n thermal3d python=3.12
conda activate thermal3d
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Download pre-trained models
```bash
./scripts/download_models.sh
```

## Dataset
This project uses the Freiburg Thermal Dataset, which contains aligned RGB and thermal images from driving scenarios. 
### Generating Pseudo-Ground Truth
```bash
./run_generate_pseudo_gt.sh --dataset_dir path/to/dataset/Freiburg --output_dir ./pseudo_gt_data --visualize
```

## Training the Model
```bash
./run_train_thermal_dustr.sh --pseudo_gt_dir ./pseudo_gt_data --epochs 100 --batch_size 8 --use_thermal_aware_loss --edge_weight 0.5 --smoothness_weight 0.3 --detail_weight 0.4
```

## Inference
```bash
python thermal_dustr_inference.py     --checkpoint checkpoints/checkpoint_name.pth     --input "path/to/thermal/image/folder/or/single/image"     --output output/path     --img_size 224 224
```
