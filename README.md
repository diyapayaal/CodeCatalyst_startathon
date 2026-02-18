# CodeCatalyst_startathon
Project Overview
Autonomous navigation in off-road environments presents unique challenges such as uneven terrain, vegetation clutter, rocks, and water bodies.
This project provides a reliable computer vision system capable of segmenting off-road scenes into 10 semantic classes, including:
-Dense Vegetation
-Uneven terrain
-Rocks and obstacles
-Water crossings 
-low contrast dirt paths
(and additional terrain-specific classes)
The model combines transformer-based feature extraction with CNN-based spatial refinement to achieve robust performance.


Our Solution
TrailNet leverages self-supervised visual intelligence from:
1. Backbone: Meta AI DINOv2 (ViT-S/14)
Pretrained Vision Transformer
Frozen weights for stable training
Strong generalization in unstructured environments
Extracts rich semantic representations
2. Segmentation Head: ConvNeXt-Based Decoder
Enhances spatial resolution
Sharp boundary segmentation
Lightweight yet powerful
This hybrid design combines:
a. Global understanding (Transformer)
b. Local precision (ConvNeXt)

Model Architecture
Input Image
→ DINOv2 Feature Extraction
→ ConvNeXt Segmentation Head
→ Pixel-wise Classification 

Performance
Trained for 40 epochs
Stable training speed: 3 iterations/sec
Hardware: NVIDIA GPU (HP Victus)
Evaluation Metrics:
Intersection over Union (IoU)
Dice Coefficient
Validation loss curves
Training statistics available at:
    train_stats/all_metrics.png

Offroad_Project
├── Offroad_Segmentation_Training_Dataset     # Annotated training data
├── Offroad_Segmentation_testImages           # Unseen images for evaluation
├── train_segmentation.py                     # Main training script
├── run_test.py                               # Inference script
├── segmentation_head.pth                     # Final trained model weights
└── train_stats                               # Graphs for IoU and Loss

Requirements
Python 3.10+
PyTorch (CUDA enabled)
Torchvision
OpenCV
Matplotlib
Pillow
tqdm
fvcore
omegaconf

Installation
1.Install PyTorch with CUDA 11.8:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
2.Install remaining dependencies:
  pip install opencv-python matplotlib pillow tqdm fvcore omegaconf


Usage
1.Train the Model
  python train_segmentation.py

2.Run Inference on Test Images
  python run_test.py

Results
After 40 epochs of training:
Consistent improvement in IoU and Dice score
Stable loss convergence
Strong generalization on unseen off-road images

Visual metrics are stored in:
   train_stats/all_metrics.png

Applications
Autonomous Off-Road Vehicles
Agricultural Robotics
Search & Rescue Robots
Defense Terrain Navigation
Forest Monitoring Systems

Future Improvements
Fine-tuning DINOv2 backbone
Multi-scale feature fusion
Real-time optimization (TensorRT deployment)
Dataset expansion with diverse terrains
Integration with LiDAR + sensor fusion

