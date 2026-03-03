Transformer-Based Semantic Segmentation for Off-Road Autonomous Navigation

Project Overview

Autonomous navigation in unstructured off-road environments presents significant perception challenges, including terrain irregularity, vegetation clutter, occlusions (logs/rocks), and dynamic lighting conditions.

This project implements a high-resolution semantic segmentation pipeline capable of classifying each pixel into 10 terrain categories, enabling robust perception for autonomous ground vehicles (UGVs).

The dataset was synthetically generated using Duality AI’s Falcon digital twin platform, providing scalable and photorealistic training data for off-road autonomy research.

Model Architecture
We designed a hybrid Transformer + CNN segmentation framework to combine global context understanding with fine-grained spatial refinement.

Architecture Pipeline

Input Image (768×768)
↓
DINOv2 ViT-S/14 (Pretrained Backbone)
↓
Patch Token Feature Extraction
↓
ConvNeXt-style Segmentation Head
↓
Bilinear Upsampling
↓
1×1 Convolution Classifier
↓
Pixel-wise Prediction (10 Classes)

Backbone

DINOv2 ViT-S/14

Self-supervised pretrained

Initially frozen for stable convergence

Later fine-tuned for domain adaptation

This enabled strong feature generalization while reducing early training instability.

Decoder Design

ConvNeXt-style refinement blocks

Batch Normalization + GELU activation

Dropout regularization

Lightweight 1×1 classifier head

The CNN head restores spatial resolution lost during transformer patch embedding.

Training Strategy

We implemented a two-phase fine-tuning protocol:

Phase	Epochs	Backbone Status
Phase1	 1–20	   Frozen
Phase2	 21–40	  Unfrozen

This Stabilizes early optimization
Prevents catastrophic forgetting
Improves domain adaptation
Increased validation mIoU by ~X%

Loss Function 
We implemented a weighted hybrid loss:

Loss=0.5×FocalLoss+0.5×DiceLoss
Motivation:

Focal Loss → handles class imbalance (e.g., Sky dominance)
Dice Loss → improves boundary precision

Result:
Better minority class recall (Logs, Rocks)
Sharper segmentation boundaries

Hyperparameters
Parameter	Value
Image Size	768 × 768
Batch Size	8
Epochs	        40
Optimizer	AdamW
Learning Rate	1e-4
Weight Decay	1e-4
Scheduler	Cosine Annealing
Classes	        10

Evaluation Metrics

Mean Intersection over Union (mIoU)
Dice Coefficient
Pixel Accuracy
Per-Class IoU
Confusion Matrix

Final Performance
Metric	Score
Validation mIoU	0.63
Dice Score	0.71
Pixel Accuracy	0.89
Inference Speed	38 ms / image (GPU)

Meets real-time requirement (<50 ms)
Strong generalization on unseen terrain

Failure Case Analysis
1️)Logs misclassified as Rocks
Cause: Similar texture distribution
Solution: Increased Dice weighting + backbone fine-tuning

2️)Dry Grass confused with Ground Clutter
Cause: Low contrast blending
Solution: Added horizontal flip Test-Time Augmentation

Inference Enhancements

Test-Time Augmentation (Horizontal Flip)
Multi-scale inference
Logit averaging

Improved robustness on unseen desert terrain conditions.

Project Structure
Offroad_Segmentation/
│
├── train_segmentation.py
├── run_test.py
├── best_segmentation_head.pth
├── train_stats/
│   ├── loss_curves.png
│   ├── iou_curves.png
│   └── metrics_dashboard.png
└── README.md

Installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python matplotlib tqdm pillow
Train
python train_segmentation.py --dataset_path "path/to/dataset"
Inference
python run_test.py

Real-World Applications

Autonomous UGV Navigation
Search & Rescue Robotics
Agricultural Terrain Monitoring
Defense Reconnaissance Systems
Outdoor Robotic Mobility

Key Contributions & Optimizations

Hybrid Focal + Dice Loss
Freeze → Unfreeze Strategy
Cosine Learning Rate Scheduling
Dropout Regularization
Multi-scale Inference
Test-Time Augmentation

Future Improvements

Multi-scale feature fusion
Deep supervision
TensorRT deployment optimization
Domain adaptation to real-world imagery
LiDAR–Vision sensor fusion

Conclusion

This project demonstrates a scalable, high-resolution segmentation framework leveraging:
Self-supervised pretrained transformer backbone
CNN-based spatial refinement
Hybrid loss design
Structured fine-tuning strategy
Real-time inference optimization
The hybrid architecture achieved stable convergence, competitive IoU performance, and real-time inference capability for off-road autonomous navigation systems.
