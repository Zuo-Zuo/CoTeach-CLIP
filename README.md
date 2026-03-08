# CoTeach-CLIP
CoTeach-CLIP: Collaborative Teacher-Student Learning for Zero-Shot 3D Point Cloud Recognition  This repository is the official implementation of CoTeach-CLIP, a novel framework designed to enhance zero-shot 3D point cloud recognition by bridging the gap between 2D vision-language models (CLIP) and 3D geometric structures.

# Overview
[Overview of the architecture.pdf](https://github.com/user-attachments/files/25823046/Overview.of.the.architecture.pdf)

CoTeach-CLIP addresses the inherent challenges in transferring CLIP's knowledge to the 3D domain, such as inter-modal asymmetry and the limitations of hand-crafted text templates. Our framework introduces three key innovations:

1. Dual-Teacher Collaborative Distillation: Employs both visual and text teachers from CLIP to jointly supervise a depth encoder, providing complementary cross-modal guidance.

2. Sparse CLIP-MoE Visual Encoder: Utilizes a Sparse Mixture-of-Experts (MoE) architecture to expand model capacity and capture diverse geometric patterns with high computational efficiency.

3. See-then-Write Strategy: Automatically generates geometry-aware descriptions from rendered depth maps using a Vision-Language Model (VLM), replacing restrictive hand-crafted templates with rich, structural text data.

# Repository Structure
<pre>
CoTeach-CLIP/
├── datasets/          # Data loading scripts (ModelNet, ScanObjectNN, ShapeNet)
├── models/            # Core architectures (CoTeach-CLIP, DPA, MoE Adapter)
├── render/            # Depth map rendering engine and view selection
├── tools/             # Data preprocessing (e.g., preparation for MoE)
├── pretraining.py     # Main pre-training script
├── zeroshot.py        # Zero-shot evaluation script
├── export_renders.py  # Utility for rendering 3D points to 2D
└── requirements.txt   # Environment dependencies
<pre>
