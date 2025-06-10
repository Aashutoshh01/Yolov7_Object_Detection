# Object Detection using YOLOv7 🚀

This repository demonstrates a complete pipeline for training and applying an object detection model using YOLOv7. It includes model training, inference on video files, result analysis, and insight extraction.

---

## 📦 Setup Instructions

### 1. Create Python Environment (Python 3.9)
```bash
conda create -n yolov7env python=3.9
```
### 2. Activate the Environment
```bash
conda activate yolov7env
```
### 3. Install PyTorch with CUDA Support
Make sure your system supports CUDA 11.3. Install the GPU-enabled version of PyTorch:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
