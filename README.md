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

## 📓 Notebook Execution

### 4. Run notebook.ipynb
- This notebook contains the end-to-end training workflow.
- Make sure to update the dataset configuration files (coco.yaml, yolov7.yaml) as needed. You can find these YAML files in the repository.
- Follow the notebook cells sequentially for training and evaluation.

## 🎥 Video Inference (Optional)

###5. Run Video-Based Inference from Notebook

Use the following command in a code cell to run inference on a video:
```bash
!python yolov7/detect.py \
  --weights runs/train/exp12/weights/best.pt \
  --source /home/ubuntu/Object_Detection_Yolov7_Project/demo_video.mp4 \
  --conf 0.25 \
  --img 640 \
  --project video_results \
  --name demo_output \
  --exist-ok
```
🔁 Note:
- Replace /home/ubuntu/Object_Detection_Yolov7_Project/demo_video.mp4 with the full path to your video file.

- Adjust --project and --name to control where the output video will be saved.

## 📊 6. Results Visualization and Analysis
## 🔎 7. Inference Video Insights
## 🧠 8. Conclusion
## 📜 9. License

Open-source for learning purposes. Use and modify freely.
