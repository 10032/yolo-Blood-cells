# YOLOv11 识别镜下细胞

基于 YOLOv11 和 PyQt6 的跨平台目标检测与标注工具，支持图片/视频/摄像头输入，提供直观的GUI操作界面。

## 主要特性

- 🖼️ **多格式支持**  
  拖放识别：支持 JPG/PNG/JPEG 图片、MP4/AVI 视频
- 🚀 **高性能检测**  
  YOLOv8 实时检测（CPU/GPU 自动切换）
- 📦 **开箱即用**  
  提供预编译 EXE 文件，无需配置环境
- 🔧 **灵活扩展**  
  支持加载自定义 YOLO 模型（*.pt 格式）

## 快速开始
下载编译好的exe文件，拖入图片进行识别
下载链接

### 环境要求

- Python 3.8+
- Windows/Linux/macOS
- NVIDIA GPU (可选，推荐用于加速)

### 安装依赖

```bash
# 基础依赖
pip install pyqt6 ultralytics opencv-python

# GPU加速支持 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

