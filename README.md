# Face Detection using YOLOv8

This repository contains an implementation of face detection using [YOLOv8](https://github.com/ultralytics/ultralytics), a state-of-the-art object detection model developed by Ultralytics. The model is fine-tuned for detecting human faces in images and videos with high accuracy and efficiency.

## Features
- Real-time face detection using YOLOv8.
- Supports image, video, and webcam input.
- Exports and runs models in various formats (ONNX, TensorRT, etc.).
- Fast and lightweight implementation.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed. Then, install the required dependencies:
```bash
pip install ultralytics opencv-python numpy torch torchvision matplotlib
```

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/face-detection-yolov8.git
cd face-detection-yolov8
```

### 2. Download the YOLOv8 Model
Download the pre-trained YOLOv8 model (or train your own) and place it in the `weights/` directory:
```bash
mkdir weights
wget -P weights https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 3. Run Face Detection
#### On an Image
```bash
python detect.py --weights weights/yolov8n.pt --source path/to/image.jpg
```

#### On a Video
```bash
python detect.py --weights weights/yolov8n.pt --source path/to/video.mp4
```

#### On Webcam
```bash
python detect.py --weights weights/yolov8n.pt --source 0
```

## Training Custom Model
To train YOLOv8 on a custom face dataset:
```bash
python train.py --data data.yaml --weights yolov8n.pt --epochs 50 --imgsz 640
```
Modify `data.yaml` to define your dataset paths and classes.

## Exporting the Model
You can export the trained model to various formats:
```bash
python export.py --weights weights/yolov8n.pt --format onnx
```

## Results
Detected faces are saved in the `runs/detect/` directory.

## Acknowledgments
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)

## License
This project is licensed under the MIT License.

