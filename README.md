# YOLO Object Detection on Custom Dataset

This project applies YOLOv8-oiv7, YOLO11m for object detection using two pretrained models and a fine-tuned version trained on a custom dataset.

## 📁 Structure
- **scripts/** – detection
- **models/** – pretrained and fine-tuned YOLO weights

- **Class.txt** - detectable classes

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/huyhduve/yolo-object-detection.git
   cd yolo-object-detection

## 🧠 Using YOLODetector as a Module

You can also use this project as a Python module in your own code.  
The main class `YOLODetector` (defined in `scripts/ObjectDetection.py`) provides easy access to model loading, detection, and visualization.

### Example
```python
from scripts.ObjectDetection import YOLODetector

# Initialize with your fine-tuned model
detector = YOLODetector()

results = detector.detect(
    image_paths="example\images\image_1.webp",
    conf_threshold=0.6, 
    display_result=True, # Enable displaying the result
    output_image_folder="example\output-labelled_images" # Set output directory for labelled images

    output_object_folder="example\output-objects", # Set output directory for object file (json)

)
