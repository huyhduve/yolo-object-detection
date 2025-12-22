import os
from ultralytics import YOLO

from scripts import ImageProcessor
# import scripts.ImageProcessor as ImageProcessor

MODEL1 = "models\yolo11m.pt"
MODEL2 = "models\yolov8l-oiv7.pt"
MODEL3 = "models\yolo-spec.pt"

class YOLODetector: 
    def __init__(self, model_base=MODEL1, model_sup=MODEL2, model_spec=MODEL3):
        self.model_base = YOLO(model_base)
        self.model_sup = YOLO(model_sup)
        self.model_spec = YOLO(model_spec)

    def detect(self, 
            image_paths, 
            device="cuda:0", 
            conf_threshold=0.3, 
            display_result=False, 
            output_image_folder=None,
            output_object_folder=None
        ):
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        
        results1 = self.model_base(image_paths, show=False, verbose=False, device = device)
        results2 = self.model_sup(image_paths, show=False, verbose=False, device = device)
        results3 = self.model_spec(image_paths, show=False, verbose=False, device = device)
        
        

        for idx, image_path in enumerate(image_paths):
            labels = []
            boxes = []
            scores = []

            for box in results1[idx].boxes:
                conf = float(box.conf[0])
                if( conf < conf_threshold):
                    continue
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                cls = int(box.cls[0])
                labels.append(self.model_base.names[cls])
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
            
            for box in results2[idx].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model_sup.names[cls]
                if(conf < conf_threshold or name in labels):
                    continue
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                labels.append(name)
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

            for box in results3[idx].boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model_spec.names[cls]
                if(conf < conf_threshold):
                    continue
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
                labels.append(name)
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

            detection = []
            for label, bbox, score in zip(labels, boxes, scores):
                detection.append({"label": label, "bbox": bbox, "conf": score})

            ImageProcessor.display(image_path, 
                                   detection, 
                                   display_result, 
                                   output_image_folder, 
                                   output_object_folder)


        return detection

