from ultralytics import YOLO
import DisplayBbox

class YOLODetector: 
    def __init__(self, model_base = "yolo11m.pt", model_sup="yolov8l-oiv7.pt", model_spec="Finetune1.pt"):
        self.model_base = YOLO(model_base)
        self.model_sup = YOLO(model_sup)
        self.model_spec = YOLO(model_spec)

    def detect(self, image_path, device="cuda", conf_threshold=0.6, display_result=False):
        results1 = self.model_base(image_path, show=False, verbose=False, device = device)
        results2 = self.model_sup(image_path, show=False, verbose=False, device = device)
        results3 = self.model_spec(image_path, show=False, verbose=False, device = device)
        
        labels = []
        boxes = []
        scores = []
        for box in results1[0].boxes:
            conf = float(box.conf[0])
            if( conf < conf_threshold):
                continue
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            cls = int(box.cls[0])
            labels.append(self.model_base.names[cls])
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
        
        for box in results2[0].boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = self.model_sup.names[cls]
            if(conf < conf_threshold or name in labels):
                continue
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            labels.append(name)
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        for box in results3[0].boxes:
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

        if display_result:
            DisplayBbox.display(image_path, boxes, labels, scores)

        return detection


