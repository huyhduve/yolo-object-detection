import os
from ultralytics import YOLO
# from ColorDetect import DominantColor
import scripts.DisplayBbox as DisplayBbox
from scripts.SaveJson import save

MODEL1 = "models\yolo11m.pt"
MODEL2 = "models\yolov8l-oiv7.pt"
MODEL3 = "models\yolo-spec.pt"

class YOLODetector: 
    def __init__(self, model_base=MODEL1, model_sup=MODEL2, model_spec=MODEL3):
        self.model_base = YOLO(model_base)
        self.model_sup = YOLO(model_sup)
        self.model_spec = YOLO(model_spec)

    def detect(self, image_paths, vid, device="cuda:0", conf_threshold=0.2, display_result=False, object_folder=None):
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

            if display_result:
                DisplayBbox.display(image_path, boxes, labels, scores)
            
            if object_folder is not None:
                filename = os.path.basename(image_path)
                name, _ = os.path.splitext(filename)
                save(vid, name, detection, object_folder + "\\" + name + ".json")

        return detection



if __name__ == "__main__":
    IMG_FOLDER = r"D:\Python\CLIP-DEMO\L22_V029\027.jpg"
    # OBJECT_FOLDER = r"D:\Python\New_object_file"
    # BATCH_SIZE = 32
    yolo_detector = YOLODetector()
    yolo_detector.detect([IMG_FOLDER], "e", display_result=True)

    # for subdir in os.listdir(IMG_FOLDER):
        
    #     img_path = os.listdir(os.path.join(IMG_FOLDER, subdir))
    #     os.makedirs(f"{OBJECT_FOLDER}\\{subdir}", exist_ok=True)

    #     for i in range(0, len(img_path), BATCH_SIZE):
    #         batch_paths = [os.path.join(IMG_FOLDER, subdir, img) for img in img_path[i:i+BATCH_SIZE]]
            # yolo_detector.detect(batch_paths,subdir, object_folder=f"{OBJECT_FOLDER}\\{subdir}")
    #     print(f"{subdir} done")
