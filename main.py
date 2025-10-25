from GroundingDino import DinoDetect
from DisplayBbox import display
from YOLO11 import YOLODetector
from Masking import SAMSegment
from ColorDetect import DominantColor
import SaveJson


IMAGE_PATH = "D:\AIC\L21_a\L21_V001\\0.webp"
dino_detector = DinoDetect()
yolo_detector = YOLODetector()
SAM_detector = SAMSegment()

obj = yolo_detector.detect(IMAGE_PATH, display_result=True) + dino_detector.detect(IMAGE_PATH, display_result=True)

bbox_list = [det["bbox"] for det in obj]
mask_list = SAM_detector.detect(IMAGE_PATH, bbox_list, display_result=True)

for idx, det in enumerate(obj):
    det["color"] = DominantColor(IMAGE_PATH, det["bbox"])

for info in obj:
    print(info)
# SaveJson.save("Test", obj, "D:\Python\CLIP-DEMO\\Test.json")

