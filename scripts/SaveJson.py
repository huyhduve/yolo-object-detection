import json 
from collections import Counter
# from pathlib import Path

def save(name, detections, output_path):
    class_names = [det["label"] for det in detections]
    counts = dict(Counter(class_names))

    json_data = {
        "Vid_id": name, 
        "detections": detections,
        "object_counts": counts,
        "total_objects": len(detections)
    }

    # output_file = Path(output_path)
    # output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

