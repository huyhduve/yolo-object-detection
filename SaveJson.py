import json 
from collections import Counter

def save(vid_id, frame_id, detections, output_path):
    # Count number of items per class
    class_names = [det["label"] for det in detections]
    counts = dict(Counter(class_names))

    # Prepare JSON structure
    json_data = {
        "Vid_id": vid_id,
        "Keyframe_id": frame_id,
        "detections": detections,
        "object_counts": counts,
        "total_objects": len(detections)
    }

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    save("vr", [], "D:\Python\CLIP-DEMO\\Test2.json")
