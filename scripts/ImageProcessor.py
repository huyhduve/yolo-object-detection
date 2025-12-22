from PIL import Image, ImageDraw, ImageFont
from scripts.SaveJson import save
import random
import os 
from pathlib import Path 
# generate random RGB tuple
def random_rgb():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))

def display(image_path, 
            detection,
            display_result=False, 
            output_image_folder=None,
            output_object_folder=None
        ):

    img_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for det in detection:
        box = det["bbox"]
        labels = det["label"]
        scores = det["conf"]
        
        x1, y1, x2, y2 = box
        color = random_rgb()
        draw.rectangle([x1, y1, x2, y2], outline= color, width=2)

        label_text = ""
        if labels:
            label_text += str(labels)
        if scores:
            label_text += f"{scores:.2f}"

        if label_text:
            text_size = draw.textbbox((x1, y1), label_text, font=font)
            draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill="purple")
            draw.text((x1, y1), label_text, fill="white", font=font)

    
    if display_result:
        print("[INFO]: Displaying image with detections...")
        img.show()
 

    if output_image_folder is not None:
        print("[INFO]: Saving labelled image of", os.path.basename(image_path), "to", output_image_folder)
        if not os.path.exists(output_image_folder):
            print("[INFO]: Creating output image folder at", output_image_folder)
            os.makedirs(output_image_folder)
        
        filename = os.path.basename(image_path)
        output_path = Path(output_image_folder)
        img.save(str(output_path / filename))
    
    if output_object_folder is not None:
        print("[INFO]: Saving detected objects from", os.path.basename(image_path))
        
        if not os.path.exists(output_object_folder):
            print("[INFO]: Creating output object folder at", output_object_folder)
            os.makedirs(output_object_folder)
            
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
    
        save(name, detection, output_object_folder + f"/{name}.json")
    
    return img

