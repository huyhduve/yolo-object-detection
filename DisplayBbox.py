from PIL import Image, ImageDraw, ImageFont
import random

# generate random RGB tuple
def random_rgb():
    return (random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255))

def display(image_path, boxes, labels=None, scores=None):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        color = random_rgb()
        draw.rectangle([x1, y1, x2, y2], outline= color, width=2)

        label_text = ""
        if labels:
            label_text += str(labels[i])
        if scores:
            label_text += f" {scores[i]:.2f}"

        if label_text:
            text_size = draw.textbbox((x1, y1), label_text, font=font)
            draw.rectangle([text_size[0], text_size[1], text_size[2], text_size[3]], fill="purple")
            draw.text((x1, y1), label_text, fill="white", font=font)

    # Show image
    img.show()

    return img

if __name__ == "__main__":
    # Example usage
    image_path = "D:/Python/CLIP-DEMO/Images/cat.jpg"
    boxes = [[50, 50, 200, 200]]
    labels = ["Cat"]
    scores = [0.95]

    display(image_path, boxes, labels, scores)