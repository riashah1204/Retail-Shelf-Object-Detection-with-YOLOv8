from ultralytics import YOLO
import gradio as gr
from PIL import Image
from collections import Counter

model = YOLO("app/best.pt")

def detect_classify(image):
    results = model(image)[0]
    boxes = results.boxes

    if boxes is not None and len(boxes.cls) > 0:
        class_ids = boxes.cls.tolist()
        names = results.names
        labels = [names[int(cls_id)] for cls_id in class_ids]
        label_counts = Counter(labels)
        count_str = ", ".join([f"{v} {k}" for k, v in label_counts.items()])
        total = sum(label_counts.values())
        final_count = f"Total Detected: {total}\nBreakdown: {count_str}"
    else:
        final_count = "No objects detected."

    annotated_img = Image.fromarray(results.plot())
    return annotated_img, final_count

demo = gr.Interface(
    fn=detect_classify,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Image(label="Detected Image"),
        gr.Label(label="Detection Summary")
    ],
    title="Retail Shelf Detector",
    description="Upload an image to detect retail shelf objects using YOLOv8."
)

demo.launch(share=True)
