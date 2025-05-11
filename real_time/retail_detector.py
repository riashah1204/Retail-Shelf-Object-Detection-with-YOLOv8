from ultralytics import YOLO
import gradio as gr
from PIL import Image
from collections import Counter

model = YOLO("real_time/best.pt")

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
        final_count = f"Total Detected: {total}\n Breakdown: {count_str}"
    else:
        final_count = "No objects detected."

    annotated_img = Image.fromarray(results.plot())
    return annotated_img, final_count

with gr.Blocks(title="Retail Shelf Detector") as demo:
    gr.Markdown("#  Retail Shelf Object Detector")
    gr.Markdown("Detect and classify items using YOLOv8!")

    image_input = gr.Image(type="pil", label="Image Input")  
    detect_btn = gr.Button(" Run Detection")

    output_img = gr.Image(label="Detection Result")
    output_label = gr.Label(label="Detected Items")

    detect_btn.click(fn=detect_classify, inputs=image_input, outputs=[output_img, output_label])

demo.launch(server_name="127.0.0.1", share=True)

