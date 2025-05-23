# Retail Shelf Object Detector Using YOLOv8
Ever wondered what's on a retail shelf just by looking at a photo? Detect and count retail products from both photos and real-time camera feeds—instantly! This smart and simple app uses YOLOv8 to analyze retail shelves, highlight every product it spots, and give you a clear count. 

The Retail Shelf Object Detector is a web app built with YOLOv8 and Gradio. It lets you upload retail shelf images or use your webcam to detect and count products in real time.

🔗 Live Demo: https://riashah1204-retail-shelf-object-detector.hf.space/

## Features

* Detects products on retail shelves using a trained YOLOv8 model
* Works with both uploaded images and live webcam feed
* Shows results with bounding boxes and class labels
* Displays item-wise count summary
* Easy-to-use interface, accessible online

## How It Works

1. Upload an image or turn on your webcam
2. The app detects and classifies retail items using YOLOv8
3. You get an annotated image and a breakdown of item counts

## Real Time Object Detection

 <img width="450" alt="image" src="https://github.com/user-attachments/assets/8176d5f0-c1f4-4132-b9f0-792c5c9c54f9" />
<img width="450" alt="image" src="https://github.com/user-attachments/assets/f0e2ae21-9990-4bfe-b8cd-157e98e1531e" />


## Object Detection of Uploaded Image by User

<img width="450" alt="image" src="https://github.com/user-attachments/assets/5ca1c2e2-e229-47ab-b40f-c0d70378e6e3" />


## Run Locally

Download or clone the repository:
git clone https://github.com/riashah1204/Retail-Shelf-Object-Detection-with-YOLOv8

Install the required libraries:
pip install -r requirements.txt

Start the app:
python app.py
