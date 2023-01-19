from ultralytics import YOLO
import cv2

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train4/weights/best.pt")  # load a pretrained model (recommended for training)
# Train the model
results = model.predict(source="coco/Benchmark/okay")
print(len(results))