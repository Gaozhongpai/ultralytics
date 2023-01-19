from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train4/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.val(data="ultralytics/yolo/data/datasets/hand.yaml", 
                      epochs=100, 
                      imgsz=640)