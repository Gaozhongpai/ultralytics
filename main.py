from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="ultralytics/yolo/data/datasets/hand.yaml", 
                      epochs=100, 
                      imgsz=640,
                      device='0,1,2')