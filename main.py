from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO('yolov8m.yaml').load('./runs/detect/BodyHand/weights/last.pt')  # build from YAML and transfer weights
# Train the model
results = model.train(data="ultralytics/yolo/data/datasets/JointBP_BodyHands.yaml", 
                      epochs=100, 
                      imgsz=640,
                      device='0,1')