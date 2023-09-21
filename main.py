from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO('yolov5l6.yaml').load('yolov5l6u.pt') # .load('./runs/detect/BodyHand2/weights/best.pt')  # build from YAML and transfer weights
# model = YOLO('./runs/detect/HumanParts10/weights/last.pt') # .load('yolov5l6u.pt') # .load('./runs/detect/BodyHand2/weights/best.pt') 
# Train the model
results = model.train(data="ultralytics/yolo/data/datasets/JointBP_CrowdHuman_face.yaml", 
                      epochs=100, 
                      imgsz=1536, ## 640
                      name="CrowdHuman-yolov5l6-1536r",
                      batch=12,
                    #   resume=True,
                      device='0,1,2,3')
