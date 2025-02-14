from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="H0htomGiaGj1b3IrzueF")
project = rf.workspace("cardetectioniklb5").project("psio1")
version = project.version(3)
dataset = version.download("yolov5")
model = YOLO('yolov5n.pt')  

results = model.train(
    data=f'{dataset.location}/data.yaml',  
    epochs=20,  
    imgsz=640,  
    batch=4,   
    device='cpu'   
)

metrics = model.val()
model.save('best_car_detection_project.pt') 