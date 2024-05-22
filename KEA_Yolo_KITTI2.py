import torch
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sort import *


# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
model = torch.hub.load('C:/Users/syhoo/Code/yolov5','yolov5x', 'yolov5x.pt',source='local') 
#ImagePath = 'C:\\Users\\syhoo\\VScode\\Python_Project\\KITTI\\2011_09_26_drive_0011_sync\\2011_09_26\\2011_09_26_drive_0011_sync\\image_02\\data'
vid = cv2.VideoCapture('test_img/KITTI_test6.mp4')
mot_tracker = Sort()


colours = np.random.rand(32, 3)*255 #used only for display
while(True):
    ret, image_show = vid.read()
    preds = model(image_show)
    detections = preds.pred[0].cpu().numpy()


    # Filter detections where 6th column value is between 0 and 7
    # Persion,Bicycle,Car,Motorbike,Airplane,Bus,Train, Truck
    detections = detections[(detections[:, 5] >= 0) & (detections[:, 5] <= 7)]

    #print("preds",preds.xyxy[0])
    #print("detections : ", detections)
    #print("6th column of detections: ", detections[:, 5])
    track_bbs_ids = mot_tracker.update(detections)
    for j in range(len(track_bbs_ids.tolist())):
        coords = track_bbs_ids.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]),int(coords[2]),int(coords[3])
        name_idx = int(coords[4])
        name = "ID : {}".format(str(name_idx))
        color = colours[name_idx % len(colours)]
        cv2.rectangle(image_show,(x1,y1),(x2,y2),color,2)
        cv2.putText(image_show,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
        
    cv2.imshow('Image',image_show)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
