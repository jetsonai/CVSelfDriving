import torch
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sort import *


### Camera coordinate to vehicle coordinate(ISO)
def RotX(Pitch):
      Pitch = np.deg2rad(Pitch)
      return [[1, 0, 0], [0, math.cos(Pitch), -math.sin(Pitch)], [0, math.sin(Pitch), math.cos(Pitch)]]

def RotY(Yaw):
      Yaw = np.deg2rad(Yaw)
      return [[math.cos(Yaw), 0, math.sin(Yaw)], [0, 1, 0], [-math.sin(Yaw), 0, math.cos(Yaw)]]

def RotZ(Roll):
      Roll = np.deg2rad(Roll)
      return [[math.cos(Roll), -math.sin(Roll), 0], [math.sin(Roll), math.cos(Roll), 0], [0, 0, 1]]


### Camera parameter setting
ImageSize = (375, 1242)
FocalLength = (721.5377, 721.5377)
PrinciplePoint = (609.5593, 172.854)
IntrinsicMatrix = ((FocalLength[0], 0, 0), (0, FocalLength[1], 0), (PrinciplePoint[0], PrinciplePoint[1], 1))
Height = 1.65
Pitch = 0
Yaw = 0
Roll = 0

### Bird's eye view setting
DistAhead = 40
SpaceToOneSide = 7
BottomOffset = 0

OutView = (BottomOffset, DistAhead, -SpaceToOneSide, SpaceToOneSide)
OutImageSize = [math.nan, 500]

WorldHW = (abs(OutView[1]-OutView[0]), abs(OutView[3]-OutView[2]))

Scale = (OutImageSize[1]-1)/WorldHW[1]
ScaleXY = (Scale, Scale)

OutDimFrac = Scale*WorldHW[0]
OutDim = round(OutDimFrac)+1
OutImageSize[0] = OutDim

### Homography Matrix Compute

#Translation Vector
Rotation = np.matmul(np.matmul(RotZ(-Yaw),RotX(90-Pitch)),RotZ(Roll))
TranslationinWorldUnits = (0, 0, Height)
Translation = [np.matmul(TranslationinWorldUnits, Rotation)]

#Rotation Matrix
RotationMatrix = np.matmul(RotY(180), np.matmul(RotZ(-90), np.matmul(RotZ(-Yaw), np.matmul(RotX(90-Pitch), RotZ(Roll)))))

#Camera Matrix
CameraMatrix = np.matmul(np.r_[RotationMatrix, Translation], IntrinsicMatrix)
CameraMatrix2D = np.r_[[CameraMatrix[0]], [CameraMatrix[1]], [CameraMatrix[3]]]

#Compute Vehicle-to-Image Projective Transform
VehicleHomo = np.linalg.inv(CameraMatrix2D)

AdjTransform = ((0, -1, 0), (-1, 0, 0), (0, 0, 1))
BevTransform = np.matmul(VehicleHomo, AdjTransform)

DyDxVehicle = (OutView[3], OutView[1])
tXY = [a*b for a,b in zip(ScaleXY, DyDxVehicle)]

#test = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
ViewMatrix = ((ScaleXY[0], 0, 0), (0, ScaleXY[1], 0), (tXY[0]+1, tXY[1]+1, 1))

T_Bev = np.matmul(BevTransform, ViewMatrix)
T_Bev = np.transpose(T_Bev)

toOriginalImage = np.linalg.inv(np.transpose(T_Bev))
Trans = np.matmul(toOriginalImage, VehicleHomo)


def class_to_label(x):
    # x 숫자 레이블 -> 문자열 레이블로 반환
    classes = model.names
    return classes[int(x)]

def calculate_center_and_distance(CenterPoint, T_Bev, Trans):
    Center = np.r_[CenterPoint[0],np.shape(CenterPoint)[0]]
    V_center = np.dot(T_Bev,Center)
    V_dist = V_center / V_center[2]
    Dist = np.matmul(V_dist, Trans)
    return V_dist[0:2], Dist[0:2]


# Model
model = torch.hub.load('C:/Users/syhoo/Code/yolov5','yolov5x', 'yolov5x.pt',source='local') 


vid = cv2.VideoCapture('test_img/KITTI_test6.mp4')
mot_tracker = Sort()

colours = np.random.rand(32, 3)*255 #used only for display

while(True):
    ret, image_show = vid.read()
    preds = model(image_show)
    detections = preds.pred[0].cpu().numpy()

    src = image_show
    BirdEyeView = cv2.warpPerspective(src, T_Bev, (OutImageSize[1], OutImageSize[0]))


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
        xCenter = (x2-x1)/2+x1
        yBottom = y2
        
        #print("xCenter=",xCenter,"yBottom=",yBottom)
        CenterPoint = [[xCenter,yBottom]]
        CenterPoint = [[xCenter,yBottom]]
        V_center, Dist = calculate_center_and_distance(CenterPoint, T_Bev, Trans)
        V_center_x_int = int(V_center[0])
        V_center_y_int = int(V_center[1])
        
        # Draw the coordinates on the Bird's Eye View image
        cv2.putText(BirdEyeView, "V_center_x: " + str(V_center_x_int), (V_center_x_int, V_center_y_int), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(BirdEyeView, "V_center_y: " + str(V_center_y_int), (V_center_x_int, V_center_y_int + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        X_dist = Dist[0]
        Y_dist = Dist[1]        
       
        name_idx = int(coords[4])
        name = "ID : {}".format(str(name_idx))
        color = colours[name_idx % len(colours)]
        cv2.rectangle(image_show,(x1,y1),(x2,y2),color,2)
        #cv2.putText(image_show,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
        cv2.putText(image_show, class_to_label(detections[j,5])
                    + ': ' + "{:.1f}".format(X_dist) + ', ' + "{:.1f}".format(Y_dist),
                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)       
    cv2.imshow('Image',image_show)
    cv2.imshow('BirdEyeView', BirdEyeView)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()



