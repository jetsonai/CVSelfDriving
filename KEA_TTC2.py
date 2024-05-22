import torch
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from sort import *
import time
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

def BEV():
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
        SpaceToOneSide = 4
        BottomOffset = 0

        OutView = (BottomOffset, DistAhead, -SpaceToOneSide, SpaceToOneSide)
        OutImageSize = [math.nan, 250]

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
        return Trans, T_Bev

# Model
model = torch.hub.load('C:/Users/syhoo/Code/yolov5','yolov5x', 'yolov5x.pt',source='local') 
#ImagePath = 'C:\\Users\\syhoo\\VScode\\Python_Project\\KITTI\\2011_09_26_drive_0011_sync\\2011_09_26\\2011_09_26_drive_0011_sync\\image_02\\data'
vid = cv2.VideoCapture('test_img/KITTI_test6.mp4')
#frame_rate = vid.get(cv2.CAP_PROP_FPS)
#time_between_frames = 1.0 / frame_rate
prev_time = time.time()
mot_tracker = Sort()
Trans,T_Bev = BEV()
print(Trans)

def class_to_label(x):
     classes = model.names
     return classes[int(x)]

def calculate_center_and_distance(CenterPoint, T_Bev, Trans):
    Center = np.r_[CenterPoint[0],np.shape(CenterPoint)[0]]
    V_center = np.dot(T_Bev,Center)
    V_dist = V_center / V_center[2]
    Dist = np.matmul(V_dist, Trans)
    return V_dist[0:2], Dist[0:2]

colours = np.random.rand(32, 3)*255 #used only for display
prev_X_dist = {}
while(True):
    current_time = time.time()
    ret, image_show = vid.read()
    preds = model(image_show)
    detections = preds.pred[0].cpu().numpy()
    detections = detections[(detections[:, 5] >= 0) & (detections[:, 5] <= 7)]
    #print("detections : ", detections)
    #preds.xyxy[0]
    #preds.pandas().xyxy[0]
    track_bbs_ids = mot_tracker.update(detections)


    for j in range(len(track_bbs_ids.tolist())):
        coords = track_bbs_ids.tolist()[j]
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]),int(coords[2]),int(coords[3])
        name_idx = int(coords[4])
        name = "ID : {}".format(str(name_idx))
        color = colours[name_idx % len(colours)]

        xCenter = (x1-x2)/2 + x1
        yBottom = y2
        
        CenterPoint = [[xCenter, yBottom]]
        V_center, Dist = calculate_center_and_distance(CenterPoint, T_Bev, Trans)
        V_center_x_int = int(V_center[0])
        V_center_y_int = int(V_center[1])
        
        X_dist = Dist[0]
        Y_dist = Dist[1] 
        # Calculate speed and TTC if we have a previous X_dist for this ID
        if name_idx in prev_X_dist and -5 <= Y_dist <= 5:
            if name_idx in prev_X_dist:
                prev_dist = prev_X_dist[name_idx]
                time_between_frames = current_time - prev_time  # 프레임 사이의 실제 시간 계산
                speed = abs(X_dist - prev_dist)/time_between_frames  # Speed in X direction
                TTC = X_dist / speed if speed != 0 else float('inf')
                print("TTC for ID {}: {}".format(name_idx, TTC))

        # Update the X_dist for this ID
        prev_X_dist[name_idx] = X_dist   
        #color = colours[name_idx]
        cv2.rectangle(image_show,(x1,y1),(x2,y2),color,2)
        cv2.putText(image_show,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)
        cv2.putText(image_show, class_to_label(detections[j,5])
                    + ': ' + "{:.1f}".format(X_dist) + ', ' + "{:.1f}".format(Y_dist),
                    (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)        
    cv2.imshow('Image',image_show)    
    prev_time = current_time  # 현재 시간을 이전 시간으로 설정

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()



