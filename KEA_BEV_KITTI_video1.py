import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


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

### Main
### Main
vid = cv2.VideoCapture('Python_code/KEA_0523/test_img/video/KITTI_test6.mp4')

while(True):
      ret, src = vid.read()
      if not ret:
            break
      BirdEyeView = cv2.warpPerspective(src, T_Bev, (OutImageSize[1], OutImageSize[0]))

      cv2.imshow("BEV", BirdEyeView)
      if cv2.waitKey(1) & 0xFF == ord('q'):
            break
vid.release()
cv2.destroyAllWindows()