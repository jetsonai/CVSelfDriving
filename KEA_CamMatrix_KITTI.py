import numpy as np
import math
import cv2

def read_calibration_file(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.strip().split(': ')
            data[key] = np.array([float(x) for x in value.split()])

    return data

calib_path = "C:\\Users\\syhoo\\VScode\\Python_Project\\KITTI\\2011_09_26_calib\\2011_09_26"
calib_data = read_calibration_file(calib_path + '/calib_cam_to_cam.txt')

# Extract Cam2's calibration parameters
cam2_params = calib_data['P_rect_02'].reshape(3, 4)
focal_length = (cam2_params[0, 0],cam2_params[1, 1])
principal_point = (cam2_params[0, 2], cam2_params[1, 2])
print("Focal Length: ", focal_length)
print("Principal Point: ", principal_point)
pitch = calib_data['pitch'][1]  # Cam2's pitch value
yaw = calib_data['yaw'][1]      # Cam2's yaw value
roll = calib_data['roll'][1]    # Cam2's roll value

print("Focal Length: ", focal_length)
print("Principal Point: ", principal_point)
print("Pitch: ", pitch)
print("Yaw: ", yaw)
print("Roll: ", roll)