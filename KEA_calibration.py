import numpy as np
import cv2

# Define the number of inner corners in the checkerboard
num_corners_x = 9
num_corners_y = 6

# Create arrays to store object points and image points from all images
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Generate the coordinates of the corners in the checkerboard
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)

# Read the images and find the corners
images = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with your image filenames

for filename in images:
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the corners of the checkerboard
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

    # If corners are found, add object points and image points
    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (num_corners_x, num_corners_y), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients
np.save('camera_matrix.npy', mtx)
np.save('distortion_coeffs.npy', dist)
# Undistort the images using the camera matrix and distortion coefficients
undistorted_images = []
for filename in images:
    img = cv2.imread(filename)
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    undistorted_images.append(undistorted_img)

# Save the undistorted images
for i, undistorted_img in enumerate(undistorted_images):
    cv2.imwrite(f'undistorted_image{i+1}.jpg', undistorted_img)
# Path: Python_code/undistort.py
# import cv2
# import numpy as np
# import glob
# # Load the camera matrix and distortion coefficients
# mtx = np.load('camera_matrix.npy')