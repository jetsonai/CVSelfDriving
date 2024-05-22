import cv2
import numpy as np
import matplotlib.pyplot as plt

def nothing(x):
    pass

def Color_space_trackbar(image):
    color_spaces = ['HSV', 'RGB']
    channel_names = {'HSV': ['H', 'S', 'V'], 'RGB': ['R', 'G', 'B']}
    trackbars = ['_min', '_max']

    masks = {}

    for color_space in color_spaces:
        for i in range(len(channel_names[color_space])):
            cv2.namedWindow(color_space + ' - ' + channel_names[color_space][i])
            for trackbar in trackbars:
                initial_value = 0 if trackbar == '_min' else 255
                trackbar_name = channel_names[color_space][i] + trackbar
                cv2.createTrackbar(trackbar_name, color_space + ' - ' + channel_names[color_space][i], initial_value, 255, nothing)

    while True:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        color_space_images = [hsv, rgb, ycrcb, lab]

        for i, color_space in enumerate(color_spaces):
            color_mask = None
            for j in range(len(channel_names[color_space])):
                lower = cv2.getTrackbarPos(channel_names[color_space][j] + '_min', color_space + ' - ' + channel_names[color_space][j])
                upper = cv2.getTrackbarPos(channel_names[color_space][j] + '_max', color_space + ' - ' + channel_names[color_space][j])
                mask = cv2.inRange(color_space_images[i][:,:,j], lower, upper)
                masks[color_space + channel_names[color_space][j]] = mask
                color_mask = mask if color_mask is None else cv2.bitwise_and(color_mask, mask)
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                _, binary_image = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                binary_image_bgr = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
                combined_image = np.vstack((binary_image_bgr, masked_image))
                cv2.imshow(color_space + ' - ' + channel_names[color_space][j], combined_image)
            color_mask_image = cv2.bitwise_and(image, image, mask=color_mask)
            cv2.imshow(color_space + ' - Combined', color_mask_image)

        # Combine masks from different color spaces
        combined_mask = None
        for key in masks:
            combined_mask = masks[key] if combined_mask is None else cv2.bitwise_and(combined_mask, masks[key])
        combined_masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
        cv2.imshow('Combined', combined_masked_image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    image = cv2.imread('test_img/test1.jpg')
    resized_image = cv2.resize(image, (400, 200))
    Color_space_trackbar(resized_image)