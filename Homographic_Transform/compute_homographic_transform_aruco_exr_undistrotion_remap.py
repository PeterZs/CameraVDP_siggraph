import cv2
import numpy as np
import os
import json
import pyexr

with open(f'calibration.json', 'r') as fp:  # Please set it by yourself
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

# Set Aruco dictionary and parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Pattern rows, columns, side lengths, and spacing
rows = 22  # Please set it by yourself
cols = 40  # Please set it by yourself
marker_length = 48  # Please set it by yourself
marker_spacing = 48  # Please set it by yourself
origin_x = (cols * marker_length + (cols - 1) * marker_spacing) / 2
origin_y = (rows * marker_length + (rows - 1) * marker_spacing) / 2
x_bias = 0
y_bias = 0
# Read Images
root_img_path = r''
img_name_list = ['ArUco_mean.exr']
img_path_list = [os.path.join(root_img_path, i) for i in img_name_list]

for img_path in img_path_list:
    obj_points = []  # 3D world coordinate point
    img_points = []  # 2D image coordinate points
    img_ID = img_path.split('\\')[-1].split('.')[0]
    exr = pyexr.open(img_path)
    exr_data = exr.get()
    image = ((exr_data / 20000)**(1/2.2) * 255).clip(0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resize_scale = 1
    if resize_scale == 1:
        gray_resize = gray.copy()
    else:
        new_height = round(image.shape[0] / resize_scale)
        new_width = round(image.shape[1] / resize_scale)
        gray_resize = cv2.resize(gray, (new_width, new_height))
    blur_kernel_size = 1 # Sub-pixel structure will seriously affect the accuracy. Please use the blur index appropriately to get the best result (commonly used ones are 7, 13)
    if blur_kernel_size == 1:
        gray_blur = gray_resize
    else:
        gray_blur = cv2.GaussianBlur(gray_resize, (blur_kernel_size, blur_kernel_size), 0)
    corners, ids, rejectedImgPoints = detector.detectMarkers(gray_blur)

    if ids is not None:
        for i, corner in enumerate(corners):
            corner *= resize_scale
            points = corner[0].astype(int)  # Convert corners to integer pixel coordinates
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=1)

            marker_id = ids[i][0]

            # Calculate the row and column position of the mark
            row = marker_id // cols
            col = marker_id % cols

            # Calculate the center coordinates of the marker
            center_x = col * (marker_length + marker_spacing) + marker_length / 2 - origin_x + x_bias
            center_y = (rows - row - 1) * (marker_length + marker_spacing) + marker_length / 2 - origin_y + y_bias

            # Calculate the coordinates of the four corners of the marker
            bias = 0 #1/2
            top_left = np.array([center_x - marker_length / 2 + bias, center_y + marker_length / 2 - bias, 0], dtype=np.float32)
            top_right = np.array([center_x + marker_length / 2 - bias, center_y + marker_length / 2 - bias, 0], dtype=np.float32)
            bottom_right = np.array([center_x + marker_length / 2 - bias, center_y - marker_length / 2 + bias, 0], dtype=np.float32)
            bottom_left = np.array([center_x - marker_length / 2 + bias, center_y - marker_length / 2 + bias, 0], dtype=np.float32)

            # Calculate posture
            obj_points.extend([top_left, top_right, bottom_right, bottom_left])
            img_points.extend(corner[0])

            # Draw the ID at the center of the marker
            center = tuple(corner[0].mean(axis=0).astype(int))
            cv2.putText(image, f"{marker_id}", center, cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 10, cv2.LINE_AA)
        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)
        retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, length=1000, thickness=30)
        json_data = {'obj_points': obj_points.tolist(), 'img_points': img_points.tolist(), 'retval': retval,
                     'rvec': rvec.tolist(), 'tvec': tvec.tolist()}
        with open(os.path.join(root_img_path, f'homography_aruco_result_exr.json'), 'w') as fp:
            json.dump(json_data, fp)
    else:
        print("No markers detected.")

    # 显示结果
    output_path = f'{img_ID}_rz{resize_scale}_blur{blur_kernel_size}.png'
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(root_img_path, output_path), img_bgr)
    print(os.path.join(root_img_path, output_path))
