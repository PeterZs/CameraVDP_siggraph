import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
import json
import HDRutils
square_size = 64
checkerboard_size_row = 31
checkerboard_size_col = 57
x, y = np.mgrid[0:checkerboard_size_col, 0:checkerboard_size_row]
objp = np.zeros((checkerboard_size_row * checkerboard_size_col, 3), np.float32)
objp[:, :2] = np.column_stack((x.ravel(), y.ravel()))
objp[:, 0] *= square_size
objp[:, 1] *= square_size
center_x = (np.max(objp[:, 0]) + np.min(objp[:, 0])) / 2
center_y = (np.max(objp[:, 1]) + np.min(objp[:, 1])) / 2
objp[:, 0] -= center_x
objp[:, 1] -= center_y

objpoints = []
imgpoints = []

image_root_path = rf''
save_clibrate_image_path = image_root_path + '_calibrate_show'
os.makedirs(save_clibrate_image_path, exist_ok=True)
images = glob.glob(os.path.join(image_root_path, 'DSC*.ARW'))
save_images = True
for fname in tqdm(images):
    img_data = HDRutils.imread(fname, color_space='raw')
    img = (img_data/30).clip(0,255).astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = 11
    gray_blur = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    ret, corners = cv2.findChessboardCorners(gray_blur, (checkerboard_size_row, checkerboard_size_col))
    if ret == True:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
        adjusted_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(adjusted_corners)
        if save_images:
            cv2.drawChessboardCorners(img, (checkerboard_size_row, checkerboard_size_col), adjusted_corners, ret)
            cv2.imwrite(os.path.join(save_clibrate_image_path, fname.split('\\')[-1].split('.')[0]+'.png'), img)
    else:
        print('Cannot find the checker board!')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('ret: ', ret)
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))
re_projection_error = mean_error / len(objpoints)
json_data = {'ret': ret, 'mtx': mtx.tolist(), 'dist': dist.tolist(), 're_projection_error': re_projection_error}
with open(f'calibration.json', 'w') as outfile:
    json.dump(json_data, outfile)
