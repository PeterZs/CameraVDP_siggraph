import os
import HDRutils
import numpy as np
import json
import cv2
from tqdm import tqdm

image_root_path = r'' # Please set it by yourself
image_save_path = image_root_path + '_uncertainty_HDR_MTF_wiener_Vignetting_Undistortion'
os.makedirs(image_save_path, exist_ok=True)
files_list = [] # Please set it by yourself
save_exr_name_list = [] # Please set it by yourself
if len(files_list) != len(save_exr_name_list):
    raise ValueError('The lengths are not equal!')

with open(f'Camera_calibration/calibration.json', 'r') as fp: # Please set it by yourself
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

vignetting_scaler_RGB = np.load(r'Vignetting/vignetting_map.npz')['vignetting'] # Please set it by yourself
for gather_index in tqdm(range(len(files_list))):
    files = files_list[gather_index]
    save_exr_name = save_exr_name_list[gather_index]
    files = [os.path.join(image_root_path, i) for i in files]  # RAW input files
    HDR_img_mean, HDR_img_V, unsaturated = HDRutils.merge_uncertainty(files, demosaic_first=False, color_space='raw',
                                                                      arq_no_demosiac=True, mtf_json='MTF/mtf.json',  # Please set it by yourself
                                                                      wiener=True)
    HDR_img_mean[HDR_img_mean < 0] = 0
    HDR_img_V[HDR_img_V < 0] = 0
    HDR_img_mean = HDR_img_mean / vignetting_scaler_RGB
    HDR_img_V = HDR_img_V / vignetting_scaler_RGB ** 2
    h, w = HDR_img_mean.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1)
    dst_mean = cv2.remap(HDR_img_mean, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    dst_V = cv2.remap(HDR_img_V, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    dst_V[dst_V < 0] = 0
    x, y, w, h = roi
    dst_mean = dst_mean[y:y + h, x:x + w]
    dst_V = dst_V[y:y + h, x:x + w]
    dst_shape = dst_mean.shape
    print('Shape DST:', dst_shape)
    HDR_img_mean = dst_mean
    HDRutils.imwrite(os.path.join(image_save_path, f'{save_exr_name}_mean.exr'), HDR_img_mean)
    HDR_img_var = dst_V
    HDRutils.imwrite(os.path.join(image_save_path, f'{save_exr_name}_var.exr'), HDR_img_var)



