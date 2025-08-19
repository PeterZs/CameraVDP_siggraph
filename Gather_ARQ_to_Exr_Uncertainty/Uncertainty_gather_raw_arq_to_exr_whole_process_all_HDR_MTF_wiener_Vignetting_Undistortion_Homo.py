import os
import HDRutils
import numpy as np
import json
import cv2
from tqdm import tqdm
import pyexr
image_root_path = r'' # Please set it by yourself
image_save_path = image_root_path + '_uncertainty_full_all'
os.makedirs(image_save_path, exist_ok=True)
files_list = [] # Please set it by yourself
save_exr_name_list = [] # Please set it by yourself
if len(files_list) != len(save_exr_name_list):
    raise ValueError('The lengths are not equal!')

with open(f'calibration.json', 'r') as fp: # Please set it by yourself
    camera_calibration_result = json.load(fp)
camera_matrix = np.array(camera_calibration_result['mtx'])
dist_coeffs = np.array(camera_calibration_result['dist'])

vignetting_scaler_RGB = np.load(r'vignetting_map.npz')['vignetting'] # Please set it by yourself

k_scale = 3 # o_scale + 1, o is in original paper
display_width_pixel = 3840
display_height_pixel = 2160
# You can make it faster using differnt scale (if you only care about the image center)
central_crop_width_pixel = display_width_pixel #/ 5
central_crop_height_pixel = display_height_pixel #/ 5
before_homo_crop_ratio_w = 1 #/4
before_homo_crop_ratio_h = 1 #/4
json_file_name = r'homography_aruco_result_exr.json' # Please set it by yourself
with open(json_file_name, 'r') as fp:
    homography_aruco_result = json.load(fp)
obj_points = np.array(homography_aruco_result['obj_points'])
obj_points[:, 0] = obj_points[:, 0] + display_width_pixel / 2 #Move the starting position from the midpoint to the upper left corner and reverse the y-axis coordinate.
obj_points[:, 1] = -obj_points[:, 1]
obj_points[:, 1] = obj_points[:, 1] + display_height_pixel / 2

obj_points = obj_points * (k_scale - 1)
img_points = np.array(homography_aruco_result['img_points'])
display_points = obj_points[:, :2]
homography_matrix, mask = cv2.findHomography(img_points, display_points)

display_image_record_width = round(display_width_pixel * (k_scale - 1) + 1)
display_image_record_height = round(display_height_pixel * (k_scale - 1) + 1)
central_crop_display_image_record_width = round(central_crop_width_pixel * (k_scale - 1) + 1)
central_crop_display_image_record_height = round(central_crop_height_pixel * (k_scale - 1) + 1)
crop_width = int(before_homo_crop_ratio_w * display_image_record_width)
crop_height = int(before_homo_crop_ratio_h * display_image_record_height)
start_x = (display_image_record_width - crop_width) // 2
start_y = (display_image_record_height - crop_height) // 2
translate_mat = np.array([
        [1, 0, -start_x],
        [0, 1, -start_y],
        [0, 0, 1]
    ], dtype=np.float32)
homography_matrix_cropped = translate_mat @ homography_matrix
H_inv = np.linalg.inv(homography_matrix_cropped)
warp_output_size = (crop_width, crop_height)
warp_w, warp_h = warp_output_size
display_image_record_height = crop_height
display_image_record_width = crop_width
# 1. Generate output image plane coordinate grid (note warp_output_size)
grid_x, grid_y = np.meshgrid(np.arange(warp_w), np.arange(warp_h))  # shape: (H, W)
grid = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)  # (H, W, 2)
grid_flat = grid.reshape(-1, 2)              # (H*W, 2)
grid_cv = grid_flat[:, np.newaxis, :]        # (H*W, 1, 2)
grid_homogeneous = cv2.convertPointsToHomogeneous(grid_cv).reshape(-1, 3).T  # (3, H*W)
# 2. Inverse perspective transform to undistorted image space
back_proj = H_inv @ grid_homogeneous
back_proj = (back_proj[:2] / back_proj[2:]).T.reshape((warp_h, warp_w, 2)).astype(np.float32)  # (H, W, 2)

h, w = 5320, 7968
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
x, y, w_roi, h_roi = roi
back_proj[..., 0] += x
back_proj[..., 1] += y

mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, newcameramtx, (w, h), cv2.CV_32FC1)
# 3. Interpolation remap mapping (based on mapx/mapy)
mapx_full = cv2.remap(mapx, back_proj[..., 0].astype(np.float32), back_proj[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR)
mapy_full = cv2.remap(mapy, back_proj[..., 0].astype(np.float32), back_proj[..., 1].astype(np.float32), interpolation=cv2.INTER_LINEAR)

for gather_index in tqdm(range(len(files_list))):
    files = files_list[gather_index]
    save_exr_name = save_exr_name_list[gather_index]
    files = [os.path.join(image_root_path, i) for i in files]  # RAW input files
    HDR_img_mean, HDR_img_V, unsaturated = HDRutils.merge_uncertainty(files, demosaic_first=False, color_space='raw',
                                                                      arq_no_demosiac=True, mtf_json='MTF/mtf.json', # Please set it by yourself
                                                                      wiener=True)
    HDR_img_mean[HDR_img_mean < 0] = 0
    HDR_img_V[HDR_img_V < 0] = 0
    HDR_img_mean = HDR_img_mean / vignetting_scaler_RGB
    HDR_img_V = HDR_img_V / vignetting_scaler_RGB ** 2
    # h, w = HDR_img_mean.shape[:2]
    # print('h, w:', h, w)
    # 4. Final remap: Do it once directly on the original HDR image
    aligned_RGB_image_mean = cv2.remap(HDR_img_mean, mapx_full, mapy_full, interpolation=cv2.INTER_LINEAR)
    aligned_RGB_image_var = cv2.remap(HDR_img_V, mapx_full, mapy_full, interpolation=cv2.INTER_LINEAR)

    start_x = (display_image_record_width - central_crop_display_image_record_width) // 2
    start_y = (display_image_record_height - central_crop_display_image_record_height) // 2
    cropped_image_RGB_linear_mean = aligned_RGB_image_mean[start_y:start_y + central_crop_display_image_record_height,
                                    start_x:start_x + central_crop_display_image_record_width]
    cropped_image_RGB_linear_var = aligned_RGB_image_var[start_y:start_y + central_crop_display_image_record_height,
                                   start_x:start_x + central_crop_display_image_record_width]
    cropped_image_RGB_linear_mean_plot = cropped_image_RGB_linear_mean.copy()
    cropped_image_RGB_linear_mean[cropped_image_RGB_linear_mean < 0] = 0
    cropped_image_RGB_linear_var[cropped_image_RGB_linear_var < 0] = 0
    # cropped_image_rgb_encoded = (cropped_image_RGB_linear_mean_plot / 20000) ** (1 / 2.2)
    # cropped_image_rgb_clipped = np.clip(cropped_image_rgb_encoded, 0, 1) * 255
    # cropped_image_rgb_uint8 = cropped_image_rgb_clipped.round().astype(np.uint8)
    # cv2.imwrite(os.path.join(image_save_path, f'RGB_HT{save_exr_name}_k{k_scale}.png'), cropped_image_rgb_uint8[:, :, ::-1])
    pyexr.write(os.path.join(image_save_path, f'RGB_HT{save_exr_name}_k{k_scale}_mean.exr'), cropped_image_RGB_linear_mean)
    pyexr.write(os.path.join(image_save_path, f'RGB_HT{save_exr_name}_k{k_scale}_var.exr'), cropped_image_RGB_linear_var)



