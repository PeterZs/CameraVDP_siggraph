import numpy as np
import os
import json
import glob
import HDRutils
from tqdm import tqdm
import cv2

def generate_white_steps(n_steps=10):
    step_values = [round(i * 255 / (n_steps - 1)) for i in range(n_steps)]
    white_steps = [(v, v, v) for v in step_values]
    return white_steps

colorchecker_colors = generate_white_steps(11)

root_path = r'' # Please set it by yourself
RAW_files = glob.glob(os.path.join(root_path, '*.ARQ'))
vignetting_scaler_RGB = np.load('vignetting_map.npz')['vignetting'] # Please set it by yourself
crop_h, crop_w = 500, 500
R_mean_list = []
G_mean_list = []
B_mean_list = []
R_var_list = []
G_var_list = []
B_var_list = []
for RAW_file_index in tqdm(range(len(RAW_files))):
    RAW_file = RAW_files[RAW_file_index]
    img_data_4_C = HDRutils.imread(RAW_file, color_space='raw')
    img_data_RGB = np.stack([img_data_4_C[...,0],
                             (img_data_4_C[...,1]+img_data_4_C[...,3])/2,
                             img_data_4_C[...,2]], axis=-1)
    img_data_V = img_data_RGB/vignetting_scaler_RGB
    center_y, center_x = img_data_V.shape[0] // 2, img_data_V.shape[1] // 2
    start_y = center_y - crop_h // 2
    end_y = center_y + crop_h // 2
    start_x = center_x - crop_w // 2
    end_x = center_x + crop_w // 2
    img_data_V_crop = img_data_V[start_y:end_y, start_x:end_x, :]
    R_mean = float(img_data_V_crop[..., 0].mean())
    G_mean = float(img_data_V_crop[..., 1].mean())
    B_mean = float(img_data_V_crop[..., 2].mean())\

    sigma = 101  # You can adjust the blur level as needed
    # Use cv2.GaussianBlur to smooth each channel (ksize = 0 will automatically choose the appropriate size based on sigma)
    R_blur = cv2.GaussianBlur(img_data_V_crop[..., 0], ksize=(0, 0), sigmaX=sigma)
    G_blur = cv2.GaussianBlur(img_data_V_crop[..., 1], ksize=(0, 0), sigmaX=sigma)
    B_blur = cv2.GaussianBlur(img_data_V_crop[..., 2], ksize=(0, 0), sigmaX=sigma)
    # Subtract the smoothed image from the original image to get the high-frequency residual image
    R_residual = img_data_V_crop[..., 0] - R_blur
    G_residual = img_data_V_crop[..., 1] - G_blur
    B_residual = img_data_V_crop[..., 2] - B_blur
    # Calculate the variance of the residual graph (reflecting high-frequency energy)
    R_var = float(np.var(R_residual))
    G_var = float(np.var(G_residual))
    B_var = float(np.var(B_residual))

    R_mean_list.append(R_mean)
    G_mean_list.append(G_mean)
    B_mean_list.append(B_mean)
    R_var_list.append(R_var)
    G_var_list.append(G_var)
    B_var_list.append(B_var)
json_data_dict = {'R_mean_list': R_mean_list,
                  'G_mean_list': G_mean_list,
                  'B_mean_list': B_mean_list,
                  'R_var_list': R_var_list,
                  'G_var_list': G_var_list,
                  'B_var_list': B_var_list}
with open(f'Noise_model_bright.json', 'w') as f:
    json.dump(json_data_dict, f)
