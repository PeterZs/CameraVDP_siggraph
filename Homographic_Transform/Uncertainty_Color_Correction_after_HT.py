import pyexr
import numpy as np
import sys
sys.path.append('../../CameraVDP_siggraph') # Path of CameraVDP_siggraph
from Color_correction.RGB_2_XYZ_color_correction import transform_RGB_2_XYZ_color_correction_uncertainty_cov_chunk
from tqdm import tqdm
import os
import HDRutils

root_path = r''
image_name_list = []
k_scale = 3
for img_name in tqdm(image_name_list):
    img_mean_path = os.path.join(root_path, f'RGB_HT{img_name}_k{k_scale}_mean.exr') # Please set it by yourself
    img_var_path = os.path.join(root_path, f'RGB_HT{img_name}_k{k_scale}_var.exr')
    img_mean_exr = pyexr.open(img_mean_path).get()# [5278, 7921, 3]
    img_mean_exr[img_mean_exr < 1] = 1
    img_var_exr = pyexr.open(img_var_path).get()  # [5278, 7921, 3]
    img_var_exr[img_var_exr < 0] = 0
    img_std_exr = img_var_exr ** 0.5
    XYZ_linear_mean, XYZ_linear_cov = transform_RGB_2_XYZ_color_correction_uncertainty_cov_chunk(RGBs_mean=img_mean_exr,
                                                                                             RGBs_std=img_std_exr,
                                                                                             mode='Sony_a7R3_FE35_F20_Eizo', # Please set it by yourself
                                                                                             expand=False,
                                                                                             chunk_size=100)
    XYZ_linear_mean[XYZ_linear_mean < 1] = 1
    HDRutils.imwrite(os.path.join(root_path, f'{img_name}_XYZ_linear_k{k_scale}_mean.exr'), XYZ_linear_mean) # Please set it by yourself
    np.savez_compressed(os.path.join(root_path, f'{img_name}_XYZ_linear_k{k_scale}_cov.npz'), cov_matrix=XYZ_linear_cov)


