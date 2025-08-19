import sys
sys.path.append('../../CameraVDP_siggraph') # Path of CameraVDP_siggraph
import os
import HDRutils
from tqdm import tqdm

image_root_path = r'' # Please set it by yourself (where your ARQ files are)
image_save_path = image_root_path + '_uncertainty_HDR'
os.makedirs(image_save_path, exist_ok=True)
files_list = [] # Please set it by yourself (where your ARQ files are) [[f'DSC{str(i).zfill(5)}_PSMS.ARQ', f'DSC{str(i + 4).zfill(5)}_PSMS.ARQ'] for i in range(0, 5, 8)]
save_exr_name_list = [] # Please set it by yourself (your exr files name)
if len(files_list) != len(save_exr_name_list):
    raise ValueError('The lengths are not equal!')
for gather_index in tqdm(range(len(files_list))):
    files = files_list[gather_index]
    save_exr_name = save_exr_name_list[gather_index]
    files = [os.path.join(image_root_path, i) for i in files]  # RAW input files
    HDR_img_mean, HDR_img_var, unsaturated = HDRutils.merge_uncertainty(files, demosaic_first=False, color_space='raw', arq_no_demosiac=True)
    HDR_img_mean[HDR_img_mean < 0] = 0
    HDR_img_var[HDR_img_var < 0] = 0
    HDRutils.imwrite(os.path.join(image_save_path, f'{save_exr_name}_mean.exr'), HDR_img_mean)
    HDRutils.imwrite(os.path.join(image_save_path, f'{save_exr_name}_var.exr'), HDR_img_var)