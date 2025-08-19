import pyexr
import os
import numpy as np
import glob
from tqdm import tqdm
import json

json_data_dict = {}
image_root_path = rf"" # Please set it by yourself Where you 30 .exr color checker images are
exr_images = glob.glob(os.path.join(image_root_path, '*.exr'))

crop_size = 1000

for exr_file_name in tqdm(exr_images):
    exr = pyexr.open(exr_file_name)
    exr_data = exr.get()
    exr_data[exr_data < 0] = 0

    # center crop
    h, w, c = exr_data.shape
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    exr_data_cropped = exr_data[start_h:start_h+crop_size, start_w:start_w+crop_size, :]

    mean_RGB = np.mean(exr_data_cropped, axis=(0, 1))
    exr_file_name_key = os.path.basename(exr_file_name).split('.')[0]
    json_data_dict[exr_file_name_key] = mean_RGB.tolist()

with open(f'MeanRGB_Camera_Colorchecker.json', 'w') as outfile: # Please set it by yourself
    json.dump(json_data_dict, outfile, indent=4)
