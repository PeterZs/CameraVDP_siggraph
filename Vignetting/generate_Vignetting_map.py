import os
import numpy as np
import pyexr
import cv2
import matplotlib.pyplot as plt
from glob import glob
import argparse


def read_exr_file(filepath):
    return pyexr.open(filepath).get()  # shape (H, W, C)


def process_vignetting_map(img, blur_ksize=101, blur_sigma=0):
    vignette_map = np.zeros_like(img)
    for c in range(3):
        blurred = cv2.GaussianBlur(img[..., c], (blur_ksize, blur_ksize), blur_sigma)
        peak = np.max(blurred)
        vignette_map[..., c] = blurred / peak  # normalization
    return np.clip(vignette_map, 0, 1)


def plot_vignetting_map(vignetting_map, title):
    channels = ['Red', 'Green', 'Blue']
    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(vignetting_map[..., i], cmap='inferno')
        plt.colorbar()
        plt.title(f'{title} - {channels[i]}')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root_path', type=str, required=True)
    args = parser.parse_args()
    folder = args.image_root_path
    # folder = r'F:\sony_pictures_new\Psycho_Stimulus_V'  # change to your path
    # filenames = ['V_white_3.exr', 'V_white_4.exr']
    filenames = ['V_white_1.exr', 'V_white_2.exr', 'V_white_3.exr', 'V_white_4.exr']
    filepaths = [os.path.join(folder, f) for f in filenames]

    # Read and process each file
    images = {f: process_vignetting_map(read_exr_file(p), blur_ksize=1) for f, p in zip(filenames, filepaths)}

    # # Plot individual Vignetting Maps
    # for f, v_map in images.items():
    #     plot_vignetting_map(v_map, title=f)

    # Compute the final RGB Vignetting Map
    R_vignette = np.mean(
        [images['V_white_1.exr'][..., 0], images['V_white_2.exr'][..., 0], images['V_white_3.exr'][..., 0],
         images['V_white_4.exr'][..., 0]], axis=0)
    G_vignette = np.mean(
        [images['V_white_1.exr'][..., 1], images['V_white_2.exr'][..., 1], images['V_white_3.exr'][..., 1],
         images['V_white_4.exr'][..., 1]], axis=0)
    B_vignette = np.mean(
        [images['V_white_1.exr'][..., 2], images['V_white_2.exr'][..., 2], images['V_white_3.exr'][..., 2],
         images['V_white_4.exr'][..., 2]], axis=0)
    # R_vignette = np.mean([images['V_white_3.exr'][..., 0], images['V_white_4.exr'][..., 0]], axis=0)
    # G_vignette = np.mean([images['V_white_3.exr'][..., 1], images['V_white_4.exr'][..., 1]], axis=0)
    # B_vignette = np.mean([images['V_white_3.exr'][..., 2], images['V_white_4.exr'][..., 2]], axis=0)
    R_vignette = R_vignette / R_vignette.max()
    G_vignette = G_vignette / G_vignette.max()
    B_vignette = B_vignette / B_vignette.max()
    final_vignette = np.stack([R_vignette, G_vignette, B_vignette], axis=-1)
    # plot_vignetting_map(final_vignette, title='Final RGB Vignetting Map')

    # Save the result
    np.savez(os.path.join(folder, 'vignetting_map.npz'), vignetting=final_vignette)
