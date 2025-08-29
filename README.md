# How to Use CameraVDP from Scratch?

⚠️ **Warning:** You may need a large amount of storage space (in our experiments, we used more than 100GB for storing RAW images).

---

## I. Camera Calibration Stage

### 1. Calibrate Camera Intrinsics and Distortion Parameters
- Calibration is **camera-lens-focal length specific**. Different combinations require separate calibration.
- For different lenses, adjust parameters carefully so that the checkerboard occupies roughly 70% of the lens area at a fixed distance.

```bash
python Camera_calibration/plot_checker_board.py --square_size 64 --rows 32 --cols 58
```

- Capture ARW-format images of the checkerboard from fixed distances and different angles (at least 9 images).  
- Ensure the checkerboard is always fully visible and occupies ≤90% of the image for accurate calibration.

Estimate intrinsics and distortion (square size must match, rows/cols must be smaller by 1 than checkerboard):  
```bash
python calibrate_checker_board.py   # Set parameters, image paths, and output path first
```

If using a macro camera (subpixels of the display visible), use:  
```bash
python calibrate_checker_board_refine.py   # Set parameters, image paths, and output path first
```

👉 Output: `calibration.json` or `calibration_refine.json` containing keys:  
- `"ret"`  
- `"mtx"` (camera intrinsics)  
- `"dist"` (distortion parameters)  
- `"re_projection_error"`  

---

### 2. Calibrate Camera MTF (SFR) Curve
- MTF is also **camera-lens-focal length specific**.  
- Prepare a high-contrast black/white slanted edge (e.g., Siemens star center, must be optically precise).

1. Capture ARQ images (Pixel Shift Multi Shooting fused by Sony official app).  
2. Convert to `.exr`:  
```bash
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR.py
```

3. Use the generated `_mean.exr` to compute MTF:  
```bash
python MTF/compute_MTF.py your_mean.exr   # Adjust exposure parameters to avoid over/under exposure
```

👉 Output: `mtf.json` with keys: `"R"`, `"G"`, `"B"`, `"Y"` (MTF parameters for each channel).

---

### 3. Calibrate Camera Vignetting Map
- Requires a uniform light source (ideally integrating sphere, alternatively a flat-field calibrated display).

```bash
python Vignetting/plot_full_screen.py
```

- Capture ARQ images at multiple exposures, merge into `.exr`:  
```bash
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener.py
```

- Recommended: capture 4 positions × 3 exposures = 12 ARQ files, then average.  
```bash
python Vignetting/generate_Vignetting_map.py
```

👉 Output: `vignetting_map.npz`

---

### 4. Calibrate Camera Color Matrix
- Run color checker stimulus:  
```bash
python Color_correction/plot_color_checker_30colors_each.py --rect_width 2000 --rect_height 2000
```

- Measure each patch using both camera and photometer (e.g., specbos):  
```bash
python Color_correction/specbos_measure.py --output measure_specbos.json
```

- Capture ARQ images, merge into `.exr`:  
```bash
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py
```

- Compute mean RGB values:  
```bash
python Color_correction/compute_mean_RGB_value_from_exr_center_crop.py
```

- Find RGB→XYZ matrix:  
```bash
python Color_correction/find_RGB_to_XYZ_matrix.py
```

👉 Output: `Camera_Colorchecker_RGB2XYZ.json`

---

### 5. Estimate Camera Noise Parameters
1. Bright noise model (RAW variance vs. mean → quantum efficiency):  
```bash
python Camera_calibration/compute_noise_model_bright.py
```

2. Dark noise model (lens covered, variance vs. gain → read & ADC noise):  
```bash
python Camera_calibration/compute_noise_model_dark.py
```

3. Fit parameters:  
```bash
python Camera_calibration/Fit_Noise_Model_together.py
```

👉 Outputs:  
- `Noise_model_bright.json`, `Noise_model_dark.json`  
- `Fit_parameters_bright.json`, `Fit_parameters_dark.json` (keys: `k`, `sigma_read`, `sigma_adc` for RGB channels)

⚠️ Add these parameters into `HDRutils/merge_uncertainty.py` (`imread_merge_arq_no_demosaic_uncertainty`, rows 173–175).

---

## II. Display Capture Stage

### 1. Estimate Camera Extrinsics (Homography)
- Generate OpenCV ArUco patterns:  
```bash
python Homographic_Transform/plot_opencv_aruco.py
```

- Capture ARQ images (fixed camera position), merge into `.exr`:  
```bash
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py
```

👉 Output: `homography_aruco_result_exr.json`

---

### 2. End-to-End Capture (RGB mean & variance)
```bash
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion_Homo.py
```

---

### 3. Color Conversion (RGB→XYZ, with uncertainty)
```bash
python Uncertainty_Color_Correction_after_HT.py
```

👉 Output: `_mean.exr` and `_cov.npz` (XYZ mean and covariance).

---

## III. Display VDP Perceptual Evaluation Stage

### 1. Perceptual Score with Uncertainty
```bash
python run_ColorVideoVDP/run_ColorVideoVDP_uncertainty.py
```

### 2. Difference Visibility Heatmap
```bash
python run_ColorVideoVDP/run_ColorVideoVDP_heatmap.py
```

---

✅ You now have the full CameraVDP pipeline with uncertainty-aware perceptual evaluation.
