# How to Use CameraVDP from Scratch?

**Warning:** You may require substantial storage space. In our experiments, more than 100 GB of storage was used for storing RAW images.

I. Camera Calibration Phase
1. **Intrinsic and Distortion Parameter Calibration**
   Calibrate the intrinsic parameters and distortion coefficients of the camera (note: this is specific to the camera-lens-focal length combination, and each configuration requires independent calibration). For different lenses, parameters must be carefully adjusted such that the checkerboard occupies approximately 70% of the frame at a fixed distance:
   ```bash
   python Camera_calibration/plot_checker_board.py --square_size 64 --rows 32 --cols 58
   ```
   Capture ARW-format images at a roughly fixed distance, with at least 9 images taken from different angles, ensuring the checkerboard fully covers the field of view (ideally ≤90% of the image area for higher accuracy).

   For typical cases (when subpixels of the display are not clearly visible in the image), estimate the intrinsic and distortion parameters:
   ```bash
   python calibrate_checker_board.py  # Set parameters, image paths, and storage paths before execution
   ```

   For macro imaging (where display subpixels are visible), use the refined calibration method:
   ```bash
   python calibrate_checker_board_refine.py  # Set parameters, image paths, and storage paths before execution
   ```

   The process will generate a `calibration.json` or `calibration_refine.json` file containing four keys: `"ret"`, `"mtx"`, `"dist"`, and `"re_projection_error"`. Here, `"mtx"` denotes the intrinsic matrix and `"dist"` the distortion parameters.

2. **MTF (SFR) Calibration**
   Calibrate the Modulation Transfer Function (SFR curve), which is camera-lens-focal length dependent. Prepare a high-contrast slanted edge (we used the center portion of a Siemens star). Ensure that the black region is sufficiently dark and the white region sufficiently bright (printing is inadequate; optical-grade materials are required).

   Capture ARQ-format images (Pixel Shift Multi Shooting merged results, provided by the official Sony app), then aggregate them into `.exr` format:
   ```bash
   python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR.py
   ```

   Compute the MTF using the generated `_mean.exr` file:
   ```bash
   python MTF/compute_MTF.py your_mean.exr
   ```
   Adjust the `exposure` parameter to avoid overexposure or underexposure.

   The process will generate `mtf.json`, containing keys `"R"`, `"G"`, `"B"`, `"Y"`, representing MTF parameters for red, green, blue, and luminance, respectively.

3. **Vignetting Calibration**
   Prepare a nearly uniform light source (ideally an integrating sphere, or alternatively, a flat-field calibrated display):
   ```bash
   python Vignetting/plot_full_screen.py
   ```
   Capture ARQ-format images without focusing on the display plane (move the camera closer to the display while retaining the focal setting used in steps 1 and 2). Aggregate into `.exr` format:
   ```bash
   python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener.py
   ```

   Recommended procedure: capture images at four display positions with three exposure levels (12 ARQ images total), then average results:
   ```bash
   python Vignetting/generate_Vignetting_map.py
   ```

   The output will be `vignetting_map.npz`. Avoid using the blur setting (`blur_ksize=1`), as internal system issues must be considered.

4. **Color Matrix Calibration**
   This calibration is camera-lens-display dependent. Generate a 30-color checker stimulus:
   ```bash
   python Color_correction/plot_color_checker_30colors_each.py --rect_width 2000 --rect_height 2000
   ```
   Measure each stimulus using both the camera and a photometer (capable of providing XYZ tristimulus values). Example code with Specbos:
   ```bash
   python Color_correction/specbos_measure.py --output measure_specbos.json
   ```

   Capture ARQ-format images and aggregate them into `.exr` format:
   ```bash
   python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py
   ```

   Compute central RGB means:
   ```bash
   python Color_correction/compute_mean_RGB_value_from_exr_center_crop.py
   ```

   Finally, compute the RGB-to-XYZ transformation matrix:
   ```bash
   python Color_correction/find_RGB_to_XYZ_matrix.py
   ```

   This produces `Camera_Colorchecker_RGB2XYZ.json`.

5. **Noise Parameter Estimation**
   Noise parameters are estimated in two steps:
   - Bright-field noise: fit the variance–mean relationship from uniform field images in RAW format to estimate quantum efficiency:
     ```bash
     python Camera_calibration/compute_noise_model_bright.py
     ```
   - Dark-field noise: capture dark images (lens covered) at varying gains and fit a quadratic model to estimate read noise and ADC noise:
     ```bash
     python Camera_calibration/compute_noise_model_dark.py
     ```

   Outputs: `Noise_model_bright.json` and `Noise_model_dark.json`.

   Then, fit combined parameters:
   ```bash
   python Camera_calibration/Fit_Noise_Model_together.py
   ```

   Final outputs: `Fit_parameters_bright.json` and `Fit_parameters_dark.json`, containing per-channel parameters (`k`, `sigma_read`, `sigma_adc`).

   **Important:** Insert these parameters into `HDRutils/merge_uncertainty.py` at `imread_merge_arq_no_demosaic_uncertainty` (rows 173–175).

II. Display Capture Phase
1. **Extrinsic Calibration (Homography Transformation)**
   Generate OpenCV ArUco patterns, capture ARQ-format images at a fixed camera position, and aggregate into `.exr` format:
   ```bash
   python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py
   ```

   Output: `homography_aruco_result_exr.json`.

2. **End-to-End RGB Mean and Variance Computation**
   ```bash
   python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion_Homo.py
   ```

3. **RGB-to-XYZ Conversion with Uncertainty (Optional, Computationally Intensive)**
   ```bash
   python Uncertainty_Color_Correction_after_HT.py
   ```

   Outputs: `_mean.exr` and `_cov.npz` (mean and covariance in XYZ color space).

III. Display VDP Perceptual Evaluation Phase
1. **Compute Perceptual Scores with Uncertainty Distribution**
   ```bash
   python run_ColorVideoVDP/run_ColorVideoVDP_uncertainty.py
   ```

2. **Generate Difference Visibility Heatmaps**
   ```bash
   python run_ColorVideoVDP/run_ColorVideoVDP_heatmap.py
   ```
