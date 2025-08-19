# 如何从头开始使用CameraVDP? 警告：你可能需要大量的空间（在实验中我们使用了超过100GB的空间存储RAW图像）
I. 相机校准阶段
1. 校准相机的内参和畸变参数(请注意，这个是相机-镜头-焦距独立的，不同组合需要不同校准)，注意对于不同镜头,参数需要详细调整使得在固定距离下checkerboard大致占据镜头70%面积：
python Camera_calibration/plot_checker_board.py --square_size 64 --rows 32 --cols 58
然后使用相机拍摄ARW格式图片，在大约固定的距离处，不同的角度至少9张以上，始终覆盖住checkerboard的所有内容（注意checkerboard覆盖面积相对越大<=90%，校准越准确）。
然后对于一般情况（相机图像中没有明显的显示器子像素）估计相机的内参和畸变参数(注意里面的square_size和checkerboard一样，checkerboard_size_row和checkerboard_size_col应该比checkerboard的小1)：
python calibrate_checker_board.py #(不要直接运行，请先设定参数、图像路径、储存路径)
如果你使用微距的摄像机，则你会在图像中发现可以直接拍到显示器的子像素，此时，请使用下面的代码：
python calibrate_checker_board_refine.py #(不要直接运行，请先设定参数、图像路径、储存路径)

上述操作结束后，你应该获得一个calibration.json/calibration_refine.json文件，此文件包含了四个键（"ret","mtx","dist","re_projection_error"）, 其中"mtx"为相机内参, "dist"为畸变参数

2. 校准相机的MTF(SFR)曲线(请注意，这个是相机-镜头-焦距独立的，不同组合需要不同校准)
在测量前先准备一个黑/白斜边（我们实验中使用的是西门子星图中心部分），要求黑色部分足够黑，白色部分足够白（因此打印的斜边并不行，要达到光学材质级别）
先使用相机在步骤1中的距离处拍摄ARQ格式图片（注意，是Pixel Shift Multi Shooting的模式的融合结果, SONY官方App提供）聚合为.exr格式的文件：
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR.py # 请自己指定路径以及不同曝光度的图片（建议2个曝光度以上）
使用上一步生成的_mean.exr进行MTF计算：
python MTF/compute_MTF.py your_mean.exr #请修改exposure参数避免使得显示的过曝或过暗

上述操作结束后，你应该获得一个mtf.json文件，包含四个键（"R","G","B","Y"）, 其代表红、绿、蓝、亮度的MTF曲线参数

3. 校准相机的减晕（Vignetting）图(请注意，这个是相机-镜头-焦距独立的，不同组合需要不同校准)
在测量前要准备一个几乎完全各处发光强度相同的刺激（理想情况下，使用积分球，但是也可以使用一个平场校准的显示器做替代）:
python Vignetting/plot_full_screen.py
使用相机拍摄ARQ格式图片（如果使用显示器，请务必不要对焦到显示器平面，方法是使用你在步骤1/2中使用的焦距，然后将照相机大幅度搬近显示器，但不再重新对焦）聚合为.exr格式的文件：
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener.py # 请自己指定路径以及不同曝光度的图片（建议3个曝光度以上）
建议的设置中，在显示器的不同地方拍摄四个白色exr文件（即总共4位置*3曝光度=12个ARQ文件），最终的Vignetting使用其平均生成：
python Vignetting/generate_Vignetting_map.py
不建议使用blur设置（即设置blur_ksize=1），因为我们希望同时考虑相机系统内部问题

上述操作结束后，你应该获得一个vignetting_map.npz文件

4. 校准相机的颜色矩阵（请注意，这个是相机-镜头-显示器独立的，不同组合需要不同校准）
python Color_correction/plot_color_checker_30colors_each.py --rect_width 2000 --rect_height 2000
应使用相机和光度计（可以提供XYZ三色值）同时对上面代码中每一个刺激进行测量，此处我们提供使用specbos测量的python代码：
python Color_correction/specbos_measure.py --output measure_specbos.json #路径请自己指定
使用相机拍摄ARQ格式图片（使用你在步骤1/2/3中使用的焦距，然后将照相机大幅度搬近显示器，但不再重新对焦）聚合为.exr格式的文件：
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py # 请自己指定路径以及不同曝光度的图片（建议3个曝光度以上）
计算相机拍摄图片对应的中心RGB均值：
python Color_correction/compute_mean_RGB_value_from_exr_center_crop.py
最后一步，计算颜色转换矩阵：
python Color_correction/find_RGB_to_XYZ_matrix.py

上述操作结束后，你应该获得一个Camera_Colorchecker_RGB2XYZ.json文件

5. 摄像机噪声参数估计
噪声参数分为两个步骤：
（1）拍摄不同亮度的均匀场图像，并拟合RAW图像方差与均值间的线性关系，以估计量子效率:
python Camera_calibration/compute_noise_model_bright.py
（2）在遮挡相机镜头的情况下（暗场条件），改变增益，拟合方差与增益间的二次函数，以估计read噪声和ADC噪声:
python Camera_calibration/compute_noise_model_dark.py
上述操作结束后，你应该获得一个Noise_model_bright.json和一个Noise_model_dark.json

然后拟合参数:
python Camera_calibration/Fit_Noise_Model_together.py
最终你应该获得一个Fit_parameters_bright.json和一个Fit_parameters_dark.json, 内部应该包含RGB通道分别对应的k, sigma_read和sigma_adc

！！！请将这些参数填写到HDRutils/merge_uncertainty.py的imread_merge_arq_no_demosaic_uncertainty（row 173-175）

II. 显示器拍摄阶段
1. 首先要估计相机外参(以进行单应变换)
首先生成OpenCV ArUco Pattern们：
使用相机拍摄ARQ格式图片聚合为.exr格式的文件（注意此处相机位置被定死，最终你拍摄的所有东西都必须在这个位置）：
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion.py # 请自己指定路径以及不同曝光度的图片（建议3个曝光度以上）

上述操作结束后，你应该获得一个homography_aruco_result_exr.json文件

2. 然后直接端到端的计算拍摄的RGB_mean.exr和RGB_var.exr:
python Gather_ARQ_to_Exr_Uncertainty/Uncertainty_gather_raw_arq_to_exr_whole_process_all_HDR_MTF_wiener_Vignetting_Undistortion_Homo.py # 请自己指定路径以及不同曝光度的图片（建议3个曝光度以上）

3. 颜色转换RGB->XYZ（如果考虑不确定度，很费时间，因此独立出来）
python Uncertainty_Color_Correction_after_HT.py

上述操作结束后，你应该获得一个_mean.exr和一个_cov.npz的文件，这就是在XYZ颜色空间的均值和协方差矩阵

III. 显示器VDP感知评估阶段
# 输出 带不确定度分布 的分数 (请务必完成I中的步骤5噪声参数估计和填写)：
python run_ColorVideoVDP/run_ColorVideoVDP_uncertainty.py #请不要直接运行，请设定好观察距离、Test和Reference等等

# 如果你想输出 差异可见性 热图：
python run_ColorVideoVDP/run_ColorVideoVDP_heatmap.py #请不要直接运行，请设定好观察距离、Test和Reference等等

