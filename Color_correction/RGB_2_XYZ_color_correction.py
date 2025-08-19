import numpy as np
import json
from tqdm import tqdm

global Matrix_RGB2XYZ
global Matrix_RGB_expanded_2XYZ

def expand_rgb_features(rgb_array):
    R, G, B = rgb_array[..., 0], rgb_array[..., 1], rgb_array[..., 2]
    RG_sqrt = np.sqrt(R * G)
    RB_sqrt = np.sqrt(R * B)
    GB_sqrt = np.sqrt(G * B)
    return np.stack([R, G, B, RG_sqrt, RB_sqrt, GB_sqrt], axis=-1)

def transform_RGB_2_XYZ_color_correction(RGBs, mode, expand=True):
    global Matrix_RGB2XYZ
    global Matrix_RGB_expanded_2XYZ

    with open(rf'Color_correction/{mode}_Camera_Colorchecker_RGB2XYZ.json') as json_file:  # Please set it by yourself
        data = json.load(json_file)
    Matrix_RGB2XYZ = np.array(data['Matrix_RGB2XYZ'])
    Matrix_RGB_expanded_2XYZ = np.array(data['Matrix_RGB_expanded_2XYZ'])

    if expand:
        RGBs = expand_rgb_features(RGBs)
        XYZs = RGBs @ Matrix_RGB_expanded_2XYZ.T
    else:
        XYZs = RGBs @ Matrix_RGB2XYZ.T
    return XYZs

def transform_RGB_2_XYZ_color_correction_uncertainty(RGBs_mean, RGBs_std, expand=True):
    original_shape = RGBs_mean.shape[:-1]
    RGBs_mean = RGBs_mean.reshape(-1, 3)
    RGBs_std = RGBs_std.reshape(-1, 3)
    R, G, B = RGBs_mean[:, 0], RGBs_mean[:, 1], RGBs_mean[:, 2]
    σ_R, σ_G, σ_B = RGBs_std[:, 0], RGBs_std[:, 1], RGBs_std[:, 2]
    if expand:
        sqrt_RG = np.sqrt(np.clip(R * G, 1e-8, None))
        sqrt_RB = np.sqrt(np.clip(R * B, 1e-8, None))
        sqrt_GB = np.sqrt(np.clip(G * B, 1e-8, None))
        RGBs_mean_exp = np.stack([R, G, B, sqrt_RG, sqrt_RB, sqrt_GB], axis=-1)

        eps = 1e-8
        J = np.zeros((RGBs_mean.shape[0], 6, 3))  # [N, 6, 3]
        J[:, 0, 0] = 1
        J[:, 1, 1] = 1
        J[:, 2, 2] = 1
        J[:, 3, 0] = 0.5 * G / (sqrt_RG + eps)
        J[:, 3, 1] = 0.5 * R / (sqrt_RG + eps)
        J[:, 4, 0] = 0.5 * B / (sqrt_RB + eps)
        J[:, 4, 2] = 0.5 * R / (sqrt_RB + eps)
        J[:, 5, 1] = 0.5 * B / (sqrt_GB + eps)
        J[:, 5, 2] = 0.5 * G / (sqrt_GB + eps)

        Sigma_RGB = np.zeros((RGBs_mean.shape[0], 3, 3))
        Sigma_RGB[:, 0, 0] = σ_R ** 2
        Sigma_RGB[:, 1, 1] = σ_G ** 2
        Sigma_RGB[:, 2, 2] = σ_B ** 2

        Sigma_expanded = np.einsum('nij,njk,nlk->nil', J, Sigma_RGB, J)

        A = Matrix_RGB_expanded_2XYZ
        XYZs = RGBs_mean_exp @ A.T  # [N, 3]

        A_exp = np.broadcast_to(A[None, :, :], (RGBs_mean.shape[0], 3, 6))
        Sigma_XYZ = np.einsum("nij,njk,nlk->nil", A_exp, Sigma_expanded, A_exp)

    else:
        A = Matrix_RGB2XYZ
        XYZs = RGBs_mean @ A.T  # [N, 3]
        Sigma_RGB = np.zeros((RGBs_mean.shape[0], 3, 3))
        Sigma_RGB[:, 0, 0] = σ_R ** 2
        Sigma_RGB[:, 1, 1] = σ_G ** 2
        Sigma_RGB[:, 2, 2] = σ_B ** 2
        A_exp = np.broadcast_to(A[None, :, :], (RGBs_mean.shape[0], 3, 3))
        Sigma_XYZ = np.einsum("nij,njk,nlk->nil", A_exp, Sigma_RGB, A_exp)
    XYZs_std = np.sqrt(np.clip(np.stack([
        Sigma_XYZ[:, 0, 0],
        Sigma_XYZ[:, 1, 1],
        Sigma_XYZ[:, 2, 2],
    ], axis=-1), 0, None))  # [N, 3]

    XYZs_mean = XYZs.reshape(*original_shape, 3)
    XYZs_std = XYZs_std.reshape(*original_shape, 3)
    return XYZs_mean, XYZs_std

def transform_RGB_2_XYZ_color_correction_uncertainty_cov(RGBs_mean, RGBs_std, expand=True):
    original_shape = RGBs_mean.shape[:-1]
    RGBs_mean = RGBs_mean.reshape(-1, 3)
    RGBs_std = RGBs_std.reshape(-1, 3)
    R, G, B = RGBs_mean[:, 0], RGBs_mean[:, 1], RGBs_mean[:, 2]
    σ_R, σ_G, σ_B = RGBs_std[:, 0], RGBs_std[:, 1], RGBs_std[:, 2]
    if expand:
        sqrt_RG = np.sqrt(np.clip(R * G, 1e-8, None))
        sqrt_RB = np.sqrt(np.clip(R * B, 1e-8, None))
        sqrt_GB = np.sqrt(np.clip(G * B, 1e-8, None))
        RGBs_mean_exp = np.stack([R, G, B, sqrt_RG, sqrt_RB, sqrt_GB], axis=-1)

        eps = 1
        J = np.zeros((RGBs_mean.shape[0], 6, 3))  # [N, 6, 3]
        J[:, 0, 0] = 1
        J[:, 1, 1] = 1
        J[:, 2, 2] = 1
        J[:, 3, 0] = 0.5 * G / (sqrt_RG + eps)
        J[:, 3, 1] = 0.5 * R / (sqrt_RG + eps)
        J[:, 4, 0] = 0.5 * B / (sqrt_RB + eps)
        J[:, 4, 2] = 0.5 * R / (sqrt_RB + eps)
        J[:, 5, 1] = 0.5 * B / (sqrt_GB + eps)
        J[:, 5, 2] = 0.5 * G / (sqrt_GB + eps)

        Sigma_RGB = np.zeros((RGBs_mean.shape[0], 3, 3))
        Sigma_RGB[:, 0, 0] = σ_R ** 2
        Sigma_RGB[:, 1, 1] = σ_G ** 2
        Sigma_RGB[:, 2, 2] = σ_B ** 2

        A = Matrix_RGB_expanded_2XYZ
        XYZs = RGBs_mean_exp @ A.T  # [N, 3]

        A_exp = np.broadcast_to(A[None, :, :], (RGBs_mean.shape[0], 3, 6))
        AJ = np.einsum("nij,njk->nik", A_exp, J)
        Sigma_XYZ = np.einsum("nij,njk,nlk->nil", AJ, Sigma_RGB, AJ)

    else:
        A = Matrix_RGB2XYZ
        XYZs = RGBs_mean @ A.T  # [N, 3]
        Sigma_RGB = np.zeros((RGBs_mean.shape[0], 3, 3))
        Sigma_RGB[:, 0, 0] = σ_R ** 2
        Sigma_RGB[:, 1, 1] = σ_G ** 2
        Sigma_RGB[:, 2, 2] = σ_B ** 2
        A_exp = np.broadcast_to(A[None, :, :], (RGBs_mean.shape[0], 3, 3))
        Sigma_XYZ = np.einsum("nij,njk,nlk->nil", A_exp, Sigma_RGB, A_exp)

    XYZs_mean = XYZs.reshape(*original_shape, 3)
    XYZs_cov = Sigma_XYZ.reshape(*original_shape, 3, 3)
    return XYZs_mean, XYZs_cov

def transform_RGB_2_XYZ_color_correction_uncertainty_chunk(RGBs_mean, RGBs_std, expand=True, chunk_size=500):
    H, W = RGBs_mean.shape[:2]
    XYZs_mean = np.zeros((H, W, 3), dtype=np.float32)
    XYZs_std = np.zeros((H, W, 3), dtype=np.float32)

    for i in tqdm(range(0, H, chunk_size)):
        i_end = min(i + chunk_size, H)
        RGB_chunk = RGBs_mean[i:i_end].reshape(-1, 3)
        STD_chunk = RGBs_std[i:i_end].reshape(-1, 3)
        XYZ_mean_chunk, XYZ_std_chunk = transform_RGB_2_XYZ_color_correction_uncertainty(RGB_chunk, STD_chunk, expand=expand)
        XYZs_mean[i:i_end] = XYZ_mean_chunk.reshape(i_end - i, W, 3)
        XYZs_std[i:i_end] = XYZ_std_chunk.reshape(i_end - i, W, 3)

    return XYZs_mean, XYZs_std

def transform_RGB_2_XYZ_color_correction_uncertainty_cov_chunk(RGBs_mean, RGBs_std, mode, expand=True, chunk_size=500):
    H, W = RGBs_mean.shape[:2]
    XYZs_mean = np.zeros((H, W, 3), dtype=np.float32)
    XYZs_cov = np.zeros((H, W, 3, 3), dtype=np.float32)

    global Matrix_RGB2XYZ
    global Matrix_RGB_expanded_2XYZ
    with open(rf'Color_correction/{mode}_Camera_Colorchecker_RGB2XYZ.json') as json_file:  # Please set it by yourself
        data = json.load(json_file)
    Matrix_RGB2XYZ = np.array(data['Matrix_RGB2XYZ'])
    Matrix_RGB_expanded_2XYZ = np.array(data['Matrix_RGB_expanded_2XYZ'])

    for i in tqdm(range(0, H, chunk_size)):
        i_end = min(i + chunk_size, H)
        RGB_chunk = RGBs_mean[i:i_end].reshape(-1, 3)
        STD_chunk = RGBs_std[i:i_end].reshape(-1, 3)
        XYZ_mean_chunk, XYZ_std_chunk = transform_RGB_2_XYZ_color_correction_uncertainty_cov(RGB_chunk, STD_chunk, expand=expand)
        XYZs_mean[i:i_end] = XYZ_mean_chunk.reshape(i_end - i, W, 3)
        XYZs_cov[i:i_end] = XYZ_std_chunk.reshape(i_end - i, W, 3, 3)\

    return XYZs_mean, XYZs_cov

if __name__ == '__main__':
    ## Monte Carlo validation accuracy
    symbol = 'Sony_a7R3_FE35_F20_Eizo'
    expand = True
    N_1 = 300
    N_2 = 500
    rgb_mean = np.array([300.0, 500.0, 300.0])  # Mean of R, G, B
    rgb_std = np.array([20.0, 30.0, 10.0])  # Standard deviation of R, G, B
    RGBs = np.random.normal(loc=rgb_mean, scale=rgb_std, size=(N_1, N_2, 3))  # shape: (10000, 3)
    XYZs_mean, XYZs_cov = transform_RGB_2_XYZ_color_correction_uncertainty_cov_chunk(RGBs_mean=rgb_mean[None,None,...], RGBs_std=rgb_std[None,None,...], mode=symbol, expand=expand)
    print('Mine:', XYZs_mean, XYZs_cov)
    XYZs_real = transform_RGB_2_XYZ_color_correction(RGBs=RGBs, expand=expand, mode=symbol)
    XYZs_real_flat = XYZs_real.reshape(-1,3)
    XYZs_real_mean = XYZs_real_flat.mean(axis=0)
    XYZs_real_std = XYZs_real_flat.std(axis=0)
    XYZs_real_cov = np.cov(XYZs_real_flat, rowvar=False)
    print('Real:', XYZs_real_mean, XYZs_real_cov)

