import pyexr
import numpy as np
import os
import cv2
import torch
import utils
import math
import json
from lpyr_dec import weber_contrast_pyr, lpyr_dec_2
from csf import castleCSF
from torchvision.transforms import GaussianBlur
from torch.functional import Tensor
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelParameters:
    CSF_castleCSF: Any
    csf_sigma: torch.Tensor
    sensitivity_correction: torch.Tensor
    mask_p: torch.Tensor
    mask_c: torch.Tensor
    mask_q: torch.Tensor
    pu_dilate: float
    pu_blur: Any = None
    pu_padsize: int = 0
    xcm_weights: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    beta: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    image_int: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    ch_chrom_w: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    baseband_weight: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    beta_tch: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    beta_sch: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    jod_a: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    jod_exp: torch.Tensor = field(default_factory=lambda: torch.tensor([]))
    d_max: torch.Tensor = field(default_factory=lambda: torch.tensor([]))

def load_exr(filename):
    exr = pyexr.open(filename)
    data = exr.get()
    data[data < 0] = 0
    return data

def generate_scaled_coords(central_crop_width, central_crop_height, k_scale):
    x_coords = np.arange(central_crop_width)
    y_coords = np.arange(central_crop_height)
    scaled_x_coords = (x_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    scaled_y_coords = (y_coords * (k_scale - 1) + (k_scale - 1) / 2).astype(int)
    return scaled_x_coords, scaled_y_coords

def create_mask_from_exr(data, scaled_x, scaled_y, threshold):
    grid_x, grid_y = np.meshgrid(scaled_x, scaled_y, indexing='xy')
    values = np.mean(data[grid_y, grid_x], axis=-1)
    mask = (values > threshold).astype(np.uint8)
    return mask

def reset_CVVDP_parameters(param_path = r'cvvdp_parameters.json'):
    ## Load Parameters
    with open(param_path, 'r') as fp:
        parameters = json.load(fp)
    CSF_castleCSF = castleCSF(csf_version=parameters['csf'], device=device)
    pu_dilate = parameters['pu_dilate']
    pu_blur = GaussianBlur(int(pu_dilate * 4) + 1, pu_dilate)
    pu_padsize = int(pu_dilate * 2)

    MP_set = ModelParameters(
        CSF_castleCSF=CSF_castleCSF,
        csf_sigma=torch.as_tensor(parameters['csf_sigma'], device=device),
        sensitivity_correction=torch.as_tensor(parameters['sensitivity_correction'], device=device),
        mask_p=torch.as_tensor(parameters['mask_p'], device=device),
        mask_c=torch.as_tensor(parameters['mask_c'], device=device),
        mask_q=torch.as_tensor(parameters['mask_q'], device=device),
        pu_dilate=pu_dilate,
        pu_blur=pu_blur,
        pu_padsize=pu_padsize,
        xcm_weights=torch.as_tensor(parameters['xcm_weights'], device=device, dtype=torch.float32),
        beta=torch.as_tensor(parameters['beta'], device=device),
        image_int=torch.as_tensor(parameters['image_int'], device=device),
        ch_chrom_w=torch.as_tensor(parameters['ch_chrom_w'], device=device),
        baseband_weight=torch.as_tensor(parameters['baseband_weight'], device=device),
        beta_tch=torch.as_tensor(parameters['beta_tch'], device=device),
        beta_sch=torch.as_tensor(parameters['beta_sch'], device=device),
        jod_a=torch.as_tensor(parameters['jod_a'], device=device),
        jod_exp=torch.as_tensor(parameters['jod_exp'], device=device),
        d_max=torch.as_tensor(parameters['d_max'], device=device)
    )
    return MP_set


if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

LMS2006_to_DKLd65 = (
        (1.000000000000000, 1.000000000000000, 0),
        (1.000000000000000, -2.311130179947035, 0),
        (-1.000000000000000, -1.000000000000000, 50.977571328718781))
XYZ_to_LMS2006 = (
        (0.187596268556126, 0.585168649077728, -0.026384263306304),
        (-0.133397430663221, 0.405505777260049, 0.034502127690364),
        (0.000244379021663, -0.000542995890619, 0.019406849066323))

# Load Images
root_path = r'' # Please set it by yourself
save_path = r'' # Please set it by yourself
os.makedirs(save_path, exist_ok=True)
test_name_list = [''] # Please set it by yourself
k_scale = 3

Predict = True
MonteCarlo = True # Here the MonteCarlo will be applied both to the CVVDP parameters and also the Noise

##Display Setting
edge_crop_w = 2000 #2000
edge_crop_h = 1000 #1000
resolution = [7681-2*edge_crop_w,4321-2*edge_crop_h]
diagonal_size_inches = 27 / (7681**2+4321**2)**0.5 * ((7681-2*edge_crop_w)**2+(4321-2*edge_crop_h)**2)**0.5
heatmap_type = 'raw'
ar = resolution[0]/resolution[1]
height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
display_size_m = (ar*height_mm/1000, height_mm/1000)

viewing_distance_meters_list = [1.0000, 1.1429, 1.3333, 1.6000, 2.0000]

MP_set = reset_CVVDP_parameters(param_path = r'cvvdp_parameters.json')


def phase_uncertainty(M):
    if MP_set.pu_dilate != 0 and M.shape[-2] > MP_set.pu_padsize and M.shape[-1] > MP_set.pu_padsize:
        M_pu = MP_set.pu_blur.forward(M) * (10 ** MP_set.mask_c)
    else:
        M_pu = M * (10 ** MP_set.mask_c)
    return M_pu

def safe_pow( x:Tensor, p ):
    epsilon = torch.as_tensor(0.00001, device=x.device)
    return (x + epsilon) ** p - epsilon ** p

def mask_pool(C):
    # Cross-channel masking
    num_ch = C.shape[0]
    M = torch.empty_like(C)
    xcm_weights_re = torch.reshape((2 ** MP_set.xcm_weights), (4, 4, 1, 1, 1))[:num_ch, ...]
    for cc in range(num_ch):  # for each channel: Sust, RG, VY, Trans
        M[cc, ...] = torch.sum(C * xcm_weights_re[:, cc], dim=0, keepdim=True)
    return M

def clamp_diffs(D):
    max_v = 10 ** MP_set.d_max
    Dc = max_v * D / (max_v + D)
    return Dc

def apply_masking_model(T, R, S):
    num_ch = T.shape[0]
    ch_gain = torch.reshape(torch.as_tensor([1, 1.45, 1, 1.], device=T.device), (4, 1, 1, 1))[:num_ch, ...]
    T_p = T * S * ch_gain
    R_p = R * S * ch_gain
    M_mm = phase_uncertainty(torch.min(torch.abs(T_p), torch.abs(R_p)))
    p = MP_set.mask_p
    q = MP_set.mask_q[0:num_ch].view(num_ch, 1, 1, 1)
    M = mask_pool(safe_pow(torch.abs(M_mm), q))
    D_u = safe_pow(torch.abs(T_p - R_p), p) / (1 + M)
    D = clamp_diffs(D_u)
    return D

def lp_norm(x, p, dim=0, normalize=True, keepdim=True):
    if dim is None:
        dim = 0

    if normalize:
        if isinstance(dim, tuple):
            N = 1.0
            for dd in dim:
                N *= x.shape[dd]
        else:
            N = x.shape[dim]
    else:
        N = 1.0

    if isinstance(p, torch.Tensor):
        # p is a Tensor if it is being optimized. In that case, we need the formula for the norm
        return safe_pow(torch.sum(safe_pow(x, p), dim=dim, keepdim=keepdim) / float(N), 1 / p)
    else:
        return torch.norm(x, p, dim=dim, keepdim=keepdim) / (float(N) ** (1. / p))


def get_ch_weights(no_channels):
    per_ch_w_all = torch.stack(
        [torch.as_tensor(1., device=MP_set.ch_chrom_w.device), MP_set.ch_chrom_w, MP_set.ch_chrom_w])
    # Weights for the channels: sustained, RG, YV, [transient]
    per_ch_w = per_ch_w_all[0:no_channels].view(-1, 1, 1)
    return per_ch_w

def met2jod(Q):
    # We could use
    # Q_JOD = 10. - self.jod_a * Q**self.jod_exp
    # but it does not differentiate well near Q=0
    Q_t = 0.1
    jod_a_p = MP_set.jod_a * (Q_t ** (MP_set.jod_exp - 1.))
    Q_JOD = torch.empty_like(Q)
    Q_JOD[Q <= Q_t] = 10. - jod_a_p * Q[Q <= Q_t]
    Q_JOD[Q > Q_t] = 10. - MP_set.jod_a * (Q[Q > Q_t] ** MP_set.jod_exp)
    return Q_JOD

def do_pooling_and_jods(Q_per_ch, base_rho_band):
    no_channels = Q_per_ch.shape[0] #3
    no_frames = Q_per_ch.shape[1] #1
    no_bands = Q_per_ch.shape[2] #10
    per_ch_w = get_ch_weights(no_channels)
    # Weights for the spatial bands
    per_sband_w = torch.ones((no_channels, 1, no_bands), dtype=torch.float32, device=device)
    per_sband_w[:, 0, -1] = MP_set.baseband_weight[0:no_channels]

    Q_sc = lp_norm(Q_per_ch * per_ch_w * per_sband_w, MP_set.beta_sch, dim=2, normalize=False)  # Sum across spatial channels
    t_int = MP_set.image_int
    Q_tc = lp_norm(Q_sc, MP_set.beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels
    Q = Q_tc * t_int
    Q = Q.squeeze()
    Q_JOD = met2jod(Q)
    return Q_JOD

### Start reading image data
for test_name in test_name_list:
    test_XYZ_linear_float64 = pyexr.open(os.path.join(root_path, f'fringing_test_{test_name}_XYZ_linear_k{k_scale}_mean.exr')).get()[edge_crop_h:-edge_crop_h,edge_crop_w:-edge_crop_w,:]
    reference_XYZ_linear_float64 = pyexr.open(os.path.join(root_path, f'fringing_reference_{test_name}_XYZ_linear_k{k_scale}_mean.exr')).get()[edge_crop_h:-edge_crop_h,edge_crop_w:-edge_crop_w,:]
    test_XYZ_mean_tensor = torch.tensor(test_XYZ_linear_float64, device=device)
    test_right_mask = torch.tensor((test_XYZ_linear_float64.mean(axis=-1) > 5).astype(np.uint8), device=device)[...,None]
    reference_XYZ_mean_tensor = torch.tensor(reference_XYZ_linear_float64, device=device)
    reference_right_mask = torch.tensor((reference_XYZ_linear_float64.mean(axis=-1) > 5).astype(np.uint8), device=device)[...,None]
    test_XYZ_cov = np.load(os.path.join(root_path, f'fringing_test_{test_name}_XYZ_linear_k{k_scale}_cov.npz'))['cov_matrix'][edge_crop_h:-edge_crop_h,edge_crop_w:-edge_crop_w,:]
    reference_XYZ_cov = np.load(os.path.join(root_path, f'fringing_reference_{test_name}_XYZ_linear_k{k_scale}_cov.npz'))['cov_matrix'][edge_crop_h:-edge_crop_h,edge_crop_w:-edge_crop_w,:]
    test_XYZ_cov_tensor = torch.tensor(test_XYZ_cov, device=device)
    reference_XYZ_cov_tensor = torch.tensor(reference_XYZ_cov, device=device)
    test_XYZ_mvn = torch.distributions.MultivariateNormal(loc=test_XYZ_mean_tensor,
                                                          covariance_matrix=test_XYZ_cov_tensor)
    reference_XYZ_mvn = torch.distributions.MultivariateNormal(loc=reference_XYZ_mean_tensor,
                                                               covariance_matrix=reference_XYZ_cov_tensor)

    threshold = 1
    test_XYZ_mean_tensor = test_XYZ_mean_tensor * test_right_mask
    reference_XYZ_mean_tensor = reference_XYZ_mean_tensor * reference_right_mask
    test_XYZ_mean_tensor[test_XYZ_mean_tensor < threshold] = threshold
    reference_XYZ_mean_tensor[reference_XYZ_mean_tensor < threshold] = threshold
    H, W, C = test_XYZ_mean_tensor.shape

    # Color Space change: XYZ -> DKL
    xyz2dkl = torch.as_tensor(LMS2006_to_DKLd65, dtype=test_XYZ_mean_tensor.dtype,
                              device=test_XYZ_mean_tensor.device) @ torch.as_tensor(XYZ_to_LMS2006,
                                                                                    dtype=test_XYZ_mean_tensor.dtype,
                                                                                    device=test_XYZ_mean_tensor.device)

    test_XYZ_mean_tensor = utils.reshuffle_dims(test_XYZ_mean_tensor, in_dims='HWC', out_dims="BCFHW")
    reference_XYZ_mean_tensor = utils.reshuffle_dims(reference_XYZ_mean_tensor, in_dims='HWC', out_dims="BCFHW")
    width, height = resolution[0], resolution[1]

    temp_ch = 1
    all_ch = 2 + temp_ch

    dmap_channels = 1 if heatmap_type == "raw" else 3

    R = torch.empty((1, 6, 1, height, width), device=device)
    test_DKL_tensor = torch.einsum('bcthw,ac->bathw', test_XYZ_mean_tensor, xyz2dkl)
    reference_DKL_tensor = torch.einsum('bcthw,ac->bathw', reference_XYZ_mean_tensor, xyz2dkl)

    JOD_Predict_distance_list = []
    JOD_MC_mean_distance_list = []
    JOD_MC_std_distance_list = []
    JOD_samples_all_list = []

    for viewing_index in tqdm(range(len(viewing_distance_meters_list))):
        viewing_distance_meters = viewing_distance_meters_list[viewing_index]
        pix_deg = 2 * math.degrees(math.atan(0.5 * display_size_m[0] / resolution[0] / viewing_distance_meters))
        display_ppd = 1 / pix_deg

        ## Predict
        if Predict:
            lpyr = weber_contrast_pyr(width, height, display_ppd, device, contrast='weber_g1')
            heatmap_pyr = lpyr_dec_2(width, height, display_ppd, device)
            R = torch.empty((1, 6, 1, height, width), device=device)
            R[:, 0::2, :, :, :] = test_DKL_tensor
            R[:, 1::2, :, :, :] = reference_DKL_tensor
            # R.shape = [1,6,1,4321,7681]
            ## Decompose
            B_bands, L_bkg_pyr = lpyr.decompose(R[0, ...])
            rho_band = lpyr.get_freqs()
            rho_band[lpyr.get_band_count() - 1] = 0.1
            Q_per_ch_block = None
            block_N_frames = R.shape[-3]

            for bb in range(lpyr.get_band_count()):  # For each spatial frequency band
                is_baseband = (bb == (lpyr.get_band_count() - 1))
                B_bb = lpyr.get_band(B_bands, bb)
                T_f = B_bb[0::2, ...]  # Test
                R_f = B_bb[1::2, ...]  # Reference
                logL_bkg = lpyr.get_gband(L_bkg_pyr, bb)

                # Compute CSF
                rho = rho_band[bb]  # Spatial frequency in cpd
                ch_height, ch_width = logL_bkg.shape[-2], logL_bkg.shape[-1]
                S = torch.empty((all_ch, block_N_frames, ch_height, ch_width), device=device)

                for cc in range(all_ch):
                    tch = 0 if cc < 3 else 1  # Sustained or transient
                    cch = cc if cc < 3 else 0  # Y, rg, yv
                    # The sensitivity is always extracted for the reference frame
                    S[cc, :, :, :] = MP_set.CSF_castleCSF.sensitivity(rho, 0, logL_bkg[..., 1, :, :, :], cch, MP_set.csf_sigma) * 10.0 ** (MP_set.sensitivity_correction / 20.0)  # 第一个的平均值应该为2.5645

                if is_baseband:
                    D = (torch.abs(T_f - R_f) * S)
                else:
                    # dimensions: [channel,frame,height,width]
                    D = apply_masking_model(T_f, R_f, S)

                if Q_per_ch_block is None:
                    Q_per_ch_block = torch.empty((all_ch, block_N_frames, lpyr.get_band_count()), device=device)
                Q_per_ch_block[:, :, bb] = lp_norm(D, MP_set.beta, dim=(-2, -1), normalize=True,
                                                   keepdim=False)  # Pool across all pixels (spatial pooling)

                # heatmap
                t_int = MP_set.image_int
                per_ch_w = get_ch_weights(all_ch).view(-1, 1, 1, 1) * t_int
                if is_baseband:
                    per_ch_w *= MP_set.baseband_weight[0:all_ch].view(-1, 1, 1, 1)

                D_chr = lp_norm(D * per_ch_w, MP_set.beta_tch, dim=-4,
                                normalize=False)  # Sum across temporal and chromatic channels
                heatmap_pyr.set_lband(bb, D_chr)

            heatmap_block = 1. - (met2jod(heatmap_pyr.reconstruct()) / 10.)
            Q_per_ch = Q_per_ch_block
            heatmap = heatmap_block.detach().type(torch.float16).cpu()
            rho_band = lpyr.get_freqs()
            Q_jod = do_pooling_and_jods(Q_per_ch, rho_band[-1])  # 8.8581

            JOD_predict = float(Q_jod.detach().cpu())
            heatmap = np.array(heatmap)[0, 0, ...]
            heatmap_uint8 = (heatmap.clip(0, 1) * 255).astype(np.uint8)
            print('JOD Predict:', JOD_predict)  # 8.6702
            JOD_Predict_distance_list.append(JOD_predict)
            cv2.imwrite(os.path.join(save_path, f'Predict_VDP_heatmap_{test_name}_{heatmap_type}.png'), heatmap_uint8)

        if MonteCarlo:
            JOD_samples = []
            std_map_mean = None
            std_map_M2 = None
            MC_cvvdp_param_range = range(-1,21,1)
            num_samples = 100 # In most cases, 10 samples are sufficient.
            for MC_index in tqdm(MC_cvvdp_param_range):
                if MC_index >= 0:
                    MC_param_path = rf'CVVDP_MC_parameters/cvvdp_parameters_xrdavid_split{MC_index:02d}.json'
                    MP_set = reset_CVVDP_parameters(MC_param_path)
                else:
                    MP_set = reset_CVVDP_parameters(param_path = r'cvvdp_parameters.json')
                for i in range(num_samples):
                    lpyr = weber_contrast_pyr(width, height, display_ppd, device, contrast='weber_g1')
                    R = torch.empty((1, 6, 1, height, width), device=device)
                    test_XYZ_sampled = test_XYZ_mvn.sample()
                    test_XYZ_sampled = test_XYZ_sampled * test_right_mask
                    test_XYZ_sampled[test_XYZ_sampled < threshold] = threshold
                    reference_XYZ_sampled = reference_XYZ_mvn.sample()
                    reference_XYZ_sampled = reference_XYZ_sampled * reference_right_mask
                    reference_XYZ_sampled[reference_XYZ_sampled < threshold] = threshold
                    test_XYZ_sampled = utils.reshuffle_dims(test_XYZ_sampled, in_dims='HWC', out_dims="BCFHW")
                    reference_XYZ_sampled = utils.reshuffle_dims(reference_XYZ_sampled, in_dims='HWC', out_dims="BCFHW")
                    test_DKL_tensor_sampled = torch.einsum('bcthw,ac->bathw', test_XYZ_sampled, xyz2dkl)
                    reference_DKL_tensor_sampled = torch.einsum('bcthw,ac->bathw', reference_XYZ_sampled, xyz2dkl)
                    # --- Step 2: Rerun the quality assessment process ---
                    R[:, 0::2, :, :, :] = test_DKL_tensor_sampled
                    R[:, 1::2, :, :, :] = reference_DKL_tensor_sampled
                    B_bands, L_bkg_pyr = lpyr.decompose(R[0, ...])
                    rho_band = lpyr.get_freqs()
                    rho_band[lpyr.get_band_count() - 1] = 0.1
                    Q_per_ch_block = None
                    block_N_frames = R.shape[-3]

                    for bb in range(lpyr.get_band_count()):
                        is_baseband = (bb == (lpyr.get_band_count() - 1))
                        B_bb = lpyr.get_band(B_bands, bb)
                        T_f = B_bb[0::2, ...]
                        R_f = B_bb[1::2, ...]
                        logL_bkg = lpyr.get_gband(L_bkg_pyr, bb)
                        S = torch.empty((all_ch, block_N_frames, *logL_bkg.shape[-2:]), device=device)
                        for cc in range(all_ch):
                            tch = 0 if cc < 3 else 1
                            cch = cc if cc < 3 else 0
                            S[cc, :, :, :] = MP_set.CSF_castleCSF.sensitivity(rho_band[bb], 0,
                                                                              logL_bkg[..., 1, :, :, :], cch,
                                                                              MP_set.csf_sigma) * \
                                             10.0 ** (MP_set.sensitivity_correction / 20.0)

                        if is_baseband:
                            D = (torch.abs(T_f - R_f) * S)
                        else:
                            D = apply_masking_model(T_f, R_f, S)

                        if Q_per_ch_block is None:
                            Q_per_ch_block = torch.empty((all_ch, block_N_frames, lpyr.get_band_count()), device=device)
                        Q_per_ch_block[:, :, bb] = lp_norm(D, MP_set.beta, dim=(-2, -1), normalize=True, keepdim=False)
                    JOD = do_pooling_and_jods(Q_per_ch_block, rho_band[-1])
                    print('Sampled Qulaity Score:', JOD)
                    # --- Step 3: Store samples ---
                    JOD_samples.append(JOD.item())

            # --- Step 4: Calculate covariance ---
            JOD_samples = np.array(JOD_samples)
            JOD_mean = np.mean(JOD_samples)
            JOD_std = np.std(JOD_samples, ddof=1)
            JOD_MC_mean_distance_list.append(float(JOD_mean))
            JOD_MC_std_distance_list.append(float(JOD_std))
            JOD_samples_all_list.append(JOD_samples.tolist())

    if MonteCarlo:
        json_dict = {'viewing_distance_meters_list': viewing_distance_meters_list,
                     'JOD_Predict_distance_list': JOD_Predict_distance_list,
                     'JOD_MC_mean_distance_list': JOD_MC_mean_distance_list,
                     'JOD_MC_std_distance_list': JOD_MC_std_distance_list,
                     'JOD_samples_all_list': JOD_samples_all_list}
    elif Predict:
        json_dict = {'viewing_distance_meters_list': viewing_distance_meters_list,
                     'JOD_Predict_distance_list': JOD_Predict_distance_list,
                     'JOD_samples_all_list': JOD_samples_all_list}

    with open(os.path.join(root_path, f'MC_result.json'), 'w') as f:
        json.dump(json_dict, f)

