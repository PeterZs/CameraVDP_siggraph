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

def load_exr(filename):
    exr = pyexr.open(filename)
    data = exr.get()
    data[data < 0] = 0
    return data
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# Load Images
root_path = r''
test_file = ''
test_exr_XYZ = load_exr(os.path.join(root_path, test_file))
threshold = 1 #设置最小值为1
test_exr_XYZ[test_exr_XYZ < threshold] = threshold
H,W,C = test_exr_XYZ.shape
test_file_pure_name = test_file.split('.')[0]
mask_map = f'mask_{test_file_pure_name}.png'
mask_map_data = cv2.imread(os.path.join(root_path, mask_map), cv2.IMREAD_GRAYSCALE)
mask_map_data[mask_map_data == 255] = 1
mask_map_data = cv2.resize(mask_map_data, (W, H), interpolation=cv2.INTER_LINEAR)
masked_XYZ_values = test_exr_XYZ[mask_map_data == 1].astype(np.float64)
average_XYZ = masked_XYZ_values.mean(axis=0).astype(np.float32)
reference_exr_XYZ = np.stack([mask_map_data] * 3, axis=-1) * average_XYZ

test_exr_XYZ_tensor = torch.tensor(test_exr_XYZ, device=device)
reference_exr_XYZ_tensor = torch.tensor(reference_exr_XYZ, device=device)

##Display Setting
resolution = [7681,4321]
viewing_distance_meters = 1
diagonal_size_inches = 55
heatmap_type = 'raw'
ar = resolution[0]/resolution[1]
height_mm = math.sqrt( (diagonal_size_inches*25.4)**2 / (1+ar**2) )
display_size_m = (ar*height_mm/1000, height_mm/1000)
pix_deg = 2*math.degrees(math.atan( 0.5*display_size_m[0]/resolution[0]/viewing_distance_meters))
display_ppd = 1/pix_deg

## Load Parameters
with open('cvvdp_parameters.json', 'r') as fp:
    parameters = json.load(fp)
CSF_castleCSF = castleCSF(csf_version=parameters['csf'], device=device)
csf_sigma = torch.as_tensor( parameters['csf_sigma'], device=device)
sensitivity_correction = torch.as_tensor(parameters['sensitivity_correction'], device=device)
mask_p = torch.as_tensor(parameters['mask_p'], device=device)
mask_c = torch.as_tensor(parameters['mask_c'], device=device)
mask_q = torch.as_tensor(parameters['mask_q'], device=device)
pu_dilate = parameters['pu_dilate']
if pu_dilate>0:
    pu_blur = GaussianBlur(int(pu_dilate*4)+1, pu_dilate)
    pu_padsize = int(pu_dilate*2)
xcm_weights = torch.as_tensor(parameters['xcm_weights'], device=device, dtype=torch.float32)
beta = torch.as_tensor(parameters['beta'], device=device)
image_int = torch.as_tensor(parameters['image_int'], device=device)
ch_chrom_w = torch.as_tensor(parameters['ch_chrom_w'], device=device)
baseband_weight = torch.as_tensor(parameters['baseband_weight'], device=device)
beta_tch = torch.as_tensor(parameters['beta_tch'], device=device)
beta_sch = torch.as_tensor(parameters['beta_sch'], device=device)
jod_a = torch.as_tensor( parameters['jod_a'], device=device)
jod_exp = torch.as_tensor(parameters['jod_exp'], device=device)
d_max = torch.as_tensor(parameters['d_max'], device=device ) # Clamping of difference values

def phase_uncertainty(M):
    if pu_dilate != 0 and M.shape[-2] > pu_padsize and M.shape[-1] > pu_padsize:
        M_pu = pu_blur.forward(M) * (10 ** mask_c)
    else:
        M_pu = M * (10 ** mask_c)
    return M_pu

def safe_pow( x:Tensor, p ):
    epsilon = torch.as_tensor(0.00001, device=x.device)
    return (x + epsilon) ** p - epsilon ** p

def mask_pool(C):
    # Cross-channel masking
    num_ch = C.shape[0]
    M = torch.empty_like(C)
    xcm_weights_re = torch.reshape((2 ** xcm_weights), (4, 4, 1, 1, 1))[:num_ch, ...]
    for cc in range(num_ch):  # for each channel: Sust, RG, VY, Trans
        M[cc, ...] = torch.sum(C * xcm_weights_re[:, cc], dim=0, keepdim=True)
    return M

def clamp_diffs(D):
    max_v = 10 ** d_max
    Dc = max_v * D / (max_v + D)
    return Dc

def apply_masking_model(T, R, S):
    num_ch = T.shape[0]
    ch_gain = torch.reshape(torch.as_tensor([1, 1.45, 1, 1.], device=T.device), (4, 1, 1, 1))[:num_ch, ...]
    T_p = T * S * ch_gain
    R_p = R * S * ch_gain
    M_mm = phase_uncertainty(torch.min(torch.abs(T_p), torch.abs(R_p)))
    p = mask_p
    q = mask_q[0:num_ch].view(num_ch, 1, 1, 1)
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
        [torch.as_tensor(1., device=ch_chrom_w.device), ch_chrom_w, ch_chrom_w])
    # Weights for the channels: sustained, RG, YV, [transient]
    per_ch_w = per_ch_w_all[0:no_channels].view(-1, 1, 1)
    return per_ch_w

def met2jod(Q):
    # We could use
    # Q_JOD = 10. - self.jod_a * Q**self.jod_exp
    # but it does not differentiate well near Q=0
    Q_t = 0.1
    jod_a_p = jod_a * (Q_t ** (jod_exp - 1.))
    Q_JOD = torch.empty_like(Q)
    Q_JOD[Q <= Q_t] = 10. - jod_a_p * Q[Q <= Q_t]
    Q_JOD[Q > Q_t] = 10. - jod_a * (Q[Q > Q_t] ** jod_exp)
    return Q_JOD

def do_pooling_and_jods(Q_per_ch, base_rho_band):
    no_channels = Q_per_ch.shape[0] #3
    no_frames = Q_per_ch.shape[1] #1
    no_bands = Q_per_ch.shape[2] #10
    per_ch_w = get_ch_weights(no_channels)
    # Weights for the spatial bands
    per_sband_w = torch.ones((no_channels, 1, no_bands), dtype=torch.float32, device=device)
    per_sband_w[:, 0, -1] = baseband_weight[0:no_channels]

    Q_sc = lp_norm(Q_per_ch * per_ch_w * per_sband_w, beta_sch, dim=2, normalize=False)  # Sum across spatial channels
    t_int = image_int
    Q_tc = lp_norm(Q_sc, beta_tch, dim=0, normalize=False)  # Sum across temporal and chromatic channels
    Q = Q_tc * t_int
    Q = Q.squeeze()
    Q_JOD = met2jod(Q)
    return Q_JOD


## Predict
test_exr_XYZ_tensor = utils.reshuffle_dims(test_exr_XYZ_tensor, in_dims='HWC', out_dims="BCFHW") #mean=159.2148
reference_exr_XYZ_tensor = utils.reshuffle_dims(reference_exr_XYZ_tensor, in_dims='HWC', out_dims="BCFHW")
width, height = resolution[0], resolution[1]
lpyr = weber_contrast_pyr(width, height, display_ppd, device, contrast='weber_g1')
heatmap_pyr = lpyr_dec_2(width, height, display_ppd, device)
temp_ch = 1
all_ch = 2+temp_ch

dmap_channels = 1 if heatmap_type == "raw" else 3

met_colorspace='DKLd65' # This metric uses DKL colourspaxce with d65 whitepoint

R = torch.empty((1, 6, 1, height, width), device=device)
LMS2006_to_DKLd65 = (
  (1.000000000000000,   1.000000000000000,                   0),
  (1.000000000000000,  -2.311130179947035,                   0),
  (-1.000000000000000,  -1.000000000000000,  50.977571328718781) )
XYZ_to_LMS2006 = (
   ( 0.187596268556126,   0.585168649077728,  -0.026384263306304 ),
   (-0.133397430663221,   0.405505777260049,   0.034502127690364 ),
   (0.000244379021663,  -0.000542995890619,   0.019406849066323 ) )

# Color Space change: XYZ -> DKL
xyz2dkl = torch.as_tensor( LMS2006_to_DKLd65, dtype=test_exr_XYZ_tensor.dtype, device=test_exr_XYZ_tensor.device) @ torch.as_tensor( XYZ_to_LMS2006, dtype=test_exr_XYZ_tensor.dtype, device=test_exr_XYZ_tensor.device)
test_DKL_tensor = torch.einsum('bcthw,ac->bathw', test_exr_XYZ_tensor, xyz2dkl)
reference_DKL_tensor = torch.einsum('bcthw,ac->bathw', reference_exr_XYZ_tensor, xyz2dkl)
R[:,0::2, :, :, :] = test_DKL_tensor
R[:,1::2, :, :, :] = reference_DKL_tensor
# R.shape = [1,6,1,4321,7681]
## Decompose
B_bands, L_bkg_pyr = lpyr.decompose(R[0,...])
rho_band = lpyr.get_freqs()
rho_band[lpyr.get_band_count()-1] = 0.1
Q_per_ch_block = None
block_N_frames = R.shape[-3]

for bb in range(lpyr.get_band_count()): # For each spatial frequency band
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
        S[cc, :, :, :] = CSF_castleCSF.sensitivity(rho, 0, logL_bkg[..., 1, :, :, :], cch, csf_sigma) * 10.0 ** (sensitivity_correction / 20.0) #第一个的平均值应该为2.5645

    if is_baseband:
        D = (torch.abs(T_f - R_f) * S)
    else:
        # dimensions: [channel,frame,height,width]
        D = apply_masking_model(T_f, R_f, S)

    if Q_per_ch_block is None:
        Q_per_ch_block = torch.empty((all_ch, block_N_frames, lpyr.get_band_count()), device=device)
    Q_per_ch_block[:, :, bb] = lp_norm(D, beta, dim=(-2, -1), normalize=True, keepdim=False)  # Pool across all pixels (spatial pooling)

    # heatmap
    t_int = image_int
    per_ch_w = get_ch_weights(all_ch).view(-1, 1, 1, 1) * t_int
    if is_baseband:
        per_ch_w *= baseband_weight[0:all_ch].view(-1, 1, 1, 1)

    D_chr = lp_norm(D * per_ch_w, beta_tch, dim=-4, normalize=False)  # Sum across temporal and chromatic channels
    heatmap_pyr.set_lband(bb, D_chr)

heatmap_block = 1.-(met2jod(heatmap_pyr.reconstruct())/10.)

# Q_per_ch_block [3,1,10]
# heatmap_block [1,1,4321,7681]
Q_per_ch = Q_per_ch_block
heatmap = heatmap_block.detach().type(torch.float16).cpu()
rho_band = lpyr.get_freqs()
Q_jod = do_pooling_and_jods(Q_per_ch, rho_band[-1]) #8.8581

JOD = Q_jod
heatmap = np.array(heatmap)[0,0,...]
heatmap_uint8 = (heatmap * 255).astype(np.uint8)
print(JOD)
cv2.imwrite(os.path.join(root_path, f'heatmap_thr_{threshold}_{test_file_pure_name}_{heatmap_type}.png'), heatmap_uint8)