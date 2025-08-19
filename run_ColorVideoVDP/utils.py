import numpy as np
import torch
import torch.nn.functional as Func
from torch.functional import Tensor

class ImGaussFilt():
    def __init__(self, sigma, device):
        self.filter_size = 2 * int(np.ceil(2.0 * sigma)) + 1
        self.half_filter_size = (self.filter_size - 1) // 2

        x = np.arange(-self.half_filter_size, self.half_filter_size + 1)
        y = np.arange(-self.half_filter_size, self.half_filter_size + 1)
        xx, yy = np.meshgrid(x, y)
        gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma * sigma))
        gaussian /= gaussian.sum()
        self.K = torch.tensor(gaussian, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        X = 1

    def run(self, img):

        if len(img.shape) == 2:
            img_4d = img.reshape((1, 1, img.shape[0], img.shape[1]))
        else:
            img_4d = img

        pad = (
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,
            self.half_filter_size,)

        img_4d = Func.pad(img_4d, pad, mode='reflect')
        return Func.conv2d(img_4d, self.K)[0, 0]

def reshuffle_dims( T: Tensor, in_dims: str, out_dims: str ) -> Tensor:
    in_dims = in_dims.upper()
    out_dims = out_dims.upper()

    assert len(in_dims) == T.dim(), "The in_dims string must have as many characters as there are dimensions in T"

    # Find intersection of two strings
    inter_dims = ""
    for kk in range(len(out_dims)):
        if in_dims.find(out_dims[kk]) != -1:
            inter_dims += out_dims[kk]

    # First, squeeze out the dimensions that are missing in the output
    sq_dims = []
    new_in_dims = ""
    for kk in range(len(in_dims)):
        if inter_dims.find(in_dims[kk]) == -1: # The dimension is missing in the output
            sq_dims.append(kk)
            assert T.shape[kk] == 1, "Only the dimensions of size 1 can be skipped in the output"
        else:
            new_in_dims += in_dims[kk]
    in_dims = new_in_dims
    # For the compatibility with PyTorch pre 2.0, squeeze dims one by one
    sq_dims.sort(reverse=True)
    for kk in sq_dims:
        T = T.squeeze(dim=kk)

    # First, permute into the right order
    perm = [0] * len(inter_dims)
    for kk in range(len(inter_dims)):
        ind = in_dims.find(inter_dims[kk])
        assert ind != -1, 'Dimension "{}" missing in the target dimensions: "{}"'.format(in_dims[kk],out_dims)
        perm[kk] = ind
    T_p = T.permute(perm)

    # Add missing dimensions
    out_sh = [1] * len(out_dims)
    for kk in range(len(out_dims)):
        ind = inter_dims.find(out_dims[kk])
        if ind != -1:
            out_sh[kk] = T_p.shape[ind]

    return T_p.reshape( out_sh )

if __name__ == '__main__':
    Filter = ImGaussFilt(sigma=55.0521, device=torch.device('cpu'))
