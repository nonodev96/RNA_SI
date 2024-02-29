import torch
from skimage import io as skimage


def imreadImg(path, opt):
    x = skimage.imread(path)
    x = x.astype(float)
    if x.ndim == 2:
        x = x[:, :, None, None]
        max_x = x.max()
        x = x.transpose(2, 3, 0, 1) / max_x
    elif x.ndim == 3:
        x = x[:, :, :, None]
        x = x.transpose(3, 2, 0, 1)
        max_x = x.max()
        x = x / max_x
    x = torch.from_numpy(x)
    x = x.to(opt.device).type(torch.cuda.FloatTensor)
    x = (x - 0.5) * 2
    x = x.clamp(-1, 1)
    return x, max_x
