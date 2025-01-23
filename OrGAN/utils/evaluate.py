import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader 
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import math

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    ssim_value = 0
    psnr = 0
    criterion_ssim = MS_SSIM(win_size=11, win_sigma=2, data_range=1, size_average=True, channel=1)

    for batch, (images, mask) in enumerate(dataloader):
        
        images = images.to(device=device, dtype=torch.float32)
        lungs = mask.unsqueeze(1).to(device=device, dtype=torch.float32)
        

        with torch.no_grad():

            mask_pred, _ = net(images)
            ssim_value += criterion_ssim(mask_pred, lungs).item()
            psnr += PSNR(mask_pred.cpu().detach().numpy(), lungs.cpu().detach().numpy())

    net.train()

    if num_val_batches == 0:
        return psnr, ssim_value
    return psnr / num_val_batches, ssim_value / num_val_batches

def PSNR(original, pred):
    mse = np.mean((original - pred) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR