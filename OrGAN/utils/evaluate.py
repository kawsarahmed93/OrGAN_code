
import torch
from tqdm import tqdm
from pytorch_msssim import MS_SSIM


def psnr_batch_torch(pred, target, max_pixel=1.0, eps=1e-8):
    """
    pred, target shape: [B, C, H, W]
    Returns PSNR for each image in batch: shape [B]
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))

    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse + eps))

    # If mse is exactly zero, set PSNR to 100
    psnr = torch.where(
        mse == 0,
        torch.tensor(100.0, device=pred.device, dtype=pred.dtype),
        psnr
    )

    return psnr


def evaluate(net, dataloader, device):
    net.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_images = 0

    criterion_ssim = MS_SSIM(
        win_size=11,
        win_sigma=2,
        data_range=1,
        size_average=False,   # important for batch_size > 1
        channel=1
    ).to(device)

    # progress_bar = tqdm(dataloader, desc="Validation", leave=True)

    with torch.no_grad():
        for batch, (images, mask) in enumerate(dataloader):

            images = images.to(device=device, dtype=torch.float32)
            lungs = mask.unsqueeze(1).to(device=device, dtype=torch.float32)

            mask_pred, _ = net(images)

            # Ensure both are [B, C, H, W]
            if mask_pred.ndim == 3:
                mask_pred = mask_pred.unsqueeze(1)

            if lungs.ndim == 3:
                lungs = lungs.unsqueeze(1)

            batch_size = images.shape[0]

            # PSNR per image: shape [B]
            psnr_values = psnr_batch_torch(mask_pred, lungs)

            # MS-SSIM per image: shape [B]
            ssim_values = criterion_ssim(mask_pred, lungs)

            # Accumulate image-wise sums
            total_psnr += psnr_values.sum().item()
            total_ssim += ssim_values.sum().item()
            total_images += batch_size

            # progress_bar.set_postfix({
            #     "batch_psnr": f"{psnr_values.mean().item():.4f}",
            #     "batch_ssim": f"{ssim_values.mean().item():.4f}"
            # })

    net.train()

    if total_images == 0:
        return 0, 0

    return total_psnr / total_images, total_ssim/ total_images