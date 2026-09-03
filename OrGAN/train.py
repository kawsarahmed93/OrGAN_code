#!/usr/bin/env python3
# ----------------------------------------
# OrGAN training: domain classifier (adapter) + discriminator
# ----------------------------------------

import os
# Must be set before the first CUDA call (i.e. before torch is imported by
# any of the modules below) - required by torch.use_deterministic_algorithms
# for cuBLAS ops to be reproducible on GPU.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from argparse import ArgumentParser

from model.gen import OrGAN
from model.dnet import CNN, GANLoss
from utils.dataset import XrayDataset, TwoStreamBatchSampler, TXDataset
from utils.evaluate import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import albumentations as A
from pytorch_msssim import MS_SSIM
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import math
import cv2
import pandas as pd

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

import torch.backends.cudnn as cudnn
import random
import time
import gc

cudnn.benchmark = False
cudnn.deterministic = True
# cudnn.deterministic only pins cuDNN's convolution algorithm choice; it does
# NOT make a run reproducible on its own - other CUDA kernels (atomic-add
# based reductions and scatters) still have nondeterministic backward passes
# without the call below, and cuBLAS additionally needs CUBLAS_WORKSPACE_CONFIG,
# set at the top of this file before torch is imported.
#
# This used to be incomplete. reflection_pad2d_backward_cuda has no
# deterministic implementation, so under warn_only=True it warned once per
# epoch and silently fell back to the nondeterministic kernel - which is why
# train-D and train-C_D never repeated across runs of the same seed while
# train-C (no discriminator) always did. model/dnet.py now routes that padding
# through ReflectConv2d, whose forward is bit-identical and whose backward is
# deterministic, so all three configs are reproducible. Verified 2026-09-02:
# a full generator + discriminator forward/backward for both D and C_D
# reports no "does not have a deterministic implementation" warnings.
#
# warn_only=True is kept so a newly introduced nondeterministic op warns
# instead of crashing a long run - check the logs for that message after
# adding any new layer or loss.
torch.use_deterministic_algorithms(True, warn_only=True)
# Prevent each DataLoader worker process from also spawning its own OpenCV
# thread pool - with num_workers>0 that oversubscribes CPUs and can make
# augmentation slower, not faster, on a shared node.
cv2.setNumThreads(0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=0,
                         help='Random seed for this run (vary this across repeats, e.g. 0/1/2)')
    parser.add_argument('--epochs', type=int, default=100,
                         help='Max training epochs (early stopping may end a run sooner)')
    parser.add_argument('--grl-lambda-max', type=float, default=0.001,
                         help='GRL alpha ceiling once ramped up (see grl_alpha in main)')
    parser.add_argument('--grl-e0', type=int, default=10,
                         help='Epoch at which the GRL alpha ramp starts (alpha=0 before this)')
    parser.add_argument('--grl-ramp-epochs', type=int, default=20,
                         help='Epochs over which alpha ramps from 0 to grl-lambda-max')
    parser.add_argument('--model-suffix', type=str, default='',
                         help='Appended to model_iter, e.g. to separate sweep runs (C_D3_seed0<suffix>)')
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def save_with_retry(save_func, path, retries=5, delay=2.0):
    """Runs save_func(path), retrying on transient network-filesystem errors.

    The shared storage this repo runs on intermittently reports a freshly
    created directory as missing to a subsequent low-level file open (seen
    as 'Remote I/O error' / 'Parent directory ... does not exist' even right
    after os.makedirs succeeded). Retrying after a short delay, re-creating
    the directory each time, works around that without masking a real
    missing-path bug (which would still fail after all retries).
    """
    parent_dir = os.path.dirname(path)
    last_err = None
    for attempt in range(retries):
        try:
            os.makedirs(parent_dir, exist_ok=True)
            save_func(path)
            return
        except (RuntimeError, OSError) as e:
            last_err = e
            if attempt < retries - 1:
                print(f'Save to {path} failed ({e}); retrying ({attempt + 1}/{retries})...')
                time.sleep(delay)
    raise last_err


# ----------------------------------------
# Utility functions
# ----------------------------------------

def PSNR(original, pred):
    mse = np.mean((original - pred) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class RandomZeroPad(A.DualTransform):
    def __init__(self, min_pad=0, max_pad=40, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.min_pad = min_pad
        self.max_pad = max_pad

    def get_params(self):
        return {
            "top": random.randint(self.min_pad, self.max_pad),
            "bottom": random.randint(self.min_pad, self.max_pad),
            "left": random.randint(self.min_pad, self.max_pad),
            "right": random.randint(self.min_pad, self.max_pad),
        }

    def apply(self, img, top=0, bottom=0, left=0, right=0, **params):
        return cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)

    def apply_to_mask(self, mask, top=0, bottom=0, left=0, right=0, **params):
        return cv2.copyMakeBorder(mask, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)


# ----------------------------------------
# Main
# ----------------------------------------

def main():

    args = parse_args()
    set_seed(args.seed)

    BATCH_SIZE_TRAIN = 24
    L_BATCH_SIZE_TRAIN = 12
    BATCH_SIZE_TEST = 6
    learning_rate = 0.001
    dlearning_rate = 0.0001
    epochs = args.epochs
    early_stop_patience = 20
    model_iter = f'C_D3_seed{args.seed}{args.model_suffix}'
    NUM_WORKERS = 8

    # GRL schedule (Brion et al. 2021, Comput Biol Med 131:104269, Eq. 4):
    # alpha=0 for the first GRL_E0 epochs (classifier and generator each learn
    # their own task independently), then ramps linearly to GRL_LAMBDA_MAX
    # over GRL_RAMP_EPOCHS, reaching full ceiling by epoch GRL_E0+GRL_RAMP_EPOCHS
    # = 30 by default (comfortably before this run's ~epoch 45-90 saturation
    # window). GRL_LAMBDA_MAX=1.0 (full reversal) was tried and measured worse
    # than discriminator-only on both PSNR and FID/KID (see C_D3_seed0 run),
    # even though closs reached the ln(2)=0.693 confusion equilibrium by
    # ~epoch 14 at alpha still << 1 - i.e. this classifier is fooled easily,
    # so driving alpha all the way to 1 and sustaining it for 60+ epochs is
    # pure over-regularization of the encoder with no further confusion to
    # buy. Default backs off to Brion's own choice (lambda_max=0.001), which
    # is what produced this project's best C+D FID (<70) historically; the
    # true optimum is unconfirmed between 0.001 and 1.0, hence the CLI
    # override below for sweeping it. closs stays unweighted in `loss` below
    # - GRL_LAMBDA_MAX is the only damping knob, since it (unlike an outer
    # weight on closs) throttles only the reversed gradient reaching the
    # encoder and leaves the classifier's own training at full strength,
    # matching the DANN design intent.
    GRL_E0 = args.grl_e0
    GRL_RAMP_EPOCHS = args.grl_ramp_epochs   # alpha reaches ceiling by epoch GRL_E0 + GRL_RAMP_EPOCHS
    GRL_LAMBDA_MAX = args.grl_lambda_max

    def grl_alpha(epoch):
        progress = max(0.0, epoch - GRL_E0) / GRL_RAMP_EPOCHS
        return min(GRL_LAMBDA_MAX, GRL_LAMBDA_MAX * progress)


    os.makedirs('../OrGAN-files/model_weights/' + model_iter, exist_ok=True)

    train_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
            ], p=0.8),
            A.Affine(scale=(0.92, 1.05), translate_percent=0, rotate=0, shear=0, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
            RandomZeroPad(min_pad=0, max_pad=40, p=0.5),
            A.OneOf([
                A.RandomGamma(gamma_limit=(70.0, 160.0), p=0.6),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.Blur(blur_limit=(3, 7), p=0.1),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.1),
            ], p=0.6),
            A.Resize(width=512, height=512, p=1.0),
        ]
    )
    train_transforms_realx = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.07, scale_limit=0.1, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5),
            ], p=0.8),
            A.Resize(width=512, height=512, p=1.0),
        ]
    )
    test_transforms = A.Compose([A.Resize(width=512, height=512, p=1.0)])

    train_xray_directory = '../OrGAN-files/data/Train/Xray'
    train_xray_filenames = sorted(os.listdir(train_xray_directory))
    l_id = 2700
    label_idx = list(range(l_id))
    unlabel_idx = list(range(l_id, len(train_xray_filenames)))

    test_xray_directory = '../OrGAN-files/data/Test/Xray'
    test_xray_filenames = sorted(os.listdir(test_xray_directory))

    dataset_train = XrayDataset(train_xray_filenames, train_xray_directory, l_id, transform=train_transforms, r_transform=train_transforms_realx, MS=True)
    dataset_test = TXDataset(test_xray_filenames, test_xray_directory, transform=test_transforms, MS=True)

    batch_sampler = TwoStreamBatchSampler(label_idx, unlabel_idx, BATCH_SIZE_TRAIN, BATCH_SIZE_TRAIN - L_BATCH_SIZE_TRAIN)

    def worker_init_fn(worker_id):
        worker_seed = args.seed * 100 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=worker_init_fn, generator=loader_generator,
    )
    test_loader = DataLoader(
        dataset_test, BATCH_SIZE_TEST, shuffle=False, drop_last=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=worker_init_fn,
    )

    model = OrGAN()
    dnet = CNN()

    optimizer_G = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer_D = torch.optim.AdamW(dnet.parameters(), lr=dlearning_rate, weight_decay=1e-4)

    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, 'min', factor=0.5, patience=10, min_lr=1e-7)
    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_G, mode="max", factor=0.5, patience=5, threshold=0.02, threshold_mode="abs", cooldown=2, min_lr=1e-5
    )

    # Loss functions
    mae = torch.nn.L1Loss()
    criterionGAN = GANLoss()
    nll = torch.nn.NLLLoss()
    ssim = MS_SSIM(win_size=11, win_sigma=2, data_range=1, size_average=True, channel=1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)

    dnet = nn.DataParallel(dnet, device_ids=[0])
    dnet = dnet.to(device)

    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999), use_buffers=True)

    best_psnr_score = 0
    best_ssim_score = 0
    epochs_without_improvement = 0

    column_names = ["TrPSNR", "TePSNR", "TrSSIM", "TeSSIM", "GTrloss", "GTrSuploss", "GTrAdloss", "GTrcloss", "DTrloss"]
    df = pd.DataFrame(columns=column_names)

    for epoch in range(epochs):
        model.train()

        current_alpha = grl_alpha(epoch)
        model.module.set_grl_alpha(current_alpha)

        epoch_loss = 0
        epoch_sloss = 0
        epoch_closs = 0
        epoch_aloss = 0
        epoch_dloss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        count = 0
        pbar = tqdm(train_loader, dynamic_ncols=True)
        for batch, (images, masks) in enumerate(pbar):

            lungs = masks[:L_BATCH_SIZE_TRAIN]
            lungs = lungs.unsqueeze(1).to(device).float()
            images = images.to(device).float()
            l_image = images[:L_BATCH_SIZE_TRAIN]
            ul_image = images[L_BATCH_SIZE_TRAIN:]

            dnet.train()
            for p in dnet.parameters():
                p.requires_grad_(True)

            with torch.no_grad():
                l_pred, _ = model(l_image)
                ul_pred, _ = model(ul_image)

            all_pred = torch.cat([l_pred, ul_pred], dim=0)
            ul_d = dnet(all_pred.detach())
            ul_dloss = criterionGAN(ul_d, False)

            l_d = dnet(lungs)
            l_dloss = criterionGAN(l_d, True)

            dloss = 0.5 * (ul_dloss + l_dloss)

            optimizer_D.zero_grad()
            dloss.backward()
            optimizer_D.step()

            epoch_dloss += dloss.item()
            del ul_d, ul_dloss, l_d, l_dloss, dloss

            l_pred, l_log = model(l_image)
            ul_pred, ul_log = model(ul_image)

            l_domain = torch.zeros(L_BATCH_SIZE_TRAIN).to(device).long()
            ul_domain = torch.ones(BATCH_SIZE_TRAIN - L_BATCH_SIZE_TRAIN).to(device).long()

            l_nll_loss = nll(l_log, l_domain)
            ul_nll_loss = nll(ul_log, ul_domain)

            sloss = mae(l_pred, lungs) + (1 - ssim(l_pred, lungs))
            closs = 0.5 * (l_nll_loss + ul_nll_loss)

            for p in dnet.parameters():
                p.requires_grad_(False)
            dnet.eval()

            all_pred = torch.cat([l_pred, ul_pred], dim=0)
            ul_d = dnet(all_pred)
            aloss = criterionGAN(ul_d, True)

            loss = sloss + 0.01 * aloss + closs

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            ema_model.update_parameters(model)

            ssim_value = ssim(l_pred, lungs)
            psnr_value = PSNR(lungs.cpu().detach().numpy(), l_pred.cpu().detach().numpy())

            epoch_psnr += psnr_value
            epoch_ssim += ssim_value.item()
            epoch_loss += loss.item()
            epoch_sloss += sloss.item()
            epoch_aloss += aloss.item()
            epoch_closs += closs.item()

            count += 1
            pbar.set_postfix({'epoch': f'{epoch}', 'loss': f'{loss.item():.4f}', 'vP': f'{best_psnr_score:.2f}'})

            del l_pred, lungs, images, loss, ssim_value, l_nll_loss, ul_nll_loss, sloss, closs
            gc.collect()
            torch.cuda.empty_cache()

        psnr_score, ssim_score = evaluate(ema_model, test_loader, device)
        print('EMA | VPSNR(dB): ' + str(round(psnr_score, 3)) + ' | VSSIM: ' + str(round(ssim_score, 3)))

        if psnr_score > best_psnr_score:
            epochs_without_improvement = 0
            save_with_retry(lambda p: torch.save(ema_model.module.module.state_dict(), p), '../OrGAN-files/model_weights/' + model_iter + '/best_ema.ckpt')
            save_with_retry(lambda p: torch.save(dnet.module.state_dict(), p), '../OrGAN-files/model_weights/' + model_iter + '/dnet.ckpt')
            best_psnr_score = psnr_score
        else:
            epochs_without_improvement += 1

        if ssim_score > best_ssim_score:
            # SSIM improvements reset the patience clock as well as PSNR ones.
            # This looks redundant but is load-bearing: test PSNR can dip for
            # ~25 epochs and then recover to a new best (see D3_seed1, which
            # peaked at ep43, declined through ep70, then set its true best of
            # 27.985 at ep75). SSIM keeps improving through those dips, so the
            # reset here is what keeps the run alive long enough to find the
            # recovery. A PSNR-only clock at patience=20 stops that run at
            # ep63 and costs 0.064 dB, while saving only ~9 epochs per run.
            epochs_without_improvement = 0
            save_with_retry(lambda p: torch.save(ema_model.module.module.state_dict(), p), '../OrGAN-files/model_weights/' + model_iter + '/best_ema_ssim.ckpt')
            save_with_retry(lambda p: torch.save(dnet.module.state_dict(), p), '../OrGAN-files/model_weights/' + model_iter + '/dnet_ssim.ckpt')
            best_ssim_score = ssim_score

        if epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

        temp_dloss = epoch_dloss / count
        scheduler_G.step(psnr_score)
        scheduler_D.step(temp_dloss)

        row = [round(epoch_psnr / count, 3), psnr_score, round(epoch_ssim / count, 3), ssim_score,
               round(epoch_loss / count, 3), round(epoch_sloss / count, 3), round(epoch_aloss / count, 3),
               round(epoch_closs / count, 3), round(epoch_dloss / count, 3)]
        df.loc[len(df)] = row

        print('Epoch ' + str(epoch))
        print(' TPsnr: ' + str(round(epoch_psnr / count, 3)) + ' TSsim: ' + str(round(epoch_ssim / count, 3)) +
              ' Tloss: ' + str(round(epoch_loss / count, 3)) + ' dloss: ' + str(round(epoch_dloss / count, 3)) +
              ' adv_loss: ' + str(round(epoch_aloss / count, 3)) + ' sloss: ' + str(round(epoch_sloss / count, 3)) +
              ' c_loss: ' + str(round(epoch_closs / count, 3)))
        print(f'G_lr: {optimizer_G.param_groups[0]["lr"]:.7f} | D_lr: {optimizer_D.param_groups[0]["lr"]:.7f} | GRL_alpha: {current_alpha:.4f}')

    save_with_retry(lambda p: df.to_csv(p), '../OrGAN-files/model_weights/' + model_iter + '/epoch_data.csv')


# ----------------------------------------
# Entry point
# ----------------------------------------

if __name__ == "__main__":
    main()
