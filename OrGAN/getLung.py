#!/usr/bin/env python3
# ----------------------------------------
# Lung Mask Inference Script
# ----------------------------------------

from argparse import ArgumentParser

import json
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import numpy as np
import torch
import albumentations as A

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.gen import OrGAN
from utils.dataset import BXDataset

# No determinism setup is needed here (unlike the train-* scripts): inference
# runs under model.eval() + no_grad() with a deterministic Resize as the only
# transform, so there is no dropout, no augmentation RNG, and no backward pass
# - the reflection_pad2d nondeterminism that affects train-D / train-C_D is
# backward-only and cannot reach this script.


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--ckpt', type=str,
                        default='../OrGAN-files/model_weights/C_D3_seed2/best_ema.ckpt',
                        help='Generator checkpoint to run. Pin this explicitly rather than '
                             'letting it track whichever run last wrote the file')
    parser.add_argument('--images', type=str, default='../Classifier/NIH-CXR/images',
                        help='Directory of input chest X-rays')
    parser.add_argument('--out', type=str, default='../Classifier/NIH-CXR/lungs',
                        help='Directory to write lung masks into')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader workers (no seeding concerns - no augmentation here)')
    return parser.parse_args()


# ----------------------------------------
# Main
# ----------------------------------------

def main():

    args = parse_args()

    # Albumentations transform
    transform = A.Compose(
        [
            A.Resize(width=512, height=512, p=1.0),
        ]
    )

    # Parameters
    BATCH_SIZE_TEST = 1
    test_xray_directory = args.images
    save_path = args.out
    os.makedirs(save_path, exist_ok=True)

    # Dataset & Loader
    test_xray_filenames = sorted(os.listdir(test_xray_directory))
    dataset_test = BXDataset(
        test_xray_filenames,
        test_xray_directory,
        transform=transform,
        MS=True,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = OrGAN()
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    written = 0
    skipped = 0

    for images, name in tqdm(
    test_loader,
    desc="Generating lung masks",
    total=len(test_loader),
    ):
        filename = name[0]
        out_path = os.path.join(save_path, filename)

        # Skip if already processed. NOTE: this makes the script a no-op once
        # the directory is fully populated, so switching --ckpt does NOT
        # regenerate anything - move the old masks aside first, or you will
        # silently end up with a set produced by two different generators.
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue

        images = images.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred, _ = model(images)

        mask_pred_np = (
            mask_pred.cpu()
            .detach()
            .numpy()
            .squeeze(0)
            .squeeze(0)
        )

        # The generator's OutConv ends in Sigmoid, so mask_pred is already in
        # [0, 1] and maps directly onto [0, 255]. Do NOT use cv2.normalize with
        # NORM_MINMAX here: that rescales every mask by its own extremes, so a
        # mask whose true peak is 0.6 and one whose peak is 1.0 both come out
        # at 255, discarding absolute intensity and making it inconsistent
        # across the dataset that feeds the downstream classifier.
        mask_uint8 = np.clip(mask_pred_np * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(out_path, mask_uint8)
        written += 1

    print(f'Done. {written} masks written, {skipped} skipped (already present).')
    if written == 0 and skipped:
        print('NOTE: nothing was regenerated - every output already existed. '
              'If you meant to switch checkpoints, move the old masks aside first.')


# ----------------------------------------
# Entry point
# ----------------------------------------

if __name__ == "__main__":
    main()
