# OrGAN

Organ-level image separation in projection radiographs. OrGAN takes a chest
radiograph and produces a lung-only image — an organ-specific view intended to
be read alongside the conventional radiograph, not in place of it.

The framework is a U-Net generator trained with three objectives: a supervised
loss against CT-derived lung projections, a gradient-reversal domain classifier
that aligns simulated and real X-ray features, and a two-scale PatchGAN
discriminator.

## Layout

```
train.py                    training (supervised + domain adapter + discriminator)
getLung.py                  inference: generate lung images for a directory of X-rays
model/gen.py                U-Net generator with the domain classifier head
model/gen_parts.py          generator blocks and the domain classifier
model/dnet.py               two-scale spectral-norm PatchGAN + GAN loss
model/gradient_reversal/    gradient reversal layer
utils/dataset.py            datasets, augmentation, two-stream batch sampler
utils/evaluate.py           PSNR / MS-SSIM evaluation
Classifier/                 downstream evaluation (see Classifier/README.md)
```

`Classifier/` holds the thoracic disease classifier used to test whether the
generated lung images retain diagnostic information, and whether pairing them
with the original radiograph helps. It has its own README, and depends on lung
images produced by `getLung.py` below.

## Data layout

Paths are relative to the working directory, with data expected one level up:

```
../OrGAN-files/data/Train/Xray     simulated (CT-derived) + real chest X-rays, .npy
../OrGAN-files/data/Train/Lungs    lung labels for the simulated subset, .npy
../OrGAN-files/data/Test/Xray      held-out simulated validation X-rays
../OrGAN-files/data/Test/Lungs     held-out lung labels
../OrGAN-files/model_weights/      checkpoints are written here
```

The first `l_id` entries of `Train/Xray` (sorted) are the labelled simulated
pairs; the remainder are unlabelled real radiographs. Set `l_id` in `train.py`
to match your split.

## Training

```bash
python train.py --seed 0
```

Useful flags:

| flag | default | meaning |
|---|---|---|
| `--seed` | 0 | run seed; vary across repeats |
| `--epochs` | 100 | maximum epochs (early stopping may end sooner) |
| `--grl-lambda-max` | 0.001 | gradient-reversal alpha ceiling |
| `--grl-e0` | 10 | epoch the alpha ramp starts |
| `--grl-ramp-epochs` | 20 | epochs to ramp alpha to its ceiling |
| `--model-suffix` | `''` | appended to the run directory name |

Validation PSNR and MS-SSIM are computed each epoch on an EMA copy of the
generator (decay 0.999), and the best-PSNR checkpoint is what `getLung.py`
loads. Per-epoch metrics are written to `epoch_data.csv` in the run directory.

## Inference

```bash
python getLung.py \
  --ckpt ../OrGAN-files/model_weights/<run>/best_ema.ckpt \
  --images /path/to/xrays \
  --out /path/to/lungs
```

Already-generated outputs are skipped, so the script is a no-op once the output
directory is full — move previous results aside before switching checkpoints.
A `lungs_provenance.json` sidecar records which checkpoint produced a mask set.

## Reproducibility

Runs are deterministic for a fixed seed: seeds are set for torch, CUDA, numpy
and Python, DataLoader workers are seeded explicitly, cuDNN is pinned, and
`CUBLAS_WORKSPACE_CONFIG` is set before torch is imported. `model/dnet.py`
provides `ReflectConv2d`, a reflection-padded convolution whose forward is
bit-identical to `padding_mode="reflect"` but whose backward is deterministic —
PyTorch's native `reflection_pad2d_backward_cuda` has no deterministic
implementation and was the last source of run-to-run drift.

Adversarial training can occasionally collapse from an unlucky initialisation:
the discriminator wins within the first epoch and never gives ground, leaving
`DTrloss` pinned near 0.02 instead of the ~0.65 equilibrium. It is visible by
epoch 2 — check `DTrloss` and restart with a different seed if it appears.

## Requirements

```bash
pip install -r requirements.txt
```

## License

MIT.
