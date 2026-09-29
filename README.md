# OrGAN: Towards organ-level image separation in projection radiographs using generative AI

<p align="justify"> This repo contains the supported pytorch code to reproduce the results of "OrGAN: Towards organ-level image separation in projection radiographs using generative AI" Article. </p>

# Abstract

<p align="justify">
Chest X-rays are inexpensive and widely accessible, particularly in rural and resource-limited regions, and are used as a preliminary tool for diagnosing fractures, cardiomegaly, pneumonia, pneumothorax, and other lung diseases. However, overlapping anatomical structures can obscure subtle abnormalities, reduce diagnostic accuracy and increase inter-observer variability. To address this challenge, we present OrGAN, a domain-adversarial generative framework that separates a single organ (here, the lung) from a projection radiograph, providing an organ-specific view that complements the conventional radiograph. OrGAN pairs a U-Net generator, supervised directly against a lung projection derived from computed tomography, with a gradient-reversal domain adapter and a patchGAN discriminator that adapt it to real radiographs. On the simulated validation set it reached a peak signal-to-noise ratio of 28.07 ± 0.28 dB and a multiscale structural similarity of 0.920 ± 0.005, and on the VinDr-CXR test set a Fréchet Inception Distance of 73.72 ± 5.55 and a Kernel Inception Distance of 0.0908 ± 0.0084 (mean ± s.d. across runs). We then applied OrGAN without fine-tuning to an unseen dataset, NIH-CXR, and trained disease classifiers on the resulting images. A classifier trained on the generated lung image alone (D121-L) reached a mean area under the curve (AUC) across 14 findings within one percentage point of a lung-segmented radiograph (D121-S). When the generated image was used alongside the radiograph (FN-XL), classification recovered to the level of the radiograph alone (D121-X) (84.52% against 84.57%), and localization of expert bounding-box annotations improved at every overlap threshold, from +9.2% at an intersection over union (IoU) of 0.1 to +38.3% at IoU 0.7, in five of five cross-validation folds (Holm-corrected p = 0.007). The generated lung image acts as a spatial prior for the lung field. Ten certified radiologists rated the generated lung X-rays positively, endorsing most strongly their use alongside the conventional radiograph. This simple yet powerful GAN-based approach opens a new direction of research in medical imaging. OrGAN offers a solution for organ-level image separation from projection radiographs, potentially inspiring multi-organ separation and organ-focused image analysis.
</p>

# Proposed Architecture: OrGAN
![Architecture](images/OrGA.png)

OrGAN has three components:

- **Image generator** `G` — a U-Net. Encoder blocks are two 3×3 convolutions with batch normalisation and ReLU, downsampled by 2×2 max pooling; decoder blocks upsample, concatenate the skip connection, and repeat the same pair of convolutions. The final decoder block emits a four-channel map, **L₉**, which feeds both the domain adapter and the output layer (1×1 convolution, sigmoid).
- **Domain adapter** `C` — a gradient reversal layer (GRL) on L₉, then five strided convolutional blocks, adaptive average pooling and a two-layer dense head with dropout. The GRL is the identity on the forward pass and negates the gradient, scaled by λ_c, on the backward pass. λ_c is held at 0 for 10 epochs, ramped linearly over 20 epochs to 0.001, then held.
- **Discriminator** `D` — a two-scale PatchGAN with spectral normalisation, instance normalisation and LeakyReLU(0.2). D₁ runs at full input resolution, D₂ on patches; the adversarial loss is averaged over both scales. `D` is used in training only and excluded at inference.

The objective is `L = L_s + L_c + λ_a · L_a`, where the supervised loss `L_s` is MAE + MS-SSIM against the paired CT-derived lung projection, `L_c` is the domain classification loss, and `λ_a = 0.01`.

# Results

Mean ± s.d. across independently trained runs.

| | metric | value |
|---|---|---|
| Simulated validation set | PSNR | 28.07 ± 0.28 dB |
| | MS-SSIM | 0.920 ± 0.005 |
| VinDr-CXR test set | FID | 73.72 ± 5.55 |
| | KID | 0.0908 ± 0.0084 |

Downstream evaluation on **NIH-CXR**, a dataset OrGAN never saw, applied without fine-tuning. Mean AUC over the 14 findings, five-fold patient-grouped cross-validation:

| configuration | input | mean AUC (%) |
|---|---|---|
| D121-X | chest radiograph | 84.57 ± 0.20 |
| D121-B | bone-suppressed radiograph | 84.63 ± 0.30 |
| D121-S | segmented radiograph | 81.86 ± 0.20 |
| D121-L | **generated lung image** | 80.88 ± 0.40 |
| FN-XL | radiograph **+** generated lung image | 84.52 |

The generated lung image alone loses 0.98 percentage points more than lung segmentation, a non-generative operation. Supplying it alongside the radiograph leaves classification unchanged but improves localisation of the expert bounding boxes at every IoU threshold — +9.2% at IoU 0.1 rising to +38.3% at IoU 0.7 — in five of five folds (Holm-corrected p = 0.007), without any localisation supervision.

# Datasets

For OrGAN training:
  We have used real X-ray data from the publicly available VinBigDr-CXR dataset: [Link](https://vindr.ai/datasets/cxr) </br>
  Additionally, we have created a simulated dataset of chest X-rays with lung (label) from the publicly available LUNA16 CT scan dataset: [Link](https://luna16.grand-challenge.org/Download/)</br>

The experiments are conducted on three publicly available datasets, </br>
VinDr-CXR test set : [Link](https://vindr.ai/datasets/cxr)</br>
National Institutes of Health (NIH) Chest X-ray Dataset : [Link](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)</br>
FracAtlas Dataset : [Link](https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628)</br>

## Splits

- **Simulated (LUNA16).** 888 CT volumes → 779 with at least 128 slices → 680 after outlier removal (InceptionV3 features, cosine distance from the centroid beyond three standard deviations; all five bone-weighted variants of a rejected volume are dropped together), giving 3,400 images. Split **at the volume level**, so no patient contributes to both sides: 540 volumes (2,700 images) train, 140 volumes (700 images) validation.
- **VinDr-CXR.** The official split — 15,000 train (unlabeled, for domain adaptation), 3,000 test. The qualitative examples, FID/KID and the radiologist assessment all come from the test set.
- **NIH-CXR and FracAtlas.** Inference only; never seen at any stage of OrGAN training. For the classifier, the 880 images carrying bounding boxes are held out for localisation, and the remaining 111,240 are split into five folds grouped by patient identifier.

# Dataset Preparation
1) Use the CT2Xray-process.ipynb to generate simulated dataset from LUNA16 CT scans for training OrGAN. You can download the preprocessed dataset from here. [Link](https://figshare.com/s/49b395a6c9a883cfeb8f)
2) Use the VinBiG-process.ipynb to process the real X-ray dicom files for training OrGAN.

Images are resampled to 512 × 512. Chest X-rays are standardised with fixed dataset-level mean and standard deviation per domain; the lung labels are normalised to [0, 1]. Global standardisation keeps relative intensity consistent within a dataset — for a new dataset, estimate the constants from a small representative subset. Falling back to each image's own mean and standard deviation works but gives up the common intensity reference.

# Train OrGAN
For training OrGAN, use the following steps:
1) After preparing the simulated dataset in the previous step, split the data into train and test and move them to the OrGAN/data/Train and OrGAN/data/Test folders.
2) After preparing the real dataset (VinDr-CXR train set), move the data to OrGAN/data/Train/Xray folder.
3) Finally, run the training script:

```bash
cd OrGAN
python train.py --seed 0
```

Key flags: `--seed` (vary across repeats), `--epochs` (default 100, early stopping may end a run sooner), and `--grl-lambda-max` / `--grl-e0` / `--grl-ramp-epochs` for the gradient-reversal schedule. Validation PSNR and MS-SSIM are computed each epoch on an EMA copy of the generator, per-epoch metrics are written to `epoch_data.csv`, and the best-PSNR checkpoint is saved for inference. See `OrGAN/README.md` for the full description.

Training settings used in the paper: AdamW, learning rate 1e-3 (generator) and 1e-4 (discriminator), weight decay 1e-4, reduce-on-plateau schedulers, batch size 24 split evenly between simulated and real X-rays, EMA decay 0.999, maximum 100 epochs with early stopping after 20 epochs without improvement in either validation PSNR or MS-SSIM, on a single NVIDIA A100-SXM4 80 GB. All metrics and the released weights come from the EMA generator.

Runs are seeded for PyTorch, CUDA, NumPy and Python's `random`, and DataLoader workers are seeded through an explicit generator and a worker init function. Seed-to-seed variation is still expected; the paper reports mean ± s.d. over repeats rather than a single run.

# Pre-trained Model Weight and Preprocessed training data
Pre-trained model weight of the best model and preprocessed training data can be downloaded from here. [Fileshare Link](https://figshare.com/s/49b395a6c9a883cfeb8f) 

# Inference on Real X-ray:
For inference on chest X-rays: 
1) Download the model weight and place it inside folder: "OrGAN/model_weights/".
2) Place some real X-ray dicom files inside folder: "OrGAN/data/Xray_real/real/"
3) Run the OrGAN/inferenceRealX.ipynb.

To generate lung images for a whole directory instead of one at a time, use `OrGAN/getLung.py`:

```bash
cd OrGAN
python getLung.py --ckpt <checkpoint> --images <xray_dir> --out <lung_dir>
```

# Downstream Classifier

`Classifier/` contains the thoracic disease classifier used to test whether the generated lung images retain diagnostic information, and whether pairing them with the original radiograph helps. Three configurations are trained and evaluated identically under 5-fold cross-validation: chest X-ray only (`xray_base`), generated lung image only (`lung_base`), and a two-stream gated fusion of both (`proposed`). See `Classifier/README.md`.

These correspond to **D121-X**, **D121-L** and **FN-XL** in the paper. The paper additionally reports bone-suppressed and lung-segmented comparators (D121-B, D121-S, FN-XB, FN-XS) and two gating controls (FN-XX, FN-X0); those inputs are produced by third-party tools — [xU-NetFullSharp](https://doi.org/10.1016/j.bspc.2024.106983) for bone suppression and [HybridGNet](https://doi.org/10.1109/TMI.2022.3224660) / [CheXmask](https://doi.org/10.13026/3705-zg36) for the lung contour — and are not redistributed here.

Localisation is measured against the 984 NIH bounding-box annotations. The attended region is a Grad-CAM++ map thresholded at its 80th percentile, so every configuration attends to the same fraction of the image and can only gain by placing that area better, not by enlarging it. Absolute values are therefore not comparable with box-regression detectors; only the comparison across configurations, which share the protocol exactly, is meaningful.

# Comparison 
![Comparison](images/Video.gif)

# Intended use

The organ image is a **secondary visualisation aid**, read alongside the conventional radiograph and never in place of it. It is derived from the radiograph itself and adds no information beyond the learned priors; what changes is the representation. Nodules, masses, emphysematous change and pneumothorax are attenuated relative to the radiograph, so the radiograph remains the primary reference for those findings. We do not propose it for autonomous use, primary interpretation, or triage. Use is restricted to adult chest radiographs unless the model is retrained, and outputs should be gated by an out-of-domain check.

# Citing OrGAN

```bibtex
@article{ahmed2026organ,
  title   = {OrGAN: Towards organ-level image separation in projection radiographs using generative AI},
  author  = {Ahmed, Md. Kawsar and Islam, Mohammad Tariqul and Zunaed, Mohammad and Jacob, Mathews and Hasan, Taufiq},
  year    = {2026}
}
```

# License

MIT. See [LICENSE](LICENSE).
