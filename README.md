# OrGAN: Towards organ-level image separation in projection radiographs using generative AI 
<p align="justify"> This repo contains the supported pytorch code to reproduce the results of "OrGAN: Towards organ-level image separation in projection radiographs using generative AI" Article. </p>

# Abstract

<p align="justify"> 
Chest X-rays are widely used to diagnose conditions such as fractures, pneumonia, COPD, asthma, pneumothorax, and other lung-related diseases. However, overlapping anatomical structures often obscure critical abnormalities, reducing diagnostic accuracy and leading to high inter-observer variability. Organ-level image separation from X-rays has the potential to excel organ-focused diagnostics in radiographs, surpassing the limitations of traditional segmentation. While significant research has been done on organ segmentation in projection radiographs, organ-level image separation remains underexplored. To address this gap, we propose OrGAN, a generative adversarial framework designed to separate organ-level images from projection radiographs. This study focuses on isolating lung tissue images from chest X-rays, providing radiologists with organ-specific insights to complement conventional radiographs. OrGAN was trained on two data sources- simulated X-rays (labeled) generated from CT volumes and real X-rays (unlabeled). A modified U-Net based image generator, inspired by domain adversarial algorithms, was employed for image separation, while a CNN based discriminator further guided the image translation process. OrGAN achieved a PSNR of 28.3 dB and a Multiscale SSIM of 0.935 on the simulated validation set. Qualitative assessments on public real X-ray datasets further demonstrated the model’s adaptability to unseen and out-of-domain data. The generated lung X-rays from the VinDr-CXR test set had a Fréchet Inception Distance (FID) score of 0.0112, indicating a high similarity between the generated images and the ground truth distribution.  Additionally, qualitative assessments of the clinical usefulness of lung X-rays were conducted by 10 radiologists from multiple hospitals in Bangladesh. The radiologists significantly agreed that the lung X-ray images enhanced the visibility of lung features and are helpful in diagnosis when used alongside conventional radiographs. This simple yet powerful GAN-based approach addresses a long-standing challenge in medical imaging. OrGAN offers a novel solution for multi-organ image separation from projection radiographs, paving the way for organ-focused disease diagnosis using X-rays.
</p>

# Proposed Architecture: OrGAN
![Architecture](images/OrGAN.png)

# Datasets

For OrGAN training:
  We have used real X-ray data from the publicly available VinBigDr-CXR dataset: [Link](https://vindr.ai/datasets/cxr) </br>
  Additionally, we have created a simulated dataset of chest X-rays with lung (label) from the publicly available LUNA16 CT scan dataset: [Link](https://luna16.grand-challenge.org/Download/)</br>

The experiments are conducted on three publicly available datasets, </br>
VinDr-CXR test set : [Link](https://vindr.ai/datasets/cxr)</br>
National Institutes of Health (NIH) Chest X-ray Dataset : [Link](https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset)</br>
FracAtlas Dataset : [Link](https://figshare.com/articles/dataset/The_dataset/22363012?file=43283628)</br>

# Dataset Preparation
1) Use the CT2Xray-process.ipynb to generate simulated dataset from LUNA16 CT scans for training OrGAN.
2) Use the VinBiG-process.ipynb to process the real X-ray dicom files for training OrGAN.

# Train OrGAN
For training OrGAN, use the following steps:
1) After preparing the simulated dataset in the previous step, split the data into train and test and move them to the OrGAN/data/Train and OrGAN/data/Test folders.
2) After preparing the real dataset (VinDr-CXR train set), move the data to OrGAN/data/Train/Xray folder.
3) Finally, run the OrGAN/train.ipynb

# Trained Model Weight
[Drive Link](#) Will be uploaded soon ...

# Inference on Real X-ray:
For inference on chest X-rays: 
1) Download the model weight and place it inside folder: "OrGAN/model_weights/1/".
2) Place some real X-ray dicom files inside folder: "OrGAN/data/Xray_real/real/"
3) Run the OrGAN/inferenceRealX.ipynb.

# Comparison 
![Comparison](images/Video.gif)

# Citing OrGAN
