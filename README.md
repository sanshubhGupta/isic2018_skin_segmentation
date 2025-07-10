# isic2018_skin_segmentation
Skin lesion segmentation using U-Net on ISIC 2018 dataset

# 🧬 Skin Lesion Segmentation using U-Net on ISIC 2018

This project performs binary segmentation of skin lesions using a U-Net architecture with a pretrained ResNet34 encoder. The model is trained and evaluated on the **ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection** dataset (Task 1).

<p align="center">
  <img src="outputs/example_preds.png" width="700"/>
</p>

---

## 📌 Project Overview

- **Goal:** Automatically segment skin lesions from dermoscopic images.
- **Dataset:** [ISIC 2018 - Task 1: Lesion Segmentation](https://challenge.isic-archive.com/landing/2018/)
- **Model:** U-Net with ResNet34 encoder from `segmentation_models_pytorch`
- **Frameworks:** PyTorch, Albumentations, Gradio (for demo)

---

## 🧠 Model Architecture

- U-Net architecture (encoder-decoder with skip connections)
- **Encoder:** Pretrained ResNet34 (ImageNet)
- **Input Size:** 256×256 RGB images
- **Output:** 1-channel binary segmentation mask
- **Loss Function:** BCEWithLogits + Dice Loss
- **Metric:** IoU (Jaccard Index)

---

## 📊 Results

### Model Predictions
![Prediction 0](outputs/predictions/sample_0.png)
![Prediction 1](outputs/predictions/sample_1.png)
![Prediction 2](outputs/predictions/sample_2.png)

...

## 🗂️ Folder Structure

