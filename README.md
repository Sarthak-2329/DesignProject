# Optimization of CNN Using Transfer Learning

A deterministic, resource-efficient transfer learning pipeline built on **VGG16**, optimized for image classification on **Fashion-MNIST**. The project focuses not only on accuracy but also on **reproducibility and stable inference latency**, making it suitable for real-time and embedded AI deployment.

## 📌 Overview

Modern CNN architectures (VGG, ResNet, Inception) deliver high accuracy but are computationally expensive, limiting their use on edge devices, IoT systems, and embedded platforms. This project addresses two challenges simultaneously:

1. **High Accuracy** — via transfer learning and targeted optimization techniques.
2. **Deterministic, Stable Inference** — via controlled threading, fixed random seeds, and capped GPU memory usage.

The result is a VGG16-based model with a lightweight custom CNN head that achieves **93.69% test accuracy** while maintaining **stable ~21 ms inference latency** under a **2 GB VRAM cap**.

## 🏗️ Architecture

- **Backbone:** Pre-trained VGG16 (ImageNet weights)
  - Early convolutional blocks frozen
  - `block4` and `block5` fine-tuned for the target domain
- **Custom Classifier Head:**
  - 512-unit Dense layer
  - Batch Normalization
  - ReLU activation
  - Dropout (0.5)
  - 10-class Softmax output

## 📊 Dataset

- **Fashion-MNIST** — 70,000 grayscale images (28×28)
- Preprocessing:
  - Normalized
  - Resized to 48×48
  - Converted to RGB (3-channel) to match VGG16 input requirements
- **Data Augmentation:** horizontal flips, rotations (0°, 90°, 180°, 270°), brightness perturbations

## ⚙️ Training Configuration

- **Optimizer:** Adam (learning rate = 1e-4)
- **Callbacks:**
  - `ReduceLROnPlateau` (factor 0.5, monitors validation accuracy)
  - `EarlyStopping`
- **Reproducibility controls:**
  - Fixed random seeds
  - Controlled TensorFlow thread parallelism
  - GPU memory capped at 2 GB

## 🚀 Optimization Pipeline

Each technique was applied incrementally, with measurable accuracy gains at each stage:

| Stage | Technique / Configuration | Accuracy (%) |
|-------|----------------------------|--------------|
| 1 | Baseline CNN (no transfer learning) | 89.79 |
| 2 | + Transfer Learning (VGG16 pretrained) | 91.69 |
| 3 | + Fine-tuning VGG block4 & block5 | 92.29 |
| 4 | + Data Augmentation (flip, rotation, brightness) | 92.79 |
| 5 | + LR Scheduling (ReduceLROnPlateau) | 92.99 |
| 6 | + BatchNorm & Dropout (Regularized Head) | 93.19 |
| 7 | + Test-Time Augmentation (8-view TTA) | **93.69** |

### Test-Time Augmentation (TTA)

At inference, multiple stochastic transformations (flips, rotations) are applied to each test image, and predictions are averaged — improving robustness without retraining.

| Evaluation Type | Test Accuracy (%) |
|------------------|--------------------|
| Standard (No TTA) | 93.05 |
| 8-View TTA | **93.69** |

## 📈 Results

- **Training Accuracy:** 96.68%
- **Validation Accuracy:** 93.05%
- **Final Test Accuracy (with TTA):** 93.69%

### Resource Profiling (Under 2 GB VRAM Cap)

| Metric | Mean | Std | Median | P90 |
|--------|------|-----|--------|-----|
| Latency (ms) | 21.1 | 0.2 | 21.1 | 21.4 |
| GPU Utilization (%) | 51.7 | 8.0 | 51.0 | 58.0 |
| CPU Utilization (%) | 55.7 | 8.3 | 51.8 | 60.0 |
| VRAM (MB) | 2157 | 0.0 | 2157 | 2157 |
| RAM (%) | 38.5 | 0.2 | 38.6 | 39.0 |

The system runs at an average of **~48 FPS** with minimal variance — confirming suitability for real-time applications requiring predictable latency.

## 🔑 Key Techniques

- **Transfer Learning** — Reuse of low-level ImageNet features (edges, textures) to accelerate convergence and reduce overfitting.
- **Selective Fine-Tuning** — Unfreezing `block4` and `block5` to adapt deeper, task-specific representations.
- **Data Augmentation** — Improves generalization to unseen orientations and lighting conditions.
- **Learning Rate Scheduling** — Adaptive LR reduction on validation plateau for smoother convergence.
- **Regularization** — Batch Normalization + Dropout to reduce overfitting in the classifier head.
- **Test-Time Augmentation** — Ensemble-style prediction averaging at inference for a robustness boost.
- **Deterministic Inference** — Fixed seeds, capped GPU memory, and controlled threading for reproducible, stable performance.

## 🔮 Future Work

- Adaptive quantization strategies for further efficiency
- Exploration of more compact architectures (e.g., **MobileNet**, **EfficientNet-Lite**) for deployment on highly constrained hardware

## 👥 Authors

- Surale Sarthak Dattatray (202351143)
- Raunak Anand (202351120)
- Chavda Himesh Ravjibhai (202351022)
- Vaghela Ronak Nandikishorbhai (202351152)

## 📚 References

1. Pan, S. J., & Yang, Q. — *A Survey on Transfer Learning*, IEEE TKDE, 2010
2. Yosinski, J. et al. — *How transferable are features in deep neural networks?*, NeurIPS, 2014
3. Howard, J., & Ruder, S. — *Universal Language Model Fine-Tuning for Text Classification (ULMFiT)*, ACL, 2018
4. Raghu, M. et al. — *Transfusion: Understanding Transfer Learning for Medical Imaging*, NeurIPS, 2019
5. Tan, M., & Le, Q. V. — *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML, 2019
6. TensorFlow — *Deterministic Operations*, 2023
7. Psutil Library — *System and Process Utilities*, [github.com/giampaolo/psutil](https://github.com/giampaolo/psutil), 2024
