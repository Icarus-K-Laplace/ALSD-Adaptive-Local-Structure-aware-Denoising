# ALSD-Adaptive-Local-Structure-aware-Denoising
ALSD-Framework: A white-box image denoising system based on robust statistics and Wiener filtering. It features automated noise estimation (MAD), structure-aware feature extraction, and local SNR-guided fusion to balance noise suppression and detail preservation without deep learning or manual tuning.
# ALSD: Adaptive Local Structure-aware Denoising

### A "White-Box" Denoising Framework Based on Robust Statistics and Signal Processing

> **Developed independently by an undergraduate student.**
> Unlike trends that blindly apply Deep Learning to every problem, this project revisits classical Signal Processing. It aims to solve the "Denoising vs. Detail Preservation" trade-off using **mathematically explainable** and **training-free** methods.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 💡 Design Philosophy (My Thinking Process)

During my research on infrared image restoration, I identified three limitations in standard engineering filters:
1.  **Manual Parameter Tuning**: Methods like Gaussian/Bilateral require manually setting `sigma`, which is impractical for varying scenes.
2.  **Edge Blurring**: Isotropic filters kill details; standard anisotropic filters often leave "staircase" artifacts.
3.  **Lack of Adaptability**: They treat flat regions and textured regions with the same kernel.

**My Solution:**
I designed ALSD based on the **Wiener Filtering principle** (Minimum Mean Square Error). Instead of learning from data (CNNs), ALSD learns from the **image's own statistics** in real-time.

1.  **Robust Estimation**: Used **MAD (Median Absolute Deviation)** instead of Standard Deviation to estimate noise, making it robust against outliers and edges.
2.  **Local SNR**: Calculated pixel-wise Signal-to-Noise Ratio to dynamically adjust the fusion weight.
3.  **Structure Awareness**: Guided the filtering strength using gradient magnitude.

---

## 📐 Methodological Core

The core update rule is derived from the Wiener Filter:

$$ \hat{I}(x) = w(x) \cdot I_{obs}(x) + (1 - w(x)) \cdot I_{filt}(x) $$

Where the adaptive weight $w(x)$ is determined by the local signal variance $\sigma_s^2$ and noise variance $\sigma_n^2$:

$$ w(x) = \frac{\sigma_s^2(x)}{\sigma_s^2(x) + \sigma_n^2} $$

*   **Flat Region** ($\sigma_s \to 0$) $\Rightarrow w \to 0$ (Strong Denoising)
*   **Edge Region** ($\sigma_s \gg \sigma_n$) $\Rightarrow w \to 1$ (Detail Preservation)

*(See `docs/mathematical_model.md` for full derivation)*

---

## 📊 Performance

Tested on standard synthetic noise (Gaussian $\sigma=25$):

| Metric | Gaussian Filter | Bilateral Filter | **ALSD (Ours)** |
| :--- | :---: | :---: | :---: |
| **PSNR (dB)** | 28.12 | 29.45 | **30.82** |
| **SSIM** | 0.78 | 0.82 | **0.88** |
| **Explainability** | Low | Medium | **High (White-box)** |
| **Training** | N/A | N/A | **None (Zero-shot)** |

---

## 🚀 Usage

```bash
# 1. Install
pip install -r requirements.txt

# 2. Run Demo
python main.py
