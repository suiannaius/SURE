# Uncertainty-Supervised Interpretable and Robust Evidential Segmentation

This repository contains the official implementation of the **MICCAI 2025 paper**:  
**"Uncertainty-Supervised Interpretable and Robust Evidential Segmentation"**.

## 🧠 Overview

In this work, we propose a novel uncertainty supervision framework guided by human-intuitive principles. Instead of treating uncertainty as a by-product of prediction, we explicitly supervise the uncertainty using interpretable patterns derived from human reasoning. Additionally, we introduce new evaluation **metrics (UCC and UR)** to evaluate the **interpretability and robustness of model uncertainty**.

---

## 📊 Metrics: UCC and UR

The proposed metrics are implemented in `utilities/count_pixels.py`, with the following functions:

- `count_pixels_d`
- `count_pixels_d_chunk`
- `count_pixels_mu`
- `count_pixels_grad`
- `count_corr_mu`

These functions are used to calculate the **Uncertainty Correlation Coefficient (UCC)** and **Uncertainty Ratio (UR)** metrics.

### 📌 Input Requirements

All input tensors for these functions — such as **uncertainty**, **gradient**, and **distance maps** — must be **flattened 1D tensors of shape `[N, 1]`**.
