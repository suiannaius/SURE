# Uncertainty-Supervised Interpretable and Robust Evidential Segmentation

This repository contains the official implementation of the **MICCAI 2025 paper**:  
**"Uncertainty-Supervised Interpretable and Robust Evidential Segmentation"**.

## 🧠 Overview

In this work, we propose a novel uncertainty supervision framework guided by human-intuitive principles. Instead of treating uncertainty as a by-product of prediction, we explicitly supervise the uncertainty maps using interpretable patterns derived from human reasoning. Additionally, we introduce new evaluation **metrics (UCC and UR)** to quantify the **interpretability and robustness of model uncertainty**.

---

## 📊 Metrics: UCC and UR

The proposed metrics are implemented in `utilities/count_pixels.py`, with the following functions:

- `count_pixels_d`
- `count_pixels_d_chunk`
- `count_pixels_mu`
- `count_pixels_grad`
- `count_corr_mu`

These functions are used to calculate the **Uncertainty Consistency Coefficient (UCC)** and **Uncertainty Ranking (UR)** metrics under different signal settings (e.g., gradient, distance map, mean prediction).

### 📌 Input Requirements

All input tensors for these functions — such as **uncertainty**, **gradient**, and **distance maps** — must be **flattened 1D tensors of shape `[N, 1]`**.

---

## 🧮 Metric Definitions

Let \( g_i \) be the value of some reference signal (e.g., distance or gradient) at pixel \( i \), and \( u_i \) be the predicted uncertainty. \( R(\cdot) \) denotes rank within a batch \( B \), and \( \overline{R(\cdot)} \) is the average rank. The formulas for the metrics are:

### UCC (Uncertainty Consistency Coefficient)

\[
\text{UCC}_{[g]} = \frac{\sum\limits_{i \in B} (R(g_i)-\overline{R(g)})(R(u_i)-\overline{R(u)})}{\sqrt{\sum\limits_{i \in B} (R(g_i)-\overline{R(g)})^2 \sum\limits_{i \in B} (R(u_i)-\overline{R(u)})^2}}
\]

This measures the **rank correlation** between the uncertainty and reference signal \( g \).

---

### UR (Uncertainty Ranking)

\[
\text{UR}_{[g]} = \frac{\sum\limits_{i,j \in B, i\neq j}\mathbbm{1}_{((g_i-g_j)(u_i-u_j)\leq 0)}}{\sum\limits_{i,j \in B}\mathbbm{1}_{(i\neq j)}}
\]

This captures the **pairwise ranking consistency** between the reference signal \( g \) and predicted uncertainty \( u \).
