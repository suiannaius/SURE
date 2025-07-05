import torch
import math
import numpy as np


def spearman_correlation(x, y):
    x_rank = x.argsort().argsort()
    y_rank = y.argsort().argsort()
    corr_matrix = np.corrcoef(x_rank, y_rank)
    return corr_matrix[0, 1]


def count_pixels_sigma(u_noised_sigma1, u_noised_sigma2, sigma1, sigma2, d, device, threshold=4):
    u_noised_sigma1, u_noised_sigma2 = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u_noised_sigma1, u_noised_sigma2])
    delta_u = u_noised_sigma2[d <= threshold] - u_noised_sigma1[d <= threshold ]  # [NHW,]
    delta_sigma = sigma2 - sigma1  # [1,]
    mul = delta_u * delta_sigma  # [NHW,]
    num = (mul > 0).sum()
    ratio = num / np.sum(d <= threshold)
    check_ratio(ratio)
    return ratio


def count_corr_sigma(u_noised_sigma1, u_noised_sigma2, sigma1, sigma2, d, device, threshold=4):
    u_noised_sigma1, u_noised_sigma2 = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u_noised_sigma1, u_noised_sigma2])
    u_sigma1_all = (u_noised_sigma1[d <= threshold])
    u_sigma2_all = (u_noised_sigma2[d <= threshold])
    sigma_np = np.array([sigma1, sigma2])
    corr = 0.
    for i in range(len(u_sigma1_all)):
        u_np = np.array([u_sigma1_all[i].cpu().float(), u_sigma2_all[i].cpu().float()])
        corr += spearman_correlation(u_np, sigma_np)
    corr /= len(u_sigma1_all)
    return corr


def count_pixels_d(u, u_noised, d, batch_size, device, threshold=4):
    u, u_noised, d = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u, u_noised, d])
    u = u.view(batch_size, -1)  # [N, HW]
    u_noised = u_noised.view(batch_size, -1)  # [N, HW]
    d = d.view(batch_size, -1)  # [N, HW]
    ratio = 0.0
    corr = 0.0
    for i in range(batch_size):
        u_boundary = u[i, :][d[i, :] <= threshold]  # [3504] [12988]
        u_noised_boundary = u_noised[i, :][d[i, :] <= threshold]  # [3504]
        d_boundary = d[i, :][d[i, :] <= threshold]  # [3504]
        L = len(d_boundary)
        delta_u = u_noised_boundary - u_boundary  # [3504]
        corr += spearman_correlation(d_boundary.cpu().numpy(), delta_u.cpu().numpy())
        d_diff = torch.triu(d_boundary.unsqueeze(0) - d_boundary.unsqueeze(0).t(), diagonal=0)
        delta_u_diff = torch.triu(delta_u.unsqueeze(0) - delta_u.unsqueeze(0).t(), diagonal=0)
        mul = d_diff * delta_u_diff
        num = (mul < 0).sum() + ((d_diff == 0) & (delta_u_diff == 0)).sum() - (L * (L + 1) / 2)
        ratio += num / (L * (L - 1) / 2)
        check_ratio(ratio)
    return ratio / batch_size, corr / batch_size


def count_pixels_grad(u, grad, d, batch_size, device, threshold=1, interval=0.04):
    u, grad, d = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u, grad, d])
    u = u.view(batch_size, -1)  # [N, HW]
    grad = grad.view(batch_size, -1)  # [N, HW]
    d = d.view(batch_size, -1)  # [N, HW]
    ratio = 0.0
    corr = 0.0
    for i in range(batch_size):
        u_boundary = u[i, :][d[i, :] <= threshold]  # [3504]
        grad_boundary = grad[i, :][d[i, :] <= threshold]  # [3504]
        L = len(u_boundary)
        if L == 0 or L == 1:
            continue
        
        grad_sorted, idx_sorted = torch.sort(grad_boundary)
        u_sorted = u_boundary[idx_sorted]
        g_min, g_max = grad_sorted[0], grad_sorted[-1]
        
        target_vals = torch.arange(g_min, g_max + interval, interval, device=device)  # [S]
        diff = torch.abs(grad_sorted.unsqueeze(1) - target_vals.unsqueeze(0))  # [L, S]
        idx_nearest = torch.argmin(diff, dim=0)  # [S]
        idx_sample = torch.unique(idx_nearest)
        u_boundary = u_sorted[idx_sample]
        grad_boundary = grad_sorted[idx_sample]
        L = len(u_boundary)

        corr_i = spearman_correlation(u_boundary.cpu().numpy(), grad_boundary.cpu().numpy())
        u_diff = torch.triu(u_boundary.unsqueeze(0) - u_boundary.unsqueeze(0).t(), diagonal=0)
        grad_diff = torch.triu(grad_boundary.unsqueeze(0) - grad_boundary.unsqueeze(0).t(), diagonal=0)
        
        mul = u_diff * grad_diff
        num = (mul < 0).sum() + ((grad_diff == 0) & (u_diff == 0)).sum() - (L * (L + 1) / 2)
        ratio += num / (L * (L - 1) / 2)
        check_ratio(ratio)
        corr += corr_i
    return ratio / batch_size, corr / batch_size


def count_pixels_d_chunk(u, u_noised, d, batch_size, device, threshold=4, chunk_size=100):
    u, u_noised, d = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u, u_noised, d])
    u = u.view(batch_size, -1)  # [N, HW]
    u_noised = u_noised.view(batch_size, -1)  # [N, HW]
    d = d.view(batch_size, -1)  # [N, HW]

    ratio = 0.0
    corr = 0.0
    for i in range(batch_size):
        u_boundary = u[i, :][d[i, :] <= threshold]  # [3504]
        u_noised_boundary = u_noised[i, :][d[i, :] <= threshold]  # [3504]
        d_boundary = d[i, :][d[i, :] <= threshold]  # [3504]
        L = len(d_boundary)
        
        delta_u = u_noised_boundary - u_boundary  # [3504]
        corr += spearman_correlation(d_boundary.cpu().numpy(), delta_u.cpu().numpy())

        num = 0.0
        loop = math.ceil(L / chunk_size)
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            N = end - start
            if N == 0 or N == 1:
                continue
            else:
                d_chunk = torch.triu(d_boundary[start:end].unsqueeze(0) - d_boundary[start:end].unsqueeze(0).t(), diagonal=0)  # [chunk_size, chunk_size]
                delta_u_chunk = torch.triu(delta_u[start:end].unsqueeze(0) - delta_u[start:end].unsqueeze(0).t(), diagonal=0)  # [chunk_size, chunk_size]
                mul = d_chunk * delta_u_chunk  # [chunk_size, chunk_size]
                num = (mul < 0).sum() + ((d_chunk == 0) & (delta_u_chunk == 0)).sum() - (N * (N + 1) / 2)
                check_ratio(num / (N * (N - 1) / 2))
                ratio += num / (N * (N - 1) / 2) / loop
    return ratio / batch_size, corr / batch_size


def check_ratio(ratio):
    if not (0 <= ratio <= 1):
        raise ValueError(f"ratio must be between 0 and 1 (inclusive), but got {ratio}")