import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric


def nu_loss_mu(u, u_noised_mu1, u_noised_mu2, mu1, mu2, d, indexes, device, threshold=4, mask=None):
    u, u_noised_mu1, u_noised_mu2 = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u, u_noised_mu1, u_noised_mu2])
    batch_size = len(indexes)
    patch_num = len(indexes[0])
    noised_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    mu1_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    mu2_loss_batch = torch.zeros((batch_size, patch_num), device=device)
    for i in range(batch_size):
        for j in range(patch_num):
            y, x = indexes[i][j]  # Note: (y, x) order

            d_mask = (d[i,j] <= threshold).float()

            # noised patches
            delta_u_noised = u_noised_mu2[i, :, y, x] - u_noised_mu1[i, :, y, x]
            delta_mu = torch.tensor(mu2 - mu1, device=device)

            delta_u_mu1 = u_noised_mu1[i, :, y, x] - u[i, :, y, x]
            delta_u_mu2 = u_noised_mu2[i, :, y, x] - u[i, :, y, x]

            if mask is not None:
                noised_loss_batch[i, j] = (delta_u_noised * delta_mu * d_mask) * mask[i, :, y, x]
                mu1_loss_batch[i, j] = (delta_u_mu1 * mu1 * d_mask) * mask[i, :, y, x]

                
                mu2_loss_batch[i, j] = (delta_u_mu2 * mu2 * d_mask) * mask[i, :, y, x]
            else:
                noised_loss_batch[i, j] = delta_u_noised * delta_mu * d_mask
                # original patch - mu1 patch
                # original patch - mu2 patch
                mu1_loss_batch[i, j] = delta_u_mu1 * mu1 * d_mask
                mu2_loss_batch[i, j] = delta_u_mu2 * mu2 * d_mask

    loss = torch.tensor(0.0).to(device)
    mask_noised = (noised_loss_batch < 0).float()
    if torch.sum(mask_noised) != 0:
        loss += torch.sum(mask_noised * noised_loss_batch) / torch.sum(mask_noised)

    mask_mu1 = (mu1_loss_batch < 0).float()
    if torch.sum(mask_mu1) != 0:
        loss += torch.sum(mask_mu1 * mu1_loss_batch) / torch.sum(mask_mu1)

    mask_mu2 = (mu2_loss_batch < 0).float()
    if torch.sum(mask_mu2) != 0:
        loss += torch.sum(mask_mu2 * mu2_loss_batch) / torch.sum(mask_mu2)

    return - loss


def nu_loss_d(u, u_noised_mu1, u_noised_mu2, d, indexes, device, threshold=4, mask=None):
    u, u_noised_mu1, u_noised_mu2 = map(lambda x: torch.tensor(x, device=device) if isinstance(x, np.ndarray) else x.to(device), [u, u_noised_mu1, u_noised_mu2])
    batch_size = len(indexes)
    patch_num = len(indexes[0])
    loss = torch.tensor(0.0).to(device)
    for i in range(batch_size):
        delta_u1 = torch.zeros(patch_num, device=device)
        delta_u2 = torch.zeros(patch_num, device=device)
        for j in range(patch_num):
            if d[i, j] <= threshold:
                y, x = indexes[i][j]
                if mask is not None and mask[i, :, y, x] == 0:
                    # 当前标签不置信
                    continue
                delta_u1[j] = u_noised_mu1[i, :, y, x] - u[i, :, y, x]
                delta_u2[j] = u_noised_mu2[i, :, y, x] - u[i, :, y, x]
            else:
                # 距离太远，直接delta_u置0
                continue

        # 获取上三角不包括对角线
        d_diff = torch.triu(d[i,:].unsqueeze(0) - d[i,:].unsqueeze(0).t(), diagonal=0)
        delta_u1_diff = torch.triu(delta_u1.unsqueeze(0) - delta_u1.unsqueeze(0).t(), diagonal=0)
        delta_u2_diff = torch.triu(delta_u2.unsqueeze(0) - delta_u2.unsqueeze(0).t(), diagonal=0)

        delta1 = d_diff * delta_u1_diff
        delta2 = d_diff * delta_u2_diff

        mask1 = (delta1 > 0).float()
        mask2 = (delta2 > 0).float()

        if torch.sum(mask1) != 0:
            loss += torch.sum(delta1 * mask1) / torch.sum(mask1)
        if torch.sum(mask2) != 0:
            loss += torch.sum(delta2 * mask2) / torch.sum(mask2)

    loss /= batch_size
    
    return loss


def nu_loss_far(u, u_noised_mu1, u_noised_mu2, distance_map, device, threshold=4, mask=None):
    distance_map_tensor = torch.from_numpy(distance_map).to(device).unsqueeze(1)
    d_mask = (distance_map_tensor > threshold).float()
    if mask is not None:
        d_mask *= mask.squeeze(1)
    loss = torch.sum((u + u_noised_mu1 + u_noised_mu2) * d_mask) / torch.sum(d_mask)

    return loss


def gradient_loss(u, g, distance_map, batch_size, device, delta=1, sample_size=None):
    # batch-wise
    N = batch_size
    g = g.reshape(N, -1)
    u = u.reshape(N, -1)
    distance_map = distance_map.reshape(N, -1)

    loss = torch.tensor(0.0, device=device)
    for b in range(N):
        g_batch = g[b,:]
        u_batch = u[b,:]
        distance_map_batch = distance_map[b,:]

        # We can get the 'boundary' according to distance_map. (The region where d <= delta is defined as 'boundary'.)
        gradient_on_boundary_batch = g_batch[distance_map_batch <= delta].to(device)
        uncertainty_on_boundary_batch = u_batch[distance_map_batch <= delta].to(device)

        del g_batch, u_batch, distance_map_batch
        torch.cuda.empty_cache()

        n = len(gradient_on_boundary_batch)
        if n < 2:
            # If there are less than 2 samples, the loss is zero
            loss += torch.tensor(0.0, device=device)
            continue
        if sample_size is None:
            sample_size = n

        # Randomly sample
        total_pairs = n * (n - 1) // 2
        sample_size = min(sample_size, total_pairs)  # Avoid sampling more pairs than exist
        sampled_i = torch.randint(0, n, (sample_size,), device=device)
        sampled_j = torch.randint(0, n, (sample_size,), device=device)

        mask_valid = sampled_i != sampled_j
        sampled_i = sampled_i[mask_valid]
        sampled_j = sampled_j[mask_valid]

        batch_size = 20000  # Based on available GPU memory.

        ranking_loss = torch.tensor(0.0, device=device)

        # Process batches
        for start_idx in range(0, len(sampled_i), batch_size):
            end_idx = min(start_idx + batch_size, len(sampled_i))
            batch_i = sampled_i[start_idx:end_idx]
            batch_j = sampled_j[start_idx:end_idx]

            batch_gradient_diff = gradient_on_boundary_batch[batch_i] - gradient_on_boundary_batch[batch_j]
            batch_uncertainty_diff = uncertainty_on_boundary_batch[batch_i] - uncertainty_on_boundary_batch[batch_j]

            product = batch_gradient_diff * batch_uncertainty_diff
            mask = (product > 0).float()
            if torch.sum(mask) == 0:
                pass
            else:
                ranking_loss += torch.sum(product * mask) / torch.sum(mask)
            torch.cuda.empty_cache()
        loss += ranking_loss
    loss /= N

    return loss


def Dice(y_true, y_pred, epsilon, device):
    """
    Input tensor:
    y_true: [NHWD, 4]
    y_pred: [NHWD, 4]
    alpha: [NHWD, 4]
    """
    y_true = y_true.to(device) # [NHWD, C]
    y_pred = y_pred.to(device) # [NHWD, C]
    smooth = torch.zeros(y_pred.size(-1), dtype=torch.float32).fill_(0.00001).to(device) # [C, ]
    ones = torch.ones(y_pred.shape).to(device) # [NHWD, C]
    class_mask = y_true + epsilon
    P = y_pred
    P_ = ones - P
    class_mask_ = ones - class_mask
    TP = P * class_mask
    FP = P * class_mask_
    FN = P_ * class_mask

    A = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)
    A = torch.clamp(A, min=0.2, max=0.8)
    B = 1 - A
    num = torch.sum(TP, dim=(0)).float()
    den = num + A * torch.sum(FP, dim=(0)).float() + B * torch.sum(FN, dim=(0)).float()
    dice = num / (den + smooth) # (C, )

    return dice

def DiceLoss(y_true, y_pred, epsilon, device):
    return 1 - Dice(y_true, y_pred, epsilon, device)

def KL(alp, c, device):
    assert torch.all(alp >= 1), "alp needs to be greater than or equal to 1."
    alp = alp.to(device) # [NHWD,C]
    S_alp = torch.sum(alp, dim=1, keepdim=True) # [NHWD,1]
    beta = torch.ones((1, c)).to(device) # [1,C]
    S_beta = torch.sum(beta, dim=1, keepdim=True) # [1,1]
    lnB = torch.lgamma(S_alp) - torch.sum(torch.lgamma(alp), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alp)
    dg1 = torch.digamma(alp)
    kl = torch.sum((alp - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni

    return kl

def edl_loss(y_true, alpha, num_classes, current_epoch, total_epoch, annealing_steps, device, loss_type):
    assert loss_type in ['log', 'digamma', 'mse'], f"'loss_type' should be 'log', 'digamma' or 'mse', but got {loss_type}."
    y_true = y_true.to(device) # [NHWD, 4]
    alpha = alpha.to(device) # [NHWD, 4]
    S = torch.sum(alpha, dim=-1, keepdim=True) # [NHWD, 1]
    prob = alpha / S # [NHWD, 4]
    if loss_type == 'log':
        L_ACE = torch.sum(torch.mul(y_true, (torch.log(S) - torch.log(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'digamma':
        L_ACE = torch.sum(torch.mul(y_true, (torch.digamma(S) - torch.digamma(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'mse':
        L_err = torch.sum((y_true - prob) ** 2, dim=-1, keepdim=True) # [NHWD, 1]
        L_var = alpha * (S - alpha) / (S * S * (S + 1)) # [NHWD, 4]
        L_ACE = torch.sum(L_err + L_var, dim=-1, keepdim=True) # [NHWD, 1]
    alp = (alpha - 1) * (1 - y_true) + 1 # [NHWD, 4]
    annealing_coef = min(1, current_epoch / annealing_steps)
    L_KL = KL(alp, num_classes, device) # [NHWD, 1]
    L_DICE = DiceLoss(y_true, prob, epsilon=1e-3, device=device) # [C,]
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_epoch * current_epoch)  # from 'annealing_start' to '1'
    loss = torch.mean(L_ACE  + annealing_coef * L_KL + (1 - annealing_AU)*L_DICE)
    L_ACE = torch.mean(L_ACE)
    L_KL = torch.mean(L_KL)
    L_DICE = torch.mean(L_DICE)

    return loss, L_ACE, L_KL, L_DICE


def edl_loss_a(y_true, alpha, num_classes, current_epoch, total_epoch, annealing_steps, device, loss_type):
    assert loss_type in ['log', 'digamma', 'mse'], f"'loss_type' should be 'log', 'digamma' or 'mse', but got {loss_type}."
    y_true = y_true.to(device) # [NHWD, 4]
    alpha = alpha.to(device) # [NHWD, 4]
    S = torch.sum(alpha, dim=-1, keepdim=True) # [NHWD, 1]
    prob = alpha / S # [NHWD, 4]
    if loss_type == 'log':
        L_ACE = torch.sum(torch.mul(y_true, (torch.log(S) - torch.log(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'digamma':
        L_ACE = torch.sum(torch.mul(y_true, (torch.digamma(S) - torch.digamma(alpha))), dim=-1, keepdim=True) # [NHWD, 1]
    elif loss_type == 'mse':
        L_err = torch.sum((y_true - prob) ** 2, dim=-1, keepdim=True) # [NHWD, 1]
        L_var = alpha * (S - alpha) / (S * S * (S + 1)) # [NHWD, 4]
        L_ACE = torch.sum(L_err + L_var, dim=-1, keepdim=True) # [NHWD, 1]
    alp = (alpha - 1) * (1 - y_true) + 1 # [NHWD, 4]
    annealing_coef = min(1, current_epoch / annealing_steps)
    L_KL = KL(alp, num_classes, device) # [NHWD, 1]
    L_DICE = DiceLoss(y_true, prob, epsilon=1e-3, device=device) # [C,]

    loss = torch.mean(10*L_ACE  + annealing_coef * L_KL + 10*L_DICE)
    L_ACE = torch.mean(L_ACE)
    L_KL = torch.mean(L_KL)
    L_DICE = torch.mean(L_DICE)

    return loss, L_ACE, L_KL, L_DICE

def Hausdorff_Distance(y_true, y_pred, device):
    y_true = y_true.to(device)
    y_pred = y_pred.to(device)
    hd_95 = HausdorffDistanceMetric(include_background=True, reduction='none', percentile=95)
    hd_95(y_pred, y_true)
    result = hd_95.aggregate()

    return result
    