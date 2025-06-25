import torch
import math
import numpy as np
import surface_distance as surfdist
import torch.nn.functional as F
import utilities.numpyfunctions as np_fn
from mindspore.nn.metrics import MeanSurfaceDistance
from utilities.color import apply_color_map
from utilities import binary


def hd_score(y_true, y_pred, eps_max=100, eps=1e-8, voxelspacing=None):
    if (y_true.sum()==0) and (y_pred.sum()==0):
        hd = eps
    elif (y_true.sum()!=0) and (y_pred.sum()==0):
        hd = eps_max
    elif (y_true.sum()==0) and (y_pred.sum()!=0):
        hd = eps
    else:
        hd = binary.hd95(y_true, y_pred, voxelspacing=voxelspacing)
    return hd


def soft_hd95(y_true, y_pred, voxelspacing=None, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    hd = np.zeros(num_classes - 1)
    
    if voxelspacing is not None:
        assert voxelspacing.shape[0]==y_true.shape[0] and voxelspacing.shape[1]==2, f"Check the dimension of spacing."
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t, voxelspacing=voxelspacing[i])
    else:
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t)
    return hd / y_true.shape[0]


def SDice(y_true, y_pred, epsilon, device=torch.device('cuda')):
    y_true = y_true.to(device) # [NHWD, 1]
    y_pred = y_pred.to(device) # [NHWD, 1]
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice_coef = 2 * intersection / (union + epsilon) # [1]
    return dice_coef


def calculate_dice(y_true, y_pred, epsilon, device=torch.device('cuda'), num_classes=None):
    y_true_soft = torch.argmax(y_true, axis=1, keepdims=True)
    y_pred_soft = torch.argmax(y_pred, axis=1, keepdims=True)
    dice = torch.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        dice[i] = SDice((y_pred_soft == i + 1).float(), (y_true_soft == i + 1).float(), epsilon, device) # [NHW, 1], [NHW, 1]
    return dice


def calculate_dice_Refuge(y_true, y_pred, epsilon, device=torch.device('cuda')):
    y_true_soft = torch.argmax(y_true, axis=1, keepdims=True)
    y_pred_soft = torch.argmax(y_pred, axis=1, keepdims=True)

    dice_DISC = SDice((y_pred_soft == 2).float(), (y_true_soft == 2).float(), epsilon, device) # [NHWD, 1], [NHWD, 1]
    dice_CUP = SDice((y_pred_soft != 0).float(), (y_true_soft != 0).float(), epsilon, device)
    return dice_DISC, dice_CUP


def _calculate_ueo(prob, targets, uncertainty, threshold):

    error = (prob != targets).astype(float) # [NHW, 1]
    UEO = []
    for t in threshold:
        thresholded_uncertainty = (uncertainty > t).astype(float) # [NHW, 1]
        ueo = np_fn.dice(error, thresholded_uncertainty)
        UEO.append(ueo)
    ueo = max(UEO)
    index = UEO.index(ueo)
    thresholded_uncertainty_used = (uncertainty > threshold[index]).astype(float)
    # print("uncertainty_threshold: ", threshold[index])
    return ueo, error, thresholded_uncertainty_used


def calculate_ueo(prob, targets, uncertainty, threshold, num_classes=None):
    """
    prob: [NHW, C]
    targets: [NHW, C]
    """
    prob = torch.argmax(prob, axis=1, keepdims=True) # [NHW, 1]
    targets = torch.argmax(targets, axis=1, keepdims=True) # [NHW, 1]
    
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()
    
    ueo = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        ueo[i], error, uncertainty_thresholded = _calculate_ueo((prob == i + 1).astype(float), (targets == i + 1).astype(float), uncertainty, threshold)

    return ueo


def calculate_ueo_Refuge(prob, targets, uncertainty, threshold, batch_idx=None, batch_size=None, writer=None, img_shape=None):
    """
    prob: [NHWD, C]
    targets: [NHWD, C]
    """
    prob = torch.argmax(prob, axis=1, keepdims=True) # [NHWD, 1]
    targets = torch.argmax(targets, axis=1, keepdims=True) # [NHWD, 1]
    
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()
    
    ueo_CUP, _, _ = _calculate_ueo((prob == 2).astype(float), (targets == 2).astype(float), uncertainty, threshold)
    ueo_DISC, _, _ = _calculate_ueo((prob != 0).astype(float), (targets != 0).astype(float), uncertainty, threshold)

    if writer is not None:
        pass

    return ueo_DISC, ueo_CUP


def _calculate_ece(prob, targets):
    # prob [NHWD, 1], targets [NHWD, 1], prob contains values range from 0 to 1, targets are binary
    ece = np_fn.ece_binary(prob, targets)
    return ece


def calculate_ece(prob, targets, num_classes):
    """
    Input:
    prob: [NHWD, C]
    targets: [NHWD, C]
    """
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    ece = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        prob_ = prob[:, i]
        targets_ = targets[:, i]
        ece[i] = _calculate_ece(prob_, targets_)

    return ece


def calculate_ece_Refuge(prob, targets):
    """
    Input:
    prob: [NHWD, C]
    targets: [NHWD, C]
    """
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    prob_CUP = prob[:, 2]
    prob_DISC = prob[:, 1] + prob[:, 2]
    targets_CUP = targets[:, 2]
    targets_DISC = targets[:, 1] + targets[:,2]
    ece_CUP = _calculate_ece(prob_CUP, targets_CUP)
    ece_DISC = _calculate_ece(prob_DISC, targets_DISC)
    return ece_DISC, ece_CUP


def calculate_assd_and_hd95(y_pred, y_true, spacing, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    assd = np.zeros(num_classes - 1)
    hd95 = np.zeros(num_classes - 1)
    for i in range(y_true.shape[0]):
        for j in range(num_classes - 1):
            o = y_pred[i, ...] == j + 1
            t = y_true[i, ...] == j + 1
            sd = surfdist.compute_surface_distances(t, o, spacing_mm=spacing[i])
            assd[j] += surfdist.compute_average_surface_distance(sd)[1]  # pred to gt
            hd95[j] += surfdist.compute_robust_hausdorff(sd, 95)
    
    return assd / y_true.shape[0], hd95 / y_true.shape[0]


def Uentropy(logits, c):
    pred = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logits = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    u_all = -pred * logits / math.log(c)
    NU = torch.sum(u_all[:, 1:u_all.shape[1], :, :], dim=1)
    return NU


def soft_hd95(y_true, y_pred, voxelspacing=None, num_classes=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    hd = np.zeros(num_classes - 1)
    
    if voxelspacing is not None:
        assert voxelspacing.shape[0]==y_true.shape[0] and voxelspacing.shape[1]==2, f"Check the dimension of spacing."
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t, voxelspacing=voxelspacing[i])
    else:
        for i in range(y_true.shape[0]):
            for j in range(num_classes - 1):
                o = y_pred[i, ...] == j + 1
                t = y_true[i, ...] == j + 1
                hd[j] += hd_score(o, t)
    return hd / y_true.shape[0]


def soft_hd95_Refuge(y_true, y_pred, voxelspacing=None):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    hd95_DISC, hd95_CUP = 0.0, 0.0
    for i in range(y_true.shape[0]):
        # Whole Tumor (label: 1, 2, 3)
        o = y_pred[i, ...] > 0
        t = y_true[i, ...] > 0
        hd95_DISC += hd_score(o, t)

        o = (y_pred[i, ...] == 2)
        t = (y_true[i, ...] == 2)
        hd95_CUP += hd_score(o, t)
    return hd95_DISC / y_true.shape[0], hd95_CUP / y_true.shape[0]


def calculate_dice_Refuge(y_true, y_pred, epsilon, device=torch.device('cuda')):
    y_true_soft = torch.argmax(y_true, axis=1, keepdims=True)
    y_pred_soft = torch.argmax(y_pred, axis=1, keepdims=True)

    dice_DISC = SDice((y_pred_soft == 2).float(), (y_true_soft == 2).float(), epsilon, device) # [NHWD, 1], [NHWD, 1]
    dice_CUP = SDice((y_pred_soft != 0).float(), (y_true_soft != 0).float(), epsilon, device)
    return dice_DISC, dice_CUP


def calculate_ueo_Refuge(prob, targets, uncertainty, threshold):
    """
    prob: [NHWD, C]
    targets: [NHWD, C]
    """
    prob = torch.argmax(prob, axis=1, keepdims=True) # [NHWD, 1]
    targets = torch.argmax(targets, axis=1, keepdims=True) # [NHWD, 1]
    
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    uncertainty = uncertainty.cpu().detach().numpy()
    
    ueo_CUP, _, _ = _calculate_ueo((prob == 2).astype(float), (targets == 2).astype(float), uncertainty, threshold)
    ueo_DISC, _, _ = _calculate_ueo((prob != 0).astype(float), (targets != 0).astype(float), uncertainty, threshold)

    return ueo_DISC, ueo_CUP


def calculate_ece_Refuge(prob, targets):
    """
    Input:
    prob: [NHWD, C]
    targets: [NHWD, C]
    """
    prob = prob.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    prob_CUP = prob[:, 2]
    prob_DISC = prob[:, 1] + prob[:, 2]
    targets_CUP = targets[:, 2]
    targets_DISC = targets[:, 1] + targets[:,2]
    ece_CUP = _calculate_ece(prob_CUP, targets_CUP)
    ece_DISC = _calculate_ece(prob_DISC, targets_DISC)
    return ece_DISC, ece_CUP
