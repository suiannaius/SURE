import torch
import logging
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
from torch.nn.modules.loss import _Loss

def expand_target(x, n_class,mode='softmax'):
    """
        Converts NxDxHxW label image to NxCxDxHxW, where each label is stored in a separate channel
        :param input: 4D input image (NxDxHxW)
        :param C: number of channels/labels
        :return: 5D output image (NxCxDxHxW)
        """
    assert x.dim() == 4
    shape = list(x.size())
    shape.insert(1, n_class)
    shape = tuple(shape)
    xx = torch.zeros(shape)
    if mode.lower() == 'softmax':
        xx[:, 1, :, :, :] = (x == 1)
        xx[:, 2, :, :, :] = (x == 2)
        xx[:, 3, :, :, :] = (x == 3)
    if mode.lower() == 'sigmoid':
        xx[:, 0, :, :, :] = (x == 1)
        xx[:, 1, :, :, :] = (x == 2)
        xx[:, 2, :, :, :] = (x == 3)
    return xx.to(x.device)

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)

def Dice(output, target, eps=1e-5):
    target = target.float()
    num = 2 * (output * target).sum()
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn

def KL(alpha, c):
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    beta = torch.ones((1, c)).cuda()
    # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.permute(0,4,1,2,3)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

class DiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(DiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        else:
            preds = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)

        class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask.scatter_(1, targets, 1.)

        ones = torch.ones(preds.shape).to(preds.device)
        P_ = ones - P
        class_mask_ = ones - class_mask

        TP = P * class_mask
        FP = P * class_mask_
        FN = P_ * class_mask

        smooth = smooth.to(preds.device)
        self.alpha = FP.sum(dim=(0)) / ((FP.sum(dim=(0)) + FN.sum(dim=(0))) + smooth)

        self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
        #print('alpha:', self.alpha)
        self.beta = 1 - self.alpha
        num = torch.sum(TP, dim=(0)).float()
        den = num + self.alpha * torch.sum(FP, dim=(0)).float() + self.beta * torch.sum(FN, dim=(0)).float()

        dice = num / (den + smooth)

        if not self.reduce:
            loss = torch.ones(C).to(dice.device) - dice
            return loss
        loss = 1 - dice
        if weight is not False:
            loss *= weight.squeeze(0)
        loss = loss.sum()
        if self.size_average:
            if weight is not False:
                loss /= weight.squeeze(0).sum()
            else:
                loss /= C

        return loss
class SDiceLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
        super(SDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.size_average = size_average
        self.reduce = reduce

    def forward(self, preds, targets,weight_map=None):
        N = preds.size(0)
        C = preds.size(1)
        num_class = C
        if preds.ndim==5:
            preds = preds.permute(0, 2, 3, 4, 1)
        else:
            preds = preds.permute(0, 2, 3, 1)
        pred = preds.contiguous().view(-1, num_class)
        # pred = F.softmax(pred, dim=1)
        ground = targets.view(-1, num_class)
        n_voxels = ground.size(0)
        if weight_map is not None:
            weight_map = weight_map.view(-1)
            weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
            ref_vol = torch.sum(weight_map_nclass * ground, 0)
            intersect = torch.sum(weight_map_nclass * ground * pred, 0)
            seg_vol = torch.sum(weight_map_nclass * pred, 0)
        else:
            ref_vol = torch.sum(ground, 0)
            intersect = torch.sum(ground * pred, 0)
            seg_vol = torch.sum(pred, 0)
        dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
        # dice_loss = 1.0 - torch.mean(dice_score.data[1:dice_score.shape[0]])
        k = -torch.log(dice_score)
        # 1. mean-loss
        dice_mean_score = torch.mean(-torch.log(dice_score))
        # 2. sum-loss
        # dice_score1 = torch.sum(dice_score,0)
        # dice_mean_score1 = -torch.log(torch.sum(dice_score,0))
        # dice_mean_score = torch.mean(-torch.log(dice_score.data[1:dice_score.shape[0]]))
        return dice_mean_score
        # dice_score = torch.mean(-torch.log(dice_score))
        # return dice_score

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()

        if alpha is None:
            self.alpha = torch.ones(class_num, 1).cuda()
        else:
            self.alpha = alpha

        self.gamma = gamma
        self.size_average = size_average

    def forward(self, preds, targets, weight=False):
        N = preds.size(0)
        C = preds.size(1)

        preds = preds.permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        targets = targets.view(-1, 1)

        log_P = F.log_softmax(preds, dim=1)
        P = torch.exp(log_P)
        # P = F.softmax(preds, dim=1)
        # log_P = F.log_softmax(preds, dim=1)
        # class_mask = torch.zeros(preds.shape).to(preds.device) + 1e-8
        class_mask = torch.zeros(preds.shape).to(preds.device)  # problem
        class_mask.scatter_(1, targets, 1.)
        # number = torch.unique(targets)
        alpha = self.alpha[targets.data.view(-1)] # problem alpha: weight of data
        # alpha = self.alpha.gather(0, targets.view(-1))

        probs = (P * class_mask).sum(1).view(-1, 1)  # problem
        log_probs = (log_P * class_mask).sum(1).view(-1, 1)

        # probs = P.gather(1,targets.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        # log_probs = log_P.gather(1,targets.view(-1,1))

        batch_loss = -alpha * (1-probs).pow(self.gamma)*log_probs
        if weight is not False:
            element_weight = weight.squeeze(0)[targets.squeeze(0)]
            batch_loss = batch_loss * element_weight

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss

def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_score = torch.mean(-torch.log(dice_score))
    return dice_score

def TDice(output, target,criterion_dl):
    dice = criterion_dl(output, target)
    return dice

def U_entropy(logits,c):
    pc = F.softmax(logits, dim=1)
    logits = F.log_softmax(logits, dim=1)
    pc = pc.view(-1, 1)
    u = -pc* logits/c
    return u

def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    if input_tensor.ndim == 5:
        input_tensor = input_tensor.permute(0, 2, 3, 4, 1)
    else:
        input_tensor = input_tensor.permute(0, 2, 3, 1)
    # input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def dce_evidence_u_loss(p, alpha, c, current_step, lamda_step, total_step, eps, disentangle, evidence, backbone_pred):
    # c: class number
    criterion_dl = DiceLoss()
    # notes: may be use SDiceloss:
    # criterion_dl = SDiceLoss()

    # soft_p = get_soft_label(soft_p, c)
    if alpha.ndim == 5:
        soft_p = p.unsqueeze(1)
    else:
        soft_p = p

    L_dice = TDice(evidence, soft_p, criterion_dl)

    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    # KL loss
    annealing_coef = min(1, current_step / lamda_step)
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)
    # AU Loss
    pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
    uncertainty = c / S
    target = p.view(-1, 1)
    acc_match = torch.reshape(torch.eq(pred_cls, target).float(), (-1, 1))
    if disentangle:
        acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + eps)
        inacc_certain = - torch.log((1 - pred_scores) * uncertainty + eps)
    else:
        acc_uncertain = - pred_scores * torch.log(1 - uncertainty + eps)
        inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + eps)
    L_AU = annealing_AU * acc_match * acc_uncertain + (1 - annealing_AU) * (1 - acc_match) * inacc_certain

    return (L_ace  + L_KL + (1 - annealing_AU)*L_dice + L_AU)

def dce_evidence_loss(p, alpha, c, current_step, lamda_step, total_step,eps,disentangle,evidence,backbone_pred):


    criterion_dl = SDiceLoss()

    if alpha.ndim == 5:
        soft_p = p.unsqueeze(1)
        # soft_p = p
    else:
        soft_p = p

    soft_p = get_soft_label(soft_p,c) # for SDiceloss

    L_dice = TDice(evidence, soft_p, criterion_dl)

    alpha = alpha.view(alpha.size(0), alpha.size(1), -1)  # [N, C, HW]
    alpha = alpha.transpose(1, 2)  # [N, HW, C]
    alpha = alpha.contiguous().view(-1, alpha.size(2))
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    label = label.view(-1, c)
    # digama loss
    L_ace = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    # log loss
    # labelK = label * (torch.log(S) -  torch.log(alpha))
    # L_ace = torch.sum(label * (torch.log(S) -  torch.log(alpha)), dim=1, keepdim=True)

    # KL loss
    annealing_coef = min(1, current_step / lamda_step)
    annealing_start = torch.tensor(0.01, dtype=torch.float32)
    annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / total_step * current_step)
    alp = E * (1 - label) + 1
    L_KL = annealing_coef * KL(alp, c)

    return (L_ace + L_dice + L_KL)


def dce_loss(p, alpha, c, global_step, annealing_step):
    criterion_dl = DiceLoss()
    L_dice =  TDice(alpha,p,criterion_dl)

    return L_dice

def softmax_dice(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss torch.nn.BCELoss()
    '''

    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

def softmax_dice2(output, target):
    '''
    The dice loss for using softmax activation function
    :param output: (b, num_class, d, h, w)
    :param target: (b, d, h, w)
    :return: softmax dice loss
    '''
    loss0 = Dice(output[:, 0, ...], (target == 0).float())
    loss1 = Dice(output[:, 1, ...], (target == 1).float())
    loss2 = Dice(output[:, 2, ...], (target == 2).float())
    loss3 = Dice(output[:, 3, ...], (target == 3).float())

    return loss1 + loss2 + loss3 + loss0, 1-loss1.data, 1-loss2.data, 1-loss3.data


def sigmoid_dice(output, target):
    '''
    The dice loss for using sigmoid activation function
    :param output: (b, num_class-1, d, h, w)
    :param target: (b, d, h, w)
    :return:
    '''
    loss1 = Dice(output[:, 0, ...], (target == 1).float())
    loss2 = Dice(output[:, 1, ...], (target == 2).float())
    loss3 = Dice(output[:, 2, ...], (target == 3).float())

    return loss1 + loss2 + loss3, 1-loss1.data, 1-loss2.data, 1-loss3.data

def Generalized_dice(output, target, eps=1e-5, weight_type='square'):
    if target.dim() == 4:  #(b, h, w, d)
        target[target == 4] = 3  #transfer label 4 to 3
        target = expand_target(target, n_class=output.size()[1])  #extend target from (b, h, w, d) to (b, c, h, w, d)

    output = flatten(output)[1:, ...]  # transpose [N,4，H,W,D] -> [4，N,H,W,D] -> [3, N*H*W*D] voxels
    target = flatten(target)[1:, ...]  # [class, N*H*W*D]

    target_sum = target.sum(-1)  # sub_class_voxels [3,1] -> 3个voxels
    if weight_type == 'square':
        class_weights = 1. / (target_sum * target_sum + eps)
    elif weight_type == 'identity':
        class_weights = 1. / (target_sum + eps)
    elif weight_type == 'sqrt':
        class_weights = 1. / (torch.sqrt(target_sum) + eps)
    else:
        raise ValueError('Check out the weight_type :', weight_type)

    # print(class_weights)
    intersect = (output * target).sum(-1)
    intersect_sum = (intersect * class_weights).sum()
    denominator = (output + target).sum(-1)
    denominator_sum = (denominator * class_weights).sum() + eps

    loss1 = 2*intersect[0] / (denominator[0] + eps)
    loss2 = 2*intersect[1] / (denominator[1] + eps)
    loss3 = 2*intersect[2] / (denominator[2] + eps)

    return 1 - 2. * intersect_sum / denominator_sum, loss1, loss2, loss3


class CWKLLoss(nn.Module):
    def __init__(self):
        super(CWKLLoss, self).__init__()

    def forward(self, PT, PS, weights):
        """
        计算类别感知自适应加权KL散度损失（CWKL）。

        参数:
        PT (torch.Tensor): 教师网络的预测概率分布（batch_size, num_classes）。
        PS (torch.Tensor): 学生网络的预测概率分布（batch_size, num_classes）。
        weights (torch.Tensor): 自适应权重（batch_size, num_classes）。

        返回:
        torch.Tensor: CWKL损失。
        """
        # 确保输入的概率分布是经过softmax的
        PT = F.softmax(PT, dim=1)
        PS = F.softmax(PS, dim=1)
        
        # 计算每个样本的CWKL损失
        cwkl = weights * PT * (torch.log(PT + 1e-10) - torch.log(PS + 1e-10))
        
        # 对所有类别的损失求和，然后对所有样本的损失取平均值
        cwkl_loss = cwkl.sum(dim=1).mean()
        
        return cwkl_loss


# 定义逐像素的 KL 散度计算函数
def kl_divergence_pixelwise(p, q):
    eps = 1e-10
    p = np.clip(p, eps, 1.0)  # 将 p 的值限制在 [eps, 1.0] 范围内
    q = np.clip(q, eps, 1.0)  # 将 q 的值限制在 [eps, 1.0] 范围内
    return np.sum(p * np.log(p / q), axis=0)

