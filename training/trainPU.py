import math
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from torch.autograd import Variable

from model.TrustworthySeg_criterions import get_soft_label, SDiceLoss, CWKLLoss, kl_divergence_pixelwise
from training.metrics import calculate_dice
from utilities.color import apply_color_map, apply_heatmap
from utilities.utils import compute_gradient, compute_distance_map, generate_blurred_images


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def model_PU(x, model):

    y = model.sample_m(x,m=8,testing=True)
    return y


def Uentropy(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logpc = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    # u_all1 = -pc * logpc / c
    u_all = -pc * logpc / math.log(c)
    # max_u = torch.max(u_all)
    # min_u = torch.min(u_all)
    # NU1 = torch.sum(u_all, dim=1)
    # k = u_all.shape[1]
    # NU2 = torch.sum(u_all[:, 0:u_all.shape[1]-1, :, :], dim=1)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU

def train(args,model,optimizer,epoch,train_loader,criterion_dl,writer,device,visualization_in_writer=True):
    
    model.train()
    loss_meter = AverageMeter()
    if 'ACDC' in args.dataset:
        all_dice_RV = AverageMeter()
        all_dice_Myo = AverageMeter()
        all_dice_LV = AverageMeter()
    elif 'Refuge' in args.dataset:
        all_dice_DISC = AverageMeter()
        all_dice_CUP = AverageMeter()
    
    C = args.num_classes
    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # [N,1,H,W], [N,C,H,W]
        N, _, H, W = images.size()
        adjust_learning_rate(optimizer, epoch, args.num_epochs, args.lr)

        if args.backbone == 'PU':
            model.train()
            onehot_target = labels.float()
            model.forward(images, onehot_target, training=True)
            elbo = model.elbo(onehot_target)
            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
                    model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss

            model.eval()
            logits = model_PU(images, model)

        elif args.backbone == 'Flow' or args.backbone == 'Glow':
            onehot_target = labels.float()
            model.forward(images, onehot_target, training=True)
            _,_,_,elbo = model.elbo(onehot_target,use_mask=False,analytic_kl=False)

            reg_loss = l2_regularisation(model.posterior) + l2_regularisation(model.prior) + l2_regularisation(
                    model.fcomb.layers)
            loss = -elbo + 1e-5 * reg_loss

            model.eval()
            logits = model_PU(images, model)
            
        elif args.backbone == 'UDrop':
            logits = model(images)

            outputs = F.softmax(logits,1)
            _, target = torch.max(labels, dim=1, keepdim=True)
            soft_target = get_soft_label(target, args.num_classes) # for SDiceloss: mean loss
            loss = criterion_dl(outputs, soft_target)  # for SDiceloss

        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        
        u = Uentropy(logits, C)

        targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW, C]
        prob_view = logits
        prob = logits.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW, C]

        optimizer.zero_grad()
        loss.backward()
        nan_detected = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of {name}")
                nan_detected = True
                break
        # if args.backbone == 'Flow' or args.backbone == 'Glow':
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            
        if nan_detected:
            break
        
        optimizer.step()

        if 'ACDC' in args.dataset:
            dice_RV, dice_Myo, dice_LV = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
            all_dice_RV.update(dice_RV)
            all_dice_Myo.update(dice_Myo)
            all_dice_LV.update(dice_LV)
        elif 'Refuge' in args.dataset:
            dice_DISC, dice_CUP = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
            all_dice_DISC.update(dice_DISC)
            all_dice_CUP.update(dice_CUP)

        loss_meter.update(loss.item())

        # Visualization  
        if visualization_in_writer:
            distance_map = compute_distance_map(labels)
            blurred_images = generate_blurred_images(images, sigma_blur=0.0, device=device)  
            gradient_blurred = compute_gradient(blurred_images) # [NHW,1]
            delta = 1
            boundary_mask = (distance_map <= delta).reshape(N, -1) 
            if batch_idx % 10 == 0:
                for i in range(N):
                    soft_label = torch.argmax(labels, dim=1) # [N,H,W]
                    label_slice = soft_label[i, :, :].detach().cpu().numpy()  # [H,W]
                    label_color = apply_color_map(label_slice) # [H,W,3]
                    label_color = label_color.transpose(2, 0, 1)  # [3,H,W]
                    writer.add_image(f'(Train) Ground Truth/{batch_idx*args.batch_size+i}', label_color.astype(np.uint8), dataformats='CHW')

                    prob_indices = torch.argmax(prob_view, dim=1) # [N,H,W]
                    prob_slice = prob_indices[i, :, :].detach().cpu().numpy() # [H,W]
                    prob_color = apply_color_map(prob_slice) # [H,W,3]
                    prob_color = prob_color.transpose(2, 0, 1)  # [3,H,W]
                    writer.add_image(f'(Train) Predicted Mask/{batch_idx*args.batch_size+i}', prob_color, global_step=epoch, dataformats='CHW')

                    u_view = u.view(N, H, W, 1).squeeze(1) # [N,H,W]
                    u_slice = u_view[i, :, :].detach().cpu().numpy()  # [H,W]
                    u_color = apply_heatmap(u_slice) # [H,W,3]
                    u_color = u_color.transpose(2, 0, 1)  # [3,H,W]
                    writer.add_image(f'(Train) Uncertainty_original/{batch_idx*args.batch_size+i}', u_color, global_step=epoch, dataformats='CHW')
    
                    current_mask = boundary_mask[i, :] # [HWD,]
                    u_scatter = u.view(N, -1)[i, :][current_mask].unsqueeze(-1).clone().detach().cpu().numpy()
                    g_scatter = gradient_blurred.view(N, -1)[i, :][current_mask].unsqueeze(-1).unsqueeze(-1).clone().detach().cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    plt.scatter(u_scatter, g_scatter, color='b', label="Gradient-Uncertainty")
                    plt.title("Gradient-Uncertainty")
                    plt.xlabel("Uncertainty")
                    plt.ylabel("Gradient")
                    plt.legend()
                    writer.add_figure(f'(Train) Gradient-Uncertainty/{batch_idx*args.batch_size+i}', plt.gcf(), global_step=epoch)
                    plt.close()

        # if nan_detected:
        #     print(f"Terminating training due to NaN gradients in batch {batch_idx}")
        #     if 'ACDC' in args.dataset:
        #         return loss_meter.avg, all_dice_RV.avg, all_dice_Myo.avg, all_dice_LV.avg
        #     elif 'Refuge' in args.dataset:
        #         return loss_meter.avg, all_dice_DISC.avg, all_dice_CUP.avg
    
    if 'ACDC' in args.dataset:
        return loss_meter.avg, all_dice_RV.avg, all_dice_Myo.avg, all_dice_LV.avg, nan_detected
    elif 'Refuge' in args.dataset:
        return loss_meter.avg, all_dice_DISC.avg, all_dice_CUP.avg, nan_detected


