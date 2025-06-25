from matplotlib import pyplot as plt
import numpy as np
import torch

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

def train(args,model,optimizer,epoch,train_loader,writer,device,visualization_in_writer=True):
    model.train()
    loss_meter = AverageMeter()
    if 'ACDC' in args.dataset:
        all_dice_RV = AverageMeter()
        all_dice_Myo = AverageMeter()
        all_dice_LV = AverageMeter()
    elif 'Refuge' in args.dataset:
        all_dice_DISC = AverageMeter()
        all_dice_CUP = AverageMeter()
    step = 0
    dt_size = len(train_loader.dataset)
    C = args.num_classes
    for batch_idx, (images, labels, spacing) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device) # [N,1,H,W], [N,C,H,W]
        N, _, H, W = images.size()
        adjust_learning_rate(optimizer, epoch, args.end_epochs, args.lr)
        step += 1
        # refresh the optimizer

        evidences, loss = model(images, labels, epoch, args.mode,args.dataset)

        # print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
        
        # compute gradients and take step
        optimizer.zero_grad()
        loss.requires_grad_(True).backward()
        optimizer.step()
        loss_meter.update(loss.item())

        alpha = (evidences + 1).permute(0, 2, 3, 1).contiguous().view(-1, C)
        S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
        u = args.num_classes / S # [NHW,1]
        prob = alpha / S # [NHW,C]
        # For visualization
        prob_view = prob.view(N, H, W, C).permute(0, 3, 1, 2)

        targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW, C]

        if 'ACDC' in args.dataset:
            dice_RV, dice_Myo, dice_LV = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
            all_dice_RV.update(dice_RV)
            all_dice_Myo.update(dice_Myo)
            all_dice_LV.update(dice_LV)
        elif 'Refuge' in args.dataset:
            dice_DISC, dice_CUP = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=C)
            all_dice_DISC.update(dice_DISC)
            all_dice_CUP.update(dice_CUP)

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

    if 'ACDC' in args.dataset:
        return loss_meter.avg, all_dice_RV.avg, all_dice_Myo.avg, all_dice_LV.avg
    elif 'Refuge' in args.dataset:
        return loss_meter.avg, all_dice_DISC.avg, all_dice_CUP.avg