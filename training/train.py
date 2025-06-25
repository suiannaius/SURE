import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from training.criterions import nu_loss_d, nu_loss_mu, nu_loss_far, gradient_loss
from utilities.color import apply_color_map, apply_heatmap
from utilities.utils import compute_gradient, compute_distance_map, generate_noisy_images, summarize_tensor, sample_class_wise_noised_patch_images, generate_blurred_images, run_forward_to_get_u
from training.metrics import calculate_dice, calculate_dice_Refuge
from utilities.color import apply_color_map, apply_heatmap


L_gradient = gradient_loss
L_noise_mu = nu_loss_mu
L_noise_d = nu_loss_d
L_noise_far = nu_loss_far
visualization_in_writer = False
binary_threshold = 0.3
default_mu = 1.0
default_d = 1.0
default_far = 1.0

annealing_hsd = False


def train(model,
          dataloader,
          optimizer,
          num_classes,
          criterion,
          current_epoch,
          total_epoch,
          annealing_steps,
          device,
          loss_type,
          batch_size,
          writer,
          good_model_step,
          num_patch,
          d_threshold,
          d_eps,
          epsilon,
          beta=0.0, 
          gamma=0.0,
          use_grad_clip=False,
          sample_size=None,
          visualization_in_writer=visualization_in_writer,
          dataset=None,
          **kwargs):   
    model.train()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_kl_loss = 0.0
    running_grad_loss = 0.0
    running_noise_loss = 0.0
    if 'ACDC' in dataset:
        running_dice_RV = 0.0
        running_dice_Myo = 0.0
        running_dice_LV = 0.0
        k = 3
    elif 'Refuge' in dataset:
        running_dice_DISC = 0.0
        running_dice_CUP = 0.0
        k = 7
    
    running_noise_loss_d = 0.0
    running_noise_loss_mu = 0.0
    running_noise_loss_far = 0.0
    C = num_classes
    for batch_idx, (images, labels, spacing) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device) # [N,M,H,W], [N,C,H,W]
        # summarize_tensor(images, name='Images')
        targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW,C]       
        optimizer.zero_grad()
        N, _, H, W = images.size()
        pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW,C]
        evidence = F.softplus(pred)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
        u = num_classes / S # [NHW,1]
        prob = alpha / S # [NHW,C]
        # For visualization
        prob_view = prob.view(N, H, W, C).permute(0, 3, 1, 2)
        edl_loss, term_ace, kl_loss, dice_loss = criterion(targets, 
                                                           alpha, 
                                                           num_classes=num_classes, 
                                                           current_epoch=current_epoch, 
                                                           total_epoch=total_epoch,
                                                           annealing_steps=annealing_steps, 
                                                           device=device, 
                                                           loss_type=loss_type)
        if 'ACDC' in dataset:
            dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=num_classes)
            dice_RV, dice_Myo, dice_LV = dice[0], dice[1], dice[2]
        elif 'Refuge' in dataset:
            dice_DISC, dice_CUP = calculate_dice_Refuge(targets, prob, epsilon=1e-5, device=device)
        del pred, evidence, alpha, S, prob
        torch.cuda.empty_cache()

        annealing_start = torch.tensor(0.01, dtype=torch.float32)
        annealing_AU = annealing_start * torch.exp(-torch.log(annealing_start) / (total_epoch - good_model_step) * (current_epoch - good_model_step))
        
        # Visualization  
        if visualization_in_writer and (kwargs.get('fold') == 0):
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
                    writer.add_image(f'(Train) Ground Truth/{batch_idx*batch_size+i}', label_color.astype(np.uint8), dataformats='CHW')
                    del soft_label, label_slice, label_color
                    torch.cuda.empty_cache()

                    prob_indices = torch.argmax(prob_view, dim=1) # [N,H,W]
                    prob_slice = prob_indices[i, :, :].detach().cpu().numpy() # [H,W]
                    prob_color = apply_color_map(prob_slice) # [H,W,3]
                    prob_color = prob_color.transpose(2, 0, 1)  # [3,H,W]
                    writer.add_image(f'(Train) Predicted Mask/{batch_idx*batch_size+i}', prob_color, global_step=current_epoch, dataformats='CHW')
                    del prob_indices, prob_slice, prob_color
                    torch.cuda.empty_cache()

                    u_view = u.view(N, H, W, 1).squeeze(1) # [N,H,W]
                    u_slice = u_view[i, :, :].detach().cpu().numpy()  # [H,W]
                    u_color = apply_heatmap(u_slice) # [H,W,3]
                    u_color = u_color.transpose(2, 0, 1)  # [3,H,W]
                    writer.add_image(f'(Train) Uncertainty_Original/{batch_idx*batch_size+i}', u_color, global_step=current_epoch, dataformats='CHW')
    
                    current_mask = boundary_mask[i, :] # [HWD,]
                    u_scatter = u.view(N, -1)[i, :][current_mask].unsqueeze(-1).clone().detach().cpu().numpy()
                    g_scatter = gradient_blurred.view(N, -1)[i, :][current_mask].unsqueeze(-1).unsqueeze(-1).clone().detach().cpu().numpy()
                    plt.figure(figsize=(8, 6))
                    plt.scatter(u_scatter, g_scatter, color='b', label="Gradient-Uncertainty")
                    plt.title("Gradient-Uncertainty")
                    plt.xlabel("Uncertainty")
                    plt.ylabel("Gradient")
                    plt.legend(loc="upper right")
                    writer.add_figure(f'(Train) Gradient-Uncertainty/{batch_idx*batch_size+i}', plt.gcf(), global_step=current_epoch)
                    plt.close()
                    del u_view, u_slice, u_color, current_mask, u_scatter, g_scatter
                    torch.cuda.empty_cache()
        
        if (beta == 0.0 and gamma == 0.0) or (current_epoch < good_model_step):
            loss = edl_loss
            noise_loss = torch.tensor(0.0, device=device)
            grad_loss = torch.tensor(0.0, device=device)
        else:
            if gamma != 0 :
                with torch.no_grad():
                    noisy_images = generate_noisy_images(images, device, mu=0.5, sigma=0.3)  # [N,M,H,W]
                    noisy_u = run_forward_to_get_u(model, noisy_images, num_classes)  # [NHW,1]

                    if annealing_hsd:
                        noised_uncertainty_mask=torch.ones_like(noisy_u, device=noisy_u.device).view(N, H, W ,1).squeeze(-1)
                    else:
                        noised_uncertainty_mask = ((noisy_u-u).float()<=0).view(N, H, W ,1).squeeze(-1)
                    distance_map = compute_distance_map(labels).reshape(N, H, W)
                    mu1, mu2, d, noised_images_mu1, noised_images_mu2, indexes = sample_class_wise_noised_patch_images(images, 
                                                                                                                       distance_map, 
                                                                                                                       noised_uncertainty_mask, 
                                                                                                                       labels,
                                                                                                                       k=k,
                                                                                                                       num_patch=num_patch,
                                                                                                                       device=device, 
                                                                                                                       threshold=d_threshold)
                evidence_noised_mu1 = F.softplus(model(noised_images_mu1).permute(0, 2, 3, 1).contiguous().view(-1, C))
                evidence_noised_mu2 = F.softplus(model(noised_images_mu2).permute(0, 2, 3, 1).contiguous().view(-1, C))
                alpha_mu1 = evidence_noised_mu1 + 1
                alpha_mu2 = evidence_noised_mu2 + 1
                S_mu1 = torch.sum(alpha_mu1, dim=1, keepdim=True) # [NHW,1]
                S_mu2 = torch.sum(alpha_mu2, dim=1, keepdim=True) # [NHW,1]
                uncertainty_noised_mu1 = num_classes / S_mu1  # [NHW,1]
                uncertainty_noised_mu2 = num_classes / S_mu2
                
                noise_loss_mu = kwargs.get('coefficient_mu', default_mu) * L_noise_mu(u.view(N, 1, H, W).clone().detach(), uncertainty_noised_mu1.view(N, 1, H, W), uncertainty_noised_mu2.view(N, 1, H, W), mu1, mu2, d, indexes, device=device, threshold=d_threshold)
                noise_loss_d = kwargs.get('coefficient_d', default_d) * L_noise_d(u.view(N, 1, H, W).clone().detach(), uncertainty_noised_mu1.view(N, 1, H, W), uncertainty_noised_mu2.view(N, 1, H, W), d, indexes, device=device, threshold=d_threshold)
                # noise_loss_far = kwargs.get('coefficient_far', default_far) * L_noise_far(u.view(N, 1, H, W).clone().detach(), uncertainty_noised_mu1.view(N, 1, H, W), uncertainty_noised_mu2.view(N, 1, H, W), distance_map, device=device, threshold=d_threshold)
                noise_loss_far = kwargs.get('coefficient_far', default_far) * L_noise_far(u.view(N, 1, H, W), uncertainty_noised_mu1.view(N, 1, H, W), uncertainty_noised_mu2.view(N, 1, H, W), distance_map, device=device, threshold=d_threshold)
                
                del evidence_noised_mu1, evidence_noised_mu2, alpha_mu1, alpha_mu2, S_mu1, S_mu2, uncertainty_noised_mu1, uncertainty_noised_mu2
                torch.cuda.empty_cache()

                running_noise_loss_d += noise_loss_d.item()
                running_noise_loss_mu += noise_loss_mu.item()
                running_noise_loss_far += noise_loss_far.item()

                if noise_loss_mu != 0 and noise_loss_d != 0:
                    noise_loss = noise_loss_mu + noise_loss_d + noise_loss_far
                
                elif noise_loss_mu != 0 and noise_loss_d == 0:
                    noise_loss = noise_loss_mu + noise_loss_far
                
                elif noise_loss_d != 0  and noise_loss_mu == 0:
                    noise_loss = noise_loss_d + noise_loss_far
                
                else:
                    noise_loss = torch.tensor(0.0, device=device)
            
            else:
                noise_loss = torch.tensor(0.0, device=device)
            writer.add_scalar('(Train) Noise_loss-no annealing', noise_loss, current_epoch)
            noise_loss = noise_loss * annealing_AU
            
            if beta != 0:
                distance_map = compute_distance_map(labels)
                blurred_images = generate_blurred_images(images, sigma_blur=0.0, device=device)  
                gradient_blurred = compute_gradient(blurred_images) # [NHWD,1]
                grad_loss = L_gradient(u, gradient_blurred, distance_map, batch_size, device=device, sample_size=sample_size)
                
            else:
                grad_loss = torch.tensor(0.0, device=device)
            
            writer.add_scalar('(Train) Grad_loss-no annealing', grad_loss, current_epoch)
            grad_loss = grad_loss * annealing_AU
            
            if grad_loss == 0.0 and noise_loss == 0.0:
                loss = edl_loss
            
            elif grad_loss == 0.0 and noise_loss != 0.0:
                loss = edl_loss + gamma * noise_loss
            
            elif grad_loss != 0.0 and noise_loss == 0.0:
                loss = edl_loss + beta * grad_loss
            
            else:
                loss = edl_loss + beta * grad_loss + gamma * noise_loss

        loss.backward()
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        running_dice_loss += dice_loss.item()
        running_kl_loss += kl_loss.item()
        running_noise_loss += noise_loss.item()
        running_grad_loss += grad_loss.item()
        if 'ACDC' in dataset:
            running_dice_RV += dice_RV.item()
            running_dice_Myo += dice_Myo.item()
            running_dice_LV += dice_LV.item()
        elif 'Refuge' in dataset:
            running_dice_DISC += dice_DISC.item()
            running_dice_CUP += dice_CUP.item()

    writer.add_scalar('(Train) Loss', running_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Dice', running_dice_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss KL', running_kl_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Grad-annealing', running_grad_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Noise', running_noise_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Noise-d', running_noise_loss_d / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Noise-mu', running_noise_loss_mu / len(dataloader), current_epoch)
    writer.add_scalar('(Train) Loss Noise-far', running_noise_loss_far / len(dataloader), current_epoch)
    if 'ACDC' in dataset:
        writer.add_scalar('(Train) SDice-RV', running_dice_RV / len(dataloader), current_epoch)
        writer.add_scalar('(Train) SDice-Myo', running_dice_Myo / len(dataloader), current_epoch)
        writer.add_scalar('(Train) SDice-LV', running_dice_LV / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_RV / len(dataloader), running_dice_Myo / len(dataloader), running_dice_LV / len(dataloader)
    elif 'Refuge' in dataset:
        writer.add_scalar('(Train) SDice-DISC', running_dice_DISC / len(dataloader), current_epoch)
        writer.add_scalar('(Train) SDice-CUP', running_dice_CUP / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_DISC / len(dataloader), running_dice_CUP / len(dataloader)

    
def validate(model,
             dataloader,
             num_classes,
             criterion,
             current_epoch,
             device,
             loss_type,
             writer,
             annealing_steps,
             total_epoch,
             dataset=None,
             **kwargs):
    model.eval()
    running_loss = 0.0
    running_dice_loss = 0.0
    running_kl_loss = 0.0
    
    if 'ACDC' in dataset:
        running_dice_RV = 0.0
        running_dice_Myo = 0.0
        running_dice_LV = 0.0
    elif 'Refuge' in dataset:
        running_dice_DISC = 0.0
        running_dice_CUP = 0.0
    
    C = num_classes
    with torch.no_grad():
        for batch_idx, (images, labels, spacing) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C)
            
            # N, _, H, W = images.size()
            pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C)
            evidence = F.softplus(pred)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            
            edl_loss, term_ace, kl_loss, dice_loss = criterion(targets, 
                                                               alpha, 
                                                               num_classes=num_classes, 
                                                               current_epoch=current_epoch, 
                                                               total_epoch=total_epoch,
                                                               annealing_steps=annealing_steps, 
                                                               device=device, 
                                                               loss_type=loss_type)
            if 'ACDC' in dataset:
                dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=num_classes)
                running_dice_RV += dice[0].item()
                running_dice_Myo += dice[1].item()
                running_dice_LV += dice[2].item()
            elif 'Refuge' in dataset:
                dice_DISC, dice_CUP = calculate_dice_Refuge(targets, prob, epsilon=1e-5, device=device)
                running_dice_DISC += dice_DISC.item()
                running_dice_CUP += dice_CUP.item()
            
            running_loss += edl_loss.item()
            running_dice_loss += dice_loss.item()
            running_kl_loss += kl_loss.item()
            
    writer.add_scalar('(Validate) Loss', running_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Validate) Loss Dice', running_dice_loss / len(dataloader), current_epoch)
    writer.add_scalar('(Validate) Loss KL', running_kl_loss / len(dataloader), current_epoch)
    
    if dataset == 'ACDC':
        writer.add_scalar('(Validate) SDice-RV', running_dice_RV / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-Myo', running_dice_Myo / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-LV', running_dice_LV / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_RV / len(dataloader), running_dice_Myo / len(dataloader), running_dice_LV / len(dataloader)
    elif dataset == 'Refuge':
        writer.add_scalar('(Validate) SDice-DISC', running_dice_DISC / len(dataloader), current_epoch)
        writer.add_scalar('(Validate) SDice-CUP', running_dice_CUP / len(dataloader), current_epoch)
        return running_loss / len(dataloader), running_dice_DISC / len(dataloader), running_dice_CUP / len(dataloader)
    