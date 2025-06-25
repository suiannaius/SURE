import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from utilities.color import apply_color_map, apply_heatmap
from utilities.utils import compute_gradient, compute_distance_map, generate_noisy_images, generate_blurred_images
from utilities.utils import sample_class_wise_noised_patch_images, run_forward_to_get_u
from training.metrics import soft_hd95, calculate_dice, calculate_ece, calculate_ueo, soft_hd95_Refuge, calculate_dice_Refuge, calculate_ece_Refuge, calculate_ueo_Refuge
from utilities.color import apply_color_map, apply_heatmap
from training.trainPU import model_PU, Uentropy
import ttach as tta


def inverse_func(g, k):
    return k / g

def inference(model, dataloader, num_classes, batch_size, device, writer, fold, d_threshold=4, consider_spacing=False, dataset=None, method='base'):
    model.eval()
    if 'ACDC' in dataset:
        running_dice_RV = 0.0
        running_dice_Myo = 0.0
        running_dice_LV = 0.0
        running_assd_RV = 0.0
        running_assd_Myo = 0.0
        running_assd_LV = 0.0
        running_hd95_RV = 0.0
        running_hd95_Myo = 0.0
        running_hd95_LV = 0.0
        running_ECE_RV = 0.0
        running_ECE_Myo = 0.0
        running_ECE_LV = 0.0
        running_UEO_RV = 0.0
        running_UEO_Myo = 0.0
        running_UEO_LV = 0.0
    elif 'Refuge' in dataset:
        running_dice_DISC = 0.0
        running_dice_CUP = 0.0
        running_assd_DISC = 0.0
        running_assd_CUP = 0.0
        running_hd95_DISC = 0.0
        running_hd95_CUP = 0.0
        running_ECE_DISC = 0.0
        running_ECE_CUP = 0.0
        running_UEO_DISC = 0.0
        running_UEO_CUP = 0.0
    C = num_classes
    total_samples = len(dataloader.dataset)
    print(f"Using Device: {device}")
    print('Inference in progress...')
    
    with torch.no_grad(), tqdm(total=total_samples) as progress_bar:
        if 'ACDC' in dataset:
            # _, assd_dict, _, _ = ACDC2017_Evaluation(files[fold], model)
            total_assds = np.zeros(4)
        elif 'Refuge' in dataset:
            # _, assd_dict, _, _ = Refuge_Evaluation(files, model, save_DIR=refuge_save_dir, epoch=1)
            total_assds = np.zeros(3)
        
        # num_patients = len(assd_dict)
        num_patients = 1
        # for _, assd_values in assd_dict.items():
        #     total_assds += assd_values
        average_assds = total_assds / num_patients
        for batch_idx, (images, labels, spacing) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device) # [N,M,H,W], [N,C,H,W]
            
            to_grayscale = transforms.Grayscale(num_output_channels=1)
            gray_images = to_grayscale(images)
            
            soft_label = torch.argmax(labels, dim=1) # [N,H,W]
            targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW, C]
            N, _, H, W = images.size()

            if method == 'base' or method == 'devis':
                if method == 'base':
                    pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW,C]
                    evidence = F.softplus(pred)
                elif method == 'devis':
                    evidence = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C)
                
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
                u = num_classes / S # [NHW,1]
                u_view = u.view(N, H, W, 1).squeeze(-1) # [N,H,W]
                prob = alpha / S # [NHW,C]
                # For visualization
                prob_view = prob.view(N, H, W, C).permute(0, 3, 1, 2)
                prob_indices = torch.argmax(prob_view, dim=1) # [N,H,W]

            elif method == 'pu' or method == 'flow' or method == 'glow' or method == 'udrop':
                if method == 'pu' or method == 'flow' or method == 'glow':
                    if 'Refuge-no' in dataset:
                        resized_images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
                    else:
                        resized_images = images
                    model.forward(resized_images, labels.float(), training=False)
                    logits = model_PU(resized_images, model)
                    del resized_images
                    if 'Refuge-no' in dataset:
                        logits = F.interpolate(logits, size=(512, 512), mode='nearest')
                elif method == 'udrop':
                    logits = model(images)

                if logits.ndim == 3:
                    logits = logits.unsqueeze(0)
                
                prob_view = F.softmax(logits, dim=1)
                prob_indices = torch.argmax(prob_view, dim=1)
                prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                u_view = Uentropy(logits, C)
                u = u_view.view(-1, 1)

            else:
                if method == 'eu':
                    for i in range(4):
                        logits = model[i](images)
                        if i == 0:
                            prob_view = F.softmax(logits, dim=1)
                            u_view = Uentropy(logits, C)
                        else:
                            prob_view += F.softmax(logits, dim=1)
                            u_view += Uentropy(logits, C)

                    prob_view /= 4.
                    u_view /= 4.
                    prob_indices = torch.argmax(prob_view, dim=1)
                    prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                    u = u_view.view(-1, 1)

                elif method == 'tta':
                    # defined 2 * 2 * 3 * 3 = 36 augmentations !
                    transforms_img = tta.Compose(
                        [
                            tta.HorizontalFlip(),
                            tta.Rotate90(angles=[0, 180]),
                            tta.Scale(scales=[1, 2, 4]),
                            tta.Multiply(factors=[0.9, 1, 1.1]),        
                        ]
                    )
                    tta_model = tta.SegmentationTTAWrapper(model, transforms_img)
                    logits = tta_model(images)
                    prob_view = F.softmax(logits, dim=1)
                    prob_indices = torch.argmax(prob_view, dim=1)
                    prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                    u_view = Uentropy(logits, C)
                    u = u_view.view(-1, 1)

            threshold = np.arange(0.01, 1.0, 0.01)
            
            if 'Refuge' not in dataset:
                ueo = calculate_ueo(prob, targets, u, threshold, num_classes=num_classes)
                ece = calculate_ece(prob, targets, num_classes=num_classes)
                dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=num_classes)
            
                if consider_spacing:
                    hd = soft_hd95(soft_label, prob_indices, spacing, num_classes=num_classes)
                else:
                    hd = soft_hd95(soft_label, prob_indices, num_classes=num_classes)
            else:
                ueo_DISC, ueo_CUP = calculate_ueo_Refuge(prob, targets, u, threshold)
                ece_DISC, ece_CUP = calculate_ece_Refuge(prob, targets)
                dice_DISC, dice_CUP = calculate_dice_Refuge(targets, prob, epsilon=1e-5, device=device)
                hd95_DISC, hd95_CUP = soft_hd95_Refuge(soft_label, prob_indices)
                assd_DISC, assd_CUP = average_assds[1], average_assds[2]
            if 'ACDC' in dataset:
                ueo_RV, ueo_Myo, ueo_LV = ueo[0], ueo[1], ueo[2]
                ece_RV, ece_Myo, ece_LV = ece[0], ece[1], ece[2]
                dice_RV, dice_Myo, dice_LV = dice[0], dice[1], dice[2]
                hd95_RV, hd95_Myo, hd95_LV = hd[0], hd[1], hd[2]
                assd_RV, assd_Myo, assd_LV = average_assds[1], average_assds[2], average_assds[3]
            
            # assd_RV, assd_Myo, assd_LV, hd95_RV, hd95_Myo, hd95_LV = calculate_assd_and_hd95(prob_indices, soft_label, spacing)   
            noisy_images = generate_noisy_images(images, device, mu=0.5, sigma=0.3, seed=1)  # [N,1,H,W]
            noisy_u = run_forward_to_get_u(model, noisy_images, num_classes, method, dataset)  # [NHW,1]
            noised_uncertainty_mask = torch.ones_like(noisy_u, device=noisy_u.device).view(N, H, W ,1).squeeze(-1)
            distance_map = compute_distance_map(labels).reshape(N, H, W)
            mu1, mu2, d, noised_images_mu1, noised_images_mu2, indexes = sample_class_wise_noised_patch_images(images, 
                                                                                                               distance_map, 
                                                                                                               noised_uncertainty_mask, 
                                                                                                               labels, 
                                                                                                               device=device, 
                                                                                                               threshold=d_threshold,
                                                                                                               seed=1)
            
            
            if method == 'base' or method == 'devis':
                if method == 'base':
                    evidence_noised_mu1 = F.softplus(model(noised_images_mu1)).permute(0, 2, 3, 1).contiguous().view(-1, C)
                    evidence_noised_mu2 = F.softplus(model(noised_images_mu2)).permute(0, 2, 3, 1).contiguous().view(-1, C)
                elif method == 'devis':
                    evidence_noised_mu1 = model(noised_images_mu1).permute(0, 2, 3, 1).contiguous().view(-1, C)
                    evidence_noised_mu2 = model(noised_images_mu2).permute(0, 2, 3, 1).contiguous().view(-1, C)

                alpha_mu1 = evidence_noised_mu1 + 1
                alpha_mu2 = evidence_noised_mu2 + 1
                S_mu1 = torch.sum(alpha_mu1, dim=1, keepdim=True) # [NHW,1]
                S_mu2 = torch.sum(alpha_mu2, dim=1, keepdim=True) # [NHW,1]
                uncertainty_noised_mu1 = num_classes / S_mu1
                uncertainty_noised_mu2 = num_classes / S_mu2

                u_mu1_view = uncertainty_noised_mu1.view(N, H, W, 1).squeeze(-1) # [N,H,W]
                u_mu2_view = uncertainty_noised_mu2.view(N, H, W, 1).squeeze(-1) # [N,H,W]

            elif method == 'pu' or method == 'flow' or method == 'glow' or method == 'udrop':
                if method == 'pu' or method == 'flow' or method == 'glow':
                    if 'Refuge-no' in dataset:
                        resized_noised_images_mu1 = F.interpolate(noised_images_mu1, size=(256, 256), mode='bilinear', align_corners=False)
                        resized_noised_images_mu2 = F.interpolate(noised_images_mu2, size=(256, 256), mode='bilinear', align_corners=False)
                    else:
                        resized_noised_images_mu1 = noised_images_mu1
                        resized_noised_images_mu2 = noised_images_mu2
                    logits_mu1 = model_PU(resized_noised_images_mu1, model)
                    logits_mu2 = model_PU(resized_noised_images_mu2, model)
                    del resized_noised_images_mu1, resized_noised_images_mu2
                    if 'Refuge-no' in dataset:
                        logits_mu1 = F.interpolate(logits_mu1, size=(512, 512), mode='nearest')
                        logits_mu2 = F.interpolate(logits_mu2, size=(512, 512), mode='nearest')
                elif method == 'udrop':
                    logits_mu1 = model(noised_images_mu1)
                    logits_mu2 = model(noised_images_mu2)

                if logits_mu1.ndim == 3:
                    logits_mu1 = logits_mu1.unsqueeze(0)
                if logits_mu2.ndim == 3:
                    logits_mu2 = logits_mu2.unsqueeze(0)
                
                u_mu1_view = Uentropy(logits_mu1, C)
                u_mu2_view = Uentropy(logits_mu2, C)
                uncertainty_noised_mu1 = u_mu1_view.view(-1, 1)
                uncertainty_noised_mu2 = u_mu2_view.view(-1, 1)

            else:
                if method == 'eu':
                    for i in range(4):
                        logits_mu1 = model[i](noised_images_mu1)
                        logits_mu2 = model[i](noised_images_mu2)

                        if i == 0:
                            u_mu1_view = Uentropy(logits_mu1, C)
                            u_mu2_view = Uentropy(logits_mu2, C)
                        else:
                            u_mu1_view += Uentropy(logits_mu1, C)
                            u_mu2_view += Uentropy(logits_mu2, C)

                    u_mu1_view /= 4.
                    u_mu2_view /= 4.

                    uncertainty_noised_mu1 = u_mu1_view.view(-1, 1)
                    uncertainty_noised_mu2 = u_mu2_view.view(-1, 1)

                elif method == 'tta':
                    logits_mu1 = model(noised_images_mu1)
                    logits_mu2 = model(noised_images_mu2)

                    transforms_img = tta.Compose(
                    [
                        tta.HorizontalFlip(),
                        tta.Rotate90(angles=[0, 180]),
                        tta.Scale(scales=[1, 2, 4]),
                        tta.Multiply(factors=[0.9, 1, 1.1]),        
                    ]
                    )
                    tta_model = tta.SegmentationTTAWrapper(model, transforms_img)
                    logits_mu1 = tta_model(noised_images_mu1)
                    logits_mu2 = tta_model(noised_images_mu2)

                    u_mu1_view = Uentropy(logits_mu1, C)
                    u_mu2_view = Uentropy(logits_mu2, C)
                    uncertainty_noised_mu1 = u_mu1_view.view(-1, 1)
                    uncertainty_noised_mu2 = u_mu2_view.view(-1, 1)

            noisy_u_view = noisy_u.view(N, H, W)
            # Visualization
            binary_thrshold = 0.3
            distance_map = compute_distance_map(labels)
            blurred_images = generate_blurred_images(images, sigma_blur=0.0, device=device)  
            gradient_blurred = compute_gradient(blurred_images) # [NHWD,1]
            
            gradient = compute_gradient(images) # [NHWD,1]
            gradient_binary = torch.zeros_like(gradient)
            gradient_binary[gradient > binary_thrshold] = 1
        
            gradient_blurred_binary = torch.zeros_like(gradient_blurred)
            gradient_blurred_binary[gradient_blurred > binary_thrshold] = 1
        
            writer.add_histogram("Gradient Distribution", gradient_blurred, global_step=0)
            
            delta = 1
            boundary_mask = (distance_map <= delta).reshape(N, -1)
        
            if (batch_idx % 10 == 0) and (fold == 0):
                for i in range(prob_view.shape[0]):
                    img_slice = images[i].cpu().numpy() # [M,H,W]
                    noisy_img_slice = noisy_images[i].cpu().numpy()
                    gray_img_slice = gray_images[i].cpu().numpy() # [M,H,W]
                    img_mu1_slice = noised_images_mu1[i].cpu().numpy() # [M,H,W]
                    img_mu2_slice = noised_images_mu2[i].cpu().numpy() # [M,H,W]
                    
                    prob_slice = prob_indices[i, :, :].detach().cpu().numpy() # [H,W]
                    label_slice = soft_label[i, :, :].detach().cpu().numpy()  # [H,W]
                    u_slice = u_view[i, :, :].detach().cpu().numpy()  # [H,W]
                    u_mu1_slice = u_mu1_view[i, :, :].detach().cpu().numpy()
                    u_mu2_slice = u_mu2_view[i, :, :].detach().cpu().numpy()
                    noisy_u_slice = noisy_u_view[i, :, :].detach().cpu().numpy()
                    distance_map_slice = distance_map.reshape(N, H, W)[i, :, :]
                    
                    prob_color = apply_color_map(prob_slice) # [H,W,3]
                    label_color = apply_color_map(label_slice) # [H,W,3]
                    u_color = apply_heatmap(u_slice) # [H,W,3]
                    u_mu1_color = apply_heatmap(u_mu1_slice) # [H,W,3]
                    u_mu2_color = apply_heatmap(u_mu2_slice) # [H,W,3]
                    noisy_u_color = apply_heatmap(noisy_u_slice) # [H,W,3]
                    d_color = apply_heatmap(distance_map_slice, 0, 50) # [H,W,3]
                    u_mu1_dif_color = apply_heatmap(u_mu1_slice - u_slice) # [H,W,3]
                    u_mu2_dif_color = apply_heatmap(u_mu2_slice - u_slice) # [H,W,3]
                    u_dif_color = apply_heatmap(noisy_u_slice - u_slice)
                    
                    prob_color = prob_color.transpose(2, 0, 1)  # [3,H,W]
                    label_color = label_color.transpose(2, 0, 1)  # [3,H,W]
                    u_color = u_color.transpose(2, 0, 1)  # [3,H,W]
                    u_mu1_color = u_mu1_color.transpose(2, 0, 1)  # [3,H,W]
                    u_mu2_color = u_mu2_color.transpose(2, 0, 1)  # [3,H,W]
                    noisy_u_color = noisy_u_color.transpose(2, 0, 1)  # [3,H,W]
                    d_color = d_color.transpose(2, 0, 1)  # [3,H,W]
                    u_mu1_dif_color = u_mu1_dif_color.transpose(2, 0, 1)  # [3,H,W]
                    u_mu2_dif_color = u_mu2_dif_color.transpose(2, 0, 1)  # [3,H,W]
                    u_dif_color = u_dif_color.transpose(2, 0, 1)  # [3,H,W]

                    grad_blurred_slice = gradient_blurred.view(N, H, W, 1).permute(0, 3, 1, 2)[i, :, :, :] # [1,H,W]
                    
                    writer.add_image(f'(Test) Gradient_blurred/{batch_idx*batch_size+i}', grad_blurred_slice.squeeze(0), dataformats='HW')
                    writer.add_image(f'(Test) A1. Original Image/{batch_idx*batch_size+i}', img_slice, dataformats='CHW')
                    writer.add_image(f'(Test) A2. Gray Image/{batch_idx*batch_size+i}', gray_img_slice, dataformats='CHW')
                    writer.add_image(f'(Test) A3. Noisy Image Whole/{batch_idx*batch_size+i}', noisy_img_slice, dataformats='CHW')
                    writer.add_image(f'(Test) B1. Noisy Image mu1/{batch_idx*batch_size+i}', img_mu1_slice, dataformats='CHW')
                    writer.add_image(f'(Test) B2. Noisy Image mu2/{batch_idx*batch_size+i}', img_mu2_slice, dataformats='CHW')
                    writer.add_image(f'(Test) C1. Noise mu1/{batch_idx*batch_size+i}', img_mu2_slice - img_slice, dataformats='CHW')
                    writer.add_image(f'(Test) C2. Noise mu2/{batch_idx*batch_size+i}', img_mu2_slice - img_slice, dataformats='CHW')
                    writer.add_image(f'(Test) Predicted Mask/{batch_idx*batch_size+i}', prob_color, dataformats='CHW')
                    writer.add_image(f'(Test) Ground Truth/{batch_idx*batch_size+i}', label_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty/{batch_idx*batch_size+i}', u_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty mu1-ori/{batch_idx*batch_size+i}', u_mu1_dif_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty mu2-ori/{batch_idx*batch_size+i}', u_mu2_dif_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty mu1/{batch_idx*batch_size+i}', u_mu1_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty mu2/{batch_idx*batch_size+i}', u_mu2_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty noisy/{batch_idx*batch_size+i}', noisy_u_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty noisy-ori/{batch_idx*batch_size+i}', u_dif_color, dataformats='CHW')
                    writer.add_image(f'(Test) Distance map/{batch_idx*batch_size+i}', d_color, dataformats='CHW')
                    del img_slice, noisy_img_slice, prob_slice, label_slice, prob_color, label_color, u_color, u_mu1_color, u_mu2_color
                    torch.cuda.empty_cache()
            
            if 'ACDC' in dataset:
                running_dice_RV += dice_RV.item()
                running_dice_Myo += dice_Myo.item()
                running_dice_LV += dice_LV.item()
                running_assd_RV += assd_RV
                running_assd_Myo += assd_Myo
                running_assd_LV += assd_LV
                running_hd95_RV += hd95_RV
                running_hd95_Myo += hd95_Myo
                running_hd95_LV += hd95_LV
                running_ECE_RV += ece_RV
                running_ECE_Myo += ece_Myo
                running_ECE_LV += ece_LV
                running_UEO_RV += ueo_RV
                running_UEO_Myo += ueo_Myo
                running_UEO_LV += ueo_LV
            elif 'Refuge' in dataset:
                running_dice_DISC += dice_DISC.item()
                running_dice_CUP += dice_CUP.item()
                running_assd_DISC += assd_DISC
                running_assd_CUP += assd_CUP
                running_hd95_DISC += hd95_DISC
                running_hd95_CUP += hd95_CUP
                running_ECE_DISC += ece_DISC
                running_ECE_CUP += ece_CUP
                running_UEO_DISC += ueo_DISC
                running_UEO_CUP += ueo_CUP
            progress_bar.update(images.size(0))
    if 'ACDC' in dataset:
        return (
        running_dice_RV / len(dataloader),
        running_dice_Myo / len(dataloader),
        running_dice_LV / len(dataloader),
        running_assd_RV / len(dataloader),
        running_assd_Myo / len(dataloader),
        running_assd_LV / len(dataloader),
        running_hd95_RV / len(dataloader),
        running_hd95_Myo / len(dataloader),
        running_hd95_LV / len(dataloader),
        running_ECE_RV / len(dataloader),
        running_ECE_Myo / len(dataloader),
        running_ECE_LV / len(dataloader),
        running_UEO_RV / len(dataloader),
        running_UEO_Myo / len(dataloader),
        running_UEO_LV / len(dataloader)
        )
    elif 'Refuge' in dataset:
        return (
        running_dice_DISC / len(dataloader),
        running_dice_CUP / len(dataloader),
        running_assd_DISC / len(dataloader),
        running_assd_CUP / len(dataloader),
        running_hd95_DISC / len(dataloader),
        running_hd95_CUP / len(dataloader),
        running_ECE_DISC / len(dataloader),
        running_ECE_CUP / len(dataloader),
        running_UEO_DISC / len(dataloader),
        running_UEO_CUP / len(dataloader),
        )
        

def inference_robust(model, dataloader, num_classes, batch_size, device, writer, mu, std, dataset, method='base'):
    model.eval()
    if 'ACDC' in dataset:
        assert num_classes == 4, f'num_classes should be 4, but got {num_classes}'
        running_dice_RV = 0.0
        running_dice_Myo = 0.0
        running_dice_LV = 0.0
        running_assd_RV = 0.0
        running_assd_Myo = 0.0
        running_assd_LV = 0.0
        running_hd95_RV = 0.0
        running_hd95_Myo = 0.0
        running_hd95_LV = 0.0
        running_ECE_RV = 0.0
        running_ECE_Myo = 0.0
        running_ECE_LV = 0.0
        running_UEO_RV = 0.0
        running_UEO_Myo = 0.0
        running_UEO_LV = 0.0
    elif 'Refuge' in dataset:
        assert num_classes == 3, f'num_classes should be 3, but got {num_classes}'
        running_dice_DISC = 0.0
        running_dice_CUP = 0.0
        running_assd_DISC = 0.0
        running_assd_CUP = 0.0
        running_hd95_DISC = 0.0
        running_hd95_CUP = 0.0
        running_ECE_DISC = 0.0
        running_ECE_CUP = 0.0
        running_UEO_DISC = 0.0
        running_UEO_CUP = 0.0
    
    C = num_classes
    total_samples = len(dataloader.dataset)
    print(f"Using Device: {device}")
    print('Inference in progress...')
    
    with torch.no_grad(), tqdm(total=total_samples) as progress_bar:
        if 'ACDC' in dataset:
            # _, assd_dict, _, _ = ACDC2017_Evaluation(files[fold], model)
            total_assds = np.zeros(4)
        elif 'Refuge' in dataset:
            # _, assd_dict, _, _ = Refuge_Evaluation(files, model, save_DIR=refuge_save_dir, epoch=1)
            total_assds = np.zeros(3)
        elif 'ISIC' in dataset:
            # _, assd_dict, _, _ = ISIC_Evaluation(files, model, save_DIR=isic_save_dir, epoch=1)
            total_assds = np.zeros(1)
        
        # num_patients = len(assd_dict)
        num_patients = 1
        # for _, assd_values in assd_dict.items():
        #     total_assds += assd_values
        average_assds = total_assds / num_patients
        for batch_idx, (images, labels, spacing) in enumerate(dataloader):
            # spacing = spacing.squeeze(1)
            img, labels = images.to(device), labels.to(device) # [N,1,H,W], [N,C,H,W]
            images = generate_noisy_images(img, device, mu=mu, sigma=std, seed=1)
            soft_label = torch.argmax(labels, dim=1) # [N,H,W]
            targets = labels.permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW, C]
            N, _, H, W = images.size()  
            if method == 'base' or method == 'devis':
                if method == 'base':
                    pred = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C) # [NHW,C]
                    evidence = F.softplus(pred)
                elif method == 'devis':
                    evidence = model(images).permute(0, 2, 3, 1).contiguous().view(-1, C)
                
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
                u = num_classes / S # [NHW,1]
                u_view = u.view(N, H, W, 1).squeeze(-1) # [N,H,W]
                prob = alpha / S # [NHW,C]
                # For visualization
                prob_view = prob.view(N, H, W, C).permute(0, 3, 1, 2)
                prob_indices = torch.argmax(prob_view, dim=1) # [N,H,W]

            elif method == 'pu' or method == 'flow' or method == 'glow' or method == 'udrop':
                if method == 'pu' or method == 'flow' or method == 'glow':
                    if 'Refuge-no' in dataset:
                        resized_images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
                    else:
                        resized_images = images
                    model.forward(resized_images, labels.float(), training=False)
                    logits = model_PU(resized_images, model)
                    del resized_images
                    if 'Refuge-no' in dataset:
                        logits = F.interpolate(logits, size=(512, 512), mode='nearest')
                elif method == 'udrop':
                    logits = model(images)

                prob_view = F.softmax(logits, dim=1)
                prob_indices = torch.argmax(prob_view, dim=1)
                prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                u_view = Uentropy(logits, C).squeeze(1)
                u = u_view.view(-1, 1)

            else:
                if method == 'eu':
                    for i in range(4):
                        logits = model[i](images)
                        if i == 0:
                            prob_view = F.softmax(logits, dim=1)
                            u_view = Uentropy(logits, C)
                        else:
                            prob_view += F.softmax(logits, dim=1)
                            u_view += Uentropy(logits, C)

                    prob_view /= 4.
                    u_view /= 4.
                    prob_indices = torch.argmax(prob_view, dim=1)
                    prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                    u = u_view.view(-1, 1)

                elif method == 'tta':
                    # defined 2 * 2 * 3 * 3 = 36 augmentations !
                    transforms_img = tta.Compose(
                        [
                            tta.HorizontalFlip(),
                            tta.Rotate90(angles=[0, 180]),
                            tta.Scale(scales=[1, 2, 4]),
                            tta.Multiply(factors=[0.9, 1, 1.1]),        
                        ]
                    )
                    tta_model = tta.SegmentationTTAWrapper(model, transforms_img)
                    logits = tta_model(images)
                    prob_view = F.softmax(logits, dim=1)
                    prob_indices = torch.argmax(prob_view, dim=1)
                    prob = prob_view.permute(0, 2, 3, 1).contiguous().view(-1, C)
                    u_view = Uentropy(logits, C)
                    u = u_view.view(-1, 1)

            threshold = np.arange(0.01, 1.0, 0.01)
            if 'Refuge' not in dataset:
                ueo = calculate_ueo(prob, targets, u, threshold, num_classes=num_classes)
                ece = calculate_ece(prob, targets, num_classes=num_classes)
                dice = calculate_dice(targets, prob, epsilon=1e-5, device=device, num_classes=num_classes)
            
                hd = soft_hd95(soft_label, prob_indices, num_classes=num_classes)
            else:
                ueo_DISC, ueo_CUP = calculate_ueo_Refuge(prob, targets, u, threshold)
                ece_DISC, ece_CUP = calculate_ece_Refuge(prob, targets)
                dice_DISC, dice_CUP = calculate_dice_Refuge(targets, prob, epsilon=1e-5, device=device)
                hd95_DISC, hd95_CUP = soft_hd95_Refuge(soft_label, prob_indices)
                assd_DISC, assd_CUP = average_assds[1], average_assds[2]
            
            if 'ACDC' in dataset:
                ueo_RV, ueo_Myo, ueo_LV = ueo[0], ueo[1], ueo[2]
                ece_RV, ece_Myo, ece_LV = ece[0], ece[1], ece[2]
                dice_RV, dice_Myo, dice_LV = dice[0], dice[1], dice[2]
                hd95_RV, hd95_Myo, hd95_LV = hd[0], hd[1], hd[2]
                assd_RV, assd_Myo, assd_LV = average_assds[1], average_assds[2], average_assds[3]

            if batch_idx % 10 == 0:
                for i in range(prob_view.shape[0]):
                    img_slice = images[i, :, :, :].cpu().numpy().astype(np.float32) # [M,H,W]
                    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
                    u_slice = u_view[i, :, :].detach().cpu().numpy()  # [H,W]
                    prob_slice = prob_indices[i, :, :].detach().cpu().numpy() # [H,W]
                    label_slice = soft_label[i, :, :].detach().cpu().numpy()  # [H,W]
                    prob_color = apply_color_map(prob_slice) # [H,W,3]
                    label_color = apply_color_map(label_slice) # [H,W,3]
                    u_color = apply_heatmap(u_slice) # [H,W,3]
                    prob_color = prob_color.transpose(2, 0, 1)  # [3,H,W]
                    label_color = label_color.transpose(2, 0, 1)  # [3,H,W]
                    u_color = u_color.transpose(2, 0, 1)  # [3,H,W]

                    u_slice = u_view[i, :, :].detach().cpu().numpy() # [H,W]
                    u_color = apply_heatmap(u_slice) # [H,W,3]
                    u_color = u_color.transpose(2, 0, 1)  # [3,H0,W]
                    prob_slice = prob_indices[i, :, :].detach().cpu().numpy() # [H,W]
                    prob_color = apply_color_map(prob_slice)
                    prob_color = prob_color.transpose(2, 0, 1)
                    writer.add_image(f'(Test) Predicted Mask (mu={mu})/{batch_idx*batch_size+i}', prob_color, dataformats='CHW')
                    writer.add_image(f'(Test) Ground Truth/{batch_idx*batch_size+i}', label_color, dataformats='CHW')
                    writer.add_image(f'(Test) Uncertainty (mu={mu})/{batch_idx*batch_size+i}', u_color, dataformats='CHW')
                    del img_slice, prob_slice, label_slice, prob_color, label_color, u_color
                    torch.cuda.empty_cache()
            if 'ACDC' in dataset:
                running_dice_RV += dice_RV.item()
                running_dice_Myo += dice_Myo.item()
                running_dice_LV += dice_LV.item()
                running_assd_RV += assd_RV
                running_assd_Myo += assd_Myo
                running_assd_LV += assd_LV
                running_hd95_RV += hd95_RV
                running_hd95_Myo += hd95_Myo
                running_hd95_LV += hd95_LV
                running_ECE_RV += ece_RV
                running_ECE_Myo += ece_Myo
                running_ECE_LV += ece_LV
                running_UEO_RV += ueo_RV
                running_UEO_Myo += ueo_Myo
                running_UEO_LV += ueo_LV
            elif 'Refuge' in dataset:
                running_dice_DISC += dice_DISC.item()
                running_dice_CUP += dice_CUP.item()
                running_assd_DISC += assd_DISC
                running_assd_CUP += assd_CUP
                running_hd95_DISC += hd95_DISC
                running_hd95_CUP += hd95_CUP
                running_ECE_DISC += ece_DISC
                running_ECE_CUP += ece_CUP
                running_UEO_DISC += ueo_DISC
                running_UEO_CUP += ueo_CUP
            progress_bar.update(images.size(0))
    if 'ACDC' in dataset:
        return (
        running_dice_RV / len(dataloader),
        running_dice_Myo / len(dataloader),
        running_dice_LV / len(dataloader),
        running_assd_RV / len(dataloader),
        running_assd_Myo / len(dataloader),
        running_assd_LV / len(dataloader),
        running_hd95_RV / len(dataloader),
        running_hd95_Myo / len(dataloader),
        running_hd95_LV / len(dataloader),
        running_ECE_RV / len(dataloader),
        running_ECE_Myo / len(dataloader),
        running_ECE_LV / len(dataloader),
        running_UEO_RV / len(dataloader),
        running_UEO_Myo / len(dataloader),
        running_UEO_LV / len(dataloader)
        )
    elif 'Refuge' in dataset:
        return (
        running_dice_DISC / len(dataloader),
        running_dice_CUP / len(dataloader),
        running_assd_DISC / len(dataloader),
        running_assd_CUP / len(dataloader),
        running_hd95_DISC / len(dataloader),
        running_hd95_CUP / len(dataloader),
        running_ECE_DISC / len(dataloader),
        running_ECE_CUP / len(dataloader),
        running_UEO_DISC / len(dataloader),
        running_UEO_CUP / len(dataloader),
        )
