import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import json
import numpy as np
import torch.nn.functional as F
import ttach as tta
from tqdm import tqdm
from utilities.count_pixels import count_pixels_d, count_pixels_d_chunk, count_pixels_mu, count_pixels_grad, count_corr_mu
from torch.utils.data import DataLoader
from utilities.utils import load_args, generate_noisy_images, compute_distance_map, generate_blurred_images, compute_gradient
from training.dataset import Generate_ACDC_Train_Val_Test_List, Generate_Refuge_Train_Val_Test_List, ACDC2017_Dataset, Refuge_Dataset
from model.UNet2DZoo import Unet2D, AttUnet2D, Unet2Ddrop
from model.TrustworthySeg import TMSU
from model.probabilistic_unet2D import ProbabilisticUnet2D
from model.cFlowNet import cFlowNet
from training.trainPU import model_PU, Uentropy
from torch.utils.tensorboard import SummaryWriter


def run_count(project_name, method='base'):
    
    args = load_args('./configs/' + project_name + '_config.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if method == 'base':
        print(
            f"\nDataset: {args.dataset}"
            f"\nProject Name: {project_name}"
            f"\nBackbone: {args.backbone}Net"
            f"\nBeta: {args.beta}"
            f"\nGamma: {args.gamma}"
            f"\nCoef_mu: {args.coef_mu}"
            f"\nCoef_d: {args.coef_d}"
            f"\nCoef_far: {args.coef_far}"
            f"\nRatio: {args.ratio}")
    elif method == 'devis':
        print(
            f"\nDataset: {args.dataset}"
            f"\nProject Name: {project_name}"
            f"\nBackbone: {args.model_name}Net"
            f"\nRatio: {args.ratio}")
    else:
        print(
            f"\nDataset: {args.dataset}"
            f"\nProject Name: {project_name}"
            f"\nBackbone: {args.backbone}Net"
            f"\nRatio: {args.ratio}")

    if 'ACDC' in args.dataset:
        dataset = 'ACDC'
        assert args.num_classes == 4, f"Wrong num_classes, 4 expected, got {args.num_classes}"
        fold_num = 5
        
    elif 'Refuge' in args.dataset:
        dataset = 'Refuge'
        assert args.num_classes == 3, f"Wrong num_classes, 3 expected, got {args.num_classes}"
        fold_num = 1
        
    fold_corr_mu = {f'fold_{i}': {} for i in range(fold_num)}
    fold_corr_d = {f'fold_{i}': {} for i in range(fold_num)}
    fold_corr_g = {f'fold_{i}': {} for i in range(fold_num)}
    fold_ratio_mu = {f'fold_{i}': {} for i in range(fold_num)}
    fold_ratio_d = {f'fold_{i}': {} for i in range(fold_num)}
    fold_ratio_g = {f'fold_{i}': {} for i in range(fold_num)}
    
    for fold in range(fold_num):
        if 'ACDC' in args.dataset:
            datapath = '/home/liyuzhu/MERU+EDL/ACDC2017/data/ACDC/training'
            _, Vali_ImgFiles, _ = Generate_ACDC_Train_Val_Test_List(datapath, val_ratio=0.2)
            test_dataset = ACDC2017_Dataset(Vali_ImgFiles[fold])
        elif 'Refuge' in args.dataset:
            datapath = '/home/liyuzhu/MERU+EDL/Refuge'
            _, _, Test_ImgFiles = Generate_Refuge_Train_Val_Test_List(datapath, seed=args.seed, shuffle=True)
            test_dataset = Refuge_Dataset(Test_ImgFiles, times=1, train=False)
    
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                pin_memory=True, drop_last=True, num_workers=20)
        total_samples = len(test_loader.dataset)
        num_classes = args.num_classes

        mu1, mu2 = 0.1, 0.9
        std = 0.3
        total_ratio_mu, total_ratio_d, total_ratio_grad, total_corr_mu, total_corr_d, total_corr_g = 0., 0., 0., 0., 0., 0.
        with torch.no_grad(), tqdm(total=total_samples) as progress_bar:
            print(f"Current fold: {fold}")

            if method == 'base':
                if args.backbone == 'U':
                    model = Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
                elif args.backbone == 'AU':
                    model = AttUnet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)

            elif method == 'devis':
                model = TMSU(args)

            elif method == 'pu':
                model = ProbabilisticUnet2D(input_channels=args.num_modalities, num_classes=args.num_classes, num_filters=[64, 128, 256, 512], latent_dim=2,
                                    no_convs_fcomb=4, beta=10.0).to(device)

            elif method == 'udrop':
                model = Unet2Ddrop(in_ch=args.num_modalities, out_ch=args.num_classes).to(device)

            elif method == 'flow':
                model = cFlowNet(input_channels=args.num_modalities, num_classes=args.num_classes, 
	    		    num_filters=[32,64,128,256], latent_dim=6,  
            	    no_convs_fcomb=4, num_flows=args.num_flows, 
	    		    norm=True,flow=True,glow=False).to(device)

            elif method == 'glow':
                model = cFlowNet(input_channels=args.num_modalities, num_classes=args.num_classes,
	    		    num_filters=[32,64,128,256], latent_dim=6, 
	    		    no_convs_fcomb=4, num_flows=args.num_flows,
	    		    norm=True,flow=True,glow=True).to(device)

            elif method == 'eu':
                model = nn.ModuleList([Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device) for _ in range(4)])

            elif method == 'tta':
                model = Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)

            if method == 'eu':
                for i in range(4):
                    load_name = f'./saved_models/{project_name}_fold_{fold}_{i}.pth'
                    model[i].load_state_dict(torch.load(load_name, weights_only=True))
            else:
                load_name = f'./saved_models/{project_name}_fold_{fold}.pth'
                model.load_state_dict(torch.load(load_name, weights_only=True))

            for batch_idx, (images, labels, _) in enumerate(test_loader):

                torch.manual_seed(batch_idx)
                img, labels = images.to(device), labels.to(device) # [N,1,H,W], [N,C,H,W]
                images_mu1 = generate_noisy_images(img, device, mu=mu1, sigma=std, seed=1)
                images_mu2 = generate_noisy_images(img, device, mu=mu2, sigma=std, seed=1)
                d = compute_distance_map(labels)  # [NHW, 1]
                blurred_images = generate_blurred_images(img, sigma_blur=0.0, device=device)  
                gradient_blurred = compute_gradient(blurred_images) # [NHW,1]

                if method == 'base' or method == 'devis':
                    if method == 'base':
                        pred = model(img).permute(0, 2, 3, 1).contiguous().view(-1, num_classes) # [NHW,C]
                        evidence = F.softplus(pred)

                        pred_mu1 = model(images_mu1).permute(0, 2, 3, 1).contiguous().view(-1, num_classes) # [NHW,C]
                        evidence_mu1 = F.softplus(pred_mu1)

                        pred_mu2 = model(images_mu2).permute(0, 2, 3, 1).contiguous().view(-1, num_classes) # [NHW,C]
                        evidence_mu2 = F.softplus(pred_mu2)

                    elif method == 'devis':
                        evidence = model(img).permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
                        evidence_mu1 = model(images_mu1).permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
                        evidence_mu2 = model(images_mu2).permute(0, 2, 3, 1).contiguous().view(-1, num_classes)

                    alpha = evidence + 1
                    alpha_mu1 = evidence_mu1 + 1
                    alpha_mu2 = evidence_mu2 + 1

                    S = torch.sum(alpha, dim=1, keepdim=True) # [NHW,1]
                    S_mu1 = torch.sum(alpha_mu1, dim=1, keepdim=True) # [NHW,1]
                    S_mu2 = torch.sum(alpha_mu2, dim=1, keepdim=True) # [NHW,1]

                    u = num_classes / S # [NHW,1]
                    u_mu1 = num_classes / S_mu1 # [NHW,1]
                    u_mu2 = num_classes / S_mu2 # [NHW,1]

                elif method == 'pu' or method == 'flow' or method == 'glow' or method == 'udrop':
                    if method == 'pu' or method == 'flow' or method == 'glow':
                        if 'Refuge-no' in dataset:
                            resized_images = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
                            resized_images_mu1 = F.interpolate(images_mu1, size=(256, 256), mode='bilinear', align_corners=False)
                            resized_images_mu2 = F.interpolate(images_mu2, size=(256, 256), mode='bilinear', align_corners=False)
                        else:
                            resized_images = img
                            resized_images_mu1 = images_mu1
                            resized_images_mu2 = images_mu2

                        model.forward(resized_images, labels.float(), training=False)

                        logits = model_PU(resized_images, model)
                        logits_mu1 = model_PU(resized_images_mu1, model)
                        logits_mu2 = model_PU(resized_images_mu2, model)
                        del resized_images, resized_images_mu1, resized_images_mu2
                        if 'Refuge-no' in dataset:
                            logits = F.interpolate(logits, size=(512, 512), mode='nearest')
                            logits_mu1 = F.interpolate(logits_mu1, size=(512, 512), mode='nearest')
                            logits_mu2 = F.interpolate(logits_mu2, size=(512, 512), mode='nearest')
                    elif method == 'udrop':
                        logits = model(img)
                        logits_mu1 = model(images_mu1)
                        logits_mu2 = model(images_mu2)

                    if logits.ndim == 3:
                        logits = logits.unsqueeze(0)
                    if logits_mu1.ndim == 3:
                        logits_mu1 = logits_mu1.unsqueeze(0)
                    if logits_mu2.ndim == 3:
                        logits_mu2 = logits_mu2.unsqueeze(0)
                    u = Uentropy(logits, num_classes).view(-1, 1)
                    u_mu1 = Uentropy(logits_mu1, num_classes).view(-1, 1)
                    u_mu2 = Uentropy(logits_mu2, num_classes).view(-1, 1)

                else:
                    if method == 'eu':
                        for i in range(4):
                            logits = model[i](img)
                            logits_mu1 = model[i](images_mu1)
                            logits_mu2 = model[i](images_mu2)
                            if i == 0:
                                u = Uentropy(logits, num_classes).view(-1, 1)
                                u_mu1 = Uentropy(logits_mu1, num_classes).view(-1, 1)
                                u_mu2 = Uentropy(logits_mu2, num_classes).view(-1, 1)
                            else:
                                u += Uentropy(logits, num_classes).view(-1, 1)
                                u_mu1 += Uentropy(logits_mu1, num_classes).view(-1, 1)
                                u_mu2 += Uentropy(logits_mu2, num_classes).view(-1, 1)

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
                        logits = tta_model(img)
                        logits_mu1 = tta_model(images_mu1)
                        logits_mu2 = tta_model(images_mu2)
                        u = Uentropy(logits, num_classes).view(-1, 1)
                        u_mu1 = Uentropy(logits_mu1, num_classes).view(-1, 1)
                        u_mu2 = Uentropy(logits_mu2, num_classes).view(-1, 1)

                ratio_mu = count_pixels_mu(u_mu1, u_mu2, mu1, mu2, d, device, threshold=4)
                
                corr_mu = count_corr_mu(u_mu1, u_mu2, mu1, mu2, d, device, threshold=4)
                ratio_d, corr_d = count_pixels_d_chunk(u, u_mu1, d, args.batch_size, device, threshold=4)
                ratio_grad, corr_g = count_pixels_grad(u, gradient_blurred, d, args.batch_size, 
                                               device, threshold=1)
                total_corr_mu += corr_mu.item()
                total_ratio_mu += ratio_mu.item()
                total_corr_d += corr_d.item()
                total_ratio_d += ratio_d.item()
                total_corr_g += corr_g.item()
                total_ratio_grad += ratio_grad.item()
                progress_bar.update(images.size(0))
        
        fold_corr_mu[f'fold_{fold}'] = total_corr_mu / len(test_loader)
        fold_corr_d[f'fold_{fold}'] = total_corr_d / len(test_loader)
        fold_corr_g[f'fold_{fold}'] = total_corr_g / len(test_loader)

        fold_ratio_mu[f'fold_{fold}'] = total_ratio_mu / len(test_loader)
        fold_ratio_d[f'fold_{fold}'] = total_ratio_d / len(test_loader)
        fold_ratio_g[f'fold_{fold}'] = total_ratio_grad / len(test_loader)


    corr_mus = [fold_corr_mu[f'fold_{i}'] for i in range(fold_num)]
    mean_corr_mus = np.mean(corr_mus)
    std_corr_mus = np.std(corr_mus)

    corr_ds = [fold_corr_d[f'fold_{i}'] for i in range(fold_num)]
    mean_corr_ds = np.mean(corr_ds)
    std_corr_ds = np.std(corr_ds)

    corr_gs = [fold_corr_g[f'fold_{i}'] for i in range(fold_num)]
    mean_corr_gs = np.mean(corr_gs)
    std_corr_gs = np.std(corr_gs)

    ratio_mus = [fold_ratio_mu[f'fold_{i}'] for i in range(fold_num)]
    mean_ratio_mus = np.mean(ratio_mus)
    std_ratio_mus = np.std(ratio_mus)

    ratio_ds = [fold_ratio_d[f'fold_{i}'] for i in range(fold_num)]
    mean_ratio_ds = np.mean(ratio_ds)
    std_ratio_ds = np.std(ratio_ds)

    ratio_gs = [fold_ratio_g[f'fold_{i}'] for i in range(fold_num)]
    mean_ratio_gs = np.mean(ratio_gs)
    std_ratio_gs = np.std(ratio_gs)

    fold_corr_mu['mean_and_std'] = f'{mean_corr_mus:.4f}±{std_corr_mus:.4f}'
    
    fold_corr_d['mean_and_std'] = f'{mean_corr_ds:.4f}±{std_corr_ds:.4f}'
    
    fold_corr_g['mean_and_std'] = f'{mean_corr_gs:.4f}±{std_corr_gs:.4f}'
    
    fold_ratio_mu['mean_and_std'] = f'{mean_ratio_mus:.4f}±{std_ratio_mus:.4f}'
    
    fold_ratio_d['mean_and_std'] = f'{mean_ratio_ds:.4f}±{std_ratio_ds:.4f}'
    
    fold_ratio_g['mean_and_std'] = f'{mean_ratio_gs:.4f}±{std_ratio_gs:.4f}'
    
    all_results = {
    'Corr mu': fold_corr_mu,
    'Corr d': fold_corr_d,
    'Corr g': fold_corr_g,
    'Ratio mu': fold_ratio_mu,
    'Ratio d': fold_ratio_d,
    'Ratio g': fold_ratio_g,
    }
    print(all_results)

    json_path = f'./results/count_all/{dataset}/{project_name}.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, separators=(',', ': '), ensure_ascii=False)


if __name__ == "__main__":
    
    project_name = 'Task_999_2025-06-25'
    run_count(project_name, method='base')
