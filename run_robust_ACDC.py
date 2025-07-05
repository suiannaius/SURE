import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import torch
import torch.nn as nn
import shutil
import logging
import json
import numpy as np
from torch.utils.data import DataLoader
from utilities.utils import load_args
from training.dataset import Generate_ACDC_Train_Val_Test_List, ACDC2017_Dataset
from training.inference import inference_robust
from model.UNet2DZoo import Unet2D, AttUnet2D, Unet2Ddrop
from model.TrustworthySeg import TMSU
from model.probabilistic_unet2D import ProbabilisticUnet2D
from model.cFlowNet import cFlowNet
from torch.utils.tensorboard import SummaryWriter


def run_robust(project_name, consider_spacing=False, method='base'):
    
    datapath = '/home/liyuzhu/MERU+EDL/ACDC2017/data/ACDC/training'
    args = load_args('./configs/' + project_name + '_config.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, Vali_ImgFiles, _ = Generate_ACDC_Train_Val_Test_List(datapath, val_ratio=0.2)

    mu_list = [0.1, 0.3, 0.5, 0.7, 0.9] # 0.0
    std_list = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
    
    log_file = f'./results/log/{project_name}.txt'
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    fold_dices = {f'fold_{i}': {} for i in range(5)}
    fold_assds = {f'fold_{i}': {} for i in range(5)}
    fold_hd95s = {f'fold_{i}': {} for i in range(5)}
    fold_ueos = {f'fold_{i}': {} for i in range(5)}
    fold_eces = {f'fold_{i}': {} for i in range(5)}

    logging.basicConfig(filename=log_file,
                    format = '%(asctime)s - %(name)s - %(message)s',
                    level=logging.INFO,
                    filemode='w')
    
    if method == 'base':
        logging.info(
              f"\nProject Name: {project_name}"
              f"\nBackbone: {args.backbone}Net"
              f"\nBase channels: {args.base_channels}"
              f"\nBeta: {args.beta}"
              f"\nGamma: {args.gamma}"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/inference_{project_name}")
        print(f"\nProject Name: {project_name}"
              f"\nBackbone: {args.backbone}Net"
              f"\nBase channels: {args.base_channels}"
              f"\nBeta: {args.beta}"
              f"\nGamma: {args.gamma}"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/inference_{project_name}")
    else:
        if method == 'devis':
            logging.info(
              f"\nProject Name: {project_name}"
              f"\nBackbone: {args.model_name}Net"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/inference_{project_name}")
            print(f"\nProject Name: {project_name}"
              f"\nBackbone: {args.model_name}Net"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/inference_{project_name}")
        else:
            logging.info(
              f"\nProject Name: {project_name}"
              f"\nBackbone: {args.backbone}Net"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/robust_{project_name}")
            print(f"\nProject Name: {project_name}"
              f"\nBackbone: {args.backbone}Net"
              f"\nRatio: {args.ratio}"
              f"\nNum of modalities: {args.num_modalities}"
              f"\nWriter: tensorboard --logdir=./results/writer/robust_{project_name}")
    
    for i in range(len(mu_list)):  # len(mu_list)
        mu = mu_list[i]
        std = std_list[i]
        log_dir=f'./results/writer/robust_{project_name}_mu_{mu}'
        if not os.path.exists(os.path.dirname(log_dir)):
            os.makedirs(os.path.dirname(log_dir))
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"Current mu: {mu}, std: {std}")
        for fold in range(5):
            print(f"Current fold: {fold}")
            if method == 'base':
                if args.backbone == 'U':
                    model = Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
                elif args.backbone == 'AU':
                    model = AttUnet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)

            elif method == 'devis':
                model = TMSU(args)

            elif method == 'pu':
                print(args.num_modalities)
                print(args.num_classes)
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
            val_dataset = ACDC2017_Dataset(Vali_ImgFiles[fold])
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                            pin_memory=True, drop_last=True)
            
            dice_RV, dice_Myo, dice_LV, assd_RV, assd_Myo, assd_LV, HD95_RV, HD95_Myo, HD95_LV, \
            ECE_RV, ECE_Myo, ECE_LV, UEO_RV, UEO_Myo, UEO_LV = inference_robust(model=model,
                                                                                dataloader=val_loader, 
                                                                                num_classes=args.num_classes,
                                                                                batch_size=args.batch_size,
                                                                                device=device,
                                                                                writer=writer,
                                                                                mu=mu,
                                                                                std=std,
                                                                                dataset='ACDC',
                                                                                method=method)
            
            fold_dices[f'fold_{fold}']['RV'] = dice_RV
            fold_dices[f'fold_{fold}']['Myo'] = dice_Myo
            fold_dices[f'fold_{fold}']['LV'] = dice_LV
        
            fold_assds[f'fold_{fold}']['RV'] = assd_RV
            fold_assds[f'fold_{fold}']['Myo'] = assd_Myo
            fold_assds[f'fold_{fold}']['LV'] = assd_LV
        
            fold_hd95s[f'fold_{fold}']['RV'] = HD95_RV
            fold_hd95s[f'fold_{fold}']['Myo'] = HD95_Myo
            fold_hd95s[f'fold_{fold}']['LV'] = HD95_LV
            
            fold_ueos[f'fold_{fold}']['RV'] = UEO_RV
            fold_ueos[f'fold_{fold}']['Myo'] = UEO_Myo
            fold_ueos[f'fold_{fold}']['LV'] = UEO_LV
            
            fold_eces[f'fold_{fold}']['RV'] = ECE_RV
            fold_eces[f'fold_{fold}']['Myo'] = ECE_Myo
            fold_eces[f'fold_{fold}']['LV'] = ECE_LV
            

        dice_RVs = [fold_dices[f'fold_{i}']['RV'] for i in range(5)]
        dice_Myos = [fold_dices[f'fold_{i}']['Myo'] for i in range(5)]
        dice_LVs = [fold_dices[f'fold_{i}']['LV'] for i in range(5)]
        mean_dice_RV = np.mean(dice_RVs)
        std_dice_RV = np.std(dice_RVs)
        mean_dice_Myo = np.mean(dice_Myos)
        std_dice_Myo = np.std(dice_Myos)
        mean_dice_LV = np.mean(dice_LVs)
        std_dice_LV = np.std(dice_LVs)
        fold_dices['mean_and_std'] = {
            'RV': f'{mean_dice_RV:.4f}±{std_dice_RV:.4f}',
            'Myo': f'{mean_dice_Myo:.4f}±{std_dice_Myo:.4f}',
            'LV': f'{mean_dice_LV:.4f}±{std_dice_LV:.4f}'}

        assd_RVs = [fold_assds[f'fold_{i}']['RV'] for i in range(5)]
        assd_Myos = [fold_assds[f'fold_{i}']['Myo'] for i in range(5)]
        assd_LVs = [fold_assds[f'fold_{i}']['LV'] for i in range(5)]
        mean_assd_RV = np.mean(assd_RVs)
        std_assd_RV = np.std(assd_RVs)
        mean_assd_Myo = np.mean(assd_Myos)
        std_assd_Myo = np.std(assd_Myos)
        mean_assd_LV = np.mean(assd_LVs)
        std_assd_LV = np.std(assd_LVs)
        fold_assds['mean_and_std'] = {
            'RV': f'{mean_assd_RV:.4f}±{std_assd_RV:.4f}',
            'Myo': f'{mean_assd_Myo:.4f}±{std_assd_Myo:.4f}',
            'LV': f'{mean_assd_LV:.4f}±{std_assd_LV:.4f}'}

        hd95_RVs = [fold_hd95s[f'fold_{i}']['RV'] for i in range(5)]
        hd95_Myos = [fold_hd95s[f'fold_{i}']['Myo'] for i in range(5)]
        hd95_LVs = [fold_hd95s[f'fold_{i}']['LV'] for i in range(5)]
        mean_hd95_RV = np.mean(hd95_RVs)
        std_hd95_RV = np.std(hd95_RVs)
        mean_hd95_Myo = np.mean(hd95_Myos)
        std_hd95_Myo = np.std(hd95_Myos)
        mean_hd95_LV = np.mean(hd95_LVs)
        std_hd95_LV = np.std(hd95_LVs)
        fold_hd95s['mean_and_std'] = {
            'RV': f'{mean_hd95_RV:.4f}±{std_hd95_RV:.4f}',
            'Myo': f'{mean_hd95_Myo:.4f}±{std_hd95_Myo:.4f}',
            'LV': f'{mean_hd95_LV:.4f}±{std_hd95_LV:.4f}'}
        
        ueo_RVs = [fold_ueos[f'fold_{i}']['RV'] for i in range(5)]
        ueo_Myos = [fold_ueos[f'fold_{i}']['Myo'] for i in range(5)]
        ueo_LVs = [fold_ueos[f'fold_{i}']['LV'] for i in range(5)]
        mean_ueo_RV = np.mean(ueo_RVs)
        std_ueo_RV = np.std(ueo_RVs)
        mean_ueo_Myo = np.mean(ueo_Myos)
        std_ueo_Myo = np.std(ueo_Myos)
        mean_ueo_LV = np.mean(ueo_LVs)
        std_ueo_LV = np.std(ueo_LVs)
        fold_ueos['mean_and_std'] = {
            'RV': f'{mean_ueo_RV:.4f}±{std_ueo_RV:.4f}',
            'Myo': f'{mean_ueo_Myo:.4f}±{std_ueo_Myo:.4f}',
            'LV': f'{mean_ueo_LV:.4f}±{std_ueo_LV:.4f}'}
        
        ece_RVs = [fold_eces[f'fold_{i}']['RV'] for i in range(5)]
        ece_Myos = [fold_eces[f'fold_{i}']['Myo'] for i in range(5)]
        ece_LVs = [fold_eces[f'fold_{i}']['LV'] for i in range(5)]
        mean_ece_RV = np.mean(ece_RVs)
        std_ece_RV = np.std(ece_RVs)
        mean_ece_Myo = np.mean(ece_Myos)
        std_ece_Myo = np.std(ece_Myos)
        mean_ece_LV = np.mean(ece_LVs)
        std_ece_LV = np.std(ece_LVs)
        fold_eces['mean_and_std'] = {
            'RV': f'{mean_ece_RV:.4f}±{std_ece_RV:.4f}',
            'Myo': f'{mean_ece_Myo:.4f}±{std_ece_Myo:.4f}',
            'LV': f'{mean_ece_LV:.4f}±{std_ece_LV:.4f}'}   

        all_results = {
        'Dice': fold_dices,
        'ASSD': fold_assds,
        'HD95': fold_hd95s,
        'UEO': fold_ueos,
        'ECE': fold_eces
        }
        json_path = f'./results/robust/ACDC/{project_name}.json'
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, separators=(',', ': '), ensure_ascii=False)


if __name__ == "__main__":
    
    project_name = 'Task_999_2025-06-25'
    method = 'base'  # 'base' / 'devis' / 'pu' / 'flow' / 'glow' / 'udrop'
    consider_spacing = True
    run_robust(project_name, consider_spacing=consider_spacing, method=method)
