import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import glob
import torch
import logging
import time
import argparse
import shutil
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from model.UNet2DZoo import Unet2D, AttUnet2D
from training.train import train
from utilities.weights_init import weights_init_kaiming
from utilities.utils import save_args
from training.dataset import ACDC2017_Dataset, Refuge_Dataset, Generate_ACDC_Train_Val_Test_List, Generate_Refuge_Train_Val_Test_List
from torch.utils.data import DataLoader
from training.criterions import edl_loss, edl_loss_a
from torch.utils.tensorboard import SummaryWriter


def getArgs():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser.add_argument('--dataset', required=True, type=str, default=None)
    parser.add_argument('--user', default='suian', type=str)
    parser.add_argument('--experiment', default='Supervised UnceRtainty Estimation', type=str)
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--task_id', default=999, type=int)
    parser.add_argument('--seed', default=926, type=int)
    parser.add_argument('--description', default='Noise-and-gradient-based uncertainty supervision.', type=str)
    # Training detalis
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train [default: 100]')
    parser.add_argument('--annealing_steps', default=20, type=int, help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--early_stop_steps', default=20, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta', default=0.0, type=float, help='coefficient of gradient loss [default: 0.0]')
    parser.add_argument('--gamma', default=0.0, type=float, help='coefficient of noise loss [default: 0.0]')
    parser.add_argument('--coef_mu', default=0.0, type=float, help='coefficient of noise_loss_mu [default: 0.0]')
    parser.add_argument('--coef_d', default=0.0, type=float, help='coefficient of noise_loss_d [default: 0.0]')
    parser.add_argument('--coef_far', default=0.0, type=float, help='coefficient of noise_loss_far [default: 0.0]')
    parser.add_argument('--ratio', type=float, default=1.0, help='ratio of labeled data')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes [default: 4]')
    parser.add_argument('--num_modalities', default=1, type=int, help='number of modalities [default: 1]')
    parser.add_argument('--batch_size', default=24, type=int, help="2/4/8/16")
    parser.add_argument('--warm_up', default=5, type=int, help="steps to warm up")
    parser.add_argument('--good_model_step', default=30, type=int, help="steps to get a good model")
    parser.add_argument('--num_patch', default=3, type=int, help="patches of each image to calculate noise loss")
    parser.add_argument('--d_threshold', default=4, type=int)
    parser.add_argument('--d_eps', default=0, type=int)
    parser.add_argument('--epsilon', default=0., type=float)
    parser.add_argument('--backbone', default='U', type=str, help="U/V/AU")
    parser.add_argument('--base_channels', default=8, type=int, help="base channels [default: 8]")
    parser.add_argument('--use_pretrain', action='store_true', help="Whether load parameters from pretrained models. True/False")
    parser.add_argument('--use_early_stop', action='store_true', help="Whether use early stop strategy")
    parser.add_argument('--use_grad_clip', action='store_true', help="Whether clip the gradients during training.")
    parser.add_argument('--loss_type', default='digamma', type=str, help="digamma/log/mse [default: digamma]")
    parser.add_argument('--criterion', default='edl_loss_a', type=str)
    parser.add_argument('--num_train_samples', default=None, type=int, help="Number of training samples. If None, all samples will be used for training [default: None]")
    parser.add_argument('--num_val_samples', default=None, type=int, help="Number of validating samples. If None, all samples will be used for validating [default: None]")
    # Dataset Information
    args = parser.parse_args()
    return args


def main():
    args = getArgs()
    args_name = f'./configs/Task_{args.task_id}_{args.date}_config.json'
    save_args(args, args_name)
    project_name = f'Task_{args.task_id}_{args.date}'
    if args.criterion == 'edl_loss_a':
        criterion = edl_loss_a
    elif args.criterion == 'edl_loss':
        criterion = edl_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = './logging/' + project_name + '.txt'
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    if not os.path.isfile(log_file):
        open(log_file, 'w').close()

    logging.basicConfig(filename=log_file,
                        format = '%(asctime)s - %(name)s - %(message)s',
                        level=logging.INFO,
                        filemode='w')
    logging.info('Date: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    logging.info('User: {}'.format(args.user))
    logging.info('Dataset: {}'.format(args.dataset))
    logging.info('Project name: {}'.format(project_name))
    logging.info('Writer: tensorboard --logdir=/home/{}/SURE/runs/{}_fold_x'.format(args.user, project_name))
    logging.info('Log file: {}'.format(log_file))
    logging.info('Total epochs: {}'.format(args.num_epochs))
    logging.info('Warm up steps: {}'.format(args.warm_up))
    logging.info('Annealing steps: {}'.format(args.annealing_steps))
    logging.info('Initial learning rate: {}'.format(args.lr))
    logging.info('Coefficient of Gradient Loss: {}'.format(args.beta))
    logging.info('Coefficient of Noise Loss: {}'.format(args.gamma))
    logging.info('Coefficient of noise_loss_mu: {}'.format(args.coef_mu))
    logging.info('Coefficient of noise_loss_d: {}'.format(args.coef_d))
    logging.info('Coefficient of noise_loss_far: {}'.format(args.coef_far))
    logging.info('Ratio of training data: {}'.format(args.ratio))
    logging.info('Backbone: {}Net'.format(args.backbone))
    logging.info('Base channels: {}'.format(args.base_channels))
    logging.info('Grad clip: {}'.format(args.use_grad_clip))
    logging.info('Use {} loss as evidential loss function'.format(args.loss_type))
    logging.info('Random seed: {}'.format(args.seed))

    np.random.seed(args.seed)
    
    print('Current Dataset: ', args.dataset)
    if 'ACDC' in args.dataset:
        assert args.num_classes == 4, f"Wrong num_classes, 4 expected, got {args.num_classes}"
        start_fold = 0
        end_fold = 5
        datapath = '/home/liyuzhu/MERU+EDL/ACDC2017/data/ACDC/training'
        Train_ImgFiles, _, _ = Generate_ACDC_Train_Val_Test_List(datapath, val_ratio=0.2)
    elif 'Refuge' in args.dataset:
        assert args.num_classes == 3, f"Wrong num_classes, 3 expected, got {args.num_classes}"
        start_fold = 0
        end_fold = 1
        datapath = '/home/liyuzhu/MERU+EDL/Refuge'
        Train_ImgFiles, _, _ = Generate_Refuge_Train_Val_Test_List(datapath, seed=args.seed, shuffle=True)

    for fold in range(start_fold, end_fold):
        log_dir = f'./runs/{project_name}_fold_{fold}'
        if not os.path.exists(os.path.dirname(log_dir)):
            os.makedirs(os.path.dirname(log_dir))
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        
        if args.backbone == 'U':
            model = Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
        elif args.backbone == 'AU':
            model = AttUnet2D(in_channels=args.num_modalities, out_channels=args.num_classes).to(device)
        model.apply(weights_init_kaiming)
        logging.info('Use kaiming_init.')
        pytorch_total_params = []
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total parameters: ', pytorch_total_params)
        logging.info('Total parameters: {}'.format(pytorch_total_params))
        print(f'Writer: tensorboard --logdir=/home/{args.user}/SURE/runs/{project_name}_fold_{fold}')
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        save_name = f'./saved_models/{project_name}_fold_{fold}.pth'
        
        if 'ACDC' in args.dataset:
            supervised_dataset = ACDC2017_Dataset(Train_ImgFiles[fold])
            
        elif 'Refuge' in args.dataset:
            supervised_dataset = Refuge_Dataset(Train_ImgFiles, times=1, num_classes=3)
        
        train_loader = DataLoader(supervised_dataset, batch_size=args.batch_size, shuffle=True,
                                           pin_memory=True, drop_last=True)

        logging.info(f"Current fold: {fold}")
        logging.info(f"Train dataset size: {len(supervised_dataset)}")
        print(f"Current Fold: {fold}")
        print(f"Train dataset size: {len(supervised_dataset)}") 

        for epoch in tqdm(range(args.num_epochs), desc='Epochs'):
            if 'ACDC' in args.dataset:
                train_loss, train_RV, train_Myo, train_LV = train(model=model,
                                                                  dataloader=train_loader, 
                                                                  optimizer=optimizer, 
                                                                  num_classes=args.num_classes,
                                                                  criterion=criterion,
                                                                  current_epoch=epoch,
                                                                  total_epoch=args.num_epochs,
                                                                  annealing_steps=args.annealing_steps,
                                                                  loss_type=args.loss_type,
                                                                  device=device,
                                                                  batch_size=args.batch_size,
                                                                  writer=writer,
                                                                  beta=args.beta,
                                                                  gamma=args.gamma,
                                                                  good_model_step=args.good_model_step,
                                                                  num_patch=args.num_patch,
                                                                  d_threshold=args.d_threshold,
                                                                  d_eps=args.d_eps,
                                                                  epsilon=args.epsilon,
                                                                  use_grad_clip=args.use_grad_clip,
                                                                  sample_size=int(1e7),
                                                                  coefficient_mu=args.coef_mu,
                                                                  coefficient_d=args.coef_d,
                                                                  coefficient_far=args.coef_far,
                                                                  dataset=args.dataset,
                                                                  fold=fold)
            elif 'Refuge' in args.dataset:
                train_loss, train_DISC, train_CUP = train(model=model,
                                                          dataloader=train_loader, 
                                                          optimizer=optimizer, 
                                                          num_classes=args.num_classes,
                                                          criterion=criterion,
                                                          current_epoch=epoch,
                                                          total_epoch=args.num_epochs,
                                                          annealing_steps=args.annealing_steps,
                                                          loss_type=args.loss_type,
                                                          device=device,
                                                          batch_size=args.batch_size,
                                                          writer=writer,
                                                          beta=args.beta,
                                                          gamma=args.gamma,
                                                          good_model_step=args.good_model_step,
                                                          num_patch=args.num_patch,
                                                          d_threshold=args.d_threshold,
                                                          d_eps=args.d_eps,
                                                          epsilon=args.epsilon,
                                                          use_grad_clip=args.use_grad_clip,
                                                          sample_size=int(1e7),
                                                          coefficient_mu=args.coef_mu,
                                                          coefficient_d=args.coef_d,
                                                          coefficient_far=args.coef_far,
                                                          dataset=args.dataset,
                                                          fold=fold)
            if 'ACDC' in args.dataset:
                print(f'''
                    Writer: tensorboard --logdir=/home/{args.user}/SURE/runs/{project_name}
                    Fold {fold}
                    Epoch {epoch+1}/{args.num_epochs}, project: {project_name}, lr: {optimizer.param_groups[0]['lr']}
                    beta: {args.beta}, gamma: {args.gamma}, ratio: {args.ratio}, coef_mu: {args.coef_mu}, coef_d: {args.coef_d}, coef_far: {args.coef_far}
                    Backbone: {args.backbone}
                    (Train) Loss: {train_loss:.3f}, 
                    (Train) RV: {train_RV:.3f}, Myo: {train_Myo:.3f}, LV: {train_LV:.3f}''')
                logging.info(
                    "[Fold {:d}, Epoch {:d}]  lr: {:.7f}\n"
                    "(Train) Loss: {:.3f}  Dice: RV: {:.3f}  Myo: {:.3f}  LV: {:.3f}\n".format(
                        fold,
                        epoch + 1, 
                        optimizer.param_groups[0]['lr'],
                        train_loss, train_RV, train_Myo, train_LV))
            elif 'Refuge' in args.dataset:
                print(f'''
                    Writer: tensorboard --logdir=/home/{args.user}/SURE/runs/{project_name}_fold_{fold}
                    Fold {fold}
                    Epoch {epoch+1}/{args.num_epochs}, project: {project_name}, lr: {optimizer.param_groups[0]['lr']}
                    beta: {args.beta}, gamma: {args.gamma}, ratio: {args.ratio}, coef_mu: {args.coef_mu}, coef_d: {args.coef_d}, coef_far: {args.coef_far}
                    Backbone: {args.backbone}
                    (Train) Loss: {train_loss:.3f}, 
                    (Train) DISC: {train_DISC:.3f}, CUP: {train_CUP:.3f}''')
                logging.info(
                    "[Fold {:d}, Epoch {:d}]  lr: {:.7f}\n"
                    "(Train) Loss: {:.3f}  Dice: DISC: {:.3f}  CUP: {:.3f}\n".format(
                        fold,
                        epoch + 1, 
                        optimizer.param_groups[0]['lr'],
                        train_loss, train_DISC, train_CUP))
            if epoch == args.num_epochs - 1:
                if not os.path.exists(os.path.dirname(save_name)):
                    os.makedirs(os.path.dirname(save_name))
                torch.save(model.state_dict(), save_name)
                print(f"New model for fold {fold} is saved as {save_name}")
                logging.info(f"New model for fold {fold} is saved as {save_name}")


if __name__ == "__main__":
    main()
