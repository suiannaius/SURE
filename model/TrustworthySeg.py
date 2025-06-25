import torch
import argparse
import os
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from model.TrustworthySeg_criterions import KL, ce_loss, mse_loss, dce_evidence_loss, dce_evidence_u_loss
from model.UNet2DZoo import Unet2D, AttUnet2D
from torch.autograd import Variable
from utilities.weights_init import weights_init_kaiming

class TMSU(nn.Module):

    def __init__(self, args):
        """
        :param classes: Number of classification categories
        :param modes: Number of modes
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMSU, self).__init__()
        # ---- Net Backbone ----
        num_classes = args.num_classes
        dataset = args.dataset
        # modes = args.modes
        total_epochs = args.end_epochs
        lambda_epochs = args.lambda_epochs

        if args.model_name == 'U':
            self.backbone = Unet2D(in_channels=args.num_modalities, out_channels=args.num_classes)
        elif args.model_name == 'AU':
            self.backbone = AttUnet2D(in_channels=args.num_modalities, out_channels=args.num_classes)

        self.backbone.cuda()

        self.backbone.apply(weights_init_kaiming)

        self.classes = num_classes
        self.disentangle = False
        self.eps = 1e-10
        self.lambda_epochs = lambda_epochs
        self.total_epochs = total_epochs + 1
        self.u_loss = args.Uncertainty_Loss

    def forward(self, X, y=None, global_step=None, mode='test',dataset=None):
        # X data
        # y target
        # global_step : epochs

        if y is not None:
            _, y = torch.max(y, 1, keepdim=False)

        # step zero: backbone
        backbone_output = self.backbone(X)

        # step one
        evidence = self.infer(backbone_output) # batch_size * class * image_size
        backbone_pred = F.softmax(backbone_output,1)  # batch_size * class * image_size

        # step two
        alpha = evidence + 1
        if mode == 'train':

            if self.u_loss:
                loss = dce_evidence_u_loss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs,self.total_epochs,self.eps,self.disentangle,evidence,backbone_pred)
            else:
                loss = dce_evidence_loss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs,self.total_epochs,self.eps,self.disentangle,evidence,backbone_pred)
            loss = torch.mean(loss)
            return evidence, loss
        else:
            return evidence

    def infer(self, input):
        """
        :param input: modal data
        :return: evidence of modal data
        """
        evidence = F.softplus(input)
        return evidence
