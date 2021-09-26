# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by 
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.

import re
import argparse
import os
import shutil
import time
import math

import pdb
from turtle import mode

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

from os import listdir
import os
from os.path import isfile, join
from itertools import product
import rasterio as rio
from rasterio import windows
import numpy as np
import uuid
import random
from matplotlib import colors
import rasterio as rio
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as utils
from helpers import *

args = None
best_prec1 = 0
global_step = 0



def main():
    global global_step
    global best_prec1

    # Name of the model to be trained
    if args.isMT:
    	model_name = '%s_%d_mt_ss_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	model_name = '%s_%d_ss_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))

    checkpoint_path = 'models/%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    # train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)

    # Create the model
    model = create_model(num_classes,args)

    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes,args,ema=True)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    cudnn.benchmark = True

    # Name of the model trained in Stage 1
    if args.isMT:
    	resume_fn = 'models/%s_%d_mean_teacher_split_%d_isL2_%d/checkpoint.180.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	#resume_fn = 'models/%s_%d_split_%d_isL2_%d/checkpoint.50.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
        # resume_fn = '/scratch/salman/LP-DeepSS/models_repeated/seaIce_1000_split_10_isL2_1_simple_CNN_all-centered/best.ckpt'
        resume_fn = '/home/skh018/PycharmProjects/models/mixmatch/MixMatch1.pt'

        path = "/scratch/salman/LP-DeepSS/journal_1_models/32x32/final/"
        model_path = "seaIce_4000_ss_split_10_isL2_1_cifar_cnn_1000_labeled_32x32_1_Waug_all+damarkshavn"
        resume_fn = path + model_path + "/best.ckpt"
        resume_fn = "/scratch/salman/LP-DeepSS/journal_1_models/32x32/final/fully supervised/seaIce_4000_split_10_isL2_1_cifar_cnn_40_labeled_32x32_2_Withaug/best.ckpt"
    # Load the model from Stage 1




    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)

    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    train_loader, eval_loader, _, _, _= create_data_loaders(**dataset_config, args=args)
    # Compute the starting accuracy
    # Compute the starting accuracy
    # prec1, prec5 = validate(eval_loader, model)

    ema_prec1, ema_prec5, validation_loss  = validate(eval_loader, model)

    print('Resuming from:%s' % resume_fn)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)

    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / float(labeled_minibatch_size)))
    return res


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision_ice = tp / (tp + fp + epsilon)
    recall_ice = tp / (tp + fn + epsilon)

    precision_water = tn / (tn + fn + epsilon)
    recall_water  = tn / (tn + fp + epsilon)

    precision_ave = (precision_ice + precision_water) / 2
    recall_ave = (recall_ice + recall_water) / 2

    f1_ice = 2 * (precision_ice * recall_ice) / (precision_ice + recall_ice + epsilon)

    f1_water = 2 * (precision_water * recall_water) / (precision_water + recall_water + epsilon)

    f1_ave = (f1_ice + f1_water) / 2

    # f1.requires_grad = is_training

    return f1_ave, precision_ave, recall_ave

def validate(eval_loader, model):
    class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        # print(input.shape)
        meters.update('data_time', time.time() - end)

        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target.cuda(non_blocking=True), volatile=True)

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output

        output1, _ = model(input_var)

        f1, p, r = f1_loss(target_var, output1)
        print(f1, p, r)
        class_loss = class_criterion(output1, target_var) / minibatch_size
        # print(output1.shape)
        # print(target_var.shape)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 2))
        meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
        meters.update('top1', prec1[0], labeled_minibatch_size)
        meters.update('error1', 100.0 - prec1[0], labeled_minibatch_size)
        meters.update('top5', prec5[0], labeled_minibatch_size)
        meters.update('error5', 100.0 - prec5[0], labeled_minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))

    return meters['top1'].avg, meters['top5'].avg , meters['class_loss'].avg


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()
    args.dataset = "seaIce"
    # Set the other settings
    args = load_args(args, isMT = args.isMT)
    args.labeled_batch_size = 50
    args.test_batch_size = 2000
    # args.arch = "windresnet"
    args.arch = "cifar_cnn"
    args.test_only = False
    # args.batch_size = 30
    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args.unlabels_dirs = ['unlabeled_data']
    args.current_unlabeled_dir = args.unlabels_dirs[0]
    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))

main()
