# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Changes were made by co
# Authors: A. Iscen, G. Tolias, Y. Avrithis, O. Chum. 2018.


import re
import argparse
import os
import shutil
import time
import math
import random

import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli, architectures_wideresnet
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
import matplotlib.pyplot as plt
from helpers import *

args = None
best_prec1 = 0
global_step = 0


def main():
    global global_step
    global best_prec1

    # Name of the model to be trained
    if args.isMT:
        model_name = '%s_%d_mean_teacher_split_%d_isL2_%d_%s_%s' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2), args.arch, args.num_labeledData)
    else:
        model_name = '%s_%d_split_%d_isL2_%d_%s_%s' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2), args.arch, args.num_labeledData)

    out_path = '/scratch/salman/LP-DeepSS/journal_1_models/32x32/'
    checkpoint_path = '/scratch/salman/LP-DeepSS/journal_1_models/32x32/%s' % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')
    # class_wieghts = dataset_config.pop('class_weights')
    # args.class_weights = class_wieghts
    train_loader, eval_loader, _, _,_ = create_data_loaders(**dataset_config, args=args)

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

    prec1 = 0
    prec5 = 0
    ema_prec1 = 0
    ema_prec5 = 0
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    best_epoch = 0
    results_out = open(checkpoint_path + '/results.txt', 'w')
    results_out.writelines(
        ['epoch,training,validation,loss_training,loss_validation\n'])
    for epoch in range(args.start_epoch, args.epochs):
        
        start_time = time.time()

        # Train for one epoch
        if args.isMT:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args, ema_model = ema_model)
        else:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args.lr, args)

        # Evaluate
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0 and epoch + 1 > args.warmup:
            start_time = time.time()
            print("Evaluating the primary model:")
            prec1, prec5 , validation_loss = validate(eval_loader, model, global_step, epoch + 1, isMT = args.isMT)

            if args.isMT:
                print("Evaluating the EMA model:")
                ema_prec1, ema_prec5, validation_loss  = validate(eval_loader, ema_model, global_step, epoch + 1, isMT = args.isMT)
                is_best = ema_prec1 > best_prec1
                if ema_prec1 > best_prec1:
                    best_epoch = epoch
                best_prec1 = max(ema_prec1, best_prec1)
            else:


                is_best = prec1 > best_prec1
                if prec1 > best_prec1:
                    best_epoch = epoch
                best_prec1 = max(prec1, best_prec1)

                train_loss.append(train_meter['class_loss'].avg)
                val_loss.append(validation_loss)

                train_acc.append(train_meter['top1'].avg)
                val_acc.append(prec1)
                results_out.writelines([f'{epoch}, {train_meter["top1"].avg}, {prec1}, {train_meter["class_loss"].avg}, {validation_loss}\n'] )
                # ax  = plt.subplots(1,2)
                t = range(0, train_acc.__len__())

                ax = plt.subplot(1, 2, 1)
                ax.set_title('Loss')
                plt.plot(t, train_loss, 'blue', val_loss, 'r', linewidth=3, markersize=12)
                ax = plt.subplot(1, 2, 2)
                ax.set_title('Validation. best accuracy is: {}, epoch: {}'.format(best_prec1, best_epoch))
                plt.plot(t, train_acc, 'blue', val_acc, 'r', linewidth=3, markersize=12)
                figure = plt.gcf()
                figure.set_size_inches(25, 10)
                plt.savefig('{}/acc_loss.jpg'.format(checkpoint_path))
                plt.clf()
                plt.close()


        else:
            is_best = False

        # Write to the log file and save the checkpoint
        if args.isMT:
            log.write('%d\t%.4f\t%.4f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % 
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
                ema_prec1,
                ema_prec5)
            )
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)

        else:
            log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f\n' % 
                (epoch,
                train_meter['class_loss'].avg,
                train_meter['lr'].avg,
                train_meter['top1'].avg,
                train_meter['top5'].avg,
                prec1,
                prec5,
                )
            )
            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1)

    results_out.writelines(vars(args).__str__() + f'\n best validation accuracy: {best_prec1}')

    results_out.close()

if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()
    # args.dataset = "iceTypes"
    # args.dataset = "eurosat"
    args.dataset = "seaIce"
    # args.dataset = "cifar10"
    # args.num_labeledData = "100_labeled_100x100_2"
    args.num_labeledData = "80_labeled_32x32_2_Withaug"
    args.fully_supervised = True

    # Set the other settings
    args = load_args(args, isMT = args.isMT)
    #args.arch = "simple_CNN_64"
    # args.arch = "wideresnet50_2Model"
    # args.arch = "simple_CNN"
    # args.arch = "vgg_16"
    #args.arch = "resnet18"


    args.arch = "cifar_cnn"
    # args.arch = "vgg_16"
    args.workers = 30
    # args.arch = "windresnet"
    args.batch_size = 30
    args.test_batch_size = 20
    args.lr_rampdown_epochs = 10000
    args.epochs =400
    args.checkpoint_epochs = 1
    args.evaluation_epochs = 1
    args.test_only = False
    args.lr = 0.001
    args.consistency = None
    args.warmup = 2
    print(args)
    #args.isMT = True
    #args.isL2 = True
    #args.double_output = True
    # args.exclude_unlabeled = True
    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))

    main()
