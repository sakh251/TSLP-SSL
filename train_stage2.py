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
from mpl_toolkits.mplot3d import Axes3D
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
    	model_name = '%s_%d_mt_ss_split_%d_isL2_%d_%s_%s_%s' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),args.arch,args.num_labeledData, args.num_unlabeledData)
    else:
    	model_name = '%s_%d_ss_split_%d_isL2_%d_%s_%s_%s' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),args.arch,args.num_labeledData, args.num_unlabeledData)

    checkpoint_path = '/scratch/salman/LP-DeepSS/journal_1_models/%s'  % model_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    log_file = '%s/log.txt' % checkpoint_path
    log = open(log_file, 'a')

    weights_log_file = '%s/weights_log.txt' % checkpoint_path
    weights_log = open(weights_log_file, 'a')


    # Create the dataset and loaders
    dataset_config = datasets.__dict__[args.dataset](isTwice=args.isMT)
    num_classes = dataset_config.pop('num_classes')


    # Create the model
    model = create_model(num_classes,args)

    S_model = create_S_model(num_classes,args)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    Soptimizer = torch.optim.SGD(S_model.parameters(), args.Slr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    cudnn.benchmark = True

    # Name of the model trained in Stage 1
    if args.isMT:

        resume_fn = '/home/skh018/shared/LP-DeepSS/models/%s_%d_mean_teacher_split_%d_isL2_%d_%s_%s/checkpoint..120.ckpt' % (
            args.dataset, args.num_labeled, args.label_split, int(args.isL2), args.arch, args.num_labeledData)
    else:
        if args.ss_continue == True:
            resume_fn = 'models/%s_%d_ss_split_%d_isL2_%d/checkpoint.249.ckpt' % (
            args.dataset, args.num_labeled, args.label_split, int(args.isL2))
        else:
            # resume_fn = 'models/%s_%d_split_%d_isL2_%d_%s/phase1_10_percent/checkpoint..24.ckpt' % (
            # args.dataset, args.num_labeled, args.label_split, int(args.isL2), args.arc)
            resume_fn = '/scratch/salman/LP-DeepSS/journal_1_models/32x32/' \
                        '/%s_%d_split_%d_isL2_%d_%s_%s/checkpoint..50.ckpt' % (
            args.dataset, args.num_labeled, args.label_split, int(args.isL2), args.arch, args.num_labeledData)
            # resume_fn = '/scratch/salman/LP-DeepSS/journal_1_models/32x32/seaIce_4000_ss_split_10_isL2_1_cifar_cnn_1000_labeled_32x32_1_Waug_all+damarkshavn/best_student.ckpt'

            # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    s_best_prec1 = 0
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    print('Resuming from:%s' % resume_fn)
    val_loss = []
    train_loss = []
    val_acc = []
    train_acc = []
    lp_acc = []
    best_epoch = 0
    S_best_epoch = 0
    # for un_dir in args.unlabels_dirs:
    #     args.current_unlabeled_dir = un_dir
    #     train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)


    # If Mean Teacher is turned on, create the ema model
    if args.isMT:
        ema_model = create_model(num_classes,args,ema=True)

    args.current_unlabeled_dir = args.unlabels_dirs[0]
    train_loader, eval_loader, train_loader_noshuff, train_data, Strain_loader = create_data_loaders(**dataset_config,
                                                                                      args=args)
    # Compute the starting accuracy

    prec1, prec5, Validation_error = validate(eval_loader, model, global_step, args.start_epoch, isMT=args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5, Validation_error = validate(eval_loader, ema_model, global_step, args.start_epoch, isMT=args.isMT)

    results_out = open(checkpoint_path + '/results.txt', 'w')
    results_out.writelines(
        ['epoch,training,validation,loss_training,loss_validation, LP_Accuracy\n'])
    results_out.close()
    s_val_acc = []
    s_val_loss = []
    s_train_loss = []
    s_train_acc = []
    # train_loader, eval_loader, train_loader_noshuff, train_data = create_data_loaders(**dataset_config, args=args)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % args.change_unlabeled_epochs == 0:
            args.current_unlabeled_dir = args.unlabels_dirs[epoch % 2]
            train_loader, eval_loader, train_loader_noshuff, train_data, Strain_loader = create_data_loaders(**dataset_config,
                                                                                              args=args)

        # Extract features and update the pseudolabels

        print('Extracting features...')
        feats,labels = extract_features(train_loader_noshuff, model, isMT = args.isMT)
        sel_acc, pure_weights , correct_idx, pca, pca_labels = train_data.update_plabels(feats, k = args.dfs_k, max_iter = 20, alpha = args.alpha)

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(pca[:, 0], pca[:, 1], pca[:, 2], c=pca_labels)
        plt.savefig('{}/features_{}.jpg'.format(checkpoint_path,epoch))
        # print('selection accuracy Graph: %.2f' % (sel_acc))
        # sel_acc, pure_weights , correct_idx = train_data.update_plabels_net(labels)
        # sel_acc, pure_weights , correct_idx = train_data.update_plabels_orginal()
        print('selection accuracy: %.2f' % (sel_acc))
        # print('selection accuracy: %.2f' % (sel_acc_))
        s_sel_acc = '{} , {} , '.format(sel_acc,train_data.labeled_idx.__len__())
        print(s_sel_acc)
        # print(pure_weights, " p w ", pure_weights.shape)
        weights_log.write(s_sel_acc + str(pure_weights.tolist()))
        not_correct_idx = [not a for a in correct_idx]
        ax = plt.subplot(1, 2, 1)
        ax.hist(pure_weights * 100, bins=10,
                color='blue', edgecolor='black')
        # ax.hist(pure_weights[0:train_data.labeled_idx.__len__()] * 100, bins=10,
        #         color='red', edgecolor='black')
        ax.set_title('All images')
        ax.set_xlabel('weight (min)')

        ax.set_ylabel('Number of Image')
        ax = plt.subplot(1, 2, 2)
        # ax.hist(pure_weights[0:train_data.labeled_idx.__len__()-1][correct_idx] * 100, bins=10,
        #         color='blue', edgecolor='black')
        ax.hist(pure_weights[correct_idx] * 100, bins=10,
                color='blue', edgecolor='black')
        # ax.hist(pure_weights[0:train_data.labeled_idx.__len__() - 1][not_correct_idx] * 100, bins=10,
        #         color='red', edgecolor='black')
        ax.hist(pure_weights[not_correct_idx] * 100, bins=10,
                color='red', edgecolor='black')
        ax.set_title(
            'training image (Number of label samples: {} , accuracy of label propagation for labeled samples: {})'.format(train_data.labeled_idx.__len__(),
                                                                                 sel_acc))
        ax.set_xlabel('weights')
        ax.set_ylabel('Number of Images')
        # ax.set_ylim(ymax=14000)
        figure = plt.gcf()
        figure.set_size_inches(25, 10)
        plt.savefig('{}/{}.jpg'.format(checkpoint_path,epoch), dpi=80)
        plt.clf()
        plt.close()
        #  Train for one epoch with the new pseudolabels
        if args.isMT:
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args, ema_model = ema_model)
        else:
            # train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args.lr, args)
            train_meter, global_step = train(train_loader, model, optimizer, epoch, global_step, args.lr, args)

        # Evaluate the model

            Strain_meter, Sglobal_step = train(Strain_loader, S_model, Soptimizer, epoch, global_step, args.Slr, args)
            Sprec1, Sprec5, Svalidation_loss = validate(eval_loader, S_model, global_step, epoch + 1, isMT=args.isMT)

            # print(max(s_val_accacc), "student model")
        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
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

            s_train_loss.append(Strain_meter['class_loss'].avg)
            s_val_loss.append(Svalidation_loss)

            is_best_s= Sprec1 > s_best_prec1
            if Sprec1 > s_best_prec1:
                S_best_epoch = epoch
            s_best_prec1 = max(Sprec1, s_best_prec1)

            s_val_acc.append(Sprec1)
            s_train_acc.append(Strain_meter['top1'].avg)



            train_loss.append(train_meter['class_loss'].avg)
            val_loss.append(validation_loss)

            train_acc.append(train_meter['top1'].avg)
            val_acc.append(prec1)
            lp_acc.append(sel_acc * 100)
            results_out = open(checkpoint_path + '/results.txt', 'a')
            results_out.writelines(
                [f'{epoch}, {train_meter["top1"].avg}, {prec1}, {train_meter["class_loss"].avg}, {validation_loss}, {sel_acc * 100},'
                 f'{Strain_meter["class_loss"].avg}, {Svalidation_loss}, {Strain_meter["top1"].avg}, {Sprec1} \n'])


            results_out.close()
            # ax  = plt.subplots(1,2)
            t = range(0, train_acc.__len__())

            ax = plt.subplot(1, 2, 1)
            ax.set_title('Loss')
            plt.plot(t, train_loss, 'blue', val_loss, 'r', linewidth=3, markersize=12)
            ax = plt.subplot(1, 2, 2)
            ax.set_title('Validation. best accuracy is: {}, epoch: {}'.format(best_prec1, best_epoch))
            plt.plot(t, train_acc, 'blue', val_acc, 'r', lp_acc  , 'orange', linewidth=3, markersize=12)
            figure = plt.gcf()
            figure.set_size_inches(25, 10)
            plt.savefig('{}/acc_loss_alpha{}_lr{}_NN{}_bs{}_LBS{}.jpg'.format(checkpoint_path, args.alpha, args.lr, args.dfs_k,
                        args.batch_size, args.labeled_batch_size))
            plt.clf()
            plt.close()

            ax = plt.subplot(1, 2, 1)
            ax.set_title('Loss')
            plt.plot(t, s_train_loss, 'blue', s_val_loss, 'r', linewidth=3, markersize=12)
            ax = plt.subplot(1, 2, 2)
            ax.set_title('Validation. best accuracy is: {}, epoch: {}'.format(s_best_prec1, S_best_epoch))
            plt.plot(t, s_train_acc, 'blue', s_val_acc, 'r', lp_acc, 'orange', linewidth=3, markersize=12)
            figure = plt.gcf()
            figure.set_size_inches(25, 10)
            plt.savefig(
                '{}/acc_loss_alpha{}_lr{}_NN{}_bs{}_LBS{}_student.jpg'.format(checkpoint_path, args.alpha, args.lr, args.dfs_k,
                                                                      args.batch_size, args.labeled_batch_size))
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
        #     log.write('%d,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f\n' %
        #         (epoch,
        #         train_meter['class_loss'].avg,
        #         train_meter['lr'].avg,
        #         train_meter['top1'].avg,
        #         train_meter['top5'].avg,
        #         prec1,
        #         prec5,
        #
        #         )
        #     )

            if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint_path, epoch + 1 )
                save_checkpoint_s({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': args.Sarch,
                    'state_dict': S_model.state_dict(),
                    'best_prec1': np.array(s_val_acc).max(),
                    'optimizer': Soptimizer.state_dict(),
                }, is_best_s, checkpoint_path, epoch + 1)
    results_out = open(checkpoint_path + '/results.txt', 'a')
    results_out.writelines(vars(args).__str__() + f'\n best validation accuracy: {best_prec1}')

    results_out.close()


if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()
    args.dataset = "seaIce"
    # args.dataset = "eurosat"
    # args.dataset = "cifar10"
    args.num_labeledData = "80_labeled_32x32_2_Withaug"
    args.num_unlabeledData = "after_review"
    # args.dataset = "iceTypes"

    # Set the other settings
    args = load_args(args, isMT = args.isMT)
    # args.arch = "windresnet"
    # args.arch = "simple_CNN"
    # args.Sarch = "wideresnet50_2Model"
    # args.Sarch = "windresnet"

    args.arch = "cifar_cnn"
    args.Sarch = "cifar_cnn"
    # args.arch = "wideresnet50_2Model"
    args.labeled_batch_size = 20
    args.batch_size = 40
    args.Sbatch_size = 40
    args.workers = 10
    args.epochs = 200
    args.lr_rampdown_epochs = 210
    args.dfs_k = 20
    args.alpha = 0.99
    args.ss_continue = False
    args.checkpoint_epochs = 1
    args.evaluation_epochs = 1
    args.change_unlabeled_epochs = 1000
    args.test_only = False
    args.isL2 = True
    # args.isMT = True
    # args.double_output = Truel
    # args.lr=0.0001
    # # args.Slr=0.002
    # args.Slr=0.002

    args.lr = 0.0008
    # args.Slr=0.002
    args.Slr = 0.002

    print(args)
    # args.unlabels_dirs =['unlabeled_HH_0_0' , 'unlabeled_HH_0_1', 'unlabeled_HH_1_0', 'unlabeled_HH_1_1', 'unlabeled_HH_2_0', 'unlabeled_HH_2_1']
    # args.unlabels_dirs =['unlabeled_mixed_validation_unlabeled' ]
    # args.unlabels_dirs =['unlabeled_val_half_img_back' ]
    # args.unlabels_dirs =['unlabeled_HH_1_1' ]
    args.unlabels_dirs =['unlabeled_data_without']
    args.unlabels_dirs =['unlabeled_data']

    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))

main()
