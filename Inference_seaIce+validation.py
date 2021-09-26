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
import rasterio

args = None
best_prec1 = 0
global_step = 0
import torchvision.transforms as transforms


def main():
    global global_step
    global best_prec1

    # Name of the model to be trained
    if args.isMT:
    	model_name = '%s_%d_mt_ss_split_%d_isL2_%d' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2))
    else:
    	model_name = '%s_%d_split_%d_isL2_%d_%s' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),args.arch)

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
    args.arch = "simple_CNN"
    name = "phase2_1"
    
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
    	resume_fn = 'models/%s_%d_split_%d_isL2_%d_%s/phase1_100_epochs_best_TwoClassinWater/best.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2), args.arch)
    	# resume_fn = 'models/%s_%d_ss_split_%d_isL2_%d/%s/best.ckpt' % (args.dataset,args.num_labeled,args.label_split,int(args.isL2),name)

    # Load the model from Stage 1
    assert os.path.isfile(resume_fn), "=> no checkpoint found at '{}'".format(resume_fn)
    checkpoint = torch.load(resume_fn)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

    # Compute the starting accuracy
    # prec1, prec5 = validate(eval_loader, model, global_step, args.start_epoch, isMT = args.isMT)
    if args.isMT:
        ema_prec1, ema_prec5  = validate(eval_loader, ema_model, global_step, args.start_epoch, isMT = args.isMT)

    print('Resuming from:%s' % resume_fn)


    # im_dir = './for_inference/'
    # ia_dir = '/media/salman/Windows/Users/salman/PycharmProjects/IceSea/raw_IA/aff/'
    # input = []
    # out_dir = "/media/salman/Windows/Users/salman/PycharmProjects/IceSea/extracted_patches/three_channels/32/train+val/unlabeled/"
    #
    # for f in listdir(im_dir):
    #     if isfile(join(im_dir, f)) and f.endswith(".tif"):
    #         t = (join(im_dir, f), join(ia_dir, '_'.join(f.split('_')[0:9]) + '_IA.tif'))
    #         input.append(t)

    im_dir = '/media/salman/Windows/Users/salman/PycharmProjects/IceSea/raw_images/danmarkshavn/dataset/scaled_images'
    ia_dir = '/media/salman/Windows/Users/salman/PycharmProjects/IceSea/raw_images/danmarkshavn/dataset/features'
    input = []
    for f in os.listdir(im_dir):
        if os.path.isfile(os.path.join(im_dir, f)) and f.endswith("scaled.tif"):
            t = (os.path.join(im_dir, f), os.path.join(ia_dir, '_'.join(f.split('_')[0:9])) + "/IA.img")
            input.append(t)

    for file in input:
        input_filename = file[0]
        ia_image = file[1]

        in_path = './'
        out_path = '.'

        output_filename = 'tile_{}-{}.tif'
        test_patches = []

        patch_size = 32
        # src = rio.open(input_filename)

        def get_tiles(ds, width=patch_size, height=patch_size):
            nols, nrows = ds.meta['width'], ds.meta['height']
            offsets = product(range(0, nols, width), range(0, nrows, height))
            big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)

            for col_off, row_off in offsets:
                if col_off == 1:
                    print('yes')
                window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(
                    big_window)
                transform = windows.transform(window, ds.transform)
                yield window, transform

        with rio.open(os.path.join(in_path, input_filename)) as inds:
            tile_width, tile_height = patch_size, patch_size

            meta = inds.meta.copy()
            ia = rio.open(ia_image)
            for window, transform in get_tiles(inds):
                #         print(window)checkpoint.5.ckpt
                meta['transform'] = transform
                meta['width'], meta['height'] = window.width, window.height
                outpath = os.path.join(out_path, output_filename.format(int(window.col_off), int(window.row_off)))
                t = inds.read(window=window) /255
                # t[1 , :, :] = t[0 ,: , :]
                tt = ia.read(1, window=window) / 46
                tp = np.zeros(t.shape)
                tp[0, :, :] = t[1, :, :]
                tp[1, :, :] = t[0, :, :]
                tp[2, :, :] = tt


                t = tp
                # if t.shape[2]==t.shape[1]==patch_size:
                #
                #     channels_image = np.dstack((t[0], t[1],tt))
                #     channels_image.reshape(3, patch_size, patch_size)
                #     u = np.zeros(channels_image.shape)
                #     u[:, :, 2] = channels_image[:, :, 2] / 46
                #     u[:, :, 0:2] = channels_image[:, :, 0:2] / 255
                #     u = u.astype("float32")
                #     t = u
                #     test_patches.append(t.astype('float32'))

                if t.shape[1] != patch_size:
                    t1 = np.pad(t[0], ((patch_size - t.shape[1], 0), (0, 0)), 'constant')
                    t2 = np.pad(t[1], ((patch_size - t.shape[1], 0), (0, 0)), 'constant')
                    t3 = np.pad(t[2], ((patch_size - t.shape[1], 0), (0, 0)), 'constant')
                    yy = np.dstack((t1, t2, t3)).reshape(3, patch_size, t.shape[2])
                    t = yy
                if t.shape[2] != patch_size:
                    t1 = np.pad(t[0], ((0, 0), (patch_size - t.shape[2], 0)), 'constant')
                    t2 = np.pad(t[1], ((0, 0), (patch_size - t.shape[2], 0)), 'constant')
                    t3 = np.pad(t[2], ((0, 0), (patch_size - t.shape[2], 0)), 'constant')
                    yy = np.dstack((t1, t2, t3)).reshape(3, t.shape[1], patch_size)
                    t = yy
                # t = np.rollaxis(t, 0, 3)
                test_patches.append(t.astype('float32'))

            # for index, p in enumerate(test_patches):
            #     if index == test_patches.__len__() - 1:
            #         print('tetetet')
            #     if p.shape[1] != patch_size:
            #         t1 = np.pad(p[0], ((patch_size - p.shape[1], 0), (0, 0)), 'constant')
            #         t2 = np.pad(p[1], ((patch_size - p.shape[1], 0), (0, 0)), 'constant')
            #         t3 = np.pad(p[2], ((patch_size - p.shape[1], 0), (0, 0)), 'constant')
            #         tt = np.dstack((t1, t2, t3)).reshape(3, patch_size, p.shape[2])
            #         test_patches[index] = tt
            #         p = tt
            #     if p.shape[2] != patch_size:
            #         t1 = np.pad(p[0], ((0, 0), (patch_size - p.shape[2], 0)), 'constant')
            #         t2 = np.pad(p[1], ((0, 0), (patch_size - p.shape[2], 0)), 'constant')
            #         t3 = np.pad(p[2], ((0, 0), (patch_size - p.shape[2], 0)), 'constant')
            #         tt = np.dstack((t1, t2, t3)).reshape(3, p.shape[1], patch_size)
            #         test_patches[index] = tt
            p = torch.stack([torch.tensor(i) for i in test_patches])
            # X = torch.autograd.Variable(p)
            input_dataset = utils.TensorDataset(p)
            input_loader = utils.DataLoader(input_dataset,batch_size=50)
            prediction = []
            model.eval()
            for i, batch_input in enumerate(input_loader):
                X = torch.autograd.Variable(batch_input[0])
                r,_ = model.forward(X)
                # prop = torch.exp(r)
                _,pred = r.data.topk(1, 1, True, True)
                # prediction.append(pred.t())
                for item in pred.t()[0]:
                    prediction.append(item.cpu().data.numpy())
                # for item in pred.t():
                #     if item.cpu().data.numpy().argmax() ==1 :
                #         print("dd")
                #     prediction.append(item.cpu().data.numpy)
            np.save(input_filename.split('.')[0] + '_results.npy', prediction)




            src = rio.open(input_filename)
            image = src.read()
            tp = np.zeros(image.shape)
            tp[0, :, :] = image[1, :, :]
            tp[1, :, :] = image[0, :, :]
            image = np.rollaxis(tp/255,0,3)
            num_hight = math.ceil(src.height / patch_size)
            num_width = math.ceil(src.width / patch_size)

            results = np.load(input_filename.split('.')[0] + '_results.npy', allow_pickle=True)
            predicted_list = []
            ice1 = np.zeros((patch_size, patch_size, 3), dtype='uint8')
            ice2 = np.zeros((patch_size, patch_size, 3), dtype='uint8') * 20
            ice3 = np.zeros((patch_size, patch_size, 3), dtype='uint8') * 40
            ice4 = np.zeros((patch_size, patch_size, 3), dtype='uint8') * 60
            ice5 = np.zeros((patch_size, patch_size, 3), dtype='uint8') * 80
            ice1[:, :] = [255, 128, 0]
            ice2[:, :] = [0, 0, 255]
            ice3[:, :] = [0, 255, 0]
            ice4[:, :] = [255, 0, 0]
            ice5[:, :] = [128, 0, 128]

            sea = np.zeros((patch_size, patch_size, 3), dtype='uint8')
            sea[:, :] = [100, 128, 0]


            for i in range(0, results.__len__()):
                if prediction[i] == [0]:
                    predicted_list.append(ice2)
                elif prediction[i] == [1]:
                    predicted_list.append(sea)

                else:
                    print('yesy', results[i])

            # i = 0
            # ttt=[]
            # for i in range(0,results.__len__()):
            #     if results[i]['classes']==0:
            #         ttt.append(ice * (1 -results[i]['probabilities'][0]))
            #     if results[i]['classes'] == 1:
            #         ttt.append(ice * results[i]['probabilities'][1] )
            imgs_comb = []
            j = 0
            i = 0

            for i in range(0, num_width - 1):
                imgs_comb.append(np.vstack(predicted_list[j:j + num_hight]))
                j += num_hight

            classified_image = np.hstack(imgs_comb)





            # y = cv2.filter2D(y,-1,kernel)
            # y = cv2.medianBlur(y,10)
            # plt.style.use('classic')
            plt.subplots(1, 2, figsize=(15, 15))
            plt.subplot(1, 2, 1)
            plt.imshow(classified_image)

            plt.subplot(1, 2, 2)
            plt.imshow(image)
            plt.savefig(
                '/media/salman/Windows/Users/salman/PycharmProjects/IceSea/raw_images/danmarkshavn/dataset/scaled_images/'  +  str(patch_size) + input_filename.split('/')[-1] + name + '.ckpt' + args.arch + '.tif' , dpi=100)
            print('done')
            plt.clf()
            plt.close()



if __name__ == '__main__':
    # Get the command line arguments
    args = cli.parse_commandline_args()
    args.dataset = "seaIce"
    # Set the other settings
    args = load_args(args, isMT = args.isMT)
    args.labeled_batch_size = 50
    # args.batch_size = 30
    # Use only the specified GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    print('\n\nRunning: Num labels: %d, Split: %d, GPU: %s\n\n' % (args.num_labeled,args.label_split,args.gpu_id))

main()
