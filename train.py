import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime
from time import time

import numpy
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

# from hausdorff import hausdorff_distance as hausdorff_dist
from metrics import dice_coef, iou_score, sensitivity_score, accuracy_score
from hausdorff_metric import HausdorffDistance as hausdorff_dist
import losses
from utils import str2bool, count_params
import pandas as pd
# import unet
import SSCN_64

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('SSCN_64')
model_name = 'SSCN_64'

load_weight = False  
model_pre_path = 'models/SSCN_64/model.pth'

aug = False

# arch_names = ['__name__', '__doc__', 'nn', 'F', 'Downsample_block', 'Upsample_block', 'Unet']
arch_names = list(SSCN_64.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')

IMG_PATH = glob(r"/home/trainImage/*")
MASK_PATH = glob(r"/home/trainMask/*")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=model_name, help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Unet', choices=arch_names,
                        help='model architecture: ' + ' | '.join(arch_names) + ' (default: SSCN_64_ss)')
    parser.add_argument('--deepsupervision', default=False, type=str2bool)
    parser.add_argument('--dataset', default="BraTS1819", help='dataset name')
    parser.add_argument('--input-channels', default=4, type=int, help='input channels')
    parser.add_argument('--image-ext', default='png', help='image file extension')
    parser.add_argument('--mask-ext', default='png', help='mask file extension')
    parser.add_argument('--aug', default=True, type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss', choices=loss_names,
                        help='loss: ' + ' | '.join(loss_names) + ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--early-stop', default=20, type=int, metavar='N', help='early stopping (default: 10)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')

    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强
import imgaug.augmenters as iaa
import imgaug as ia
from torchvision import transforms as tf
ia.seed(1)
def transform(images, labels):
    # images = images.cpu().numpy().astype(np.uint8)
    # labels = labels.cpu().numpy().astype(np.uint8)
    # # print(type(images), type(labels))
    # seq = iaa.Sequential([
    #         # iaa.GaussianBlur((0, 1.0)),
    #         iaa.Fliplr(0.5),
    #         iaa.Flipud(0.5),
    #         # iaa.Affine(translate_px={"x": (-40, 40)}),
    #         iaa.AdditiveGaussianNoise(scale=(10, 60)),
    # ])
    # images_aug, segmaps_aug = seq(images=images, segmentation_maps=labels)
    # images_aug = torch.from_numpy(images_aug)
    # segmaps_aug = torch.from_numpy(segmaps_aug)
    # # images_aug, segmaps_aug = images_aug.astype("float32"), segmaps_aug.astype("float32")
    # images_aug, segmaps_aug = images_aug.float(), segmaps_aug.float()

    aug = tf.Compose([tf.RandomHorizontalFlip()])
    images_aug = aug(images)
    labels_aug = aug(labels)
    return images_aug, labels_aug


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    WT_dice_coef = AverageMeter()
    TC_dice_coef = AverageMeter()
    ET_dice_coef = AverageMeter()
    sensitives = AverageMeter()
    accuracyes = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if aug:
            input, target = transform(input, target)

        input = input.to(device)
        target = target.to(device)

        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
            dice = dice_coef(outputs[-1], target)
            sensitive = sensitivity_score(outputs[-1], target)
            accuracy = accuracy_score(outputs[-1], target)
        else:
            output = model(input)  # torch.Size([32, 3, 160, 160], dtype=torch.float32)
            b, _, h, w = output.shape
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)
            
            wt_pre = output[:, 0, :, :]
            wt_label = target[:, 0, :, :]
            wt_dice = dice_coef(wt_pre.view((b, 1, h, w)), wt_label.view((b, 1, h, w)))
            tc_pre = output[:, 1, :, :]
            tc_label = target[:, 1, :, :]
            tc_dice = dice_coef(tc_pre.view((b, 1, h, w)), tc_label.view((b, 1, h, w)))
            et_pre = output[:, 2, :, :]
            et_label = target[:, 2, :, :]
            et_dice = dice_coef(et_pre.view((b, 1, h, w)), et_label.view((b, 1, h, w)))

            sensitive = sensitivity_score(output, target)
            accuracy = accuracy_score(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices.update(dice, input.size(0))
        WT_dice_coef.update(wt_dice, input.size(0))
        TC_dice_coef.update(tc_dice, input.size(0))
        ET_dice_coef.update(et_dice, input.size(0))
        # wt_hausdorff_distances.update(wt_hd, input.size(0))
        # tc_hausdorff_distances.update(tc_hd, input.size(0))
        # et_hausdorff_distances.update(et_hd, input.size(0))
        sensitives.update(sensitive, input.size(0))
        accuracyes.update(accuracy, input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('wt_dice', WT_dice_coef.avg),
        ('tc_dice', TC_dice_coef.avg),
        ('et_dice', ET_dice_coef.avg),
        # ('wt_hd', wt_hausdorff_distances.avg),
        # ('tc_hd', tc_hausdorff_distances.avg),
        # ('et_hd', et_hausdorff_distances.avg),
        ('sensitive', sensitives.avg),
        ('accuracy', accuracyes.avg),
    ])

    return log

def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices = AverageMeter()
    WT_dice_coef = AverageMeter()
    TC_dice_coef = AverageMeter()
    ET_dice_coef = AverageMeter()
    # wt_hausdorff_distances = AverageMeter()
    # tc_hausdorff_distances = AverageMeter()
    # et_hausdorff_distances = AverageMeter()
    sensitives = AverageMeter()
    accuracyes = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
                dice = dice_coef(outputs[-1], target)
                sensitive = sensitivity_score(outputs[-1], target)
                accuracy = accuracy_score(outputs[-1], target)
            else:
                output = model(input)  # torch.Size([32, 3, 160, 160], dtype=torch.float32)
                b, c, h, w = output.shape
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice = dice_coef(output, target)

                wt_pre = output[:, 0, :, :]
                wt_label = target[:, 0, :, :]
                wt_dice = dice_coef(wt_pre.view((b, 1, h, w)), wt_label.view((b, 1, h, w)))
                tc_pre = output[:, 1, :, :]
                tc_label = target[:, 1, :, :]
                tc_dice = dice_coef(tc_pre.view((b, 1, h, w)), tc_label.view((b, 1, h, w)))
                et_pre = output[:, 2, :, :]
                et_label = target[:, 2, :, :]
                et_dice = dice_coef(et_pre.view((b, 1, h, w)), et_label.view((b, 1, h, w)))

                # hd = hausdorff_dist()
                # wt_hd = hd.compute(wt_pre, wt_label)
                # tc_hd = hd.compute(tc_pre, tc_label)
                # et_hd = hd.compute(et_pre, et_label)

                sensitive = sensitivity_score(output, target)
                accuracy = accuracy_score(output, target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices.update(dice, input.size(0))
            WT_dice_coef.update(wt_dice, input.size(0))
            TC_dice_coef.update(tc_dice, input.size(0))
            ET_dice_coef.update(et_dice, input.size(0))
            # wt_hausdorff_distances.update(wt_hd, input.size(0))
            # tc_hausdorff_distances.update(tc_hd, input.size(0))
            # et_hausdorff_distances.update(et_hd, input.size(0))
            sensitives.update(sensitive, input.size(0))
            accuracyes.update(accuracy, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice', dices.avg),
        ('wt_dice', WT_dice_coef.avg),
        ('tc_dice', TC_dice_coef.avg),
        ('et_dice', ET_dice_coef.avg),
        # ('wt_hd', wt_hausdorff_distances.avg),
        # ('tc_hd', tc_hausdorff_distances.avg),
        # ('et_hd', et_hausdorff_distances.avg),
        ('sensitive', sensitives.avg),
        ('accuracy', accuracyes.avg),
    ])

    return log


def main():
    args = parse_args()
    #args.dataset = "datasets"

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_wDS' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_woDS' %(args.dataset, args.arch)

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)
    
    print('Config ----------------------------------------')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('-----------------------------------------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        # criterion = BCEDiceLoss()
        criterion = losses.__dict__[args.loss]().to(device)

    cudnn.benchmark = True

    # Data loading code
    img_paths = IMG_PATH
    mask_paths = MASK_PATH

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)
    print(" = = = > train_num : %s"%str(len(train_img_paths)))
    print(" = = = > val_num : %s"%str(len(val_img_paths)))

    # create model
    print(" = = = > creating model : %s" %args.arch)
    # model = UNet()
    model = SSCN_64.__dict__[args.arch](args)

    if load_weight:
        pretrained_dict = torch.load(model_pre_path, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict} 
        model_dict.update(pretrained_dict)  
        model.load_state_dict(model_dict)

    model = model.to(device)
    print(" = = = > model total params : %s" %str(count_params(model))) 

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # tik = time()
    # args.aug = False
    train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    # print(len(train_dataset))
    # tok = time()
    # print(tok-tik)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False)

    # log = pd.DataFrame(index=[], columns=[
    #     'epoch', 'lr', 
    #     'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_WT_hd', 'train_TC_hd', 'train_ET_hd', 'train_sensitive', 'train_accuracy', 
    #     'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_WT_hd', 'val_TC_hd', 'val_ET_hd', 'val_sensitive', 'val_accuracy',
    # ])
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 
        'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_sensitive', 'train_accuracy', 
        'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_sensitive', 'val_accuracy',
    ])

    best_iou = 0    
    trigger = 0  
    start = time()
    for epoch in range(1, args.epochs+1):
        print(' = = = > Epoch [%d/%d]' %(epoch, args.epochs))

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, val_loader, model, criterion)

        # print('Train_Loss %.4f\tTrain_IoU %.4f\tTrain_Dice %.4f\tTrain_WT_Dice %.4f\tTrain_TC_Dice %.4f\tTrain_ET_Dice %.4f\tTrain_WT_hd %.4f\tTrain_TC_hd %.4f\tTrain_ET_hd %.4f\tTrain_Sensitive %.4f\tTrain_Acc %.4f'
        #     %(train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['wt_hd'], train_log['tc_hd'], train_log['et_hd'], train_log['sensitive'], train_log['accuracy']))
        # print('Val_Loss %.4f\t\tVal_IoU %.4f\t\tVal_Dice %.4f\t\tVal_WT_Dice %.4f\tVal_TC_Dice %.4f\tVal_ET_Dice %.4f\tVal_WT_hd %.4f\tVal_TC_hd %.4f\tVal_ET_hd %.4f\tVal_Sensitive %.4f\tVal_Acc %.4f'
        #     %(val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['wt_hd'], val_log['tc_hd'], val_log['et_hd'], val_log['sensitive'], val_log['accuracy']))
        print('Train_Loss %.4f\tTrain_IoU %.4f\tTrain_Dice %.4f\tTrain_WT_Dice %.4f\tTrain_TC_Dice %.4f\tTrain_ET_Dice %.4f\tTrain_Sensitive %.4f\tTrain_Acc %.4f'
            %(train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['sensitive'], train_log['accuracy']))
        print('Val_Loss %.4f\t\tVal_IoU %.4f\t\tVal_Dice %.4f\t\tVal_WT_Dice %.4f\tVal_TC_Dice %.4f\tVal_ET_Dice %.4f\tVal_Sensitive %.4f\tVal_Acc %.4f'
            %(val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['sensitive'], val_log['accuracy']))

        writer.add_scalar('Train_Loss', scalar_value=train_log['loss'], global_step=epoch)
        writer.add_scalar('Train_IoU', scalar_value=train_log['iou'], global_step=epoch)
        writer.add_scalar('Train_Dice', scalar_value=train_log['dice'], global_step=epoch)
        writer.add_scalar('Train_WT_Dice', scalar_value=train_log['wt_dice'], global_step=epoch)
        writer.add_scalar('Train_TC_Dice', scalar_value=train_log['tc_dice'], global_step=epoch)
        writer.add_scalar('Train_ET_Dice', scalar_value=train_log['et_dice'], global_step=epoch)
        # writer.add_scalar('tTrain_WT_hd', scalar_value=train_log['wt_hd'], global_step=epoch)
        # writer.add_scalar('tTrain_TC_hd', scalar_value=train_log['tc_hd'], global_step=epoch)
        # writer.add_scalar('tTrain_ET_hd', scalar_value=train_log['et_hd'], global_step=epoch)
        writer.add_scalar('Train_Sensitive', scalar_value=train_log['sensitive'], global_step=epoch)
        writer.add_scalar('Train_Acc', scalar_value=train_log['accuracy'], global_step=epoch)

        writer.add_scalar('Val_Loss', scalar_value=val_log['loss'], global_step=epoch)
        writer.add_scalar('Val_IoU', scalar_value=val_log['iou'], global_step=epoch)
        writer.add_scalar('Val_Dice', scalar_value=val_log['dice'], global_step=epoch)
        writer.add_scalar('Val_WT_Dice', scalar_value=val_log['wt_dice'], global_step=epoch)
        writer.add_scalar('Val_TC_Dice', scalar_value=val_log['tc_dice'], global_step=epoch)
        writer.add_scalar('Val_ET_Dice', scalar_value=val_log['et_dice'], global_step=epoch)
        # writer.add_scalar('Val_WT_hd', scalar_value=val_log['wt_hd'], global_step=epoch)
        # writer.add_scalar('Val_TC_hd', scalar_value=val_log['tc_hd'], global_step=epoch)
        # writer.add_scalar('Val_ET_hd', scalar_value=val_log['et_hd'], global_step=epoch)
        writer.add_scalar('Val_Sensitive', scalar_value=val_log['sensitive'], global_step=epoch)
        writer.add_scalar('Val_Acc', scalar_value=val_log['accuracy'], global_step=epoch)


        # tmp = pd.Series([
        #     epoch, args.lr,
        #     train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['wt_hd'], train_log['tc_hd'], train_log['et_hd'], train_log['sensitive'], train_log['accuracy'], 
        #     val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['wt_hd'], val_log['tc_hd'], val_log['et_hd'], val_log['sensitive'], val_log['accuracy'], 
        # ], index=[
        #     'epoch', 'lr', 
        #     'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_WT_hd', 'train_TC_hd', 'train_ET_hd', 'train_sensitive', 'train_accuracy', 
        #     'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_WT_hd', 'val_TC_hd', 'val_ET_hd', 'val_sensitive', 'val_accuracy',
        # ])
        tmp = pd.Series([
            epoch, args.lr,
            train_log['loss'], train_log['iou'], train_log['dice'], train_log['wt_dice'], train_log['tc_dice'], train_log['et_dice'], train_log['sensitive'], train_log['accuracy'], 
            val_log['loss'], val_log['iou'], val_log['dice'], val_log['wt_dice'], val_log['tc_dice'], val_log['et_dice'], val_log['sensitive'], val_log['accuracy'], 
        ], index=[
            'epoch', 'lr', 
            'train_loss', 'train_iou', 'train_dice', 'train_WT_dice', 'train_TC_dice', 'train_ET_dice', 'train_sensitive', 'train_accuracy', 
            'val_loss', 'val_iou', 'val_dice', 'val_WT_dice', 'val_TC_dice', 'val_ET_dice', 'val_sensitive', 'val_accuracy',
        ])


        log = log.append(tmp, ignore_index=True)

        log.to_csv('models/%s/log.csv' %args.name, index=False)

        trigger += 1


        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_iou = val_log['iou']
            print(" = = = > saved best model")
            trigger = 0

        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print(" = = = > early stopping")
                break
        
        torch.cuda.empty_cache()
    end = time()
    total_time = end - start
    log_time = pd.DataFrame(index=[], columns=['Total Time/s',])
    tmp_time = pd.Series([total_time], index=['Total Time/s'])
    log_time = log_time.append(tmp_time, ignore_index=True)
    log_time.to_csv('models/%s/log_total_time.csv' %args.name, index=False)

if __name__ == '__main__':
    main()

