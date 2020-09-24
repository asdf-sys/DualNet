import argparse
import os
import time
import numpy

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
from models.net import Baseline
from datasets.cremi_dataset import cremidataset
from util import AverageMeter, save_checkpoint

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', default='H:/bs/orgCrop_xb_new',)
parser.add_argument('--epochs', default=20)
parser.add_argument('-b', '--batch-size', default=1)
parser.add_argument('--lr', '--learning-rate', default=1e-3)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--milestones', default=[10])
parser.add_argument('-lambda-list')
parser.add_argument('-linear-resolution')
parser.add_argument('-nonlinear-resolution')


def main(args):

    save_path = 'epoch={},batchsize={},lr={},lambda={},lin_reso={},non_resolu={}'.format(
        args.epochs,
        args.batch_size,
        args.lr,
        args.lambda_list,
        args.linear_resolution,
        args.nonlinear_resolution
    )
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0], std=[255]),
        transforms.Normalize(mean=[0.233], std=[0.255]),
    ])
    print("=> fetching img pairs in '{}'".format(args.data))
    train_set, test_set = cremidataset(
        args.data,
        transform=input_transform,
        split= 0.9
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set), len(train_set), len(test_set)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=min(args.batch_size, 8), pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=1, pin_memory=True, shuffle=False)

    model = Baseline(18, 18, args).cuda()
    cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    lr_para = filter(lambda p:p.requires_grad, model.parameters())
    N = sum([numpy.prod(p.size()) for p in lr_para])
    print(N)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    best_EPE = -1

    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer, epoch)
        print(train_loss)
        with torch.no_grad():
            test_loss = validate(val_loader, model)
        scheduler.step()

        if best_EPE < 0:
            best_EPE = test_loss

        is_best = test_loss < best_EPE
        best_EPE = min(test_loss, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
        },is_best, save_path )


def train(train_loader, model, optimizer, epoch):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    for i, input in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = torch.cat(input, 1).cuda()
        optimizer.zero_grad()
        model.train()
        # compute output
        loss, _, _, _ = model(input)

        losses.update(loss.item())
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'.format(epoch, i, len(train_loader), batch_time, data_time, losses))

    return losses.avg


def validate(val_loader, model):
    model.eval()
    losses=0
    for i, input in enumerate(val_loader):
        input = torch.cat(input, 1).cuda()
        loss, warp1, warp2, _ = model(input)
        losses=losses+loss
        timg, simg = input.chunk(2, dim=1)

        save_path = '../results-{}-{}-{}-{}-{}-{}/'.format(args.lambda_list[0],
                                                           args.lambda_list[1],
                                                           args.lambda_list[2],
                                                           args.lambda_list[3],
                                                           args.linear_resolution,
                                                           args.nonlinear_resolution)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if i % 10 == 0 and i<110:
            img = warp1.squeeze().cpu().numpy()
            path = save_path + str(i) + '_warp1.png'
            plt.imsave (path, img, cmap='gray')

            img = warp2.squeeze().cpu().numpy()
            path = save_path + str(i) + '_warp2.png'
            plt.imsave(path, img, cmap='gray')

            img = simg.squeeze().cpu().numpy()
            path = save_path + str(i) + '_src.png'
            plt.imsave(path, img, cmap='gray')

            img = timg.squeeze().cpu().numpy()
            path = save_path + str (i) + '_tgt.png'
            plt.imsave(path, img, cmap='gray')

    return losses/len(val_loader)


if __name__ == '__main__':
    #ablation2
    lin_reso = [128,512,256,64]
    non_reso = [512,256]
    args = parser.parse_args()
    args.lambda_list = [0.15, 0.85, 1, 0.1]
    for i in range(len(lin_reso)):
        for j in range(len(non_reso)):
            args.linear_resolution = lin_reso[i]
            args.nonlinear_resolution = non_reso[j]
            main(args)

    # ablation1b
    args = parser.parse_args()
    args.lambda_list = [0.15, 0.85, 1, 0]
    args.linear_resolution = 128
    args.nonlinear_resolution = 512
    main(args)
    # ablation1c
    args = parser.parse_args()
    args.lambda_list = [0.15, 0.85, 0, 0]
    args.linear_resolution = 128
    args.nonlinear_resolution = 512
    main(args)
    # ablation1d
    args = parser.parse_args()
    args.lambda_list = [0.15, 0, 0, 0]
    args.linear_resolution = 128
    args.nonlinear_resolution = 512
    main(args)
