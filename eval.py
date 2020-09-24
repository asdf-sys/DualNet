import torchvision.transforms as transforms
from models.net import Baseline
from ssmi import SSIM
from PIL import Image
from Flow import Flow
from time import perf_counter as p_time
import torch
import glob
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imsave

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch FlowNet Training on several datasets', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', default='H:/bs/orgCrop_xb_new',)
parser.add_argument('--epochs', default=30)
parser.add_argument('-b', '--batch-size', default=3)
parser.add_argument('--lr', '--learning-rate', default=1e-3)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--milestones', default=[15])
parser.add_argument('-lambda-list')
parser.add_argument('-linear-resolution')
parser.add_argument('-nonlinear-resolution')


def dice(src, dst):
    a = []
    b = []
    cell_size = []
    for i in range(1, 256):
        A = (abs(src - i / 255.) < 1e-3).float()
        B = (abs(dst - i / 255.) < 1e-3).float()
        cell_size.append(A.sum())
        b.append(A.sum() + B.sum())
        a.append((A * B).sum())
    a = np.array(a)
    b = np.array(b) + 1e-7
    cell_size = np.array(cell_size)
    idx_max10 = cell_size.argsort()[::-1][:10]
    a = a[idx_max10]
    b = b[idx_max10]
    dic = 2. * a / b
    # b = ((src > 1e-3).float() + (dst > 1e-3).float()).sum()
    # src[src < 1e-3] = 500
    # dst[dst < 1e-3] = 1000
    # a = (abs(src - dst) < 1e-3).sum()
    return dic.mean()


@torch.no_grad()
def Net_ver_pair():
    ssim = SSIM()



    args = parser.parse_args()
    args.lambda_list = [0.15, 0.85, 1, 0.1]
    args.linear_resolution = 64
    args.nonlinear_resolution = 512
    model = Baseline(50, 18, args)

    state_dict = torch.load(
        './epoch=20,batchsize=2,lr=0.001,lambda={},lin_reso={},non_resolu={}/model_best.pth.tar'.format(args.lambda_list, args.linear_resolution, args.nonlinear_resolution),
        map_location='cpu')

    model.load_state_dict(state_dict['state_dict'], strict=True)
    model = model.cuda()
    model.eval()

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.233], std=[0.255]),
    ])

    input_transform2 = transforms.Compose([
        transforms.ToTensor(),
    ])

    ROOT = '../testdata'
    ext = 'png'

    save_path = './lambda={},lin_reso={},non_resolu={}'.format(args.lambda_list,
                                                               args.linear_resolution,
                                                               args.nonlinear_resolution)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    warp_label_path = ROOT + r"\warp_label"
    warp_image_path = ROOT + r"\warp_image"
    gt_label_path = ROOT + r"\gt_label"
    gt_image_path = ROOT + r"\gt_image"

    gt_image2_list = glob.glob(os.path.join(gt_image_path, "*img2.{}".format(ext)))
    gt_label2_list = glob.glob(os.path.join(gt_label_path, "*img2.{}".format(ext)))
    warp_label2_list = glob.glob(os.path.join(warp_label_path, "*img2.{}".format(ext)))
    warp_image1_list= glob.glob(os.path.join(warp_image_path, "*img1.{}".format(ext)))
    warp_image2_list = glob.glob(os.path.join(warp_image_path, "*img2.{}".format(ext)))

    gt_image2_list.sort()
    warp_label2_list.sort()
    gt_label2_list.sort()
    warp_image1_list.sort()
    warp_image2_list.sort()

    gt_image2_imglist = []
    gt_label2_imglist = []
    warp_label2_imglist = []
    warp_image1_imglist = []
    warp_image2_imglist = []
    for warp_label2, gt_label2, warp_image1, warp_image2, gt_image2 in zip(warp_label2_list, gt_label2_list, warp_image1_list, warp_image2_list, gt_image2_list):
        warp_image1_imglist.append(input_transform(Image.open(warp_image1)))
        warp_image2_imglist.append(input_transform(Image.open(warp_image2)))
        warp_label2_imglist.append(input_transform2(Image.open(warp_label2)))
        gt_label2_imglist.append(input_transform2(Image.open(gt_label2)))
        gt_image2_imglist.append(input_transform(Image.open(gt_image2)))
    dice_lst = []
    time_lst = []
    ssim1_lst = []
    ssim2_lst = []
    with torch.no_grad():
        for i in range(len(warp_image1_imglist)):
            input = torch.cat([warp_image1_imglist[i], warp_image2_imglist[i]], dim=0).unsqueeze(0).cuda()
            warp_label2 = warp_label2_imglist[i].cuda()
            torch.cuda.synchronize()
            st = p_time()
            loss, warp1, warp2, flow= model(input)
            flow = Flow(flow)
            warp_label2 = flow.sample(warp_label2.unsqueeze(0))
            torch.cuda.synchronize()
            ed = p_time()
            ssi1 = ssim(warp_image1_imglist[i].unsqueeze(0), flow.sample(warp_image2_imglist[i].unsqueeze(0).cuda()).cpu())
            if ext == 'png':
                plt.imsave(os.path.join(save_path, 'warp_img2_{}.png'.format(i)), flow.sample(warp_image2_imglist[i].unsqueeze(0).cuda()).squeeze().cpu().numpy(), cmap='gray')
            else:
                imsave(os.path.join(save_path, 'warp_img2_{}.tif'.format(i)), flow.sample(warp_image2_imglist[i].unsqueeze(0).cuda()).squeeze().cpu().numpy())
            ssi2 = ssim(gt_image2_imglist[i].unsqueeze(0), flow.sample(warp_image2_imglist[i].unsqueeze(0).cuda()).cpu())
            if ext == 'png':
                plt.imsave(os.path.join(save_path, 'gt_img2_{}.png'.format(i)), gt_image2_imglist[i].squeeze().cpu().numpy(), cmap='gray')
            else:
                imsave(os.path.join(save_path, 'gt_img2_{}.tif'.format(i)), gt_image2_imglist[i].squeeze().cpu().numpy())
            ssim1_lst.append(ssi1)
            ssim2_lst.append(ssi2)
            warp_label2 = warp_label2.cpu()
            dic = dice(gt_label2_imglist[i], warp_label2)
            print("time %.3f, dice: % .3f, ssim1:%.3f, ssim2:%.3f" % (ed - st, dic, ssi1, ssi2))
            dice_lst.append(dic)
            time_lst.append(ed - st)
        dice_lst = torch.tensor(dice_lst)
        del time_lst[0]
        time_lst = torch.tensor(time_lst)
        ssim1_lst = torch.tensor(ssim1_lst)
        ssim2_lst = torch.tensor(ssim2_lst)
    print("dice: %.3f ± %.3f, time: %.3f ± %.3f, ssim1: %.3f ± %.3f, ssim2: %.3f ± %.3f" % (dice_lst.mean(), dice_lst.std(), time_lst.mean(), time_lst.std(), ssim1_lst.mean(), ssim1_lst.std(), ssim2_lst.mean(), ssim2_lst.std()))
    print('&%.3f &%.3f &%.3f'% (ssim1_lst.mean(), dice_lst.mean(), time_lst.mean()))


if __name__ == "__main__":
    Net_ver_pair()
