import argparse

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from models.net import Baseline
import glob
import matplotlib.pyplot as plt
from util import flow2rgb
import numpy as np
from PIL import Image
import warnings
from skimage.io import imsave
import os

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-data', default='../desData/test_image')
parser.add_argument('-pretrained', default='./12-26/epoch=80,batchsize=2,lr=0.001/model_best.pth.tar')


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.233], std=[0.255]),
    ])
    ref_list = glob.glob(os.path.join(args.data, '*_img1.tif'))
    mov_list = glob.glob(os.path.join(args.data, '*_img2.tif'))

    # create model
    args = parser.parse_args()
    args.lambda_list = [0.15, 0.85, 1, 0.1]
    args.linear_resolution = 128
    args.nonlinear_resolution = 512
    model = Baseline(18, 18, args)

    state_dict = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model = model.cuda()
    model.eval()
    cudnn.benchmark = True

    i = 0
    for ref_path, mov_path in zip(ref_list, mov_list):
        ref = input_transform(Image.open(ref_path))
        mov = input_transform(Image.open(mov_path))
        input = torch.cat([ref, mov], dim=0).unsqueeze(0).cuda()
        _, warp1, warp2, flow = model(input)
        timg, simg = input.chunk(2, dim=1)

        timg = (timg*0.255+0.233)*255
        simg = (simg*0.255+0.233)*255
        warp1 = (warp1*0.255+0.233)*255
        warp2 = (warp2*0.255+0.233)*255

        img = warp1.squeeze().cpu().numpy().astype(np.uint8)
        path = '../eval2/' + str(i) + '_warp1.tif'
        imsave (path, img)

        img = warp2.squeeze().cpu().numpy().astype(np.uint8)
        path = '../eval2/' + str(i) + '_warp2.tif'
        imsave(path, img)

        img = simg.squeeze().cpu().numpy().astype(np.uint8)
        path = '../eval2/' + str(i) + '_src.tif'
        imsave(path, img)

        img = timg.squeeze().cpu().numpy().astype(np.uint8)
        path = '../eval2/' + str (i) + '_tgt.tif'
        imsave(path, img)

        flow = flow.squeeze().cpu().numpy()
        rgb_flow = flow2rgb(10 * flow, max_value = None)
        image = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)

        path = '../eval2/' + str(i) + '_flow.png'
        plt.imsave(path, image)

        image = image[50:461, 50:461]
        path = '../eval2/' + str(i) + '_flowcrop.png'
        plt.imsave(path, image)

        i = i+1

if __name__ == '__main__':
    main()
