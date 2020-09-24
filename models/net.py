from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .layers import SSIM
from .encoder import Encoder
from .HR_decoder import HRDecoder
from .LR_decoder import LRDecoder


class Baseline(nn.Module):
    def __init__(self, HR_num,LR_num, args):
        super(Baseline, self).__init__()
        self.HREncoder = Encoder(HR_num)
        self.HRDecoder = HRDecoder(self.HREncoder.num_ch_enc)
        self.LREncoder = Encoder(LR_num)
        self.LRDecoder = LRDecoder(self.LREncoder.num_ch_enc)

        self.ssim = SSIM()

        self.count = 0
        self.lambda_list = args.lambda_list
        self.lin_reso = args.linear_resolution
        self.non_reso = args.nonlinear_resolution

    def forward(self, inputs):
        self.count = self.count + 1
        ref = inputs[:, :1]
        mov = inputs[:, 1:]
        warp1, affine, affine_grid = self.linear_align(ref, mov)
        warp2_list, flow_list = self.nonlinear_align(ref, warp1)
        id_grid = self.identity_grid(ref)

        loss1 = self.compute_loss(ref, warp1)
        constraint1 = self.linear_constraint(affine)
        loss2 = 0
        constraint2 = 0
        for i in range(len(warp2_list)):
            loss2 = loss2 + self.compute_loss(ref, warp2_list[i])*2**(-i)
            constraint2 = constraint2 + self.nonlinear_constraint(flow_list[i])*2**(-i)
        loss=loss1+loss2+self.lambda_list[2]*constraint1+self.lambda_list[3]*constraint2
        return loss, warp1, warp2_list[0], flow_list[0]-id_grid+affine_grid.permute([0,3,1,2])

    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_loss(self, ref, warp):
        photometric_loss = self.robust_l1(ref, warp).mean(1, True)
        ssim_loss = self.ssim(ref, warp).mean(1, True)
        loss = self.lambda_list[0] * photometric_loss + self.lambda_list[1] * ssim_loss
        return loss.mean()

    def linear_align(self, ref, mov):
        ref_resize = F.interpolate(ref, [self.lin_reso, self.lin_reso], mode="bilinear", align_corners=False)
        mov_resize = F.interpolate(mov, [self.lin_reso, self.lin_reso], mode="bilinear", align_corners=False)
        affine = self.LRDecoder(self.LREncoder(torch.cat([ref_resize, mov_resize], 1)))#b,2,3
        affine_grid = F.affine_grid(affine, ref.size())
        warp1 = F.grid_sample(mov, affine_grid, mode='bilinear', padding_mode='zeros')
        return warp1, affine, affine_grid

    def nonlinear_align(self, ref, warp1):
        if self.non_reso != 512:
            ref_resize = F.interpolate(ref, [self.non_reso, self.non_reso], mode="bilinear", align_corners=False)
            warp1_resize = F.interpolate(warp1, [self.non_reso, self.non_reso], mode="bilinear", align_corners=False)
        else:
            ref_resize = ref
            warp1_resize = warp1
        flow_list = self.HRDecoder(self.HREncoder(torch.cat([ref_resize, warp1_resize], 1)))#b,h,w,2
        flow_list = [F.interpolate(flow, [512, 512], mode="bilinear", align_corners=False) for flow in flow_list]
        warp2_list = []
        for i in range(len(flow_list)):
            id_grid = self.identity_grid(ref)
            def_grid = flow_list[i] + id_grid
            def_grid = def_grid.permute([0, 2, 3, 1])
            warp2 = F.grid_sample(warp1, def_grid)
            warp2_list.append(warp2)
        return warp2_list, flow_list

    def nonlinear_constraint(self, flow):
        b, _, h, w = flow.size()

        flow_dx, flow_dy = self.gradient(flow)

        flow_dxx, flow_dxy = self.gradient(flow_dx)
        flow_dyx, flow_dyy = self.gradient(flow_dy)

        smooth1 = torch.mean(flow_dx.abs()) + \
                  torch.mean(flow_dy.abs())

        smooth2 = torch.mean(flow_dxx.abs()) + \
                  torch.mean(flow_dxy.abs()) + \
                  torch.mean(flow_dyx.abs()) + \
                  torch.mean(flow_dyy.abs())

        return smooth1 + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy

    def linear_constraint(self, affine):
        identity = torch.zeros_like(affine).cuda()
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        loss = (affine-identity).abs().mean()
        return loss

    def identity_grid(self, ref):
        b, _, h, w = ref.size()
        x = np.arange(h).astype(np.float32) / h * 2 - 1
        y = np.arange(w).astype(np.float32) / w * 2 - 1
        xx, yy = np.meshgrid(x, y)
        xx_torch = torch.from_numpy(xx)
        xx_torch = xx_torch.unsqueeze(0).unsqueeze(1)
        yy_torch = torch.from_numpy(yy)
        yy_torch = yy_torch.unsqueeze(0).unsqueeze(1)
        grid = torch.cat((xx_torch, yy_torch), 1).repeat(b, 1, 1, 1).cuda()
        return grid