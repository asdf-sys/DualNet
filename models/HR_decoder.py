import torch
import torch.nn as nn
from .layers import Conv1x1, Conv3x3, upsample


class HRDecoder(nn.Module):
    def __init__(self,  num_ch_enc):
        super(HRDecoder, self).__init__()

        bottleneck = 256
        self.do = nn.Dropout(p=0.5)

        self.reduce4 = Conv1x1(num_ch_enc[4], 512, bias=False)
        self.reduce3 = Conv1x1(num_ch_enc[3], bottleneck, bias=False)
        self.reduce2 = Conv1x1(num_ch_enc[2], bottleneck, bias=False)
        self.reduce1 = Conv1x1(num_ch_enc[1], bottleneck, bias=False)
        self.reduce0 = Conv1x1(num_ch_enc[0], bottleneck, bias=False)

        self.iconv4 = Conv3x3(512, bottleneck)
        self.iconv3 = Conv3x3(bottleneck*2+2, bottleneck)
        self.iconv2 = Conv3x3(bottleneck*2+2, bottleneck)
        self.iconv1 = Conv3x3(bottleneck*2+2, bottleneck)
        self.iconv0 = Conv3x3(bottleneck*2+2, bottleneck)

        self.merge4 = Conv3x3(bottleneck, bottleneck)
        self.merge3 = Conv3x3(bottleneck, bottleneck)
        self.merge2 = Conv3x3(bottleneck, bottleneck)
        self.merge1 = Conv3x3(bottleneck, bottleneck)
        self.merge0 = Conv3x3(bottleneck, bottleneck)

        # flow
        self.flow4 = nn.Sequential(Conv3x3(bottleneck, 2), nn.Tanh())
        self.flow3 = nn.Sequential(Conv3x3(bottleneck, 2), nn.Tanh())
        self.flow2 = nn.Sequential(Conv3x3(bottleneck, 2), nn.Tanh())
        self.flow1 = nn.Sequential(Conv3x3(bottleneck, 2), nn.Tanh())
        self.flow0 = nn.Sequential(Conv3x3(bottleneck, 2), nn.Tanh())

    def forward(self, input_features, frame_id=0):
        scale = 0.1
        self.outputs = {}
        l0, l1, l2, l3, l4 = input_features

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.reduce4(l4)
        x4 = self.iconv4(x4)
        x4 = self.merge4(x4)
        x4 = upsample(x4)
        flow4 = self.flow4(x4) * scale

        x3 = self.reduce3(l3)
        x3 = torch.cat((x3, x4, flow4), 1)
        x3 = self.iconv3(x3)
        x3 = self.merge3(x3)
        x3 = upsample(x3)
        flow3 = self.flow3(x3) * scale

        x2 = self.reduce2(l2)
        x2 = torch.cat((x2, x3 , flow3), 1)
        x2 = self.iconv2(x2)
        x2 = self.merge2(x2)
        x2 = upsample(x2)
        flow2 = self.flow2(x2) * scale

        x1 = self.reduce1(l1)
        x1 = torch.cat((x1, x2, flow2), 1)
        x1 = self.iconv1(x1)
        x1 = self.merge1(x1)
        x1 = upsample(x1)
        flow1 = self.flow1(x1) * scale

        x0 = self.reduce0(l0)
        x0 = torch.cat((x0, x1, flow1), 1)
        x0 = self.iconv0(x0)
        x0 = self.merge0(x0)
        x0 = upsample(x0)
        flow0 = self.flow0(x0) * scale

        return [flow0,flow1,flow2,flow3,flow4]
