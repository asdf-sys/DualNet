from __future__ import absolute_import, division, print_function
import torch.nn as nn
import torch


class LRDecoder(nn.Module):
    def __init__(self, num_ch_enc, stride=1):
        super(LRDecoder, self).__init__()

        self.reduce = nn.Conv2d(num_ch_enc[-1], 256, 1)
        self.conv1 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, stride, 1)
        self.conv3 = nn.Conv2d(256, 6, 1)

        self.relu = nn.ReLU()

    def forward(self, input_features):
        f = input_features[-1]
        out = self.reduce(f)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, 2, 3)

        identity = torch.zeros_like(out)
        identity[:, 0, 0] = 1
        identity[:, 1, 1] = 1
        out = out + identity
        return out
