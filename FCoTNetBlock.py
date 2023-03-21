# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

class CoTNetLayer(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        class SELayer(nn.Module):
            def __init__(self, channel, reduction=16):
                super(SELayer, self).__init__()

                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Sequential(
                    nn.Linear(channel, channel // 16, bias=False),
                    nn.ReLU(),
                    nn.Linear(channel // 16, channel, bias=False),
                    nn.Sigmoid()
                )

            def forward(self, x):
                
                b, c, _, _ = x.size()
                y = self.avg_pool(x).view(b, c)
                y = self.fc(y).view(b, c, 1, 1)

                return x * y.expand_as(x)




        class In(nn.Module):
            
            def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
                super(In, self).__init__()


                self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

                self.branch2 = nn.Sequential(
                    BasicConv2d(in_channels, ch3x3red, kernel_size=1),
                    BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
                )

                self.branch3 = nn.Sequential(
                    BasicConv2d(in_channels, ch5x5red, kernel_size=1),
                    BasicConv2d(ch5x5red, ch5x5red, kernel_size=3, padding=1),
                    BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1) # 保证输出大小等于输入大小
                )

                self.branch4 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                    BasicConv2d(in_channels, pool_proj, kernel_size=1)
                )
                # self.branch5 = nn.Sequential(
                #     BasicConv2d(in_channels, ch7x7red, kernel_size=1),
                #     BasicConv2d(ch7x7red, ch7x7red, kernel_size=3, padding=1),
                #     BasicConv2d(ch7x7red, ch7x7red, kernel_size=3, padding=1),
                #     BasicConv2d(ch7x7red, ch7x7, kernel_size=3, padding=1)
                # )

            def forward(self, x):
                branch1 = self.branch1(x)
                branch2 = self.branch2(x)
                branch3 = self.branch3(x)
                branch4 = self.branch4(x)
                # branch5 = self.branch5(x)

                outputs = [branch1, branch2, branch3, branch4]

                return torch.cat(outputs, 1)

        class BasicConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super(BasicConv2d, self).__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.conv(x)

                return x

        factor = 8
        factor1 = 4

        self.value_embed = nn.Sequential(
            In(dim, dim // 2, dim, dim // 8, dim, dim // 8, dim // 4),
            nn.BatchNorm2d(dim)
        )


        self.attention_embed = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False), 
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        k1 = self.key_embed(x)  
        v = self.value_embed(x).view(bs, c, -1)  

        y = torch.cat([k1, x], dim=1) 
        att = self.attention_embed(y)  
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  
        k2 = F.softmax(att, dim=-1) * v  
        k2 = k2.view(bs, c, h, w)

        return x+k1+k2  

if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    cot = CoTNetLayer(dim=512, kernel_size=3)
    output = cot(input)
    print(output.shape)
