# Modified from: https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/pixelcnn/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class VerticalMaskConv2d(nn.Module):
    def __init__(self, *args, **kwags):
        super().__init__()

        self.conv = nn.Conv2d(*args, **kwags)
        # 只保留卷积核的上半部分，下半部分置为0
        h, w = self.conv.weight.shape[-2:]
        mask = torch.zeros((h, w), dtype=torch.float32)
        mask[0:h // 2 + 1] = 1
        mask = mask.reshape((1, 1, h, w))
        self.register_buffer('mask', mask, False)  # 不训练

    def forward(self, x:torch.Tensor):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)

        return conv_res

class HorizontalMaskConv2d(nn.Module):
    def __init__(self, conv_type, *args, **kwargs):
        super().__init__()
        assert conv_type in ('A', 'B')

        self.conv = nn.Conv2d(*args, **kwargs)
        # 只保留卷积核中间的左半部分
        h, w = self.conv.weight.shape[-2:]
        mask = torch.zeros((h, w), dtype=torch.float32)
        mask[h // 2, 0:w // 2] = 1  # A类不能看到中间像素(自身)
        if conv_type == 'B':  # B类能看到中间像素(自身)
            mask[h // 2, w // 2] = 1
        mask = mask.reshape((1, 1, h, w))
        self.register_buffer('mask', mask, False)
         
    def forward(self, x:torch.Tensor):
        self.conv.weight.data *= self.mask
        conv_res = self.conv(x)

        return conv_res

class GatedBlock(nn.Module):
    def __init__(self, conv_type, in_channels, p, bn=True):
        """
        - `conv_type`: 水平卷积核类型: A or B
        - `in_channels`: 输入通道数
        - `p`: 隐藏层通道数
        - `bn`: 是否BatchNorm
        """
        super().__init__()

        self.conv_type = conv_type
        self.p = p

        self.v_conv = VerticalMaskConv2d(in_channels, 2 * p, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.v_to_h_conv = nn.Conv2d(2 * p, 2 * p, 1)
        self.bn2 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_conv = HorizontalMaskConv2d(conv_type, in_channels, 2 * p, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(2 * p) if bn else nn.Identity()
        self.h_output_conv = nn.Conv2d(p, p, 1)
        self.bn4 = nn.BatchNorm2d(p) if bn else nn.Identity()

    def forward(self, v_input:torch.Tensor, h_input:torch.Tensor):
        # ---------------------------------------------------------------------
        # 垂直掩码卷积核卷积，只看到上半部分的信息
        # ---------------------------------------------------------------------
        v = self.v_conv(v_input)
        v = self.bn1(v)
        # TODO: 整体下移一行，可以理解成把上半部分的信息传递给当前水平流
        v_to_h = v[:, :, 0:-1]  # 去除最后一行
        v_to_h = F.pad(v_to_h, (0, 0, 1, 0))  # 第一行填0
        v_to_h = self.v_to_h_conv(v_to_h)

        v_to_h = self.bn2(v_to_h)

        # ---------------------------------------------------------------------
        # 垂直流门控
        # ---------------------------------------------------------------------
        v1, v2 = v[:, :self.p], v[:, self.p:]
        v1 = torch.tanh(v1)
        v2 = torch.sigmoid(v2)
        v = v1 * v2

        # ---------------------------------------------------------------------
        # 水平掩码卷积核卷积，只看到左半部分的信息
        # ---------------------------------------------------------------------
        h = self.h_conv(h_input)
        h = self.bn3(h)
        h = h + v_to_h

        # ---------------------------------------------------------------------
        # 水平流门控+输出
        # ---------------------------------------------------------------------
        h1, h2 = h[:, :self.p], h[:, self.p:]
        h1 = torch.tanh(h1)
        h2 = torch.sigmoid(h2)
        h = h1 * h2
        h = self.h_output_conv(h)
        h = self.bn4(h)
        if self.conv_type == 'B':  # 第一层不做残差连接
            h = h + h_input

        return v, h

class GatedPixelCNN(nn.Module):
    def __init__(self, n_blocks, p, linear_dim, bn=True, color_level=256):
        super().__init__()
        
        self.block1 = GatedBlock('A', 1, p, bn)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(GatedBlock('B', p, p, bn))
        self.relu = nn.ReLU()
        self.linear1 = nn.Conv2d(p, linear_dim, 1)
        self.linear2 = nn.Conv2d(linear_dim, linear_dim, 1)
        self.out = nn.Conv2d(linear_dim, color_level, 1)

    def forward(self, x:torch.Tensor):
        # input: [bs, c, h, w] -> [bs, p, h, w]*2
        v, h = self.block1(x, x)

        # 形状不变
        for block in self.blocks:
            v, h = block(v, h)

        # 最终输出要h(水平流) -> [bs, linear_dim, h, w]
        x = self.relu(h)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.out(x)

        # -> [bs, color_level, h, w]
        return x
