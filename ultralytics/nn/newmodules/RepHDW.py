import torch
import torch.nn as nn
from ultralytics.nn.modules import GhostConv
from ultralytics.nn.newmodules.common import Conv, get_bn, fuse_bn, merge_dilated_into_large_kernel, to_2tuple

#-----------------------------------------------------------------------------------------------------5改3----
class RepHDW1(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(RepHDW1, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = Conv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out

class RepHDW(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, shortcut = True, expansion = 0.5, kersize = 5,depth_expansion = 1,small_kersize = 3,use_depthwise = True):
        super(RepHDW, self).__init__()
        c1 = int(out_channels * expansion) * 2
        c_ = int(out_channels * expansion)
        self.c_ = c_
        self.conv1 = Conv(in_channels, c1, 1, 1)
        self.m = nn.ModuleList(DepthBottleneckUni(self.c_, self.c_, shortcut,kersize,depth_expansion,small_kersize,use_depthwise) for _ in range(depth))
        self.conv2 = GhostConv(c_ * (depth+2), out_channels, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x_out = list(x.split((self.c_, self.c_), 1))
        for conv in self.m:
            y = conv(x_out[-1])
            x_out.append(y)
        y_out = torch.cat(x_out, axis=1)
        y_out = self.conv2(y_out)
        return  y_out


class DepthBottleneckUni(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 shortcut=True,
                 kersize=5,
                 expansion_depth=1,
                 small_kersize=3,
                 use_depthwise=True):
        super(DepthBottleneckUni, self).__init__()

        mid_channel = int(in_channels * expansion_depth)
        self.conv1 = Conv(in_channels, mid_channel, 1)
        self.shortcut = shortcut
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid_channel, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid_channel, out_channels, kernel_size=1)
        else:
            self.conv2 = Conv(out_channels, out_channels, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        return y

class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 deploy=False,
                 attempt_use_lk_impl=True):
        super().__init__()
        if deploy:
            print('------------------------------- Note: deploy mode')
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            assert kernel_size in [3]
            self.dwconv = get_conv2d_uni(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                         dilation=1, groups=dim, bias=deploy,
                                         attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim)

    def forward(self, inputs):

        out = self.norm(self.dwconv(inputs))
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                        self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()

class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d_uni(channels, channels, kernel_size, stride=1,
                                        padding=kernel_size // 2, dilation=1, groups=channels, bias=deploy,
                                        )
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        # if kernel_size == 17:
        #     self.kernel_sizes = [5, 9, 3, 3, 3]
        #     self.dilates = [1, 2, 4, 5, 7]
        # elif kernel_size == 15:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 5, 7]
        # elif kernel_size == 13:
        #     self.kernel_sizes = [5, 7, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 11:
        #     self.kernel_sizes = [5, 5, 3, 3, 3]
        #     self.dilates = [1, 2, 3, 4, 5]
        # elif kernel_size == 9:
        #     self.kernel_sizes = [5, 5, 3, 3]
        #     self.dilates = [1, 2, 3, 4]
        # elif kernel_size == 7:
        #     self.kernel_sizes = [5, 3, 3, 3]
        #     self.dilates = [1, 1, 2, 3]
        # elif kernel_size == 5:
        #     self.kernel_sizes = [3, 3, 1]
        #     self.dilates = [1, 2, 1]
        # elif kernel_size == 3:
        #     self.kernel_sizes = [3, 1]
        #     self.dilates = [1, 1]
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [7, 5, 3]
            self.dilates = [1, 1, 1]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3]
            self.dilates = [1, 1]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]
        elif kernel_size == 3:
            self.kernel_sizes = [3, 1]
            self.dilates = [1, 1]

        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):  # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d_uni(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                         padding=origin_k.size(2) // 2, dilation=1, groups=origin_k.size(0), bias=True,
                                         attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))

def get_conv2d_uni(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    # if attempt_use_lk_impl and need_large_impl:
    #     print('---------------- trying to import iGEMM implementation for large-kernel conv')
    #     try:
    #         from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
    #         print('---------------- found iGEMM implementation ')
    #     except:
    #         DepthWiseConv2dImplicitGEMM = None
    #         print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
    #     if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
    #             and out_channels == groups and stride == 1 and dilation == 1:
    #         print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
    #         return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)