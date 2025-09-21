import torch
import torch.nn as nn
__all__ = ['C2PSA_TripleAttention', 'TripletAttention']

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,

                 bn=True, bias=False):

        super(BasicConv, self).__init__()

        self.out_channels = out_planes

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,

                              dilation=dilation, groups=groups, bias=bias)

        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.relu is not None:
            x = self.relu(x)

        return x


class ZPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):

    def __init__(self):
        super(AttentionGate, self).__init__()

        kernel_size = 7

        self.compress = ZPool()

        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)

        x_out = self.conv(x_compress)

        scale = torch.sigmoid_(x_out)

        return x * scale


class TripletAttention(nn.Module):

    def __init__(self, no_spatial=False):

        super(TripletAttention, self).__init__()

        self.cw = AttentionGate()

        self.hc = AttentionGate()

        self.no_spatial = no_spatial

        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):

        x_perm1 = x.permute(0, 2, 1, 3).contiguous()

        x_out1 = self.cw(x_perm1)

        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()

        x_out2 = self.hc(x_perm2)

        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        if not self.no_spatial:

            x_out = self.hw(x)

            x_out = 1 / 3 * (x_out + x_out11 + x_out21)

        else:

            x_out = 1 / 2 * (x_out11 + x_out21)

        return x_out


def autopad(k, p=None, d=1):  # kernel, padding, dilation

    """Pad to 'same' shape outputs."""

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""

        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.bn = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""

        return self.act(self.conv(x))


def autopad(k, p=None, d=1):  # kernel, padding, dilation

    """Pad to 'same' shape outputs."""

    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size

    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad

    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""

        super().__init__()

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        self.bn = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""

        return self.act(self.conv(x))


class PSABlock(nn.Module):
    """

    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers

    with optional shortcut connections.

    Attributes:

        attn (Attention): Multi-head attention module.

        ffn (nn.Sequential): Feed-forward neural network module.

        add (bool): Flag indicating whether to add shortcut connections.

    Methods:

        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:

        Create a PSABlock and perform a forward pass

        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)

        >>> input_tensor = torch.randn(1, 128, 32, 32)

        >>> output_tensor = psablock(input_tensor)

    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""

        super().__init__()

        self.attn = TripletAttention()

        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""

        x = x + self.attn(x) if self.add else self.attn(x)

        x = x + self.ffn(x) if self.add else self.ffn(x)

        return x


class C2PSA_TripleAttention(nn.Module):
    """

    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing

    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:

        c (int): Number of hidden channels.

        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.

        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.

        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:

        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:

        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:

        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)

        >>> input_tensor = torch.randn(1, 256, 64, 64)

        >>> output_tensor = c2psa(input_tensor)

    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""

        super().__init__()

        assert c1 == c2

        self.c = int(c1 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""

        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        b = self.m(b)

        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    # Generating Sample image

    image_size = (1, 64, 224, 224)

    image = torch.rand(*image_size)

    # Model

    model = C2PSA_TripleAttention(64, 64)

    out = model(image)

    print(out.size())


