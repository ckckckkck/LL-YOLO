import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv, DWConv, C2PSA


class SPPFC2PSA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.
        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)  # First convolution layer
        self.cv2 = Conv(c_ * 4, c2, 1, 1)  # Final convolution layer
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)  # Max pooling layer
        self.cv_residual = DWConv(c1, c_ * 4, 1, 1)  # Convolution to match the output channels for residual
        self.branch_c2psa1 = C2PSA(c_, c_, n=1, e=0.5)  
        self.branch_c2psa2 = C2PSA(c_, c_, n=1, e=0.5)  
        self.branch_c2psa3 = C2PSA(c_, c_, n=1, e=0.5)  
        #self.conv_bridge = Conv(c_ * 2, c_, 1, 1)  
        self.conv_bridge = Conv(c_ * 3, c_ * 4, 1, 1)

    def forward(self, x):
        """Forward pass with residual connection."""
        y = [self.cv1(x)]
        y.append(self.m(y[-1]))  
        branch1 = self.branch_c2psa1(y[-1])
        y.append(self.m(y[-1])) 
        branch2 = self.branch_c2psa2(y[-1])
        y.append(self.m(y[-1])) 
        branch3 = self.branch_c2psa3(y[-1])
        bridge = torch.cat([branch1, branch2,branch3], dim=1)
        bridge = self.conv_bridge(bridge)
        residual = self.cv_residual(x)  
        output = torch.cat(y, 1)
        return self.cv2(output + bridge + residual)  
