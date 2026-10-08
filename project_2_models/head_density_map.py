import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers_and_blocks import GN, DilatedBottleneck

class DensityMapDecoder(nn.Module):
    def __init__(self, in_ch=1408):
        super().__init__()
        self.bottleneck = DilatedBottleneck(in_ch, 512)
        
        # Progressive upsampling
        self.up1 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.gn1 = GN(128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.gn2 = GN(64)
        
        # FIXED: scale_factor changed from 4 to 8 to reach 512x512
        self.final_upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, features):
        # Input c5 is 16x16 for a 512x512 image
        x = self.bottleneck(features["c5"]) # 16x16
        
        x = F.relu(self.gn1(self.up1(x)))    # 32x32
        x = F.relu(self.gn2(self.up2(x)))    # 64x64
        
        x = self.final_upsample(x)           # 512x512 (Matches GT)
        return self.activation(self.conv_out(x))