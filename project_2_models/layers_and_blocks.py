import torch
import torch.nn as nn

def GN(ch: int) -> nn.GroupNorm:
    """Helper to create a GroupNorm layer with a valid number of groups."""
    g = 32
    while ch % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, ch)

class DilatedBottleneck(nn.Module):
    """
    Captures varying lesion sizes. 
    FIXED: Ensures output is exactly 'out_ch' and GN is pre-initialized on the correct device.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 6
        
        # First 5 branches
        self.d1 = nn.Conv2d(in_ch, mid, 3, padding=1, dilation=1, bias=False)
        self.d2 = nn.Conv2d(in_ch, mid, 3, padding=2, dilation=2, bias=False)
        self.d4 = nn.Conv2d(in_ch, mid, 3, padding=4, dilation=4, bias=False)
        self.d8 = nn.Conv2d(in_ch, mid, 3, padding=8, dilation=8, bias=False)
        self.d16 = nn.Conv2d(in_ch, mid, 3, padding=16, dilation=16, bias=False)
        
        # 6th branch picks up the remainder to reach exactly out_ch
        last_mid = out_ch - (5 * mid)
        self.d32 = nn.Conv2d(in_ch, last_mid, 3, padding=32, dilation=32, bias=False)
        
        # FIXED: Initialize GN here so it moves to GPU with the rest of the model
        self.bn = GN(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_cat = torch.cat([
            self.d1(x), self.d2(x), self.d4(x), 
            self.d8(x), self.d16(x), self.d32(x)
        ], dim=1)
        return self.relu(self.bn(x_cat))