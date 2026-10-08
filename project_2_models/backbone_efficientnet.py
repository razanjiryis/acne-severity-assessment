import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        base = efficientnet_b2(weights=weights)
        
        # EfficientNet-B2 hierarchical features
        self.stem = base.features[0]    
        self.layer1 = base.features[1:3] # (c2)
        self.layer2 = base.features[3:4] # (c3)
        self.layer3 = base.features[4:6] # (c4)
        self.layer4 = base.features[6:9] # (c5) - 1408 channels

    def forward(self, x):
        x0 = self.stem(x)
        c2 = self.layer1(x0) 
        c3 = self.layer2(c2) 
        c4 = self.layer3(c3) 
        c5 = self.layer4(c4) 
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}