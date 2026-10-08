import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone_efficientnet import EfficientNetB2Encoder
from .head_density_map import DensityMapDecoder

class AcneFusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.encoder = EfficientNetB2Encoder(pretrained=True)
        self.density_decoder = DensityMapDecoder(in_ch=1408)
        
        # Severity Head with Task Fusion logic
        # Inputs: 1408 (global image features) + 1 (total predicted lesion count)
        self.severity_head = nn.Sequential(
            nn.Linear(1408 + 1, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1) # Vital for Label Distribution Learning
        )

    def forward(self, x):
        features = self.encoder(x)
        
        # 1. Prediction for Counting (Density Map)
        density_map = self.density_decoder(features)
        
        # Calculate 'Predicted Count' for Fusion
        # This allows the model to 'know' that high count = high severity
        pred_count = density_map.sum(dim=(1, 2, 3), keepdim=True).view(-1, 1)
        
        # 2. Prediction for Severity (LDL Distribution)
        # Global Average Pool of deepest features
        global_features = F.adaptive_avg_pool2d(features["c5"], 1).view(x.size(0), -1)
        
        # FUSE: Combine image features + predicted count
        fusion_input = torch.cat([global_features, pred_count], dim=1)
        severity_distribution = self.severity_head(fusion_input)
        
        return density_map, severity_distribution