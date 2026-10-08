import torch
import torch.nn as nn
import torch.nn.functional as F

class AcneMultiTaskLoss(nn.Module):
    def __init__(self, lambda_sev=7.0):
        super().__init__()
        self.lambda_sev = lambda_sev
        # Weighted strategy for 90% accuracy:
        # We penalize errors on Severe (Class 2) and V.Severe (Class 3) much harder.
        self.register_buffer('class_weights', torch.tensor([1.0, 1.0, 2.5, 2.2]))

    def forward(self, pred_density, pred_ldl, gt_density, gt_ldl):
        # 1. Counting Loss (MSE)
        loss_count = F.mse_loss(pred_density, gt_density)
        
        # 2. Weighted Severity Loss (KL Divergence)
        # DEVICE FIX: Ensure weights move to GPU automatically
        weights = self.class_weights.to(pred_ldl.device)
        
        # Calculate raw KL Divergence components
        # Formula: KL(GT || Pred) = GT * (log(GT) - log(Pred))
        kl_raw = gt_ldl * (torch.log(gt_ldl + 1e-10) - torch.log(pred_ldl + 1e-10))
        
        # Apply the weights to focus the model's 'attention' on Severe cases
        weighted_kl = kl_raw * weights.view(1, -1)
        loss_sev = weighted_kl.sum() / pred_ldl.size(0)
        
        total_loss = loss_count + (self.lambda_sev * loss_sev)
        
        return {
            "total": total_loss,
            "count": loss_count,
            "severity": loss_sev
        }