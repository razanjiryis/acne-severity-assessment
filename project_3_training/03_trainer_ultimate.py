import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 1. Setup Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from project_2_models.model_acne_mtl_fusion import AcneFusionNet
from dataloader_hybrid import AcneHybridDataset
from loss_mtl import AcneMultiTaskLoss

def get_balanced_sampler(dataset):
    labels = [r['sev_gt'] for r in dataset.records]
    class_counts = np.bincount(labels)
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = weights[labels]
    return WeightedRandomSampler(samples_weights, len(samples_weights))

def train_fold(fold_idx=0, num_epochs=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = PROJECT_ROOT / "outputs" / f"fold_{fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = PROJECT_ROOT / "project_1_dataset_processing/final_manifests"
    train_ds = AcneHybridDataset(manifest_dir / f"fold_{fold_idx}_train.jsonl", augment=True)
    val_ds = AcneHybridDataset(manifest_dir / f"fold_{fold_idx}_val.jsonl", augment=False)
    
    sampler = get_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    model = AcneFusionNet(num_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=5e-2)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=1)
    
    # CRITICAL FIX: Move the criterion to the GPU
    criterion = AcneMultiTaskLoss(lambda_sev=7.0).to(device)

    print(f"\n🚀 --- STARTING WEIGHTED PUSH: FOLD {fold_idx} --- 🚀")
    best_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            imgs, gt_den, gt_ldl = batch['img'].to(device), batch['density'].to(device), batch['ldl'].to(device)
            optimizer.zero_grad()
            pred_den, pred_ldl = model(imgs)
            
            # Loss calculation
            loss_dict = criterion(pred_den, pred_ldl, gt_den, gt_ldl)
            loss_dict['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss_dict['total'].item())

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, gts = batch['img'].to(device), batch['sev_gt'].to(device)
                _, pred_ldl = model(imgs)
                correct += (pred_ldl.argmax(dim=1) == gts).sum().item()
                total += imgs.size(0)
        
        val_acc = correct / total
        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_weights.pt")
            print(f"Epoch {epoch+1:02d} | Loss: {np.mean(train_losses):.4f} | Val Acc: {val_acc*100:.2f}% ⭐")
        else:
            print(f"Epoch {epoch+1:02d} | Loss: {np.mean(train_losses):.4f} | Val Acc: {val_acc*100:.2f}%")

    return best_acc

if __name__ == "__main__":
    results = []
    for f in range(5):
        best_f = train_fold(fold_idx=f)
        results.append(best_f)
    print(f"\nFINAL CV MEAN: {np.mean(results)*100:.2f}%")