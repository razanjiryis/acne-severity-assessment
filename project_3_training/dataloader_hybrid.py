import torch
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AcneHybridDataset(Dataset):
    def __init__(self, jsonl_path, augment=False):
        with open(jsonl_path, 'r') as f:
            self.records = [json.loads(line) for line in f]
        
        self.augment = augment
        
        # 90%+ Push: Stronger Clinical Augmentations
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                # ColorJitter prevents the model from being biased by skin tone
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        
        # Load Image
        img = Image.open(r['image_path']).convert('RGB')
        img_tensor = self.transform(img)

        # Load Targets
        density = torch.from_numpy(np.load(r['density_path'])).unsqueeze(0)
        ldl_label = torch.tensor(r['ldl_label'], dtype=torch.float32)
        sev_gt = int(r['sev_gt'])

        return {
            "img": img_tensor,
            "density": density,
            "ldl": ldl_label,
            "sev_gt": sev_gt
        }