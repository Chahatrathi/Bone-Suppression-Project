import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
from models import UNetBoneSuppressor
from dataset import PairedXrayDataset

class BoneSuppressLoss(nn.Module):
    """
    Combined L1 and SSIM loss for perceptual quality.
    Ensures soft tissue details are preserved while bones are removed.
    """
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.alpha = alpha

    def ssim(self, x, y, C1=0.01**2, C2=0.03**2):
        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        mu_xx = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
        mu_yy = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
        mu_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * mu_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (mu_xx + mu_yy + C2))
        return ssim_map.mean()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim_val = 1 - self.ssim(pred, target)
        return self.alpha * l1 + (1 - self.alpha) * ssim_val

def train():
    """
    Main training loop with real-time batch progress updates.
    """
    # 1. Device Configuration (Using MPS for Mac GPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🚀 Training on: {device}")

    # 2. Path Configuration
    xray_path = "./dataset/JSRT/JSRT"
    bone_free_path = "./dataset/BSE_JSRT/BSE_JSRT"

    # 3. Safety Checks
    if not os.path.exists(xray_path) or not os.path.exists(bone_free_path):
        print(f"❌ Error: Folder structure not found.")
        return

    # 4. Data Loading
    dataset = PairedXrayDataset(xray_path, bone_free_path)
    
    if len(dataset) == 0:
        print(f"❌ Error: No images found in {xray_path}")
        return

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 5. Model Initialization
    model = UNetBoneSuppressor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = BoneSuppressLoss()

    print(f"✅ Found {len(dataset)} images. Starting training for 30 epochs...")

    # 6. Training Loop with Batch-Level Updates
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            # Move data to GPU/MPS
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Real-time progress: update terminal every batch
            # 'end=\r' keeps the text on the same line
            print(f"Epoch {epoch+1:02d}/30 | Batch {batch_idx+1:02d}/{len(loader)} | Current Loss: {loss.item():.5f}", end='\r')
        
        # Calculate and print final epoch average
        avg_loss = total_loss / len(loader)
        print(f"\n✅ Epoch {epoch+1:02d}/30 Finished | Avg Loss: {avg_loss:.5f}")
    
    # 7. Save Weights
    torch.save(model.state_dict(), "model_checkpoint.pth")
    print("✨ Training Complete! Weights saved as model_checkpoint.pth")

if __name__ == "__main__":
    train()