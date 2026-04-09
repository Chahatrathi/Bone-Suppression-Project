import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, ResizeD, ScaleIntensityd, 
    EnsureTyped, EnsureChannelFirstd, Lambdad
)
from monai.data import Dataset

# --- 1. CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 2
LEARNING_RATE = 2e-4 # Standard for Pix2Pix
L1_LAMBDA = 100      # Weight for L1 reconstruction loss
EPOCHS = 50
IMG_SIZE = (512, 512)

# --- 2. DISCRIMINATOR ARCHITECTURE (PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        def critic_block(in_c, out_c, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Takes (input_cxr + target_mask) concatenated = 2 channels
        self.model = nn.Sequential(
            critic_block(in_channels * 2, 64),
            critic_block(64, 128),
            critic_block(128, 256),
            critic_block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1) # Output 1D patch map
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# --- 3. DATA PREPARATION ---
original_dir = os.path.join(BASE_DIR, "data", "cxr")
soft_tissue_dir = os.path.join(BASE_DIR, "data", "masks")

if not os.path.exists(original_dir):
    print(f"[ERROR] Data not found.")
    sys.exit(1)

original_files = sorted([os.path.join(original_dir, f) for f in os.listdir(original_dir) if f.endswith('.png')])
soft_tissue_files = sorted([os.path.join(soft_tissue_dir, f) for f in os.listdir(soft_tissue_dir) if f.endswith('.png')])

data_dicts = [{"image": img, "label": seg} for img, seg in zip(original_files, soft_tissue_files)]

def ensure_gray(x):
    return x[0:1, :, :] if x.shape[0] > 1 else x

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys=["image", "label"], func=ensure_gray),
    ScaleIntensityd(keys=["image", "label"]),
    ResizeD(keys=["image", "label"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

train_loader = DataLoader(Dataset(data=data_dicts, transform=transforms), batch_size=BATCH_SIZE, shuffle=True)

# --- 4. MODEL INITIALIZATION ---
# Generator (Your previous ResUNet)
generator = UNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
).to(DEVICE)

# Discriminator
discriminator = Discriminator(in_channels=1).to(DEVICE)

# Optimizers & Loss
opt_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# --- 5. TRAINING LOOP ---
print("Starting Pix2Pix GAN Training...")

for epoch in range(EPOCHS):
    g_loss_accum = 0.0
    d_loss_accum = 0.0
    
    for batch_data in train_loader:
        real_x = batch_data["image"].to(DEVICE)
        real_y = batch_data["label"].to(DEVICE)
        
        # --- Train Discriminator ---
        opt_D.zero_grad()
        
        fake_y = generator(real_x)
        
        # Real loss
        pred_real = discriminator(real_x, real_y)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
        
        # Fake loss
        pred_fake = discriminator(real_x, fake_y.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        opt_D.step()
        
        # --- Train Generator ---
        opt_G.zero_grad()
        
        pred_fake = discriminator(real_x, fake_y)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) # Fool the D
        loss_G_L1 = criterion_L1(fake_y, real_y) * L1_LAMBDA # Stay close to Ground Truth
        
        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        opt_G.step()
        
        g_loss_accum += loss_G.item()
        d_loss_accum += loss_D.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | G_Loss: {g_loss_accum/len(train_loader):.4f} | D_Loss: {d_loss_accum/len(train_loader):.4f}")

# --- 6. SAVE MODELS ---
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
torch.save(generator.state_dict(), os.path.join(BASE_DIR, "models", "pix2pix_generator.pth"))
print(f"\nSuccess! GAN Model Saved.")