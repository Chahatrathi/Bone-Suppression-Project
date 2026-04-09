import os
import torch
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImaged, ResizeD, ScaleIntensityd, 
    EnsureTyped, EnsureChannelFirstd, Lambdad
)
from monai.data import Dataset, DataLoader

# 1. SETUP
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "resunet_baseline.pth")
IMG_SIZE = (512, 512)

# 2. DEFINE TRANSFORMS (Must match training exactly)
def ensure_gray(x):
    if x.shape[0] > 1: return x[0:1, :, :]
    return x

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Lambdad(keys=["image", "label"], func=ensure_gray),
    ScaleIntensityd(keys=["image", "label"]),
    ResizeD(keys=["image", "label"], spatial_size=IMG_SIZE),
    EnsureTyped(keys=["image", "label"]),
])

# 3. PREPARE DATA
original_dir = os.path.join(BASE_DIR, "data", "cxr")
soft_tissue_dir = os.path.join(BASE_DIR, "data", "masks")
files = sorted([f for f in os.listdir(original_dir) if f.endswith('.png')])
data_dicts = [{"image": os.path.join(original_dir, f), 
               "label": os.path.join(soft_tissue_dir, f)} for f in files[:3]] # Look at first 3

dataset = Dataset(data=data_dicts, transform=transforms)
loader = DataLoader(dataset, batch_size=1)

# 4. LOAD MODEL
model = UNet(
    spatial_dims=2, in_channels=1, out_channels=1,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()

# 5. INFERENCE & PLOTTING
print("Generating visualizations...")
with torch.no_grad():
    for i, batch in enumerate(loader):
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        outputs = model(inputs)

        # Convert back to CPU/Numpy for plotting
        img = inputs.cpu().numpy()[0, 0, :, :]
        gt = labels.cpu().numpy()[0, 0, :, :]
        pred = outputs.cpu().numpy()[0, 0, :, :]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original CXR (With Bones)")
        plt.imshow(img, cmap="gray")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth (Target)")
        plt.imshow(gt, cmap="gray")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("ResUNet Result (Suppressed)")
        plt.imshow(pred, cmap="gray")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()