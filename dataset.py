import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

class PairedXrayDataset(Dataset):
    def __init__(self, xray_dir, bone_free_dir, size=512):
        self.xray_paths = sorted(Path(xray_dir).glob("*.png")) + \
                         sorted(Path(xray_dir).glob("*.jpg"))
        self.bone_free_dir = Path(bone_free_dir)
        self.size = size

    def __len__(self):
        return len(self.xray_paths)

    def __getitem__(self, idx):
        xray_path = self.xray_paths[idx]
        bf_path = self.bone_free_dir / xray_path.name
        xray = cv2.imread(str(xray_path), cv2.IMREAD_GRAYSCALE)
        bf   = cv2.imread(str(bf_path),   cv2.IMREAD_GRAYSCALE)
        xray = cv2.resize(xray, (self.size, self.size)).astype(np.float32) / 255.
        bf   = cv2.resize(bf,   (self.size, self.size)).astype(np.float32) / 255.
        return (torch.from_numpy(xray).unsqueeze(0),
                torch.from_numpy(bf).unsqueeze(0))