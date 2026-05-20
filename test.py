import torch
import cv2
import numpy as np
from models import UNetBoneSuppressor

def process_xray_guided(image_path, model_path='model_checkpoint.pth'):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNetBoneSuppressor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    orig_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orig_float = orig_cv.astype(np.float32)
    img_resized = cv2.resize(orig_cv, (512, 512)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = model(tensor).squeeze().cpu().numpy()
    
    mask = cv2.resize(mask, (orig_cv.shape[1], orig_cv.shape[0]))
    # FEATURE LOCK: Preserving original gradients
    suppressed = orig_float * (1.0 - (mask * 0.96))
    final_norm = cv2.normalize(suppressed, None, orig_float.min(), orig_float.max(), cv2.NORM_MINMAX)

    # HIGH-FREQUENCY TEXTURE RECOVERY
    blur = cv2.GaussianBlur(orig_float, (0, 0), 2)
    sharp_details = orig_float - blur
    final_combined = final_norm + sharp_details
    
    final_output = np.clip(final_combined, 0, 255).astype('uint8')
    cv2.imwrite("final_output.png", final_output)
    return "final_output.png"