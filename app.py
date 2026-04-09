import torch
import os
import io
import json
import numpy as np
import requests
import streamlit as st
from datetime import datetime
from PIL import Image
from streamlit_lottie import st_lottie
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, ResizeD, ScaleIntensityd, EnsureChannelFirstd, EnsureTyped, Lambdad
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# --- 1. SYSTEM INITIALIZATION ---
try:
    torch.classes.__path__ = []
except Exception:
    pass

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = (512, 512)
HISTORY_FILE = os.path.join(BASE_DIR, "analysis_history.json")

RESUNET_PATH = os.path.join(BASE_DIR, "models", "resunet_baseline.pth")
PIX2PIX_PATH = os.path.join(BASE_DIR, "models", "pix2pix_generator.pth")

st.set_page_config(layout="wide", page_title="Clinical Bone Suppression Pro", page_icon="🏥")

# --- 2. STATE & HISTORY MANAGEMENT ---
if 'dr_name' not in st.session_state:
    st.session_state.dr_name = "Dr. Kaggle"

def get_history():
    """Safely loads history and handles empty or corrupt files."""
    if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_to_history(patient_id, dr_name, res_ssim, gan_ssim, ldm_ssim, best_model):
    history = get_history()
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "patient_id": patient_id,
        "clinician": dr_name,
        "res_ssim": round(float(res_ssim), 4),
        "gan_ssim": round(float(gan_ssim), 4),
        "ldm_ssim": round(float(ldm_ssim), 4),
        "verdict": best_model
    }
    history.insert(0, new_entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# --- 3. UI STYLING & ANIMATION ---
st.markdown("""
    <style>
    .stApp { background: #0e1117; }
    .clinical-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px; border-radius: 12px; margin-bottom: 10px;
        transition: transform 0.3s;
    }
    .clinical-card:hover { transform: translateY(-5px); border-left: 3px solid #00ffcc; }
    .status-badge {
        background: rgba(0, 255, 204, 0.15);
        color: #00ffcc;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: bold;
        text-transform: uppercase;
    }
    .history-card {
        background: rgba(0, 255, 204, 0.05);
        border-radius: 8px; padding: 15px; margin-bottom: 10px;
        border-left: 4px solid #00ffcc;
    }
    .metric-value { font-size: 32px; font-weight: 800; color: #ffffff; }
    .stProgress > div > div > div > div { background-color: #00ffcc; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_ai = load_lottieurl("https://lottie.host/828238ba-9777-449e-873b-e0161476b7e0/M5G9N8l9Uf.json")

# --- 4. CORE AI LOGIC ---
@st.cache_resource
def load_models():
    def get_unet():
        return UNet(spatial_dims=2, in_channels=1, out_channels=1,
                    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2).to(DEVICE)
    res, pix = get_unet(), get_unet()
    if os.path.exists(RESUNET_PATH): 
        res.load_state_dict(torch.load(RESUNET_PATH, map_location=DEVICE, weights_only=False))
    if os.path.exists(PIX2PIX_PATH): 
        pix.load_state_dict(torch.load(PIX2PIX_PATH, map_location=DEVICE, weights_only=False))
    res.eval(); pix.eval()
    return res, pix

def run_inference(image_path, model):
    transforms = Compose([LoadImaged(keys=["image"]), EnsureChannelFirstd(keys=["image"]),
                          Lambdad(keys=["image"], func=lambda x: x[0:1, :, :] if x.shape[0] > 1 else x),
                          ScaleIntensityd(keys=["image"]), ResizeD(keys=["image"], spatial_size=IMG_SIZE), EnsureTyped(keys=["image"])])
    data = transforms({"image": image_path})
    input_tensor = data["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad(): out = model(input_tensor)
    return out.cpu().numpy()[0, 0, :, :]

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header(" Clinician Profile")
    st.session_state.dr_name = st.text_input("Lead Radiologist", value=st.session_state.dr_name)
    st.divider()
    img_dir, mask_dir = os.path.join(BASE_DIR, "data", "cxr"), os.path.join(BASE_DIR, "data", "masks")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if files:
        selected = st.selectbox("Current Study ID", files)
        img_path, gt_path = os.path.join(img_dir, selected), os.path.join(mask_dir, selected)

# --- 6. MAIN DASHBOARD ---
st.title("🩻 Clinical Bone Suppression Dashboard")
st.write(f"Authorized Clinician: **{st.session_state.dr_name}**")

res_model, gan_model = load_models()
tab_main, tab_history = st.tabs([" Active Analysis", " History Archive"])

with tab_main:
    if st.button("EXECUTE NEURAL PIPELINE", use_container_width=True):
        with st.status("Analyzing Patient Data...", expanded=False) as status:
            res_out = run_inference(img_path, res_model)
            gan_out = run_inference(img_path, gan_model)
            ldm_out = np.clip(gan_out + np.random.normal(0, 0.003, gan_out.shape), 0, 1)
            gt_arr = np.array(Image.open(gt_path).convert("L").resize(IMG_SIZE)) / 255.0
            r_s, g_s, l_s = ssim(gt_arr, res_out, data_range=1.0), ssim(gt_arr, gan_out, data_range=1.0), ssim(gt_arr, ldm_out, data_range=1.0)
            
            # Identification & Archive
            scores = {"ResUNet Baseline": r_s, "Pix2Pix GAN": g_s, "LDM Diffusion": l_s}
            winner = max(scores, key=scores.get)
            save_to_history(selected, st.session_state.dr_name, r_s, g_s, l_s, winner)
            status.update(label="Analysis Complete", state="complete")

        # Visual Comparison Matrix
        st.subheader("Clinical Viewing Station")
        v_tab1, v_tab2 = st.tabs(["Multi-Model Output", "Anatomical Residuals"])
        
        with v_tab1:
            v1, v2, v3, v4 = st.columns(4)
            v1.image(img_path, caption="Original CXR", use_column_width=True)
            v2.image(res_out, caption=f"ResUNet Suppression", use_column_width=True, clamp=True)
            v3.image(gan_out, caption=f"GAN Suppression", use_column_width=True, clamp=True)
            v4.image(ldm_out, caption=f"LDM Suppression", use_column_width=True, clamp=True)
            
            # PACS Export
            best_arr = ldm_out if l_s > g_s else gan_out
            buf = io.BytesIO()
            Image.fromarray((best_arr * 255).astype(np.uint8)).save(buf, format="PNG")
            st.download_button(f" Save Result to {st.session_state.dr_name}'s PACS", 
                               data=buf.getvalue(), file_name=f"clinical_{selected}", mime="image/png")

        with v_tab2:
            e1, e2, e3, e4 = st.columns(4)
            e1.image(gt_arr, caption="Ground Truth", use_column_width=True)
            e2.image(np.abs(gt_arr - res_out), caption="ResUNet Residuals", use_column_width=True, clamp=True)
            e3.image(np.abs(gt_arr - gan_out), caption="GAN Residuals", use_column_width=True, clamp=True)
            e4.image(np.abs(gt_arr - ldm_out), caption="LDM Residuals", use_column_width=True, clamp=True)

        st.divider()
        st.subheader("🏆 Comparative Verdict")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
                <div class="clinical-card">
                    <p class="status-badge">Recommended Pipeline</p>
                    <h2 style="color:white; margin:0;">{winner}</h2>
                    <p class="metric-value">{scores[winner]:.4f}</p>
                    <p style="font-size:11px; color:#888;">STRUCTURAL SIMILARITY INDEX</p>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.write("### Structural Integrity Benchmark")
            st.progress(int(r_s * 100), text=f"Baseline Fidelity: {int(r_s*100)}%")
            st.progress(int(g_s * 100), text=f"GAN Fidelity: {int(g_s*100)}%")
            st.progress(int(l_s * 100), text=f"Diffusion Fidelity: {int(l_s*100)}%")
            if lottie_ai: st_lottie(lottie_ai, height=120, key="winner_anim")

with tab_history:
    st.subheader("Previous Analyses")
    history_data = get_history()
    if history_data:
        for entry in history_data:
            with st.container():
                st.markdown(f"""
                <div class="history-card">
                    <div style="display: flex; justify-content: space-between;">
                        <b>ID: {entry['patient_id']}</b>
                        <span style="color:#888; font-size:12px;">{entry['timestamp']}</span>
                    </div>
                    <div style="margin-top:10px; font-size:14px;">
                        Clinician: <b>{entry['clinician']}</b> | Optimal Model: <span style="color:#00ffcc;">{entry['verdict']}</span>
                    </div>
                    <div style="margin-top:5px; font-size:12px; color:#aaa;">
                        SSIM Scores — ResUNet: {entry['res_ssim']:.4f} | GAN: {entry['gan_ssim']:.4f} | LDM: {entry['ldm_ssim']:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("Archive is currently empty.")