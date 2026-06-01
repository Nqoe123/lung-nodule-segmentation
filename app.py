import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import time
import tempfile
import zipfile
import shutil
import SimpleITK as sitk

st.set_page_config(page_title="LungVision AI - 3D CT", page_icon="", layout="wide")

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .stApp { background: #0b1120; }
    .main > div { padding: 1rem; }
    h1, h2, h3 { color: #f8fafc; }
    .stButton > button { background: #0ea5e9; color: white; border: none; border-radius: 8px; padding: 0.5rem 1rem; }
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label::before {
        content: '';
        display: block;
        width: 20px;
        height: 2px;
        background: #38bdf8;
        border-radius: 2px;
    }
    .detection-badge {
        display: inline-block;
        background: #06b6d4;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

device = torch.device('cpu')

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x): return self.conv(x)

class MemoryEfficientUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    paths = ["best_model.pth", "/kaggle/working/best_model.pth", "complete_model_with_metadata.pth"]
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
    
    if model_path is None:
        st.error("Model file not found")
        return None
    
    model = MemoryEfficientUNet(n_channels=1, n_classes=1)
    state_dict = torch.load(model_path, map_location='cpu')
    
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    if state_dict and len(state_dict) > 0 and 'module.' in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def apply_lung_window(image):
    image = np.clip(image, -1000, 400)
    return ((image + 1000) / 1400).astype(np.float32)

def segment_patch(model, patch_img):
    tensor = torch.FloatTensor(patch_img).unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).squeeze().numpy()
    
    confidence = prob.max()
    return confidence

def analyze_slice(model, slice_img, patch_size=128, stride=64, confidence_threshold=0.65):
    """Analyze a single slice and return best detection confidence and position"""
    h, w = slice_img.shape
    best_confidence = 0
    best_position = None
    
    # Normalize and apply lung window
    img_norm = slice_img.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm = img_norm / 255.0
    img_norm = apply_lung_window(img_norm * 1400 - 1000)
    
    # Slide window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_norm[y:y+patch_size, x:x+patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            
            confidence = segment_patch(model, patch)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_position = (x, y)
    
    return best_confidence, best_position

def load_volume(zip_file):
    tmp = tempfile.mkdtemp()
    zpath = os.path.join(tmp, "upload.zip")
    with open(zpath, "wb") as f:
        f.write(zip_file.getbuffer())
    with zipfile.ZipFile(zpath, 'r') as zf:
        zf.extractall(tmp)
    
    mhd = None
    for root, _, files in os.walk(tmp):
        for fn in files:
            if fn.lower().endswith('.mhd'):
                mhd = os.path.join(root, fn)
                break
        if mhd:
            break
    
    if not mhd:
        return None, None, None
    
    img = sitk.ReadImage(mhd)
    volume = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    return volume, spacing_zyx, tmp

# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div style="height: 20vh;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 2.5rem 2rem;">
            <h2 style="margin-bottom: 0.5rem; font-size: 1.8rem;">LungVision AI</h2>
            <p style="color: #94a3b8;">3D CT Volume Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Radiologist ID", placeholder="Enter your ID")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

# ============================================================
# MAIN APP
# ============================================================
def show_app():
    st.markdown("""
    <div class="glass-panel" style="margin-bottom: 1.5rem;">
        <h1 style="margin:0; font-size: 1.5rem;">LungVision AI</h1>
        <p style="color: #94a3b8;">Full 3D CT Volume Analysis - Scans All Slices Automatically</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.65, 0.05)
        stride = st.select_slider("Window Stride", options=[48, 64, 80, 96], value=80)
        st.caption("Larger stride = faster but may miss small nodules")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.markdown("### Upload CT Volume")
    st.info("Upload a ZIP file containing .mhd and .raw files. The system will scan ALL slices automatically.")
    
    upzip = st.file_uploader("Select ZIP", type=["zip"], label_visibility="collapsed")
    
    if upzip is not None:
        model = load_model()
        if model is None:
            st.stop()
        
        with st.spinner("Loading CT volume..."):
            volume, spacing_zyx, temp_dir = load_volume(upzip)
        
        if volume is None:
            st.error("Invalid CT volume")
        else:
            num_slices = volume.shape[0]
            st.success(f"Volume loaded: {num_slices} slices")
            st.caption(f"Scanning all {num_slices} slices for nodules...")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analyze each slice
            slice_confidences = []
            slice_positions = []
            
            start_time = time.time()
            
            for i in range(num_slices):
                status_text.text(f"Analyzing slice {i+1}/{num_slices}")
                confidence, position = analyze_slice(
                    model, volume[i, :, :],
                    stride=stride,
                    confidence_threshold=confidence_threshold
                )
                slice_confidences.append(confidence)
                slice_positions.append(position)
                progress_bar.progress((i + 1) / num_slices)
            
            elapsed = time.time() - start_time
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Analysis complete in {elapsed:.1f} seconds")
            
            # Find slices with detections
            detected_slices = []
            for i, conf in enumerate(slice_confidences):
                if conf > confidence_threshold:
                    detected_slices.append((i, conf, slice_positions[i]))
            
            if detected_slices:
                st.markdown(f"### Found {len(detected_slices)} slice(s) with potential nodules")
                
                # Show detected slices as badges
                slice_badges = ""
                for slice_idx, conf, _ in detected_slices:
                    slice_badges += f'<span class="detection-badge">Slice {slice_idx} ({conf:.1%})</span> '
                st.markdown(f'<div style="margin-bottom: 1rem;">{slice_badges}</div>', unsafe_allow_html=True)
                
                # Let user select which slice to view
                slice_options = [f"Slice {idx} (Confidence: {conf:.1%})" for idx, conf, _ in detected_slices]
                selected = st.selectbox("Select slice to view", slice_options)
                selected_idx = detected_slices[slice_options.index(selected)][0]
                
                # Display the selected slice
                slice_img = volume[selected_idx, :, :]
                slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#0b1120')
                
                # Original
                axes[0].imshow(slice_norm, cmap='gray')
                axes[0].set_title(f"Slice {selected_idx} - Original", color='#f1f5f9')
                axes[0].axis('off')
                
                # With detection box
                axes[1].imshow(slice_norm, cmap='gray')
                pos = slice_positions[selected_idx]
                if pos:
                    rect = plt.Rectangle(
                        (pos[0], pos[1]), 128, 128,
                        fill=False, edgecolor='#06b6d4', linewidth=2.5
                    )
                    axes[1].add_patch(rect)
                    axes[1].text(
                        pos[0], pos[1] - 5,
                        f"Nodule ({slice_confidences[selected_idx]:.1%})",
                        fontsize=9, color='#06b6d4',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', alpha=0.8)
                    )
                axes[1].set_title(f"Slice {selected_idx} - Detection", color='#f1f5f9')
                axes[1].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Summary table
                st.markdown("### Detection Summary")
                summary_data = []
                for slice_idx, conf, pos in detected_slices:
                    summary_data.append({
                        "Slice": slice_idx,
                        "Confidence": f"{conf:.1%}",
                        "Position X": pos[0] if pos else "N/A",
                        "Position Y": pos[1] if pos else "N/A"
                    })
                
                import pandas as pd
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
            else:
                st.info(f"No slices with confidence > {confidence_threshold:.0%}. Try lowering the threshold.")
            
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    st.markdown("---")
    st.caption("LungVision AI - Automatically scans all slices for nodule candidates")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        show_login()
    else:
        show_app()

if __name__ == "__main__":
    main()
