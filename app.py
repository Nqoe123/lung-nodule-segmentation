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
    .stButton > button:hover { background: #0284c7; }
    .glass-panel {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
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
        st.error("Model file not found. Please upload best_model.pth")
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

def sliding_window_on_slice(model, slice_img, patch_size=128, stride=64, confidence_threshold=0.7):
    """Run sliding window on a single slice"""
    h, w = slice_img.shape
    detections = []
    
    # Normalize and apply lung window
    img_norm = slice_img.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm = img_norm / 255.0
    img_norm = apply_lung_window(img_norm * 1400 - 1000) if img_norm.max() > 0.1 else img_norm
    
    # Slide window
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_norm[y:y+patch_size, x:x+patch_size]
            if patch.shape != (patch_size, patch_size):
                continue
            
            confidence = segment_patch(model, patch)
            
            if confidence > confidence_threshold:
                detections.append({
                    'x': x, 'y': y, 'slice': 0,  # slice index will be set later
                    'width': patch_size, 'height': patch_size, 'confidence': confidence
                })
    
    return detections

def load_volume(zip_file):
    """Extract and load MHD/RAW volume"""
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

def process_volume_3d(model, volume, spacing_zyx, stride=64, confidence_threshold=0.7, slice_progress=None):
    """Process entire 3D volume with sliding window on each slice"""
    num_slices = volume.shape[0]
    all_detections = []
    
    for slice_idx in range(num_slices):
        if slice_progress:
            slice_progress(slice_idx, num_slices)
        
        slice_img = volume[slice_idx, :, :]
        
        # Normalize to 0-255 range for processing
        slice_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
        slice_normalized = (slice_normalized * 255).astype(np.float32)
        
        detections = sliding_window_on_slice(model, slice_normalized, stride=stride, confidence_threshold=confidence_threshold)
        
        for det in detections:
            det['slice'] = slice_idx
            all_detections.append(det)
    
    # Group detections across slices (simple approach: same x,y within tolerance)
    grouped = group_detections_across_slices(all_detections)
    
    return grouped

def group_detections_across_slices(detections, xy_tolerance=20):
    """Group detections that appear in consecutive slices at similar positions"""
    if len(detections) == 0:
        return []
    
    # Sort by slice
    detections.sort(key=lambda x: x['slice'])
    
    groups = []
    current_group = [detections[0]]
    
    for det in detections[1:]:
        last_det = current_group[-1]
        # Check if same slice or consecutive slice
        slice_diff = det['slice'] - last_det['slice']
        # Check if position is similar
        x_diff = abs(det['x'] - last_det['x'])
        y_diff = abs(det['y'] - last_det['y'])
        
        if slice_diff <= 2 and x_diff < xy_tolerance and y_diff < xy_tolerance:
            current_group.append(det)
        else:
            # Finalize current group
            if len(current_group) >= 2:  # Need at least 2 slices to consider a nodule
                avg_confidence = np.mean([d['confidence'] for d in current_group])
                groups.append({
                    'id': len(groups) + 1,
                    'slices': [d['slice'] for d in current_group],
                    'slice_range': f"{current_group[0]['slice']}-{current_group[-1]['slice']}",
                    'num_slices': len(current_group),
                    'avg_confidence': avg_confidence,
                    'position': (current_group[0]['x'], current_group[0]['y'])
                })
            current_group = [det]
    
    # Final group
    if len(current_group) >= 2:
        groups.append({
            'id': len(groups) + 1,
            'slices': [d['slice'] for d in current_group],
            'slice_range': f"{current_group[0]['slice']}-{current_group[-1]['slice']}",
            'num_slices': len(current_group),
            'avg_confidence': np.mean([d['confidence'] for d in current_group]),
            'position': (current_group[0]['x'], current_group[0]['y'])
        })
    
    return groups

def display_slice_with_detections(volume, slice_idx, detections, ax):
    """Display a single slice with detection boxes"""
    slice_img = volume[slice_idx, :, :]
    
    # Normalize for display
    slice_norm = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min() + 1e-9)
    
    ax.imshow(slice_norm, cmap='gray')
    
    # Find detections in this slice
    for det in detections:
        if slice_idx in det['slices']:
            rect = plt.Rectangle(
                (det['position'][0], det['position'][1]),
                128, 128,
                fill=False, edgecolor='#06b6d4', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                det['position'][0], det['position'][1] - 5,
                f"N{det['id']}",
                fontsize=8, color='#06b6d4',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f172a', edgecolor='#06b6d4', alpha=0.8)
            )
    
    ax.set_title(f"Slice {slice_idx}", color='#f1f5f9', fontsize=10)
    ax.axis('off')

# ============================================================
# LOGIN PAGE
# ============================================================
def show_login():
    st.markdown('<div style="height: 20vh;"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; padding: 2.5rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;"></div>
            <h2 style="margin-bottom: 0.5rem; font-size: 1.8rem;">LungVision AI</h2>
            <p style="color: #94a3b8; margin-bottom: 2rem;">3D CT Volume Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Radiologist ID", placeholder="Enter your ID")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            
            if submitted:
                if username == "radiologist" and password == "hit500":
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")

# ============================================================
# MAIN APP
# ============================================================
def show_app():
    st.markdown("""
    <div class="glass-panel" style="margin-bottom: 1.5rem; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin:0; font-size: 1.5rem;">LungVision <span style="color:#38bdf8">AI</span></h1>
            <div style="color:#94a3b8; font-size: 0.85rem;">Radiologist: """ + st.session_state.get('username', 'Guest') + """ | 3D Volume Mode</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown('<div class="glass-panel"><h3>Detection Settings</h3></div>', unsafe_allow_html=True)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.75, 0.05)
        stride = st.select_slider("Window Stride (per slice)", options=[48, 64, 80, 96], value=80)
        st.caption("Larger stride = faster but may miss nodules")
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        st.markdown("---")
        st.caption("Model trained on LUNA16 and LIDC")
        st.caption("Validation Dice: 0.8871")
        st.caption("Processing on CPU - may be slow")
    
    st.markdown('<div class="section-label">3D CT Volume Upload</div>', unsafe_allow_html=True)
    st.info("Upload a ZIP file containing .mhd and .raw files from a CT scan")
    
    upzip = st.file_uploader("Select ZIP with .mhd and .raw files", type=["zip"], label_visibility="collapsed")
    
    if upzip is not None:
        model = load_model()
        if model is None:
            st.stop()
        
        with st.spinner("Loading CT volume..."):
            volume, spacing_zyx, temp_dir = load_volume(upzip)
        
        if volume is None:
            st.error("Invalid CT volume. Ensure ZIP contains .mhd and .raw files.")
        else:
            num_slices = volume.shape[0]
            st.success(f"Volume loaded: {num_slices} slices")
            st.caption(f"Spacing: X={spacing_zyx[2]:.3f}mm, Y={spacing_zyx[1]:.3f}mm, Z={spacing_zyx[0]:.3f}mm")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            slice_progress_text = st.empty()
            
            def update_progress(current, total):
                progress_bar.progress(current / total)
                slice_progress_text.text(f"Processing slice {current}/{total}")
            
            start_time = time.time()
            
            # Process volume
            detections = process_volume_3d(
                model, volume, spacing_zyx,
                stride=stride,
                confidence_threshold=confidence_threshold,
                slice_progress=update_progress
            )
            
            elapsed_time = time.time() - start_time
            progress_bar.empty()
            slice_progress_text.empty()
            status_text.empty()
            
            st.success(f"Analysis complete in {elapsed_time:.1f} seconds. Found {len(detections)} nodule(s).")
            
            if detections:
                # Display summary
                st.markdown("### Nodule Summary")
                for det in detections:
                    st.markdown(f"""
                    <div class="glass-panel" style="margin-bottom: 0.5rem;">
                        <b>Nodule {det['id']}</b><br>
                        Slices: {det['slice_range']} ({det['num_slices']} slices)<br>
                        Confidence: {det['avg_confidence']:.1%}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Slice viewer
                st.markdown("### Slice Viewer")
                slice_idx = st.slider("Select slice to review", 0, num_slices - 1, num_slices // 2)
                
                fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0b1120')
                display_slice_with_detections(volume, slice_idx, detections, ax)
                st.pyplot(fig)
                plt.close(fig)
                
                # Export results
                import pandas as pd
                df = pd.DataFrame([{
                    "Nodule ID": d['id'],
                    "Slice Range": d['slice_range'],
                    "Number of Slices": d['num_slices'],
                    "Confidence": f"{d['avg_confidence']:.1%}"
                } for d in detections])
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Export Results (CSV)", csv, "detection_results.csv", "text/csv")
            else:
                st.info("No nodules detected.")
            
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    st.markdown("---")
    st.caption("LungVision AI - 3D CT Volume Analysis")

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
