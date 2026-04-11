# app.py - Lung Nodule Segmentation Tool
# HIT500 Capstone Project - Nqobile Maware
# Fixed version - handles any image type

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os
import gdown
from skimage.transform import resize

# ========== GOOGLE DRIVE SETUP ==========
# REPLACE THIS WITH YOUR ACTUAL FILE ID
GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"  # <-- PUT YOUR FILE ID HERE

MODEL_FILENAME = "best_unet_model.pth"

def download_model_from_drive():
    """Download model from Google Drive if not already downloaded"""
    if not os.path.exists(MODEL_FILENAME):
        with st.spinner("Downloading model from Google Drive... (first time only)"):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_FILENAME, quiet=False)
            st.success("Model downloaded successfully!")
    return MODEL_FILENAME

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Lung Nodule Segmentation",
    page_icon="🫁",
    layout="wide"
)

# ========== U-NET MODEL ARCHITECTURE ==========

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))

# ========== LOAD MODEL FROM GOOGLE DRIVE ==========

@st.cache_resource
def load_model():
    # Download model from Google Drive
    model_path = download_model_from_drive()
    
    # Load model
    model = UNet()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return model, False

# ========== FUNCTION TO CONVERT ANY IMAGE TO GRAYSCALE ==========
def convert_to_grayscale(image):
    """Convert any image to grayscale numpy array"""
    if image.mode == 'RGBA':
        # Convert RGBA to RGB first, then to grayscale
        image = image.convert('RGB')
    if image.mode != 'L':
        # Convert to grayscale
        image = image.convert('L')
    return np.array(image)

# ========== SEGMENTATION FUNCTIONS ==========

def segment_nodule(model, image_array):
    """Takes a CT image array, returns segmentation mask"""
    # Ensure 2D
    if len(image_array.shape) == 3:
        image_array = image_array[:, :, 0]
    
    # Resize to 256x256
    img_resized = resize(image_array, (256, 256))
    
    # Normalize to [0, 1]
    img_norm = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)
    
    # Convert to tensor
    input_tensor = torch.FloatTensor(img_norm).unsqueeze(0).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        mask = output.squeeze().numpy()
        mask = (mask > 0.5).astype(np.float32)
    
    # Resize mask back to original size
    return resize(mask, image_array.shape[:2])

def calculate_volume(mask, pixel_spacing_mm=0.7, slice_thickness_mm=1.25):
    """Estimate nodule volume in mm³"""
    pixel_area_mm2 = pixel_spacing_mm ** 2
    area_pixels = np.sum(mask)
    return area_pixels * pixel_area_mm2 * slice_thickness_mm

# ========== MAIN UI ==========

st.title("🫁 Lung Nodule Segmentation Tool")
st.markdown("### CNN-Based Automated Segmentation for CT Scans")
st.markdown("**HIT500 Capstone | Biomedical Engineering | Nqobile Maware**")
st.markdown("---")

# Login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Radiologist Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login", use_container_width=True):
            if username == "radiologist" and password == "hit500":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid credentials. Use: radiologist / hit500")
else:
    st.sidebar.success("✅ Logged in as: Radiologist")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    # Load model
    model, model_loaded = load_model()
    
    if model_loaded:
        st.sidebar.success("✅ Model loaded from Google Drive")
    else:
        st.sidebar.error("❌ Model failed to load")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📤 Upload CT Scan")
        uploaded = st.file_uploader("Choose CT image", type=["png", "jpg", "jpeg", "dcm"])
        
        if uploaded:
            # Open image and convert to grayscale
            image = Image.open(uploaded)
            original_array = convert_to_grayscale(image)
            
            st.subheader("📷 Original CT Scan")
            st.image(original_array, caption="Original CT Image", use_container_width=True)
            
            if st.button("🔍 Segment Nodule", type="primary"):
                with st.spinner("Segmenting..."):
                    mask = segment_nodule(model, original_array)
                    volume = calculate_volume(mask)
                    st.session_state['mask'] = mask
                    st.session_state['volume'] = volume
                    st.session_state['original'] = original_array
                st.success("Segmentation complete!")
    
    with col_right:
        st.subheader("📊 Results")
        
        if 'mask' in st.session_state:
            tab1, tab2, tab3 = st.tabs(["Mask", "Overlay", "Metrics"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(st.session_state['mask'], cmap='gray')
                ax.set_title("White = Nodule, Black = Background")
                ax.axis('off')
                st.pyplot(fig)
                
                buf = io.BytesIO()
                plt.figure(figsize=(5,5))
                plt.imshow(st.session_state['mask'], cmap='gray')
                plt.axis('off')
                plt.savefig(buf, format='png')
                buf.seek(0)
                st.download_button("Download Mask", buf, "nodule_mask.png")
            
            with tab2:
                original = st.session_state['original']
                mask = st.session_state['mask']
                if original.shape != mask.shape:
                    mask = resize(mask, original.shape)
                
                # Normalize original for display
                original_norm = (original - original.min()) / (original.max() - original.min() + 1e-8)
                
                # Create RGB overlay
                overlay = np.stack([original_norm] * 3, axis=-1)
                overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
                overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])
                overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])
                
                st.image(overlay, caption="Nodule Highlighted in RED", use_container_width=True)
            
            with tab3:
                area_pct = (np.sum(st.session_state['mask'] > 0.5) / st.session_state['mask'].size) * 100
                cola, colb, colc = st.columns(3)
                cola.metric("Volume", f"{st.session_state['volume']:.2f} mm³")
                colb.metric("Nodule Area", f"{area_pct:.2f}%")
                colc.metric("Model Dice", "0.9628")
                
                if area_pct < 1:
                    st.info("📌 Small nodule - Regular monitoring recommended")
                elif area_pct < 5:
                    st.warning("⚠️ Medium nodule - Further evaluation suggested")
                else:
                    st.error("🚨 Large nodule - Urgent consultation recommended")
    
    st.markdown("---")
    st.caption("© 2026 HIT500 Capstone | Model loaded from Google Drive")
