import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
import torch.nn.functional as F
import math
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time  # <--- NEW IMPORT FOR TIMER

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Forensic Deepfake Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# --- STYLES ---
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; }
    .metric-card {
        background-color: white; padding: 15px;
        border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. MODEL ARCHITECTURE ---

class EfficientNetV2MWithExtras(nn.Module):
    def __init__(self, num_classes=2, in_chans=3, pretrained=False):
        super().__init__()
        if pretrained:
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_v2_m(weights=weights)
        else:
            self.backbone = efficientnet_v2_m(weights=None)
        
        if in_chans != 3:
            self._modify_first_conv(in_chans)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def _modify_first_conv(self, in_chans):
        original_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(
            in_chans, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        self.backbone.features[0][0] = new_conv
    
    def forward(self, x):
        return self.backbone(x)

# --- 2. FEATURE EXTRACTORS ---

class ForensicFeatures:
    @staticmethod
    def extract_ela(image_array):
        try:
            if isinstance(image_array, np.ndarray):
                img = Image.fromarray(image_array.astype('uint8'))
            else:
                img = image_array
                
            img = img.convert('RGB')
            temp_filename = 'temp_ela_gui.jpg'
            img.save(temp_filename, 'JPEG', quality=90)
            temp_image = Image.open(temp_filename)
            
            ela_image = ImageChops.difference(img, temp_image)
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0: max_diff = 1
            scale = 255.0 / max_diff
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
            
            ela_array = np.array(ela_image.convert('L')).astype(np.float32) / 255.0
            
            if os.path.exists(temp_filename): os.remove(temp_filename)
            return ela_array
        except Exception as e:
            st.error(f"ELA Error: {e}")
            return None

    @staticmethod
    def extract_fft(image_array):
        try:
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
            
            fft = np.fft.fft2(gray.astype(np.float32))
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log(1 + np.abs(fft_shift))
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            return magnitude
        except Exception:
            return None

    @staticmethod
    def extract_prnu(image_array):
        try:
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2)
            else:
                gray = image_array
                
            kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=np.float32)
            try:
                prnu = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            except:
                prnu = gray 
            
            prnu = np.abs(prnu)
            prnu = (prnu - prnu.min()) / (prnu.max() - prnu.min() + 1e-8)
            return prnu
        except Exception:
            return None

# --- 3. PREPROCESSING ---

def process_image_pipeline(image, img_size=224, crop_pct=0.95):
    # 1. Calculate the resize dimension exactly like transforms.py
    # Logic: 320 / 0.95 = 336.8 -> 336
    scale_size = int(math.floor(img_size / crop_pct))

    # 2. Define the Geometric Transform
    # We use Resize (maintaining aspect ratio) then CenterCrop.
    # We use BILINEAR because default.yaml specifies "interpolation: bilinear"
    geometric_transform = transforms.Compose([
        transforms.Resize(scale_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(img_size),
    ])
    
    # Apply the transform to get the exact 320x320 patch the model expects
    img_transformed = geometric_transform(image)
    img_np = np.array(img_transformed) 
    
    # 3. Extract Features FROM THE TRANSFORMED IMAGE
    # (Crucial: ELA/FFT must be calculated on the final 320x320 pixels)
    fft = ForensicFeatures.extract_fft(img_np)
    ela = ForensicFeatures.extract_ela(img_transformed) 
    prnu = ForensicFeatures.extract_prnu(img_np)

    def to_tensor(feat):
        if feat is None: return torch.zeros(1, img_size, img_size)
        t = torch.from_numpy(feat).float().unsqueeze(0)
        # Safety: Ensure strict shape
        if t.shape[1] != img_size or t.shape[2] != img_size:
             t = F.interpolate(t.unsqueeze(0), size=(img_size, img_size), mode='bilinear').squeeze(0)
        return t

    tensor_fft = to_tensor(fft)
    tensor_ela = to_tensor(ela)
    tensor_prnu = to_tensor(prnu)
    
    # 5. Standard RGB Normalization (matches default.yaml mean/std)
    # mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    rgb_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_rgb = rgb_transform(img_transformed)
    
    # 6. Concatenate
    final_tensor = torch.cat([tensor_rgb, tensor_fft, tensor_ela, tensor_prnu], dim=0)
    
    return final_tensor, img_transformed, (fft, ela, prnu)

# --- 4. APP LOGIC ---

@st.cache_resource
def load_model(model_path='best.pth', num_classes=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=6, pretrained=False)
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'classes' in checkpoint:
                print(f"Dataset Classes from Checkpoint: {checkpoint['classes']}")

            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k
                if name.startswith('module.'): 
                    name = name.replace('module.', '')
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.to(device)
            model.eval()
            return model, device, True
        else:
            return None, device, False
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, device, False

def main():
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
        st.session_state.features = None
        st.session_state.ai_score = 0
        st.session_state.real_score = 0
        st.session_state.exec_time = 0.0

    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = None

    st.sidebar.title("🕵️‍♂️ Settings")

    # --- MODEL SELECTION LOGIC ---
    model_folder = "GUI_MODEL"

    # 1. Check if folder exists, if not create it
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        st.sidebar.warning(f"⚠️ Folder '{model_folder}' was missing. I created it! Please move your .pth files inside.")
        model_files = []
    else:
        # 2. Scan the folder for .pth files
        model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]
        model_files.sort()

    # 3. Handle the dropdown list
    if not model_files:
        st.sidebar.error(f"No models found in '{model_folder}'")
        selected_file = None
    else:
        # Try to auto-select "best.pth" if it exists
        try:
            default_index = model_files.index("best.pth")
        except ValueError:
            default_index = 0

        selected_file = st.sidebar.selectbox(
            "Select Model Checkpoint", 
            model_files, 
            index=default_index
        )

    # 4. Load the selected model
    if selected_file:
        full_model_path = os.path.join(model_folder, selected_file)
        model, device, loaded = load_model(full_model_path)
    else:
        model, device, loaded = None, "cpu", False
    
    # --- STATUS INDICATOR ---
    if loaded:
        st.sidebar.success(f"✅ Model Loaded: {selected_file}")
        st.sidebar.caption(f"Device: {device}")
    else:
        st.sidebar.warning("⚠️ Model not found. Demo Mode.")

    st.title("AI Image Forensic Tool")
    st.markdown("Inputs: **RGB (3) + FFT (1) + ELA (1) + PRNU (1)** -> EfficientNetV2-M")

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Input", use_column_width=True)
            
            if st.button("Run Forensic Analysis"):
                start_time = time.time()
                with st.spinner("Extracting forensic features & running inference..."):
                    input_tensor, processed_img, features = process_image_pipeline(image, img_size=224)
                    st.session_state.features = features
                    st.session_state.debug_tensor = input_tensor  
                    
                    if loaded:
                        batch = input_tensor.unsqueeze(0).to(device)
                        with torch.no_grad():
                            logits = model(batch)
                            probs = torch.softmax(logits, dim=1)

                            st.session_state.debug_info = {
                                "logits": logits.cpu().numpy().tolist()[0],
                                "probs": probs.cpu().numpy().tolist()[0]
                            }
                            
                            # Assuming Class 0 = AI ("ai"), Class 1 = Real ("real")
                            ai_score = probs[0][0].item() * 100
                            real_score = probs[0][1].item() * 100
                            
                            st.session_state.ai_score = ai_score
                            st.session_state.real_score = real_score
                    else:
                        import random
                        st.session_state.ai_score = random.uniform(0, 100)
                        st.session_state.real_score = 100 - st.session_state.ai_score
                    
                    st.session_state.analyzed = True
                    end_time = time.time()
                    st.session_state.exec_time = end_time - start_time

            if st.session_state.analyzed:
                st.write("---")
                ai_score = st.session_state.ai_score
                
                if ai_score > 50:
                    st.error(f"## 🚨 DETECTED: AI-GENERATED ({ai_score:.2f}%)")
                else:
                    st.success(f"## ✅ DETECTED: REAL ({100 - ai_score:.2f}%)")
                
                st.progress(int(ai_score))
                st.caption(f"AI Probability: {ai_score:.1f}% | Real Probability: {100 - ai_score:.1f}%")
                st.caption(f"⏱️ Inference Time: {st.session_state.exec_time:.4f} seconds")

        with col2:
            st.subheader("Forensic Channels")
            tab1, tab2, tab3 = st.tabs(["ELA", "FFT", "PRNU"])
            
            if st.session_state.analyzed and st.session_state.features:
                fft_map, ela_map, prnu_map = st.session_state.features
                
                with tab1:
                    st.image(ela_map, caption="ELA Pattern", clamp=True, use_column_width=True)
                with tab2:
                    st.image(fft_map, caption="Frequency Spectrum", clamp=True, use_column_width=True)
                with tab3:
                    st.image(prnu_map, caption="PRNU / Noise Residual", clamp=True, use_column_width=True)
            else:
                st.info("Click 'Run Forensic Analysis' to visualize features.")

            # --- ADVANCED TECHNICAL ANALYSIS SECTION ---
        if st.session_state.analyzed and st.session_state.debug_info:
            st.write("---")
            with st.expander("🔬 Advanced Technical Analysis (Debug View)", expanded=False):
            
                # SECTION 1: DATA PIPELINE
                st.markdown("### 1. 🧠 Model Input (Tensor Pipeline)")
                t_shape = st.session_state.debug_tensor.shape
                st.code(f"Tensor Shape: {t_shape} \nFormat: [Channels, Height, Width] \nSize: {t_shape[0]}x{t_shape[1]}x{t_shape[2]} = {t_shape[0]*t_shape[1]*t_shape[2]:,} floating point values")
            
                # SECTION 2: CHANNEL STATISTICS
                st.markdown("### 2. 📊 6-Channel Value Distribution")
                st.caption("This proves the forensic channels contain valid data (not just zeros).")
            
                tensor_data = st.session_state.debug_tensor.cpu().numpy()
                channels = ["Red", "Green", "Blue", "FFT (Freq)", "ELA (Error)", "PRNU (Noise)"]
            
                cols = st.columns(6)
                for i, name in enumerate(channels):
                    data = tensor_data[i]
                    with cols[i]:
                        st.metric(label=name, value=f"{data.mean():.2f}", delta=f"Max: {data.max():.1f}")
                        st.markdown(f"<div style='font-size:10px; color:gray'>Min: {data.min():.2f}<br>Std: {data.std():.2f}</div>", unsafe_allow_html=True)

                # SECTION 3: THE DECISION
                st.markdown("### 3. ⚖️ The Decision (Logits → Softmax)")
                d_info = st.session_state.debug_info
            
                c1, c2 = st.columns(2)
                with c1:
                    st.info("**Step A: Raw Logits**\n(Model's raw confidence score)")
                    st.write(d_info['logits'])
                with c2:
                    st.success("**Step B: Softmax Probability**\n(Converted to %)")
                    st.write(d_info['probs'])

                # SECTION 4: VISUAL HISTOGRAM & INTERPRETATION
                st.markdown("### 4. 📈 Forensic Feature Histogram")
                st.caption("Distribution of pixel intensities in the forensic channels.")
            
                st.info("""
                **How to interpret this graph:**
                * **FFT (Purple):** Spikes often indicate grid-like artifacts common in GANs/Diffusion models.
                * **ELA (Orange):** Flat distributions suggest consistent compression (Real). Varied distributions suggest tampering.
                * **PRNU (Green):** Wider variance typically indicates natural sensor noise (Real camera).
                """)
            
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(tensor_data[3].flatten(), bins=50, alpha=0.5, label='FFT', color='purple')
                ax.hist(tensor_data[4].flatten(), bins=50, alpha=0.5, label='ELA', color='orange')
                ax.hist(tensor_data[5].flatten(), bins=50, alpha=0.5, label='PRNU', color='green')
                ax.set_title("Forensic Channel Pixel Intensity Distribution")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # SECTION 5: REPORT GENERATION
                st.markdown("### 5. 📄 Report Generation")
                report_text = f"""
                AI FORENSIC REPORT
                ==================
                Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
                Filename: {uploaded_file.name if uploaded_file else 'Unknown'}
                Inference Time: {st.session_state.exec_time:.4f}s
            
                RESULTS
                -------
                Classification: {'AI-GENERATED' if st.session_state.ai_score > 50 else 'REAL'}
                Confidence: {max(st.session_state.ai_score, st.session_state.real_score):.2f}%
            
                TECHNICAL DETAILS
                -----------------
                Logits: {d_info['logits']}
                Probabilities: {d_info['probs']}
                """
            
                st.download_button(
                    label="📥 Download Forensic Report (TXT)",
                    data=report_text,
                    file_name="forensic_report.txt",
                    mime="text/plain"
                )
                # ---------------------------------

if __name__ == "__main__":
    main()