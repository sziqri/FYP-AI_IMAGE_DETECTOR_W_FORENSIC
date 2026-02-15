import torch
from torchvision import transforms
import torch.nn.functional as F
from features import extract_ela_feature, extract_fft_feature, extract_prnu_feature
from PIL import Image
import numpy as np
import io

# --- 1. NEW AUGMENTATION CLASS ---
class RandomJPEGCompression:
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            try:
                # Ensure image is RGB before saving as JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                quality = torch.randint(*self.quality_range, (1,)).item()
                buffer = io.BytesIO()
                img.save(buffer, 'JPEG', quality=quality)
                buffer.seek(0)
                return Image.open(buffer)
            except Exception as e:
                print(f"JPEG Compression failed: {e}")
                return img
        return img

# --- 2. FORENSIC TRANSFORM CLASS ---
class ForensicTransform:
    """Transform that handles forensic feature extraction without local functions."""
    def __init__(self, img_size: int, train: bool = True, 
                 use_fft: bool = True, use_ela: bool = True, use_prnu: bool = True):
        self.img_size = img_size
        self.train = train
        self.use_fft = use_fft
        self.use_ela = use_ela
        self.use_prnu = use_prnu
        
        # Final transform for RGB
        self.final_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _process_forensic_feature(self, feature_array):
        """Convert forensic feature array to properly shaped tensor."""
        if feature_array is None:
            return None
            
        # Ensure we have a 2D array (H, W)
        if len(feature_array.shape) == 3:
            feature_array = feature_array.mean(axis=2)  # Convert to grayscale if RGB
        
        # Convert to tensor and add channel dimension
        feature_tensor = torch.from_numpy(feature_array).float().unsqueeze(0)  # Shape: (1, H, W)
        
        # Resize to target size if needed
        if feature_tensor.shape[1] != self.img_size or feature_tensor.shape[2] != self.img_size:
            feature_tensor = F.interpolate(
                feature_tensor.unsqueeze(0),  # Add batch dimension: (1, 1, H, W)
                size=(self.img_size, self.img_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # Remove batch dimension: (1, H, W)
        
        return feature_tensor

    def __call__(self, img):
        img_transformed = img
        
        # Compute extra channels using your forensic functions
        extra_channels = []
        
        try:
            if self.use_fft:
                fft_feature = extract_fft_feature(img_transformed)
                fft_tensor = self._process_forensic_feature(fft_feature)
                if fft_tensor is not None:
                    extra_channels.append(fft_tensor)
            
            if self.use_ela:
                ela_feature = extract_ela_feature(img_transformed)
                ela_tensor = self._process_forensic_feature(ela_feature)
                if ela_tensor is not None:
                    extra_channels.append(ela_tensor)
            
            if self.use_prnu:
                prnu_feature = extract_prnu_feature(img_transformed)
                prnu_tensor = self._process_forensic_feature(prnu_feature)
                if prnu_tensor is not None:
                    extra_channels.append(prnu_tensor)
        
        except Exception as e:
            print(f"Warning: Forensic feature extraction failed: {e}")
            # Continue without forensic features
        
        # Apply final transform to RGB
        img_tensor = self.final_transform(img_transformed)  # Shape: (3, H, W)
        
        # Combine with extra channels
        if extra_channels:
            # Ensure all tensors have the same shape: (C, H, W)
            for i, tensor in enumerate(extra_channels):
                if len(tensor.shape) == 2:  # (H, W)
                    extra_channels[i] = tensor.unsqueeze(0)  # -> (1, H, W)
                elif len(tensor.shape) == 4:  # (B, C, H, W) - remove batch dim
                    extra_channels[i] = tensor.squeeze(0)  # -> (C, H, W)
            
            # Concatenate all extra channels along channel dimension
            extra_tensor = torch.cat(extra_channels, dim=0)  # Shape: (C_extra, H, W)
            
            # Concatenate with RGB
            combined_tensor = torch.cat([img_tensor, extra_tensor], dim=0)  # Shape: (3 + C_extra, H, W)
        else:
            combined_tensor = img_tensor
        
        return combined_tensor

# --- 3. BUILDER FUNCTION ---
def build_transforms_with_extras(img_size: int, train: bool = True,
                                 use_fft: bool = True, use_ela: bool = True, use_prnu: bool = True,
                                 rand_augment: bool = True, color_jitter: float = 0.0,
                                 hflip_prob: float = 0.5):
    
    # --- THESIS EXPERIMENT: Resolution Equalizer ---
    equalize_resolution = [
        # RandomCrop will cut a piece from HD images. 
        # padding_mode='reflect' helps if an image is slightly smaller than 128
        transforms.RandomCrop(128, pad_if_needed=True, padding_mode='reflect'),
        
        # Now both are 128x128. Scale both up to model size.
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomErasing(p=0.2)
    ]
    # -----------------------------------------------

    if train:
        aug = list(equalize_resolution) # Resolves 128x128 vs High-Res
        
        aug.extend([
            # 1. Spatial/Geometry Augmentation
            transforms.RandomHorizontalFlip(p=hflip_prob),
            transforms.RandomCrop(img_size, padding=4),

            # 2. Forensic Attack 1: Smoothing (Gaussian Blur)
            # Use standard transforms.GaussianBlur for compatibility
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            
            # 3. Forensic Attack 2: Quantization (JPEG)
            # Always put JPEG near the end to simulate the saving process
            RandomJPEGCompression(quality_range=(60, 100), p=0.5),
            
            # 4. Color Jitter (Optional, after forensic attacks)
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1) if color_jitter > 0 else transforms.Lambda(lambda x: x)
        ])

        if color_jitter > 0:
            aug.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter, 0.1))
        
        if rand_augment:
            try:
                from torchvision.transforms import RandAugment
                aug.append(RandAugment())
            except ImportError:
                pass
    else:
        # Validation
        aug = list(equalize_resolution)

    # The ForensicTransform wrapper
    forensic_transform = ForensicTransform(
        img_size=img_size,
        train=train,
        use_fft=use_fft,
        use_ela=use_ela, 
        use_prnu=use_prnu
    )

    # Combine: Augmentations/Degradation FIRST -> Then Extract Features
    return transforms.Compose([
        transforms.Compose(aug),  # Degrades the image
        forensic_transform        # Extracts features from the degraded image
    ])