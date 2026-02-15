import torch                 
import io                     
from PIL import Image        
from torchvision import transforms

class RandomJPEGCompression:
    def __init__(self, quality_range=(60, 100), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            # Ensure image is RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            quality = torch.randint(*self.quality_range, (1,)).item()
            buffer = io.BytesIO()
            img.save(buffer, 'JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img

def build_transforms(img_size: int, train: bool = True,
                     rand_augment: bool = True, color_jitter: float = 0.0,
                     hflip_prob: float = 0.5):
    
    # --- THESIS EXPERIMENT: Resolution Equalizer (UPDATED) ---
    # OLD STRATEGY: Resize((128, 128)) -> Destroyed noise/details (Bad for forensics)
    # NEW STRATEGY: RandomCrop(128) -> Keeps noise/details (Fair fight)
    
    equalize_resolution = [
        # 1. Cut a 128x128 piece from the HD Real Image (preserves quality)
        #    If image is already 128 (BigGAN), this effectively does nothing harmful.
        transforms.RandomCrop(128, pad_if_needed=True, padding_mode='reflect'),
        
        # 2. Scale up to model size (e.g., 320 or 384)
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

        # Color Jitter
        if color_jitter > 0:
            aug.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter, 0.1))
        
        # RandAugment
        if rand_augment:
            try:
                from torchvision.transforms import RandAugment
                aug.append(RandAugment())
            except ImportError:
                print("RandAugment not found. Skipping.")
        
        # Finalize
        train_tfms = transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return train_tfms

    else:
        # Validation Transforms
        # MUST also use Equalizer to make the test fair!
        return transforms.Compose([
            *equalize_resolution, 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])