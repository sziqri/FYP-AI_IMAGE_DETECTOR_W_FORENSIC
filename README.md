# AI Image Detector Using CNN and Forensic Features

##  Description
This repository contains the official codebase for the research project: **AI Image Detector using CNN Model and Forensic Features**. It provides a robust deep learning framework designed to classify images as either "Real" or "AI-Generated". 

Unlike standard CNN classifiers that rely solely on RGB spatial data, this system extracts deep digital forensic artifacts—specifically **Error Level Analysis (ELA)**, **Fast Fourier Transform (FFT)**, and **Photo Response Non-Uniformity (PRNU / Laplacian HPF)**. These features are concatenated with the RGB image to form a 6-channel input tensor, allowing models like MobileNet, ResNet50, and EfficientNetV2 to detect the micro-artifacts left behind by generative AI models.

This backend is specifically designed to integrate seamlessly with a Graphical User Interface (GUI) forensic tool.

##  Features
* **Multi-Channel Forensic Architecture:** Modifies standard CNN first-layers to accept 6-channel inputs (RGB + ELA + FFT + PRNU) dynamically.
* **Supported Models:** MobileNetV1, MobileNetV2, MobileNetV3-Small, ResNet50, and EfficientNetV2-M. Includes baseline replication modes and enhanced forensic modes.
* **Non-Destructive Preprocessing (`SmartCropPad`):** Avoids standard bilinear interpolation during training by using reflection padding and cropping, preserving delicate pixel-level generative artifacts.
* **Per-Model Evaluation:** Dynamically tracks validation accuracy across specific generative sources (e.g., ADM, BigGAN, Midjourney, Stable Diffusion) without needing hardcoded labels.
* **Forensic Metrics Tracking:** Calculates standard metrics alongside specific forensic metrics: **FAR** (False Acceptance Rate) and **FRR** (False Rejection Rate).
* **GUI-Ready Inference:** The `infer.py` script automatically configures itself from saved checkpoints and outputs strict JSON blocks for easy parsing by a frontend GUI.

##  Technologies Used
* **Deep Learning Framework:** PyTorch, Torchvision, `timm` (for MobileNetV1)
* **Metrics:** Torchmetrics
* **Image Processing:** Pillow (PIL), NumPy, SciPy (via NumPy FFT)
* **Configuration & Logging:** PyYAML, CSV
* **Terminal UI:** Rich (for enhanced console outputs)

---

##  Project Setup

### 1. Installation
Clone the repository and install the necessary dependencies via a virtual environment.

```bash
git clone https://github.com/yourusername/ai-image-detector.git
cd ai-image-detector

# Create and activate a virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate  
# On Windows:
venv\Scripts\activate

# Install required packages
pip install torch torchvision torchaudio torchmetrics timm numpy pillow pyyaml rich
```

### 2. Pre-trained Weights Setup
To utilize transfer learning while adapting the first convolutional layer for 6 channels, place your baseline weights in a `weights/` directory.

```bash
mkdir weights
```
*(Place specific weights like `resnet50.pth` or `efficientnet_v2_m-dc08266a.pth` here if not using PyTorch's default downloaders).*

### 3. Dataset Structure
The dataloaders (`dataset.py` and `dataset_extras.py`) require a specific hierarchical structure to track which generative models are tricking the network.

Create your `data/` folder and organize it as follows:

```text
data/
├── train/
│   ├── Real/
│   │   ├── pristine_01.jpg
│   └── Fake/
│       ├── adm/
│       │   └── fake_adm_01.jpg
│       ├── biggan/
│       ├── midjourney/
│       └── sd5/
├── val/
│   ├── Real/
│   └── Fake/
│       ├── adm/
│       └── midjourney/
```

---

##  Configuration (`configs/mobilenet.yaml`)
Training runs are controlled via YAML files. Here is an example of the setup for **MobileNetV1** using all forensic features:

```yaml
model: "mobilenet_v1"
img_size: 224           
num_classes: 2
pretrained: true

seed: 42
epochs: 100
batch_size: 32          
lr: 1e-3                
optimizer: "adam"      
amp: true

train_dir: "data/train"
val_dir: "data/val" 
num_workers: 4

# Forensic Features
use_extras: true
use_fft: true
use_ela: true
use_prnu: true

output_dir: "outputs/baseline_mobilenet_v1_withforensic"
early_stopping_patience: 10
```

You are free to config on your own. For my paper, i only toggle true/false in Forensic Features section

---

##  How to Run the Project

### 1. Start Training
To start training the model based on your configuration file:

```bash
python train.py --config configs/mobilenet.yaml
```

**Resuming / Fine-Tuning:**
If training stops, or you want to drop the learning rate for final adjustments, run:
```bash
python train.py --config configs/mobilenet.yaml --resume
```
The script will interactively ask if you want to load the `last.pth` or `best.pth` and offer a "Fine-Tuning Mode" (Constant Low LR).
This only can be utilized when you use Cosine LR Scheduling.

### 2. Understanding Output Artifacts
During training, the script generates the following inside your `output_dir`:
* **`checkpoints/`**: Contains `last.pth`, `best.pth`, and versioned best models (e.g., `best_acc_0.95_20260510.pth`).
* **`training_log.csv`**: Logs Loss, F1, FAR, FRR, and dynamically creates columns for the accuracy of each fake subset (e.g., `Acc_midjourney`, `Acc_adm`).
* **Rich Console Output**: Provides a real-time Per-Model Forensic Report showing exactly which generative AI models the network is struggling to detect.

---

## 🔎 Inference (For GUI Integration)
The `infer.py` script is designed to be executed by a frontend application (like PyQt or Streamlit). It reads the checkpoint, auto-detects if forensic features were used, dynamically reconstructs the 6-channel architecture, and processes the images.

**Command:**
```bash
python infer.py --ckpt outputs/baseline_mobilenet_v1_withforensic/checkpoints/best.pth --paths sample1.jpg sample2.png
```

**JSON Output Format:**
The script logs progress to the console but outputs a strict JSON block wrapped in identifier tags for easy frontend parsing:

```json
==================================================
FINAL_JSON_OUTPUT_START
{
  "sample1.jpg": {
    "status": "success",
    "predictions": [
      {
        "label": "AI-Generated",
        "confidence": 0.9912
      },
      {
        "label": "Real",
        "confidence": 0.0088
      }
    ],
    "pipeline": "CenterCrop/Pad 224x224"
  }
}
FINAL_JSON_OUTPUT_END
==================================================
```
