import argparse
from pathlib import Path
import json
import torch
from PIL import Image
from torchvision import transforms

from models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from transforms import build_transforms
from transforms_extra import build_transforms_with_extras

def load_checkpoint(ckpt_path: str, num_classes: int, in_chans: int = 3):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Auto-detect model type from config
    config = ckpt.get("config", {})
    model_type = config.get("model", "efficientnet_v2_m")
    
    if config.get("use_extras", False):
        model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=False)
    else:
        model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=False)
    
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    classes = ckpt.get("classes", None)
    
    # Debug info
    print(f"✅ Loaded {model_type} with {num_classes} classes, {in_chans} input channels")
    if classes:
        print(f"✅ Classes: {classes}")
    
    return model, classes

def main():
    parser = argparse.ArgumentParser(description="AI Image Detection Inference with EfficientNetV2-M")
    parser.add_argument("--paths", type=str, nargs="+", required=True, help="Image paths to analyze")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=384, help="Image size (384 for EfficientNetV2-M)")  # CHANGED DEFAULT
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")  # IMPROVED DEFAULT
    parser.add_argument("--topk", type=int, default=2, help="Number of top predictions to show")
    parser.add_argument("--use_extras", action="store_true", help="Use FFT/ELA/PRNU channels")
    parser.add_argument("--use_fft", action="store_true", help="Use FFT channel")
    parser.add_argument("--use_ela", action="store_true", help="Use ELA channel")
    parser.add_argument("--use_prnu", action="store_true", help="Use PRNU channel")
    
    args = parser.parse_args()

    device = torch.device(args.device)
    extra_ch = (1 if args.use_fft else 0) + (1 if args.use_ela else 0) + (1 if args.use_prnu else 0)
    in_chans = 3 + extra_ch if args.use_extras else 3

    # Auto-detect num_classes from checkpoint if possible
    checkpoint_info = torch.load(args.ckpt, map_location="cpu")
    num_classes = len(checkpoint_info.get("classes", ["AI", "Real"]))  # Auto-detect
    
    # Load the model from the latest checkpoint
    model, classes = load_checkpoint(args.ckpt, num_classes=num_classes, in_chans=in_chans)
    model.to(device)

    # Select the appropriate transform
    if args.use_extras:
        tfm = build_transforms_with_extras(args.img_size, train=False,
                                           use_fft=args.use_fft, use_ela=args.use_ela, use_prnu=args.use_prnu)
    else:
        tfm = build_transforms(args.img_size, train=False)

    # Load and process images
    paths = [Path(p) for p in args.paths]
    results = {}
    
    print(f"🔍 Analyzing {len(paths)} image(s) with {model.__class__.__name__}...")
    
    with torch.no_grad():
        for img_path in paths:
            try:
                img = Image.open(img_path).convert("RGB")
                x = tfm(img).unsqueeze(0).to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                conf, idx = torch.topk(probs, k=args.topk, dim=-1)
                conf, idx = conf.squeeze(0).cpu().tolist(), idx.squeeze(0).cpu().tolist()
                
                # Get class labels
                if classes:
                    labels = [classes[i] for i in idx]
                else:
                    labels = ["AI-Generated" if i == 1 else "Real" for i in idx]  # Fallback
                
                results[str(img_path)] = [{"label": l, "confidence": float(c)} for l, c in zip(labels, conf)]
                
                # Print immediate results for user feedback
                top_label = labels[0]
                top_conf = conf[0]
                print(f"📊 {img_path.name}: {top_label} ({top_conf:.2%})")
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
                results[str(img_path)] = {"error": str(e)}

    # Output final results in JSON format
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()