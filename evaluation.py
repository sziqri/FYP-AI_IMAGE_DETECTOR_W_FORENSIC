import argparse
import torch
import yaml
import csv  # <--- Added
import os   # <--- Added
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torchvision import datasets
from datetime import datetime # <--- Added

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

# Import your project modules
from src.models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from src.engine import validate
from src.transforms import build_transforms
from src.transforms_extra import build_transforms_with_extras

console = Console()

def load_best_model(ckpt_path, num_classes, device):
    console.log(f"[yellow] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    
    # Initialize Model based on Config
    if config.get("use_extras", False):
        in_chans = 3 + int(config.get("use_fft", True)) + int(config.get("use_ela", True)) + int(config.get("use_prnu", True))
        model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=False)
        transform = build_transforms_with_extras(
            img_size=config["img_size"], 
            train=False,
            use_fft=config.get("use_fft", True),
            use_ela=config.get("use_ela", True),
            use_prnu=config.get("use_prnu", True)
        )
        model_type = "Forensic (EfficientNetV2 + Extras)"
    else:
        model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=False)
        transform = build_transforms(
            img_size=config["img_size"], 
            train=False
        )
        model_type = "Standard (RGB Only)"

    # Load Weights
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model, transform, config, model_type

def main():
    parser = argparse.ArgumentParser(description="Final Thesis Evaluation Script")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pth")
    parser.add_argument("--test_dir", type=str, default="E:/fypAI/FYP/EfficientNetV2-m/data/test", help="Path to Test Data")
    # Added option to specify output filename
    parser.add_argument("--output", type=str, default="final_test_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Data
    temp_ds = datasets.ImageFolder(args.test_dir)
    classes = temp_ds.classes
    num_classes = len(classes)
    
    # 2. Load Model
    model, test_transform, config, model_type = load_best_model(args.ckpt, num_classes, device)

    # 3. Create Loader
    test_ds = datasets.ImageFolder(args.test_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=config.get("batch_size", 16), 
                             shuffle=False, num_workers=4, pin_memory=True)

    console.print(f"\n[bold blue]STARTING FINAL EVALUATION[/bold blue]")
    console.print(f"Model Type: [cyan]{model_type}[/cyan]")
    console.print(f"Test Set:   [cyan]{len(test_ds)} images[/cyan]")

    # 4. Run Evaluation
    _, metrics = validate(model, test_loader, device, num_classes)

    # 5. Print to Console (Rich Table)
    table = Table(title="Final Model Performance (Test Set)")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Score", style="magenta")

    table.add_row("Accuracy", f"{metrics['accuracy']:.2%}")
    table.add_row("Precision", f"{metrics['precision']:.2%}")
    table.add_row("Recall", f"{metrics['recall']:.2%}")
    table.add_row("F1-Score", f"{metrics['f1']:.2%}")
    table.add_row("FAR", f"{metrics['far']:.2%}")
    table.add_row("FRR", f"{metrics['frr']:.2%}")
    console.print(table)

    # --- NEW: SAVE TO CSV ---
    save_path = args.output
    file_exists = os.path.isfile(save_path)
    
    with open(save_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        # Write Header if new file
        if not file_exists:
            writer.writerow(["Timestamp", "Model_Type", "Accuracy", "Precision", "Recall", "F1", "FAR", "FRR", "Checkpoint_Path"])
            
        # Write Data
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_type,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['far']:.4f}",
            f"{metrics['frr']:.4f}",
            args.ckpt
        ])
    
    console.print(f"\n[bold green]Results saved to: {save_path}[/bold green]")

if __name__ == "__main__":
    main()