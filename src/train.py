import csv
from datetime import datetime
import os
from pathlib import Path
import yaml
import argparse
import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from rich.console import Console

from dataset import build_dataloaders
from dataset_extras import build_dataloaders_with_extras
from models.efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras
from engine import TrainState, train_one_epoch, validate
from utils import seed_everything, ensure_dir, save_checkpoint, load_checkpoint, get_timestamp

# Define accumulation steps (4 * 8 = 32 effective batch size)
ACCUMULATION_STEPS = 4 

console = Console()

def create_model(cfg, num_classes, device):
    """Create model with EfficientNetV2-M using local weights."""
    model_type = cfg.get("model", "efficientnet_v2_m")
    use_extras = cfg.get("use_extras", False)
    weights_path = Path("weights/efficientnet_v2_m-dc08266a.pth")
    
    def adapt_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if "classifier" in k or "head" in k: continue
            if k.startswith("features."): new_state_dict[f"model.{k}"] = v
            else: new_state_dict[k] = v
        return new_state_dict

    if use_extras:
        in_chans = 3 + int(cfg.get("use_fft", True)) + int(cfg.get("use_ela", True)) + int(cfg.get("use_prnu", True))
        if weights_path.exists() and cfg.get("pretrained", True):
            model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=False)
            state_dict = torch.load(weights_path, map_location='cpu')
            state_dict = adapt_state_dict(state_dict)
            
            if in_chans != 3:
                target_key = 'model.features.0.0.weight'
                if target_key in state_dict:
                    first_conv_weight = state_dict[target_key]
                    new_first_conv_weight = torch.randn(
                        first_conv_weight.shape[0], in_chans, first_conv_weight.shape[2], first_conv_weight.shape[2],
                        device=first_conv_weight.device, dtype=first_conv_weight.dtype
                    )
                    new_first_conv_weight[:, :3] = first_conv_weight
                    rgb_mean = first_conv_weight.mean(dim=1, keepdim=True)
                    for i in range(3, in_chans): new_first_conv_weight[:, i:i+1] = rgb_mean
                    state_dict[target_key] = new_first_conv_weight
            
            model.load_state_dict(state_dict, strict=False)
            console.log(f"[green] Loaded pretrained weights (Adapted keys)")
        else:
            model = EfficientNetV2MWithExtras(num_classes=num_classes, in_chans=in_chans, pretrained=cfg.get("pretrained", True))
    else:
        if weights_path.exists() and cfg.get("pretrained", True):
            model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=False)
            state_dict = torch.load(weights_path, map_location='cpu')
            state_dict = adapt_state_dict(state_dict)
            model.load_state_dict(state_dict, strict=False)
            console.log(f"[green] Loaded pretrained weights (Adapted keys)")
        else:
            model = EfficientNetV2MStandard(num_classes=num_classes, pretrained=cfg.get("pretrained", True))
            
    return model.to(device)

def main():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-M")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f: cfg = yaml.safe_load(f)

    console.log(f"[bold blue] Starting AI Image Detection Training")
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.get("use_extras", False):
        train_loader, val_loader, classes = build_dataloaders_with_extras(
            cfg["train_dir"], cfg["val_dir"], cfg["img_size"],
            cfg["batch_size"], cfg["num_workers"],
            cfg.get("rand_augment", True), cfg.get("color_jitter", 0.0), cfg.get("hflip_prob", 0.5),
            cfg.get("use_fft", True), cfg.get("use_ela", True), cfg.get("use_prnu", True)
        )
    else:
        train_loader, val_loader, classes = build_dataloaders(
            cfg["train_dir"], cfg["val_dir"], cfg["img_size"],
            cfg["batch_size"], cfg["num_workers"],
            cfg.get("rand_augment", True), cfg.get("color_jitter", 0.0), cfg.get("hflip_prob", 0.5)
        )

    # --- NEW: PROFESSIONAL CONFIGURATION DASHBOARD ---
    console.rule("[bold green] TRAINING CONFIGURATION")
    console.print(f"[cyan]Model:[/][bold magenta] {cfg.get('model', 'efficientnet_v2_m')}")
    console.print(f"[cyan]Device:[/] [bold green]{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}[/]")
    console.print(f"[cyan]Image Size:[/] {cfg.get('img_size')}")
    console.print(f"[cyan]Batch Size:[/] {cfg.get('batch_size')} (Accumulation: {ACCUMULATION_STEPS})")
    console.print(f"[cyan]Classes:[/] {len(classes)} {classes}")  # Shows count and names
    console.print(f"[cyan]Total Epochs:[/] {cfg.get('epochs', 100)}")
    console.print(f"[blue]Optimizer:[/] {cfg.get('optimizer', 'adamw').upper()}")
    console.print(f"[yellow]Learning Rate:[/] {cfg.get('lr', '5e-5')}")
    console.print(f"[yellow]Weight Decay:[/] {cfg.get('weight_decay', '1e-4')}")
    console.print(f"[yellow]Forensic Features:[/] FFT: {cfg.get('use_fft')}, ELA: {cfg.get('use_ela')}, PRNU: {cfg.get('use_prnu')}")
    console.print(f"[blue]Output Dir:[/] {cfg.get('output_dir')}")
    console.rule()
    
    model = create_model(cfg, len(classes), device)
    ckpt_dir = ensure_dir(Path(cfg.get("output_dir", "outputs_v2")) / "checkpoints")
    
    # CSV Logging
    log_file = Path(cfg.get("output_dir", "outputs_v2")) / "training_log.csv"
    file_exists = log_file.exists()
    log_f = open(log_file, "a", newline="")
    csv_writer = csv.writer(log_f)
    if not file_exists:
        csv_writer.writerow(["Timestamp", "Epoch", "Train_Loss", "Train_Acc", "Train_F1", "Val_Loss", "Val_Acc", "Val_F1", "Precision", "Recall", "FAR", "FRR", "LR"])

    if args.resume: train_state = load_checkpoint(model, ckpt_dir, device)
    else: train_state = None

    total_epochs = cfg.get("epochs", 100)
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, capturable=True)
    if train_state and train_state.get('optimizer_state'): optimizer.load_state_dict(train_state['optimizer_state'])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
    if train_state and train_state.get('scheduler_state'): scheduler.load_state_dict(train_state['scheduler_state'])

    scaler = torch.amp.GradScaler('cuda', enabled=cfg.get("amp", True) and device.type == "cuda")
    
    start_epoch = 0
    is_fine_tuning = False

    if train_state:
        state = TrainState(epoch=train_state['epoch'], best_val_acc=train_state['best_val_acc'])
        start_epoch = state.epoch
        console.rule("[bold yellow] FINE-TUNING MODE SELECTION")
        if input("Switch to FINE-TUNING (Constant Low LR)? (y/n): ").lower().strip() == 'y':
            is_fine_tuning = True
            for param_group in optimizer.param_groups: param_group['lr'] = float(cfg.get("fine_tune_lr", 1e-5))
            console.print("[bold purple] FINE-TUNING ENABLED!")
    else:
        state = TrainState(epoch=0, best_val_acc=0.0)

    
    # --- MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, total_epochs):
        state.epoch = epoch + 1
        console.log(f"\n[bold cyan]══════ Epoch {state.epoch}/{total_epochs} ══════")

        # 1. TRAIN (With updated Gradient Accumulation logic)
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, 
            num_classes=len(classes),
            accumulation_steps=ACCUMULATION_STEPS  # <-- Correctly passed here
        )
        console.log(f"  Train → Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f}")

        # 2. VALIDATE
        val_loss, val_metrics = validate(model, val_loader, device, num_classes=len(classes))
        console.log(f" Val   → Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        console.log(f"  Forensic → Recall: {val_metrics['recall']:.4f} | FAR: {val_metrics['far']:.4f}")

        current_lr = optimizer.param_groups[0]['lr']
        if not is_fine_tuning: scheduler.step()

        # 3. LOGGING & CHECKPOINT
        csv_writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), state.epoch,
            f"{train_loss:.4f}", f"{train_metrics['accuracy']:.4f}", f"{train_metrics['f1']:.4f}",
            f"{val_loss:.4f}", f"{val_metrics['accuracy']:.4f}", f"{val_metrics['f1']:.4f}",
            f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{val_metrics['far']:.4f}", f"{val_metrics['frr']:.4f}",
            f"{current_lr:.2e}"
        ])
        log_f.flush()

        is_best = val_metrics['accuracy'] > state.best_val_acc
        if is_best:
            state.best_val_acc = val_metrics['accuracy']
            console.log(f"[bold green] NEW BEST! Accuracy: {val_metrics['accuracy']:.4f}")

        save_checkpoint({
            "epoch": state.epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_acc": state.best_val_acc,
            "classes": classes,
            "config": cfg,
            "timestamp": get_timestamp()
        }, is_best=is_best, ckpt_dir=str(ckpt_dir))

    console.log(f"\n[bold green] Training complete! Best Acc: {state.best_val_acc:.4f}")
    log_f.close()

if __name__ == "__main__":
    main()