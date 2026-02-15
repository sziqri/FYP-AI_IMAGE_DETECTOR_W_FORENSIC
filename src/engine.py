# src/engine.py
from dataclasses import dataclass
from typing import Optional, Dict
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score, 
    MulticlassPrecision, MulticlassRecall,
    MulticlassConfusionMatrix
)
from rich.console import Console

console = Console()

@dataclass
class TrainState:
    epoch: int = 0
    best_val_acc: float = 0.0
    best_f1: float = 0.0

class ForensicMetrics:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        
        self.accuracy = MulticlassAccuracy(num_classes=num_classes).to(device)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
        self.precision = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
        self.recall = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        
    def update(self, logits, targets):
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, targets)
        self.f1.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.confusion_matrix.update(preds, targets)
    
    def compute(self) -> Dict[str, float]:
        accuracy = self.accuracy.compute().item()
        f1 = self.f1.compute().item()
        precision = self.precision.compute().item()
        recall = self.recall.compute().item()
        
        cm = self.confusion_matrix.compute()
        far, frr = self._compute_far_frr(cm)
        
        return {
            'accuracy': accuracy, 'f1': f1,
            'precision': precision, 'recall': recall,
            'far': far, 'frr': frr
        }
    
    def _compute_far_frr(self, confusion_matrix):
        if self.num_classes == 2:
            tn = confusion_matrix[0, 0]
            fp = confusion_matrix[0, 1]
            fn = confusion_matrix[1, 0]
            tp = confusion_matrix[1, 1]
            
            far = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            frr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far, frr = 0.0, 0.0
        return far.item(), frr.item()
    
    def reset(self):
        self.accuracy.reset()
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.confusion_matrix.reset()

# FIXED FUNCTION BELOW
def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, accumulation_steps=1):
    model.train()
    metrics = ForensicMetrics(num_classes, device)
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            
            # Normalize loss for accumulation
            loss = loss / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Step optimizer only after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Scale loss back up for logging
        total_loss += loss.item() * accumulation_steps
        metrics.update(logits, y)

    # Handle remaining gradients if total batches is not divisible by accumulation_steps
    if (i + 1) % accumulation_steps != 0:
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, len(loader))
    computed_metrics = metrics.compute()
    
    return avg_loss, computed_metrics

def validate(model, loader, device, num_classes):
    model.eval()
    metrics = ForensicMetrics(num_classes, device)
    total_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            metrics.update(logits, y)

    val_loss = total_loss / len(loader)
    computed_metrics = metrics.compute()
    
    return val_loss, computed_metrics