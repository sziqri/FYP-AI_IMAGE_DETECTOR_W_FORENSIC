from torch.utils.data import DataLoader
from torchvision import datasets
from transforms import build_transforms

def build_dataloaders(train_dir: str, val_dir: str, img_size: int, batch_size: int, num_workers: int,
                       rand_augment: bool, color_jitter: float, hflip_prob: float):
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms(img_size, True, rand_augment, color_jitter, hflip_prob))
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms(img_size, False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, train_ds.classes
