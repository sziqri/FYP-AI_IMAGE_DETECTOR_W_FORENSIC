from torch.utils.data import DataLoader
from torchvision import datasets
from transforms_extra import build_transforms_with_extras

def build_dataloaders_with_extras(train_dir: str, val_dir: str, img_size: int, batch_size: int, num_workers: int,
                                  rand_augment: bool, color_jitter: float, hflip_prob: float,
                                  use_fft: bool = True, use_ela: bool = True, use_prnu: bool = True):
    # Create the training dataset with the extra transforms
    train_ds = datasets.ImageFolder(train_dir, transform=build_transforms_with_extras(
        img_size, True, rand_augment, color_jitter, hflip_prob, use_fft, use_ela, use_prnu))
    
    # Create the validation dataset with the extra transforms
    val_ds = datasets.ImageFolder(val_dir, transform=build_transforms_with_extras(
        img_size, False, rand_augment, color_jitter, hflip_prob, use_fft, use_ela, use_prnu))

    # Create the DataLoaders for training and validation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Return DataLoader objects and class names
    return train_loader, val_loader, train_ds.classes
