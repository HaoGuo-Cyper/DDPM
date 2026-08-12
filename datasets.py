"""Dataset definitions used by the DDPM training script."""

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class MNISTCSVDataset(Dataset):
    """Read rows containing one label followed by 784 grayscale pixels."""

    def __init__(self, csv_path: str, image_size: int = 32):
        table = pd.read_csv(csv_path, header=None)
        pixels = torch.tensor(table.iloc[:, 1:].values, dtype=torch.float32)
        images = pixels.reshape(-1, 1, 28, 28) / 255.0

        images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        self.images = images * 2.0 - 1.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


class CelebAImageDataset(Dataset):
    """Recursively load CelebA JPG/PNG files without using attributes."""

    def __init__(self, root: str, transform=None):
        self.transform = transform
        root = Path(root)
        self.image_paths = sorted(
            list(root.rglob("*.jpg"))
            + list(root.rglob("*.jpeg"))
            + list(root.rglob("*.png"))
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No CelebA images found under: {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image


def create_dataset(
    name: str,
    mnist_csv: str,
    celeba_root: str,
    max_samples: int | None = None,
):
    """Create a dataset and return it with its image channels and size."""
    name = name.lower()

    if name == "mnist":
        image_channels = 1
        image_size = 32
        dataset = MNISTCSVDataset(mnist_csv, image_size)

    elif name == "celeba":
        image_channels = 3
        image_size = 64
        transform = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        dataset = CelebAImageDataset(celeba_root, transform=transform)

    else:
        raise ValueError("dataset must be 'mnist' or 'celeba'")

    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    return dataset, image_channels, image_size


def create_data_loader(
    name: str,
    batch_size: int,
    mnist_csv: str,
    celeba_root: str,
    max_samples: int | None = None,
    num_workers: int = 2,
):
    """Create the selected dataset and its shuffled training loader."""
    dataset, image_channels, image_size = create_dataset(
        name=name,
        mnist_csv=mnist_csv,
        celeba_root=celeba_root,
        max_samples=max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return loader, image_channels, image_size
