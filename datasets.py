"""CelebA dataset loader with automatic download support."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CelebA


def create_dataset(
    name,
    celeba_root,
):

    if name.lower() != "celeba":
        raise ValueError("Only CelebA is supported")


    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((64,64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5,0.5,0.5],
                [0.5,0.5,0.5]
            ),
        ]
    )


    root = Path(celeba_root)

    # If dataset exists, use it.
    # Otherwise torchvision automatically downloads it.
    dataset = CelebA(
        root=str(root),
        split="train",
        target_type="attr",
        transform=transform,
        download=True,
    )


    return dataset, 3, 64



def create_data_loader(
    name,
    batch_size,
    celeba_root,
    num_workers=4,
):

    dataset, channels, size = create_dataset(
        name,
        celeba_root
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


    return loader, channels, size
