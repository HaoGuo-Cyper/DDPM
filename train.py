"""Train one unconditional DDPM on MNIST or CelebA."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from datasets import create_data_loader
from diffusion import DenoiseDiffusion
from unet import UNet


class Config:
    device = torch.device("cuda")

    # Change this value, or use: python train.py --dataset celeba
    dataset_name = "celeba"
    mnist_csv = "dataset/mnist/mnist_train.csv"
    celeba_root = "dataset/celeba"
    max_samples = None

    output_root = "result"
    model_channels = 64
    n_steps = 1_000
    batch_size = 64
    learning_rate = 2e-5
    epochs = 1000
    sample_every = 100
    n_samples = 16
    num_workers = 2


def train_one_epoch(diffusion, data_loader, optimizer, device):
    diffusion.train()
    total_loss = 0.0
    progress = tqdm(data_loader, desc="Training")

    for batch in progress:
        # The current datasets return images; tuple support keeps this reusable.
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / len(data_loader)


def sample(diffusion, n_samples, image_channels, image_size, device):
    diffusion.eval()
    x = torch.randn(n_samples, image_channels, image_size, image_size, device=device)

    with torch.no_grad():
        for step in tqdm(
            range(diffusion.n_steps - 1, -1, -1),
            desc="Sampling",
            leave=False,
        ):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            x = diffusion.p_sample(x, t)

    return x.clamp(-1.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["mnist", "celeba"],
        default=Config.dataset_name,
    )
    args = parser.parse_args()

    config = Config()
    dataset_name = args.dataset

    batch_size = config.batch_size

    data_loader, image_channels, image_size = create_data_loader(
        name=dataset_name,
        batch_size=batch_size,
        mnist_csv=config.mnist_csv,
        celeba_root=config.celeba_root,
        max_samples=config.max_samples,
        num_workers=config.num_workers,
    )

    output_path = Path(config.output_root) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    eps_model = UNet(
        image_channels=image_channels,
        n_channels=config.model_channels,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True],
    )

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=config.n_steps,
    ).to(config.device)

    optimizer = torch.optim.Adam(diffusion.parameters(), lr=config.learning_rate)

    loss_history = []

    print(f"Dataset: {dataset_name}")
    print(f"Image shape: [{image_channels}, {image_size}, {image_size}]")
    print(f"Training samples: {len(data_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        loss = train_one_epoch(diffusion, data_loader, optimizer, config.device)
        loss_history.append(loss)
        print(f"Epoch {epoch}/{config.epochs}, loss: {loss:.6f}")

        if epoch % config.sample_every == 0:
            images = sample(
                diffusion,
                config.n_samples,
                image_channels,
                image_size,
                config.device,
            )
            save_image(
                (images + 1.0) / 2.0,
                output_path / f"sample_{epoch:04d}.png",
                nrow=4,
            )
            torch.save(diffusion.state_dict(), output_path / "model.pt")

    # Always save the final model, even when the last epoch is not a sampling epoch.
    torch.save(
    {
        "epoch": epoch,
        "model": diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss_history": loss_history,
    },
    output_path / "checkpoint.pt",
    )

    plt.plot(range(1, config.epochs + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Noise Prediction MSE")
    plt.title(f"DDPM Training Loss ({dataset_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
