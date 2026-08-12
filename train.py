"""Train DDPM with automatic CelebA download support."""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torchvision.utils import save_image

from datasets import create_data_loader
from diffusion import DenoiseDiffusion
from unet import UNet


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "celeba"

    # Put datasets/checkpoints on AutoDL data disk
    data_root = "/root/autodl-tmp/datasets"
    celeba_root = f"{data_root}/celeba"

    output_root = "result"

    model_channels = 64
    n_steps = 1000
    batch_size = 64
    learning_rate = 2e-5
    epochs = 500
    sample_every = 20
    n_samples = 16
    num_workers = 4


def train_one_epoch(diffusion, data_loader, optimizer, device):
    diffusion.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss = diffusion.loss(images)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def sample(diffusion, n_samples, image_channels, image_size, device):
    diffusion.eval()

    x = torch.randn(
        n_samples,
        image_channels,
        image_size,
        image_size,
        device=device
    )

    with torch.no_grad():
        for step in tqdm(
            range(diffusion.n_steps - 1, -1, -1),
            desc="Sampling",
            leave=False,
        ):
            t = torch.full(
                (n_samples,),
                step,
                device=device,
                dtype=torch.long,
            )
            x = diffusion.p_sample(x, t)

    return x.clamp(-1, 1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default=Config.dataset_name,
        choices=["celeba"]
    )

    parser.add_argument(
        "--data_root",
        default=Config.celeba_root,
        help="Existing CelebA path. If empty, torchvision will download it."
    )

    args = parser.parse_args()

    config = Config()

    loader, image_channels, image_size = create_data_loader(
        name=args.dataset,
        batch_size=config.batch_size,
        celeba_root=args.data_root,
        num_workers=config.num_workers,
    )


    output_path = Path(config.output_root) / args.dataset
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


    optimizer = torch.optim.Adam(
        diffusion.parameters(),
        lr=config.learning_rate
    )


    print("Device:", config.device)
    print("Dataset:", args.dataset)
    print("Image:", image_channels, image_size)
    print("Samples:", len(loader.dataset))


    for epoch in range(1, config.epochs + 1):

        loss = train_one_epoch(
            diffusion,
            loader,
            optimizer,
            config.device
        )

        print(
            f"Epoch {epoch}/{config.epochs}, loss={loss:.6f}"
        )


        if epoch % config.sample_every == 0:

            images = sample(
                diffusion,
                config.n_samples,
                image_channels,
                image_size,
                config.device,
            )

            save_image(
                (images + 1) / 2,
                output_path / f"sample_{epoch:04d}.png",
                nrow=4,
            )

            torch.save(
                diffusion.state_dict(),
                output_path / "model.pt"
            )


    torch.save(
        diffusion.state_dict(),
        output_path / "final_model.pt"
    )


if __name__ == "__main__":
    main()
