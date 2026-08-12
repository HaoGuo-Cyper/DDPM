
from pathlib import Path
import sys
import time

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import DenoiseDiffusion
from unet import UNet


class DDPMConfig:
    """DDPM sampling configuration."""

    device = torch.device("cuda")

    checkpoint_path = "result/celeba/model.pt"
    samples_path = "result/celeba/ddpm_samples.png"
    trajectory_path = "result/celeba/ddpm_trajectory.png"

    # Must be identical to the training configuration.
    image_channels = 3
    image_size = 64

    model_channels = 64
    ch_mults = [1, 2, 2, 4]
    is_attn = [False, False, False, True]

    n_steps = 1000

    # Number of final generated images.
    n_samples = 16

    # Record one intermediate state every 100 steps.
    snapshot_every = 100

    # Number of samples shown in the trajectory plot.
    trajectory_rows = 8

    # Fixed seed for reproducible sampling.
    seed = None


def create_diffusion(config: DDPMConfig) -> DenoiseDiffusion:
    """Create the same U-Net and diffusion model used during training."""

    eps_model = UNet(
        image_channels=config.image_channels,
        n_channels=config.model_channels,
        ch_mults=config.ch_mults,
        is_attn=config.is_attn,
    )

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=config.n_steps,
    )

    return diffusion.to(config.device)


def load_checkpoint(
    diffusion: DenoiseDiffusion,
    checkpoint_path: str,
    device: torch.device,
):
    """Load the trained model parameters."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Support both raw state_dict and structured checkpoints.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        checkpoint = checkpoint["model_state_dict"]

    diffusion.load_state_dict(checkpoint)
    diffusion.eval()


def ddpm_sample(
    diffusion: DenoiseDiffusion,
    n_samples: int,
    image_channels: int,
    image_size: int,
    device: torch.device,
    initial_noise: torch.Tensor | None = None,
    snapshot_every: int = 100,
):

    diffusion.eval()

    if initial_noise is None:
        x = torch.randn(
            n_samples,
            image_channels,
            image_size,
            image_size,
            device=device,
        )
    else:
        x = initial_noise.to(device).clone()

    # Record the initial Gaussian noise x_T.
    trajectory = [
        (
            diffusion.n_steps,
            x.detach().cpu().clone(),
        )
    ]

    with torch.no_grad():
        for step in tqdm(
            range(diffusion.n_steps - 1, -1, -1),
            desc="DDPM sampling",
        ):
            timesteps = torch.full(
                (n_samples,),
                step,
                device=device,
                dtype=torch.long,
            )

            # One reverse diffusion step:
            # x_t -> x_{t-1}
            x = diffusion.p_sample(
                x,
                timesteps,
            )

            # Save intermediate states.
            if step % snapshot_every == 0:
                trajectory.append(
                    (
                        step,
                        x.detach().cpu().clone(),
                    )
                )

    final_images = x.clamp(-1.0, 1.0)

    return final_images, trajectory


def save_final_images(
    images: torch.Tensor,
    output_path: str,
):
    """Save the final generated images as a grid."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Convert from [-1, 1] to [0, 1].
    images = (images + 1.0) / 2.0

    save_image(
        images,
        output_path,
        nrow=4,
    )


def save_trajectory_plot(
    trajectory,
    output_path: str,
    n_rows: int = 8,
):

    # Sampling was recorded as x_T -> x_0.
    # Reverse it for display so x_0 appears on the left.
    displayed_trajectory = list(reversed(trajectory))

    n_columns = len(displayed_trajectory)

    first_batch = displayed_trajectory[0][1]
    n_rows = min(
        n_rows,
        first_batch.shape[0],
    )

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            n_columns * 1.35,
            n_rows * 1.35,
        ),
        squeeze=False,
    )

    for column, (step, images) in enumerate(
        displayed_trajectory
    ):
        # Convert from [-1, 1] to [0, 1].
        images = (
            images.clamp(-1.0, 1.0) + 1.0
        ) / 2.0

        for row in range(n_rows):
            # MNIST has one grayscale channel.
            image_tensor = images[row]

            if image_tensor.shape[0] == 1:
                image = image_tensor[0].numpy()
                axes[row, column].imshow(
                image,
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            else:
                image = image_tensor.permute(1, 2, 0).numpy()
                axes[row, column].imshow(
                image,
                vmin=0.0,
                vmax=1.0,
            )

            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

            for spine in axes[row, column].spines.values():
                spine.set_visible(True)
                spine.set_color("black")
                spine.set_linewidth(0.7)

        if step == 0:
            title = r"$x_0$"
        elif step == trajectory[0][0]:
            title = r"$x_T$"
        else:
            title = rf"$x_{{{step}}}$"

        axes[0, column].set_title(
            title,
            fontsize=11,
            pad=6,
        )

    figure.suptitle(
        r"DDPM generation: $x_T \rightarrow x_0$ "
        r"(displayed from right to left)",
        fontsize=14,
        y=0.995,
    )

    plt.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.92,
        bottom=0.02,
        wspace=0.08,
        hspace=0.05,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    config = DDPMConfig()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch environment."
        )

    checkpoint_path = Path(config.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )



    initial_noise = torch.randn(
        config.n_samples,
        config.image_channels,
        config.image_size,
        config.image_size,
        device=config.device,
    )

    diffusion = create_diffusion(config)

    load_checkpoint(
        diffusion=diffusion,
        checkpoint_path=config.checkpoint_path,
        device=config.device,
    )

    start_time = time.perf_counter()

    final_images, trajectory = ddpm_sample(
        diffusion=diffusion,
        n_samples=config.n_samples,
        image_channels=config.image_channels,
        image_size=config.image_size,
        device=config.device,
        initial_noise=initial_noise,
        snapshot_every=config.snapshot_every,
    )

    elapsed_time = time.perf_counter() - start_time

    save_final_images(
        images=final_images,
        output_path=config.samples_path,
    )

    save_trajectory_plot(
        trajectory=trajectory,
        output_path=config.trajectory_path,
        n_rows=config.trajectory_rows,
    )

    saved_steps = [
        step for step, _ in trajectory
    ]

    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"DDPM steps: {config.n_steps}")
    print(f"Recorded steps: {saved_steps}")
    print(f"Sampling time: {elapsed_time:.2f} seconds")
    print(f"Final images: {config.samples_path}")
    print(f"Trajectory: {config.trajectory_path}")


if __name__ == "__main__":
    main()