from pathlib import Path
import time
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from diffusion import DenoiseDiffusion
from unet import UNet


class DDIMConfig:
    """DDIM sampling settings."""

    device = torch.device("cuda")

    checkpoint_path = "result/celeba/model.pt"
    output_directory = "result/celeba"

    # Must match the configuration used during DDPM training.
    image_channels = 3
    image_size = 64
    model_channels = 64
    n_steps = 1000

    # DDIM sampling settings.
    sample_steps_list = [100, 50, 20, 10]
    eta = 0.0

    n_samples = 16
    seed = None


def create_diffusion(config: DDIMConfig) -> DenoiseDiffusion:
    """Create the same network architecture used during training."""

    eps_model = UNet(
        image_channels=config.image_channels,
        n_channels=config.model_channels,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True],
    )

    diffusion = DenoiseDiffusion(
        eps_model=eps_model,
        n_steps=config.n_steps,
    ).to(config.device)

    return diffusion


def load_checkpoint(
    diffusion: DenoiseDiffusion,
    checkpoint_path: str,
    device: torch.device,
):
    """Load the trained DDPM checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    diffusion.load_state_dict(checkpoint)
    diffusion.eval()


def create_sampling_sequence(
    n_steps: int,
    sample_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Select DDIM timesteps from the original DDPM trajectory."""

    if sample_steps < 1:
        raise ValueError("sample_steps must be at least 1.")

    if sample_steps > n_steps:
        raise ValueError("sample_steps cannot be larger than n_steps.")

    sequence = torch.linspace(
        0,
        n_steps - 1,
        sample_steps,
        device=device,
    )

    return sequence.round().long()


def ddim_sample(
    diffusion: DenoiseDiffusion,
    n_samples: int,
    image_channels: int,
    image_size: int,
    sample_steps: int,
    eta: float,
    device: torch.device,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Generate samples using DDIM.

    eta = 0:
        Deterministic DDIM sampling.

    eta = 1:
        Adds stochastic noise and behaves more like DDPM sampling.
    """

    diffusion.eval()

    with torch.no_grad():
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

        sequence = create_sampling_sequence(
            n_steps=diffusion.n_steps,
            sample_steps=sample_steps,
            device=device,
        )

        # Example:
        # sequence      = [0, 20, 40, ..., 999]
        # reverse order = [999, ..., 40, 20, 0]
        reverse_sequence = list(reversed(sequence.tolist()))

        for index, current_step in enumerate(
            tqdm(reverse_sequence, desc="DDIM sampling")
        ):
            # The next point along the reverse trajectory.
            if index == len(reverse_sequence) - 1:
                previous_step = -1
            else:
                previous_step = reverse_sequence[index + 1]

            t = torch.full(
                (n_samples,),
                current_step,
                device=device,
                dtype=torch.long,
            )

            # Predict the noise in x_t.
            predicted_noise = diffusion.eps_model(x, t)

            # alpha_bar at the current timestep.
            alpha_bar_t = diffusion.alpha_bar[current_step]

            # alpha_bar at the previous selected timestep.
            # alpha_bar_{-1} is defined as 1.
            if previous_step < 0:
                alpha_bar_previous = torch.tensor(
                    1.0,
                    device=device,
                    dtype=alpha_bar_t.dtype,
                )
            else:
                alpha_bar_previous = diffusion.alpha_bar[previous_step]

            # Recover the predicted clean sample x_0:
            #
            # x_0 = (
            #     x_t - sqrt(1-alpha_bar_t) * epsilon_theta
            # ) / sqrt(alpha_bar_t)
            predicted_x0 = (
                x
                - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise
            ) / torch.sqrt(alpha_bar_t)

            # MNIST was normalized to [-1, 1].
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

            # DDIM stochasticity:
            #
            # eta = 0 -> deterministic
            # eta > 0 -> stochastic
            sigma = eta * torch.sqrt(
                ((1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t))
                * (1.0 - alpha_bar_t / alpha_bar_previous)
            )

            # Direction pointing toward x_t.
            direction_coefficient = torch.sqrt(
                torch.clamp(
                    1.0 - alpha_bar_previous - sigma ** 2,
                    min=0.0,
                )
            )

            direction = direction_coefficient * predicted_noise

            # Do not add random noise at the final step.
            if previous_step >= 0 and eta > 0:
                random_noise = torch.randn_like(x)
            else:
                random_noise = torch.zeros_like(x)

            # DDIM update:
            #
            # x_{t-1} =
            # sqrt(alpha_bar_{t-1}) * predicted_x0
            # + direction
            # + sigma * random_noise
            x = (
                torch.sqrt(alpha_bar_previous) * predicted_x0
                + direction
                + sigma * random_noise
            )

    return x.clamp(-1.0, 1.0)

def main():
    config = DDIMConfig()

    if (
        config.device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is not available."
        )

    checkpoint_path = Path(
        config.checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    output_directory = Path(
        config.output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # All samplers use exactly the same initial noise.
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

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eta: {config.eta}")
    print()

    for sample_steps in config.sample_steps_list:
        start_time = time.perf_counter()

        samples = ddim_sample(
            diffusion=diffusion,
            n_samples=config.n_samples,
            image_channels=config.image_channels,
            image_size=config.image_size,
            sample_steps=sample_steps,
            eta=config.eta,
            device=config.device,
            initial_noise=initial_noise,
        )

        elapsed_time = (
            time.perf_counter() - start_time
        )

        # Convert from [-1, 1] to [0, 1].
        samples = (samples + 1.0) / 2.0

        output_path = (
            output_directory
            / f"ddim_{sample_steps}_steps.png"
        )

        save_image(
            samples,
            output_path,
            nrow=4,
        )

        print(
            f"DDIM {sample_steps:3d} steps | "
            f"time: {elapsed_time:.2f} s | "
            f"saved: {output_path}"
        )


if __name__ == "__main__":
    main()