
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    
    """Select one schedule value per sample and reshape it for broadcasting."""
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *([1] * (len(x_shape) - 1)))

class DenoiseDiffusion(nn.Module):
    #建立噪声时间表
    def __init__(
        self,
        eps_model: nn.Module,
        n_steps: int = 1_000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.eps_model = eps_model 
        self.n_steps = n_steps    

        beta = torch.linspace(beta_start, beta_end, n_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        # Buffers follow the model between CPU/GPU and are stored in checkpoints.
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    #计算前向分布
    def q_xt_x0(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        """Return the mean and variance of q(x_t | x_0)."""
        alpha_bar_t = extract(self.alpha_bar, timesteps, x0.shape)
        mean = alpha_bar_t.sqrt() * x0
        variance = 1.0 - alpha_bar_t
        return mean, variance

    #执行前向分布
    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        """Directly sample x_t from x_0 at arbitrary timesteps."""
        if noise is None:
            noise = torch.randn_like(x0)

        mean, variance = self.q_xt_x0(x0, timesteps)
        return mean + variance.sqrt() * noise

    #DDPM训练目标 L=||eps-eps_theta||**2
    def loss(
        self,
        x0: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        """Train the network to predict the Gaussian noise added to x_0."""
        batch_size = x0.shape[0]

        timesteps = torch.randint(
            0,
            self.n_steps,
            (batch_size,),
            device=x0.device,
            dtype=torch.long,
        )

        if noise is None:
            noise = torch.randn_like(x0)

        xt = self.q_sample(x0, timesteps, noise)
        noise_theta = self.eps_model(xt, timesteps)
        return F.mse_loss(noise_theta, noise)

    #执行反向去噪
    def p_sample(
        self, 
        xt: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        
        """Sample one reverse step x_t -> x_(t-1)."""
        noise_theta = self.eps_model(xt, timesteps)

        alpha_t = extract(self.alpha, timesteps, xt.shape)
        alpha_bar_t = extract(self.alpha_bar, timesteps, xt.shape)
        beta_t = extract(self.beta, timesteps, xt.shape)

        mean = (xt - beta_t / (1.0 - alpha_bar_t).sqrt() * noise_theta) / alpha_t.sqrt()

        # The final step returns the mean without injecting fresh noise.
        noise = torch.randn_like(xt)

        #t>0 添加随机噪声，t=0 不添加随机噪声
        nonzero_mask = (timesteps != 0).float().reshape(
            timesteps.shape[0], *([1] * (xt.ndim - 1))
        )
        return mean + nonzero_mask * beta_t.sqrt() * noise