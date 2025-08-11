import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EstDiff(nn.Module):
    def __init__(self, model_dim, cond_dim, seq_len, num_timesteps=1000, beta_min=0.0001, beta_max=0.02):
        super().__init__()
        self.model_dim = model_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps

        # Define beta schedule
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.register_buffer('betas', torch.linspace(beta_min, beta_max, num_timesteps))

        # Define alphas
        alphas = 1. - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - self.alphas_cumprod))

        # Time embedding dimension
        time_embed_dim = 128

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish()
        )

        # Calculate input dimension
        # Flattened dimensions for x and condition, plus time embedding
        input_dim = model_dim * seq_len + cond_dim * seq_len + time_embed_dim

        # MLP with 3 layers of 256 neurons and Mish activation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
            nn.Linear(256, model_dim * seq_len)
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: adding noise to the data.
        x_start: [batch_size, seq_len, model_dim]
        t: [batch_size]
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self, x, cond, t):
        """
        Predict the noise in x at timestep t, conditioned on cond.
        x: [batch_size, seq_len, model_dim]
        cond: [batch_size, seq_len, cond_dim]
        t: [batch_size]
        """
        batch_size = x.shape[0]

        # Time embedding - Convert t to float to match weight type
        t_emb = self.time_embedding(t.float().unsqueeze(-1))  # [batch_size, time_embed_dim]

        # Flatten the input tensors
        x_flat = x.reshape(batch_size, -1)  # [batch_size, seq_len * model_dim]
        cond_flat = cond.reshape(batch_size, -1)  # [batch_size, seq_len * cond_dim]

        # Concatenate inputs
        combined = torch.cat([x_flat, cond_flat, t_emb], dim=1)

        # Process through MLP
        output = self.mlp(combined)

        # Reshape output back to original dimensions
        output = output.reshape(batch_size, self.seq_len, self.model_dim)

        return output

    def score_function(self, x, cond, t_idx):
        """
        Calculate score function (gradient of log probability) based on denoiser output.
        """
        # Get the current noise level
        t_val = (t_idx.float() / self.num_timesteps).view(-1, 1, 1)

        # Predict original data
        pred_x = self(x, cond, t_idx)

        # Calculate score using the formula: (denoiser(x, t) - x) / t²
        score = (pred_x - x) / (t_val ** 2)

        return score

    @torch.no_grad()
    def predict_heun(self, cond, num_samples=1, steps=50):
        """
        Sample from the model using Heun's 2nd order method.
        cond: [batch_size, seq_len, cond_dim]
        """
        device = cond.device
        batch_size = cond.shape[0]

        EPSILON = 0.002
        T = 1.0

        # Start from random noise
        xT = torch.randn(batch_size, self.seq_len, self.model_dim, device=device)
        t_steps = [(EPSILON ** (1 / 7) + (j / (steps - 1)) * (T ** (1 / 7) - EPSILON ** (1 / 7))) ** 7
                   for j in range(steps - 1, -1, -1)]

        # Convert normalized times to indices
        t_indices = [min(int(t * self.num_timesteps), self.num_timesteps - 1) for t in t_steps]

        # Initialize and start trajectory
        xi = xT
        trajectory = [xT]

        # Heun's 2nd order method
        for i in range(0, len(t_indices) - 1):
            # Current and next time indices
            t_curr_idx = torch.ones(batch_size, device=device, dtype=torch.long) * t_indices[i]
            t_next_idx = torch.ones(batch_size, device=device, dtype=torch.long) * t_indices[i + 1]

            # Current timestep value
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]

            # Calculate first direction estimate (d in toy_example)
            d = -t_curr * self.score_function(xi, cond, t_curr_idx)

            # Euler step prediction
            xi_1 = xi + (t_next - t_curr) * d

            # Calculate second direction at the predicted point
            d_ = -t_next * self.score_function(xi_1, cond, t_next_idx)

            # Final Heun update using average of both directions
            xi_1 = xi + (t_next - t_curr) / 2 * (d + d_)

            # Update for next iteration
            xi = xi_1
            trajectory.append(xi)

        return xi, (t_steps, trajectory)

    @torch.no_grad()
    def predict(self, cond, num_samples=1, steps=50):
        """
        Wrapper for prediction - uses Heun method by default
        """
        result, _ = self.predict_heun(cond, num_samples, steps)
        return result

