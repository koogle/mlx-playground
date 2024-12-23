import mlx.core as mx


class NoiseScheduler:
    """
    Manages the noise schedule for the diffusion process.
    Implements linear beta schedule as described in the DDPM paper.
    """
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=None
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = mx.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mx.cumprod(self.alphas)
        self.alphas_cumprod_prev = mx.concatenate(
            [mx.array([1.0]), self.alphas_cumprod[:-1]]
        )
        self.sqrt_recip_alphas = mx.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = mx.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = mx.random.normal(x_start.shape)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(self, denoise_model, x_start, t, noise=None):
        """Calculate the loss for denoising model training"""
        if noise is None:
            noise = mx.random.normal(x_start.shape)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        return mx.mean((noise - predicted_noise) ** 2)

    @mx.compile
    def p_sample(self, model, x, t, t_index):
        """Sample from the model at timestep t"""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = mx.random.normal(x.shape)
            return model_mean + mx.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, model, shape):
        """Generate samples from the model using the p_sample method"""
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = mx.random.normal(shape)
        imgs = []

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(
                model, img, 
                mx.array([i] * b),
                i
            )
            imgs.append(img)
        return imgs

    def sample(self, model, image_size, batch_size=16, channels=3):
        """Generate new samples from the model"""
        return self.p_sample_loop(
            model,
            shape=(batch_size, channels, image_size, image_size),
        )
