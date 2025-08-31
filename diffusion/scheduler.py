import mlx.core as mx


class NoiseScheduler:
    """
    Manages the noise schedule for the diffusion process.
    Implements linear beta schedule as described in the DDPM paper.
    """

    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device=None):
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

    def add_noise(self, x_start, t, noise=None):
        """
        Forward diffusion process - adds noise to clean images.

        This implements the key formula that lets us jump to any timestep t directly:
        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            x_start: Clean images to add noise to
            t: Timestep(s) indicating how much noise to add (0 = clean, 999 = pure noise)
            noise: Optional pre-generated noise (for reproducibility)

        Returns:
            Noisy images at timestep t
        """
        if noise is None:
            noise = mx.random.normal(x_start.shape)

        # Get the appropriate alpha values and reshape for broadcasting
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting with image dimensions
        sqrt_alphas_cumprod_t = mx.expand_dims(sqrt_alphas_cumprod_t, axis=(1, 2, 3))
        sqrt_one_minus_alphas_cumprod_t = mx.expand_dims(
            sqrt_one_minus_alphas_cumprod_t, axis=(1, 2, 3)
        )

        # Calculate noised image for any timestamp
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t, t_index):
        """Sample from the model at timestep t"""
        # Get scalar values for this timestep
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # Convert from CHW to HWC for model
        x_hwc = mx.transpose(x, (0, 2, 3, 1))
        
        # Get model prediction in HWC format
        noise_pred_hwc = model(x_hwc, t)
        
        # Convert back to CHW
        noise_pred = mx.transpose(noise_pred_hwc, (0, 3, 1, 2))

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = mx.random.normal(x.shape)
            return model_mean + mx.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, model, shape):
        """Generate samples from the model using the p_sample method"""
        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = mx.random.normal(shape)
        imgs = []

        for i in reversed(range(0, self.num_timesteps)):
            t = mx.array([i] * b)
            img = self.p_sample(model, img, t, i)
            imgs.append(img)
        return imgs

    def sample(self, model, image_size, batch_size=16, channels=3):
        """Generate new samples from the model"""
        return self.p_sample_loop(
            model,
            shape=(batch_size, channels, image_size, image_size),
        )
