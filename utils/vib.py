import flax.linen as nn
import jax.numpy as jnp
import distrax


class VIB(nn.Module):
    """Implementation of a variational information bottleneck layer.

    Attributes:
        encoder: Encoder into the latent layer (can be left unspecified).
        beta: Bottleneck strength hyperparameter.
        rep_dim: Dimensionality of the latent layer.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
    """

    encoder: nn.Module = None
    beta: float = 1.0
    rep_dim: int = 256
    log_std_min: float = -5
    log_std_max: float = 2

    @nn.compact
    def __call__(self, goal, rng, encoded=False):
        """
        Encodes a goal, and returns appropriate KL loss and metrics.

        The goal is encoded (if not already) and parameterized into a Gaussian, from
        which we sample our goal representation. We also return the KL-divergence
        between the goal distribution and a standard Gaussian prior.

        Args:
            goal: The goal to be encoded.
            rng: Random seed.
            encoded: Whether the goal is already encoded (just needs to be reparameterized).
        """
        if not encoded:
            if self.encoder is not None:
                goal = self.encoder(goal)
            else:
                raise ValueError('Cannot encode goals without a default encoder module.')

        mean, log_stds = nn.Dense(self.rep_dim)(goal), nn.Dense(self.rep_dim)(goal)

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        stds = jnp.exp(log_stds)

        dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=stds)
        z = dist.sample(seed=rng)

        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros_like(mean), scale_diag=jnp.ones_like(stds))
        kl = dist.kl_divergence(prior)

        return (
            z,
            (self.beta * kl).mean(),
            {
                'kl_means_min': mean.min(),
                'kl_means_mean': mean.mean(),
                'kl_means_max': mean.max(),
                'kl_stds_min': stds.min(),
                'kl_stds_mean': stds.mean(),
                'kl_stds_max': stds.max(),
            },
        )
