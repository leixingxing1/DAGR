from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP, ensemblize, GCMRNValue, GCIQEValue, GCBilinearValue


class GCHilbertRepresentationValue(nn.Module):
    """Value function parameterized as the Euclidean distance between state & goal representations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None):
        """
        Return the value/critic function or representation function.
        If both goals & obs are specified, then a scalar value will be returned.
        If just observations are specified, then a representation vector `phi(obs)` will be returned.
        """

        if goals is not None:
            # Value function call.
            state_rep = self.phi(observations)
            goal_rep = self.phi(goals)
            squared_dist = jnp.square(state_rep - goal_rep).sum(axis=-1)
            v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
            return v
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            phi = self.phi(observations)
            return phi.mean(axis=0) if self.ensemble else phi


class GCAsymmetricHilbertRepresentationValue(nn.Module):
    """Value function parameterized as the Euclidean distance between unique state & goal representations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None):
        """
        Return the value/critic function or representation function.
        If both goals & obs are specified, then a scalar value will be returned.
        If just observations are specified, then a representation vector `phi(obs)` will be returned.
        """

        if goals is not None:
            # Value function call.
            state_rep = self.psi(observations)
            goal_rep = self.phi(goals)
            squared_dist = ((state_rep - goal_rep) ** 2).sum(axis=-1)
            v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
            return v
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            phi = self.phi(observations)
            return phi.mean(axis=0) if self.ensemble else phi


class GCMRNRepresentationValue(nn.Module):
    """Value function parameterized as the MRN distance function between state and goal."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        self.network = GCMRNValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            ensemble=self.ensemble,
        )

    def __call__(self, observations, goals=None):
        """
        Return the value/critic function or representation function.
        If both goals & obs are specified, then a scalar value will be returned.
        If just observations are specified, then a representation vector `phi(obs)` will be returned.
        """

        if goals is not None:
            # Value function call.
            return -self.network(observations, goals, is_phi=False, info=False)
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            dummy_observation = jnp.zeros_like(observations)
            _, _, phi_g = self.network(dummy_observation, observations, is_phi=False, info=True)
            return phi_g


class GCIQERepresentationValue(nn.Module):
    """Value function parameterized as the IQE distance function between state and goal."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True

    def setup(self):
        self.network = GCIQEValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            dim_per_component=8,
            ensemble=self.ensemble,
        )

    def __call__(self, observations, goals=None):
        """
        Return the value/critic function or representation function.
        If both goals & obs are specified, then a scalar value will be returned.
        If just observations are specified, then a representation vector `phi(obs)` will be returned.
        """

        if goals is not None:
            # Value function call.
            return -self.network(observations, goals, is_phi=False, info=False)
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            dummy_observation = jnp.zeros_like(observations)
            _, _, phi_g = self.network(dummy_observation, observations, is_phi=False, info=True)
            return phi_g


class GCBilinearRepresentationValue(nn.Module):
    """Value function parameterized as the inner product between unique state and goal representations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False

    def setup(self):
        self.network = GCBilinearValue(
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            layer_norm=self.layer_norm,
            ensemble=self.ensemble,
            value_exp=self.value_exp,
            ret_mean=True,
        )

    def __call__(self, observations, goals=None):
        """
        Return the value/critic function or representation function.
        If both goals & obs are specified, then a scalar value will be returned.
        If just observations are specified, then a representation vector `phi(obs)` will be returned.
        """

        if goals is not None:
            # Value function call.
            return self.network(observations, goals, actions=None, info=False)
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            dummy_observation = jnp.zeros_like(observations)
            v, phi, psi = self.network(dummy_observation, observations, actions=None, info=True)
            return psi


def DualRepresentationValue(type):
    """Helper function that returns the appropriate value parameterization.

    Args:
        type: Type of value parameterization ('bilinear', 'hilbert', 'asymmetric', 'mrn', or 'iqe').
    """
    if type == 'bilinear':
        return GCBilinearRepresentationValue
    elif type == 'hilbert':
        return GCHilbertRepresentationValue
    elif type == 'asymmetric':
        return GCAsymmetricHilbertRepresentationValue
    elif type == 'mrn':
        return GCMRNRepresentationValue
    elif type == 'iqe':
        return GCIQERepresentationValue
    else:
        raise ValueError('Unknown representation type.')
