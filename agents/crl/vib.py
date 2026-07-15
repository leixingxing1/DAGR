from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, MLP
from utils.vib import VIB


class CRLVIBAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent.
    Uses a VIB (Variational Information Bottleneck) goal representation.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, rng, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        goal_reps, kl_loss, kl_info = self.network.select('vib')(
            batch['value_goals'], rng, encoded=False, params=grad_params
        )

        batch_size = batch['observations'].shape[0]

        if module_name == 'critic':
            actions = batch['actions']
        else:
            actions = None
        v, phi, psi = self.network.select(module_name)(
            batch['observations'],
            goal_reps,
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss + kl_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        } | kl_info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss (AWR or DDPG+BC)."""
        # Actor gradients were not found to improve performance.
        goal_reps, _, _ = self.network.select('vib')(batch['actor_goals'], rng, encoded=False)

        if self.config['actor_loss'] == 'awr':
            # AWR loss.
            v = self.network.select('value')(batch['observations'], goal_reps)
            q1, q2 = self.network.select('critic')(batch['observations'], goal_reps, batch['actions'])
            q = jnp.minimum(q1, q2)
            adv = q - v

            exp_a = jnp.exp(adv * self.config['alpha'])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
            log_prob = dist.log_prob(batch['actions'])

            actor_loss = -(exp_a * log_prob).mean()

            actor_info = {
                'actor_loss': actor_loss,
                'adv': adv.mean(),
                'bc_log_prob': log_prob.mean(),
            }
            if not self.config['discrete']:
                actor_info.update(
                    {
                        'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                        'std': jnp.mean(dist.scale_diag),
                    }
                )

            return actor_loss, actor_info
        elif self.config['actor_loss'] == 'ddpgbc':
            # DDPG+BC loss.
            assert not self.config['discrete']

            dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)
            q1, q2 = self.network.select('critic')(batch['observations'], goal_reps, q_actions)
            q = jnp.minimum(q1, q2)

            # Normalize Q values by the absolute mean to make the loss scale invariant.
            q_loss = -q.mean() / jax.lax.stop_gradient(jnp.abs(q).mean() + 1e-6)
            log_prob = dist.log_prob(batch['actions'])

            bc_loss = -(self.config['alpha'] * log_prob).mean()

            actor_loss = q_loss + bc_loss

            return actor_loss, {
                'actor_loss': actor_loss,
                'q_loss': q_loss,
                'bc_loss': bc_loss,
                'q_mean': q.mean(),
                'q_abs_mean': jnp.abs(q).mean(),
                'bc_log_prob': log_prob.mean(),
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            }
        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, critic_rng, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, new_rng = jax.random.split(rng)
        if self.config['actor_loss'] == 'awr':
            value_loss, value_info = self.contrastive_loss(batch, grad_params, new_rng, 'value')
            for k, v in value_info.items():
                info[f'value/{k}'] = v
        else:
            value_loss = 0.0

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + value_loss + actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        vib_seed, seed = jax.random.split(seed)
        goal_reps, _, _ = self.network.select('vib')(goals, vib_seed, encoded=False)

        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        sample_seed, seed = jax.random.split(seed)
        actions = dist.sample(seed=sample_seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
        ex_goals=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # NOTE: this version does not support pixel-based observations; please refer to the visual-dedicated file
        # Define value and actor networks.
        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
                action_dim=action_dim,
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=False,
            )

        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network to compute advantages (Q - V).
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=False,
            )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
            )

        vib_def = VIB(
            encoder=MLP(
                hidden_dims=config['vib_hidden_dims'],
                layer_norm=config['layer_norm'],
            ),
            beta=config['beta'],
            rep_dim=config['goalrep_dim'],
        )

        network_info = dict(
            vib=(vib_def, (ex_observations, jax.random.PRNGKey(0), False)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl_vib',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            vib_hidden_dims=(512, 512, 512),  # VIB network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            latent_dim=512,  # Latent dimension for phi and psi.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            goalrep_dim=64,  # Dimensionality of the VIB goal representation.
            beta=0.003,  # VIB strength hyperparameter.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            oraclerep=False,  # Always False; dummy option for compatibility.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
