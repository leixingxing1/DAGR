from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCBilinearValue, ActorVectorField


class GCFBCTRAAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow behavioral cloning (GCBC) agent.
    Uses a TRA-based method for goal representation.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_rep_loss(self, batch, grad_params):
        """Compute the contrastive value loss for the representation value function."""
        batch_size = batch['observations'].shape[0]

        v, phi, psi = self.network.select('contrastive')(
            batch['observations'],
            batch['rep_goals'],
            actions=None,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])

        assert logits.shape[0] == batch_size and logits.shape[1] == batch_size
        I = jnp.eye(batch_size)

        contrastive_loss = -(
            jax.nn.log_softmax(logits, axis=0) * I[..., None] + jax.nn.log_softmax(logits, axis=1) * I[..., None]
        )
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        if self.config['actor_goalrep_grad']:
            _, _, goal_reps = self.network.select('contrastive')(
                batch['observations'], batch['actor_goals'], actions=None, info=True, params=grad_params
            )
        else:
            _, _, goal_reps = self.network.select('contrastive')(
                batch['observations'], batch['actor_goals'], actions=None, info=True
            )

        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        y = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], goal_reps, x_t, t, params=grad_params)

        actor_loss = jnp.mean((pred - y) ** 2)

        actor_info = {
            'actor_loss': actor_loss,
        }

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        contrastive_loss, contrastive_info = self.contrastive_rep_loss(batch, grad_params)
        for k, v in contrastive_info.items():
            info[f'contrastive/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss + contrastive_loss
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
        _, _, goal_reps = self.network.select('contrastive')(
            observations,
            goals,
            actions=None,
            info=True,
        )
        actions = jax.random.normal(
            seed,
            (
                *observations.shape[:-1],
                self.config['action_dim'],
            ),
        )
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, goal_reps, actions, t)
            actions = actions + vels / self.config['flow_steps']
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
        assert not config['discrete']

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        # Define actor network.
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        contrastive_def = GCBilinearValue(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=True,
            ensemble=True,
            value_exp=True,
            ret_mean=True,
        )

        network_info = dict(
            contrastive=(contrastive_def, (ex_observations, ex_observations)),
            actor_flow=(actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcfbc_tra',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            discrete=False,  # Whether the action space is discrete.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            flow_steps=10,  # Number of flow steps.
            goalrep_dim=64,  # Dimensionality of the TRA goal representation.
            actor_goalrep_grad=False,  # Whether actor gradients are flowed through the goal representation.
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            oraclerep=False,  # always False; dummy option for compatibility.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_p_trajgoal=1.0,  # Unused (defined for compatibility with GCDataset).
            value_p_randomgoal=0.0,  # Unused (defined for compatibility with GCDataset).
            value_geom_sample=False,  # Unused (defined for compatibility with GCDataset).
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            rep_p_curgoal=0.0,  # Probability of using the current state as the representation goal.
            rep_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the representation goal.
            rep_p_randomgoal=0.0,  # Probability of using a random state as the representation goal.
            rep_geom_sample=True,  # Whether to use geometric sampling for future representation goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
