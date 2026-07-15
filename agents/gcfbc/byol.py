from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, MLP, ensemblize, ActorVectorField


class GCFBCBYOLAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow behavioral cloning (GCBC) agent.
    Uses BYOL-gamma for goal representations.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def byol_loss(self, batch, grad_params):
        """Compute the BYOL-gamma loss."""
        state_rep1, state_rep2 = self.network.select('phi')(batch['observations'], params=grad_params)
        state_rep = (state_rep1 + state_rep2) / 2
        next_pred1, next_pred2 = self.network.select('psi_f')(
            jnp.concatenate([state_rep, batch['actions']], axis=-1), params=grad_params
        )
        next_pred = (next_pred1 + next_pred2) / 2
        goal_rep1, goal_rep2 = self.network.select('phi')(batch['rep_goals'])
        goal_rep = (goal_rep1 + goal_rep2) / 2

        forward_loss = -jnp.sum(
            jax.nn.softmax(goal_rep, axis=-1) * jax.nn.log_softmax(next_pred, axis=-1), axis=-1
        ).mean()

        state_rep = jax.lax.stop_gradient(state_rep)
        goal_rep1, goal_rep2 = self.network.select('phi')(batch['rep_goals'], params=grad_params)
        goal_rep = (goal_rep1 + goal_rep2) / 2
        prev_pred1, prev_pred2 = self.network.select('psi_b')(goal_rep, params=grad_params)
        prev_pred = (prev_pred1 + prev_pred2) / 2

        backward_loss = -jnp.sum(
            jax.nn.softmax(prev_pred, axis=-1) * jax.nn.log_softmax(state_rep, axis=-1), axis=-1
        ).mean()

        return forward_loss + backward_loss, {
            'forward_loss': forward_loss,
            'backward_loss': backward_loss,
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        if self.config['actor_goalrep_grad']:
            goal_rep1, goal_rep2 = self.network.select('phi')(batch['actor_goals'], params=grad_params)
        else:
            goal_rep1, goal_rep2 = self.network.select('phi')(batch['actor_goals'])

        goal_reps = (goal_rep1 + goal_rep2) / 2

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

        byol_loss, byol_info = self.byol_loss(batch, grad_params)
        for k, v in byol_info.items():
            info[f'byol/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss + byol_loss
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
        goal_rep1, goal_rep2 = self.network.select('phi')(goals)
        goal_reps = (goal_rep1 + goal_rep2) / 2
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

        assert not config['discrete']

        ex_times = ex_actions[..., :1]
        action_dim = ex_actions.shape[-1]

        ex_goals = jnp.zeros(shape=(1, config['goalrep_dim']))

        # NOTE: this version does not support pixel-based observations; please refer to the visual-dedicated file.
        # Define actor network.
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        phi_def = ensemblize(MLP, 2)(
            hidden_dims=config['rep_hidden_dims'] + (config['goalrep_dim'],),
            layer_norm=True,
        )
        psi_f_def = ensemblize(MLP, 2)(
            hidden_dims=config['rep_hidden_dims'] + (config['goalrep_dim'],),
            layer_norm=True,
        )
        psi_b_def = ensemblize(MLP, 2)(
            hidden_dims=config['rep_hidden_dims'] + (config['goalrep_dim'],),
            layer_norm=True,
        )

        network_info = dict(
            phi=(phi_def, (ex_observations,)),
            psi_f=(psi_f_def, (jnp.concatenate([ex_goals, ex_actions], axis=-1))),
            psi_b=(psi_b_def, (ex_goals,)),
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
            agent_name='gcfbc_byol',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            discrete=False,  # Whether the action space is discrete.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            flow_steps=10,  # Number of flow steps.
            goalrep_dim=64,  # Dimensionality of the goal representation.
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
