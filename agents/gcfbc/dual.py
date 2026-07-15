import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCValue, ActorVectorField
from utils.dual import DualRepresentationValue


class GCFBCDualAgent(flax.struct.PyTreeNode):
    """Goal-conditioned flow behavioral cloning (GCBC) agent.
    Uses a dual goal representation.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def rep_loss(self, batch, grad_params):
        """Compute the IQL loss for the representation value function.

        The value function is parameterized by the representation, while the critic is a standard, unrestricted (MLP) critic function.
        """

        # Rep value loss.
        q1, q2 = self.network.select('target_rep_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss.
        next_v = self.network.select('rep_value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the BC actor loss."""
        goal_reps = self.network.select('rep_value')(batch['actor_goals'])

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

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = actor_loss + rep_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'rep_critic')

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
        goal_reps = self.network.select('rep_value')(goals)
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

        rep_value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=True,
        )

        rep_critic_def = GCValue(
            hidden_dims=config['rep_hidden_dims'],
            layer_norm=True,
            ensemble=True,
        )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            rep_critic=(rep_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_rep_critic'] = params['modules_rep_critic']
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcfbc_dual',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            rep_hidden_dims=(512, 512, 512),  # Representation network hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            tau=0.005,  # Target network update rate.
            discrete=False,  # Whether the action space is discrete.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            flow_steps=10,  # Number of flow steps.
            rep_expectile=0.7,  # IQL expectile for learning the representation value function.
            goalrep_dim=256,  # Dimensionality of the dual goal representation.
            rep_type='bilinear',  # Parameterization of the dual goal representation (see utils/dual.py).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            oraclerep=False,  # always False; dummy option for compatibility.
            norm=False,  # Whether to use dataset normalization.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
