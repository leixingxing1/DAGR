"""
GCIVL State-based VIP Agent with Cross-Attention Enhancement.

For state-based (non-visual) observations.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCDiscreteActor, GCValue, MLP
from utils.cross_attention import StateAwareGoalEncoder


class GCIVLVIPCrossAttnAgent(flax.struct.PyTreeNode):
    """GCIVL agent with VIP goal representation enhanced by Cross-Attention (state-based)."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def rep_loss(self, batch, grad_params):
        """Compute the VIP loss for the goal representation."""
        phi_g = self.network.select('rep')(batch['rep_final_obs'], params=grad_params)

        o_0 = self.network.select('rep')(batch['rep_init_obs'], params=grad_params)
        o_k1 = self.network.select('rep')(batch['rep_k_obs'], params=grad_params)
        o_k2 = self.network.select('rep')(batch['rep_k+1_obs'], params=grad_params)
        v0 = -jnp.sqrt(jnp.maximum(jnp.square(o_0 - phi_g).sum(axis=-1), 1e-6))
        vt1 = -jnp.sqrt(jnp.maximum(jnp.square(o_k1 - phi_g).sum(axis=-1), 1e-6))
        vt2 = -jnp.sqrt(jnp.maximum(jnp.square(o_k2 - phi_g).sum(axis=-1), 1e-6))

        vip_loss = (1 - self.config['discount']) * -v0.mean() + jax.nn.logsumexp(
            vt1 + 1 - self.config['discount'] * vt2
        )

        return vip_loss, {'vip_loss': vip_loss, 'v0': v0.mean(), 'vt1': vt1.mean(), 'vt2': vt2.mean()}

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss with Cross-Attention enhanced goal representation."""
        goal_reps = self.network.select('rep')(batch['value_goals'])
        
        # Apply Cross-Attention
        goal_reps_obs = self.network.select('cross_attention')(goal_reps, batch['observations'])
        goal_reps_next = self.network.select('cross_attention')(goal_reps, batch['next_observations'])

        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], goal_reps_next)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], goal_reps_obs)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        
        grad_goal_reps = self.network.select('rep')(batch['value_goals'])
        grad_goal_reps_aware = self.network.select('cross_attention')(
            grad_goal_reps, batch['observations'], params=grad_params
        )
        
        (v1, v2) = self.network.select('value')(batch['observations'], grad_goal_reps_aware, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss with Cross-Attention enhanced goal representation."""
        goal_reps = self.network.select('rep')(batch['actor_goals'])
        
        goal_reps_obs = self.network.select('cross_attention')(goal_reps, batch['observations'])
        goal_reps_next = self.network.select('cross_attention')(goal_reps, batch['next_observations'])

        v1, v2 = self.network.select('value')(batch['observations'], goal_reps_obs)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goal_reps_next)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        grad_goal_reps = self.network.select('rep')(batch['actor_goals'])
        grad_goal_reps_aware = self.network.select('cross_attention')(
            grad_goal_reps, batch['observations'], params=grad_params
        )
        
        dist = self.network.select('actor')(batch['observations'], grad_goal_reps_aware, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update({
                'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                'std': jnp.mean(dist.scale_diag),
            })

        return actor_loss, actor_info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss + rep_loss
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
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor with state-aware goal representation."""
        goal_reps = self.network.select('rep')(goals)
        goal_reps_aware = self.network.select('cross_attention')(goal_reps, observations)
        
        dist = self.network.select('actor')(observations, goal_reps_aware, temperature=temperature)
        actions = dist.sample(seed=seed)
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
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
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

        rep_def = MLP(
            hidden_dims=(*config['rep_hidden_dims'], config['goalrep_dim']),
            layer_norm=config['layer_norm'],
        )

        cross_attention_def = StateAwareGoalEncoder(
            num_layers=config['cross_attn_layers'],
            num_heads=config['cross_attn_heads'],
            head_dim=config['cross_attn_head_dim'],
            ffn_dim=config['cross_attn_ffn_dim'],
            num_state_tokens=config['cross_attn_state_tokens'],
            dropout_rate=config['cross_attn_dropout'],
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            rep=(rep_def, (ex_observations,)),
            cross_attention=(cross_attention_def, (ex_goals, ex_observations)),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gcivl_vip_crossattn',
            lr=3e-4,
            batch_size=1024,
            rep_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=10.0,
            const_std=True,
            discrete=False,
            goalrep_dim=256,
            cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_head_dim=64,
            cross_attn_ffn_dim=256,
            cross_attn_state_tokens=8,
            cross_attn_dropout=0.0,
            dataset_class='VIPDataset',
            oraclerep=False,
            norm=False,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
