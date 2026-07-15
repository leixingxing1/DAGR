"""
GCIVL State-based Dual Agent with Multi-Scale Difference-Aware Cross-Attention.

For state-based tasks, observations are already flat vectors.
This agent is identical to the state-based SAGE (dual_crossattn.py state version)
except the cross-attention module is MultiScaleStateAwareGoalEncoder instead of
StateAwareGoalEncoder.

Key difference from SAGE:
    SAGE:    state_flat → 8 pseudo-tokens → CrossAttn(goal_rep, tokens)
    MS-SAGE: state_flat → {16, 8, 4} pseudo-tokens → DiffCA(goal_rep, tokens, Δ)

For state-based tasks, goal_flat = batch['observations'] at goal indices,
so diff-aware attention can still identify which dimensions differ between
current state and goal state.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue
from utils.ms_cross_attention import MultiScaleStateAwareGoalEncoder


class GCIVLDualMSCrossAttnAgent(flax.struct.PyTreeNode):
    """GCIVL state-based agent: Dual + MS-SAGE cross-attention."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def rep_loss(self, batch, grad_params):
        q1, q2 = self.network.select('target_rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        next_v = self.network.select('rep_value')(batch['next_observations'], batch['value_goals'])
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1_pred, q2_pred = self.network.select('rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1_pred - q_target) ** 2 + (q2_pred - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q_target.mean(),
        }

    def value_loss(self, batch, grad_params):
        goal_reps = self.network.select('rep_value')(batch['value_goals'])

        # MS Cross-Attention with diff: pass both state obs and goal obs
        goal_reps_obs = self.network.select('ms_cross_attention')(
            goal_reps, batch['observations'], batch['value_goals']
        )
        goal_reps_next = self.network.select('ms_cross_attention')(
            goal_reps, batch['next_observations'], batch['value_goals']
        )

        (nv1_t, nv2_t) = self.network.select('target_value')(batch['next_observations'], goal_reps_next)
        next_v_t = jnp.minimum(nv1_t, nv2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], goal_reps_obs)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2_t

        grad_goal_reps = self.network.select('rep_value')(batch['value_goals'], params=grad_params)
        grad_goal_reps_aware = self.network.select('ms_cross_attention')(
            grad_goal_reps, batch['observations'], batch['value_goals'], params=grad_params
        )
        (v1, v2) = self.network.select('value')(
            batch['observations'], grad_goal_reps_aware, params=grad_params
        )
        v = (v1 + v2) / 2

        vl1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        vl2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = vl1 + vl2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        goal_reps = self.network.select('rep_value')(batch['actor_goals'])
        goal_reps_obs = self.network.select('ms_cross_attention')(
            goal_reps, batch['observations'], batch['actor_goals']
        )
        goal_reps_next = self.network.select('ms_cross_attention')(
            goal_reps, batch['next_observations'], batch['actor_goals']
        )

        v1, v2 = self.network.select('value')(batch['observations'], goal_reps_obs)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goal_reps_next)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        grad_goal_reps = self.network.select('rep_value')(batch['actor_goals'], params=grad_params)
        grad_goal_reps_aware = self.network.select('ms_cross_attention')(
            grad_goal_reps, batch['observations'], batch['actor_goals'], params=grad_params
        )
        dist = self.network.select('actor')(
            batch['observations'], grad_goal_reps_aware, params=grad_params
        )
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
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)
        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'rep_critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        goal_reps = self.network.select('rep_value')(goals)
        goal_reps_aware = self.network.select('ms_cross_attention')(
            goal_reps, observations, goals
        )
        dist = self.network.select('actor')(observations, goal_reps_aware, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goal_reps = jnp.zeros(shape=(1, config['goalrep_dim']))
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

        rep_value_def = DualRepresentationValue(type=config['rep_type'])(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'],
        )
        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
        )

        ms_token_counts = tuple(config.get('ms_token_counts', (16, 8, 4)))
        ms_cross_attention_def = MultiScaleStateAwareGoalEncoder(
            num_heads=config['cross_attn_heads'],
            head_dim=config['cross_attn_head_dim'],
            ffn_dim=config['cross_attn_ffn_dim'],
            ms_token_counts=ms_token_counts,
            dropout_rate=config.get('cross_attn_dropout', 0.0),
            layer_norm=config['layer_norm'],
            gate_init=config.get('cross_attn_gate_init', -5.0),
        )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_observations)),
            rep_critic=(rep_critic_def, (ex_observations, ex_observations, ex_actions)),
            target_rep_critic=(copy.deepcopy(rep_critic_def), (ex_observations, ex_observations, ex_actions)),
            ms_cross_attention=(ms_cross_attention_def, (ex_goal_reps, ex_observations, ex_observations)),
            value=(value_def, (ex_observations, ex_goal_reps)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goal_reps)),
            actor=(actor_def, (ex_observations, ex_goal_reps)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='gcivl_dual_ms_crossattn',
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
            rep_expectile=0.9,
            goalrep_dim=256,
            rep_type='bilinear',
            # MS Cross-Attention
            cross_attn_heads=4,
            cross_attn_head_dim=64,
            cross_attn_ffn_dim=256,
            cross_attn_dropout=0.0,
            cross_attn_gate_init=-5.0,
            ms_token_counts=(16, 8, 4),
            # Dataset
            dataset_class='GCDataset',
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
