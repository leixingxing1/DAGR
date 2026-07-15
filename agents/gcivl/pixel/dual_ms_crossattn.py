"""
GCIVL Visual Dual Agent with Multi-Scale Difference-Aware Cross-Attention (MS-SAGE).

=== Fair Comparison Guarantee ===

vs. Dual (baseline):
    - Encoder: IDENTICAL (original ImpalaEncoder, flat 512-d output)
    - Networks: IDENTICAL (same rep_value, value, actor, rep_critic)
    - Data: IDENTICAL (same GCDataset sampling)
    - Difference: adds cross_attention module between rep_value and value/actor

vs. SAGE:
    - Encoder: IDENTICAL (same ImpalaEncoder, same flat 512-d output)
    - Networks: IDENTICAL (same value, actor, rep_value, rep_critic)
    - Data: IDENTICAL
    - Difference: cross_attention internals only
        SAGE:    state_flat → 8 pseudo-tokens → CrossAttn(goal_rep, tokens)
        MS-SAGE: state_flat → {16, 8, 4} pseudo-tokens → DiffCA(goal_rep, tokens, Δ)

Note: SAGE's pixel version (dual_crossattn.py) modified ImpalaEncoder to add
spatial token output. MS-SAGE does NOT — it uses the original encoder as-is.
This is actually MORE fair than SAGE's comparison against Dual.
"""

import copy
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dual import DualRepresentationValue
from utils.networks import GCActor, GCDiscreteActor, GCValue
from utils.ms_cross_attention import MultiScaleStateAwareGoalEncoder


class GCIVLVisualDualMSCrossAttnAgent(flax.struct.PyTreeNode):
    """GCIVL Visual agent: Dual goal representation + MS-SAGE cross-attention.

    Uses original ImpalaEncoder (flat only). No encoder modifications.
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def _encode(self, encoder_name, inputs, params=None):
        """Encode inputs → flat vector. Uses original ImpalaEncoder."""
        return self.network.select(encoder_name)(inputs, params=params)

    def rep_loss(self, batch, grad_params):
        """Representation learning loss — identical to Dual, no cross-attention."""
        grad_obs = self._encode('state_encoder', batch['observations'], params=grad_params)
        grad_next_obs = self._encode('state_encoder', batch['next_observations'], params=grad_params)
        grad_goals = self._encode('goal_encoder', batch['value_goals'], params=grad_params)

        obs = jax.lax.stop_gradient(grad_obs)
        next_obs = jax.lax.stop_gradient(grad_next_obs)
        goals = jax.lax.stop_gradient(grad_goals)

        # Rep value loss
        q1, q2 = self.network.select('target_rep_critic')(obs, goals, batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(grad_obs, grad_goals, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss
        next_v = self.network.select('rep_value')(next_obs, goals)
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1_pred, q2_pred = self.network.select('rep_critic')(
            grad_obs, grad_goals, batch['actions'], params=grad_params
        )
        critic_loss = ((q1_pred - q_target) ** 2 + (q2_pred - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q_target.mean(),
        }

    def value_loss(self, batch, grad_params):
        """IVL value loss with MS cross-attention enhanced goal reps."""
        # Encode all (flat vectors from original encoder)
        grad_obs = self._encode('state_encoder', batch['observations'], params=grad_params)
        grad_next_obs = self._encode('state_encoder', batch['next_observations'], params=grad_params)
        grad_goals = self._encode('goal_encoder', batch['value_goals'], params=grad_params)

        obs = jax.lax.stop_gradient(grad_obs)
        next_obs = jax.lax.stop_gradient(grad_next_obs)
        goals = jax.lax.stop_gradient(grad_goals)

        # Dual goal representation
        goal_reps = self.network.select('rep_value')(goals)

        # MS Cross-Attention: φ_MS(g|s) with diff-aware bias
        # Pass both state_flat and goal_flat for difference computation
        goal_reps_obs = self.network.select('ms_cross_attention')(
            goal_reps, obs, goals
        )
        goal_reps_next = self.network.select('ms_cross_attention')(
            goal_reps, next_obs, goals
        )

        (nv1_t, nv2_t) = self.network.select('target_value')(next_obs, goal_reps_next)
        next_v_t = jnp.minimum(nv1_t, nv2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(obs, goal_reps_obs)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * nv2_t

        # Gradient path
        grad_goal_reps = self.network.select('rep_value')(grad_goals, params=grad_params)
        grad_goal_reps_aware = self.network.select('ms_cross_attention')(
            grad_goal_reps, grad_obs, grad_goals, params=grad_params
        )

        (v1, v2) = self.network.select('value')(grad_obs, grad_goal_reps_aware, params=grad_params)
        v = (v1 + v2) / 2

        vl1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        vl2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = vl1 + vl2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """AWR actor loss with MS cross-attention."""
        grad_obs = self._encode('state_encoder', batch['observations'], params=grad_params)
        grad_next_obs = self._encode('state_encoder', batch['next_observations'], params=grad_params)
        grad_goals = self._encode('goal_encoder', batch['actor_goals'], params=grad_params)

        obs = jax.lax.stop_gradient(grad_obs)
        next_obs = jax.lax.stop_gradient(grad_next_obs)
        goals = jax.lax.stop_gradient(grad_goals)

        goal_reps = self.network.select('rep_value')(goals)
        goal_reps_obs = self.network.select('ms_cross_attention')(goal_reps, obs, goals)
        goal_reps_next = self.network.select('ms_cross_attention')(goal_reps, next_obs, goals)

        v1, v2 = self.network.select('value')(obs, goal_reps_obs)
        nv1, nv2 = self.network.select('value')(next_obs, goal_reps_next)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        grad_goal_reps = self.network.select('rep_value')(grad_goals, params=grad_params)
        grad_goal_reps_aware = self.network.select('ms_cross_attention')(
            grad_goal_reps, grad_obs, grad_goals, params=grad_params
        )

        dist = self.network.select('actor')(grad_obs, grad_goal_reps_aware, params=grad_params)
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
        """Sample actions at inference time."""
        obs_flat = self.network.select('state_encoder')(observations)
        goals_flat = self.network.select('goal_encoder')(goals)
        goal_reps = self.network.select('rep_value')(goals_flat)
        goal_reps_aware = self.network.select('ms_cross_attention')(
            goal_reps, obs_flat, goals_flat
        )
        dist = self.network.select('actor')(obs_flat, goal_reps_aware, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goal_reps = jnp.zeros(shape=(1, config['goalrep_dim']))

        # Encoder output dim (from original ImpalaEncoder)
        if config['encoder'] in ['impala_debug']:
            encoder_output_dim = 512  # depends on stack_sizes, check if different
        elif config['encoder'] == 'impala_large':
            encoder_output_dim = 1024
        else:
            encoder_output_dim = 512

        ex_encoder_outputs = jnp.zeros(shape=(1, encoder_output_dim))

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # === Encoder: ORIGINAL, UNMODIFIED ===
        encoder_module = encoder_modules[config['encoder']]
        state_encoder_def = encoder_module(layer_norm=config['layer_norm'])
        goal_encoder_def = encoder_module(layer_norm=config['layer_norm'])

        # === Downstream networks: IDENTICAL to Dual/SAGE ===
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

        # === THE ONLY NEW MODULE: MS Cross-Attention ===
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

        # === Network assembly ===
        networks = dict(
            state_encoder=state_encoder_def,
            goal_encoder=goal_encoder_def,
            rep_value=rep_value_def,
            rep_critic=rep_critic_def,
            target_rep_critic=copy.deepcopy(rep_critic_def),
            ms_cross_attention=ms_cross_attention_def,
            value=value_def,
            target_value=copy.deepcopy(value_def),
            actor=actor_def,
        )

        network_args = dict(
            state_encoder=(ex_observations,),
            goal_encoder=(ex_observations,),
            rep_value=(ex_encoder_outputs, ex_encoder_outputs),
            rep_critic=(ex_encoder_outputs, ex_encoder_outputs, ex_actions),
            target_rep_critic=(ex_encoder_outputs, ex_encoder_outputs, ex_actions),
            ms_cross_attention=(ex_goal_reps, ex_encoder_outputs, ex_encoder_outputs),
            value=(ex_encoder_outputs, ex_goal_reps),
            target_value=(ex_encoder_outputs, ex_goal_reps),
            actor=(ex_encoder_outputs, ex_goal_reps),
        )

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
            # Agent
            agent_name='gcivl_dual_ms_crossattn_vis',
            # Learning
            lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=10.0,
            # Networks
            rep_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            const_std=True,
            discrete=False,
            # Dual representation
            rep_expectile=0.7,
            goalrep_dim=256,
            rep_type='bilinear',
            # Encoder (ORIGINAL, UNMODIFIED)
            encoder='impala_small',
            # MS Cross-Attention (THE ONLY NEW THING)
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
            p_aug=0.5,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
