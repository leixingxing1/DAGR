"""
GCIVL Visual Dual Agent with SPATIAL Cross-Attention Enhancement.

This is the FIXED version that solves the late fusion problem for puzzle tasks.

Key Changes from Original:
1. Encoder returns BOTH flat output AND spatial tokens
2. Cross-attention operates on ACTUAL spatial tokens (8x8=64 positions)
3. Goal representation can attend to specific spatial positions

Architecture:
    State Image → IMPALA → (flat_repr [B, 512], spatial_tokens [B, 64, D])
    Goal Image → IMPALA → (flat_repr [B, 512], spatial_tokens [B, 64, D])
    
    Dual Repr: φ(g) = DualValue(goal_flat)  [B, goalrep_dim]
    
    State-Aware: φ(g|s) = φ(g) + SpatialCrossAttn(φ(g), state_spatial_tokens)
    
    Policy: π(a|s, φ(g|s))

WHY THIS WORKS FOR PUZZLES:
- Original: Cross-attention on 512-dim flat vector → NO spatial info
- New: Cross-attention on 64 spatial tokens → EACH token = one 8x8 region
- Goal can "look at" specific tile positions to identify misplacements
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
from utils.spatial_cross_attention import StateAwareSpatialGoalEncoder


class GCIVLVisualDualCrossAttnAgent(flax.struct.PyTreeNode):
    """GCIVL Visual agent with Dual goal representation and SPATIAL Cross-Attention.
    
    This agent properly solves the late fusion problem by:
    1. Preserving spatial structure from CNN encoder
    2. Using real spatial tokens for cross-attention
    3. Allowing goal to attend to specific spatial positions in state
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _encode_with_spatial(self, encoder_name, inputs, params=None):
        """Encode inputs and return both flat and spatial representations."""
        return self.network.select(encoder_name)(
            inputs, 
            return_both=True,
            params=params
        )

    def rep_loss(self, batch, grad_params):
        """Compute the IQL loss for the representation value function.
        
        rep_loss uses FLAT representations only - Dual learning doesn't need spatial.
        """
        # Encode with gradients (flat only for rep learning)
        grad_obs_flat, _ = self._encode_with_spatial('state_encoder', batch['observations'], params=grad_params)
        grad_next_obs_flat, _ = self._encode_with_spatial('state_encoder', batch['next_observations'], params=grad_params)
        grad_goals_flat, _ = self._encode_with_spatial('goal_encoder', batch['value_goals'], params=grad_params)
        
        # Stop gradients
        obs_flat = jax.lax.stop_gradient(grad_obs_flat)
        next_obs_flat = jax.lax.stop_gradient(grad_next_obs_flat)
        goals_flat = jax.lax.stop_gradient(grad_goals_flat)

        # Rep value loss
        q1, q2 = self.network.select('target_rep_critic')(obs_flat, goals_flat, batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(grad_obs_flat, grad_goals_flat, params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss
        next_v = self.network.select('rep_value')(next_obs_flat, goals_flat)
        q_target = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('rep_critic')(
            grad_obs_flat, grad_goals_flat, batch['actions'], params=grad_params
        )
        critic_loss = ((q1 - q_target) ** 2 + (q2 - q_target) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q_target.mean(),
            'q_max': q_target.max(),
            'q_min': q_target.min(),
        }

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss with SPATIAL Cross-Attention.
        
        This is where the magic happens for puzzle tasks:
        - Goal representation attends to state's SPATIAL tokens
        - Each spatial token corresponds to a position in the 8x8 grid
        - Cross-attention can focus on specific positions (misplaced tiles)
        """
        # Encode - get BOTH flat and spatial
        grad_obs_flat, grad_obs_spatial = self._encode_with_spatial(
            'state_encoder', batch['observations'], params=grad_params
        )
        grad_next_obs_flat, grad_next_obs_spatial = self._encode_with_spatial(
            'state_encoder', batch['next_observations'], params=grad_params
        )
        grad_goals_flat, _ = self._encode_with_spatial(
            'goal_encoder', batch['value_goals'], params=grad_params
        )
        
        # Stop gradients
        obs_flat = jax.lax.stop_gradient(grad_obs_flat)
        obs_spatial = jax.lax.stop_gradient(grad_obs_spatial)
        next_obs_flat = jax.lax.stop_gradient(grad_next_obs_flat)
        next_obs_spatial = jax.lax.stop_gradient(grad_next_obs_spatial)
        goals_flat = jax.lax.stop_gradient(grad_goals_flat)

        # Get Dual goal representation
        goal_reps = self.network.select('rep_value')(goals_flat)
        
        # ===== KEY: Apply SPATIAL Cross-Attention =====
        # Goal rep attends to state's SPATIAL tokens (real 8x8 positions!)
        goal_reps_obs = self.network.select('spatial_cross_attention')(
            goal_reps, obs_spatial
        )
        goal_reps_next = self.network.select('spatial_cross_attention')(
            goal_reps, next_obs_spatial
        )

        # Value computation
        (next_v1_t, next_v2_t) = self.network.select('target_value')(next_obs_flat, goal_reps_next)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(obs_flat, goal_reps_obs)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        
        # Gradient computation with spatial cross-attention
        grad_goal_reps = self.network.select('rep_value')(grad_goals_flat, params=grad_params)
        grad_goal_reps_aware = self.network.select('spatial_cross_attention')(
            grad_goal_reps, 
            grad_obs_spatial,
            params=grad_params
        )
        
        (v1, v2) = self.network.select('value')(grad_obs_flat, grad_goal_reps_aware, params=grad_params)
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
        """Compute the AWR actor loss with SPATIAL Cross-Attention."""
        # Encode
        grad_obs_flat, grad_obs_spatial = self._encode_with_spatial(
            'state_encoder', batch['observations'], params=grad_params
        )
        grad_next_obs_flat, grad_next_obs_spatial = self._encode_with_spatial(
            'state_encoder', batch['next_observations'], params=grad_params
        )
        grad_goals_flat, _ = self._encode_with_spatial(
            'goal_encoder', batch['actor_goals'], params=grad_params
        )
        
        # Stop gradients
        obs_flat = jax.lax.stop_gradient(grad_obs_flat)
        obs_spatial = jax.lax.stop_gradient(grad_obs_spatial)
        next_obs_flat = jax.lax.stop_gradient(grad_next_obs_flat)
        next_obs_spatial = jax.lax.stop_gradient(grad_next_obs_spatial)
        goals_flat = jax.lax.stop_gradient(grad_goals_flat)

        # Get Dual goal representation and apply SPATIAL cross-attention
        goal_reps = self.network.select('rep_value')(goals_flat)
        goal_reps_obs = self.network.select('spatial_cross_attention')(goal_reps, obs_spatial)
        goal_reps_next = self.network.select('spatial_cross_attention')(goal_reps, next_obs_spatial)

        # Advantage computation
        v1, v2 = self.network.select('value')(obs_flat, goal_reps_obs)
        nv1, nv2 = self.network.select('value')(next_obs_flat, goal_reps_next)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Actor with spatial cross-attention
        grad_goal_reps = self.network.select('rep_value')(grad_goals_flat, params=grad_params)
        grad_goal_reps_aware = self.network.select('spatial_cross_attention')(
            grad_goal_reps, 
            grad_obs_spatial,
            params=grad_params
        )
        
        dist = self.network.select('actor')(grad_obs_flat, grad_goal_reps_aware, params=grad_params)
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
        """Update the agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
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
        """Sample actions with SPATIAL cross-attention."""
        # Get both flat and spatial representations
        obs_flat, obs_spatial = self.network.select('state_encoder')(
            observations, return_both=True
        )
        goals_flat, _ = self.network.select('goal_encoder')(
            goals, return_both=True
        )
        
        # Get Dual goal representation
        goal_reps = self.network.select('rep_value')(goals_flat)
        
        # Apply SPATIAL Cross-Attention - goal "looks at" specific spatial positions
        goal_reps_aware = self.network.select('spatial_cross_attention')(goal_reps, obs_spatial)
        
        dist = self.network.select('actor')(obs_flat, goal_reps_aware, temperature=temperature)
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
        """Create a new agent with spatial cross-attention support."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Example tensors
        ex_goal_reps = jnp.zeros(shape=(1, config['goalrep_dim']))
        
        if config['encoder'] in ['impala_debug', 'impala_small']:
            encoder_output_dim = 512
        elif config['encoder'] == 'impala_large':
            encoder_output_dim = 1024
        else:
            encoder_output_dim = 512
            
        ex_encoder_outputs = jnp.zeros(shape=(1, encoder_output_dim))
        
        # Spatial tokens: 8x8=64 tokens for 64x64 input
        num_spatial_tokens = 64
        spatial_dim = config.get('spatial_dim', 64)
        ex_spatial_tokens = jnp.zeros(shape=(1, num_spatial_tokens, spatial_dim))
        
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Create encoders with spatial support
        encoder_module = encoder_modules[config['encoder']]
        state_encoder_def = encoder_module(spatial_dim=spatial_dim)
        goal_encoder_def = encoder_module(spatial_dim=spatial_dim)

        # Networks
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

        # SPATIAL Cross-Attention module
        spatial_cross_attention_def = StateAwareSpatialGoalEncoder(
            num_layers=config['cross_attn_layers'],
            num_heads=config['cross_attn_heads'],
            head_dim=config['cross_attn_head_dim'],
            ffn_dim=config['cross_attn_ffn_dim'],
            dropout_rate=config.get('cross_attn_dropout', 0.0),
            layer_norm=config['layer_norm'],
            gate_init=config.get('cross_attn_gate_init', -5.0),
        )

        # Define all networks
        networks = dict(
            state_encoder=state_encoder_def,
            goal_encoder=goal_encoder_def,
            rep_value=rep_value_def,
            rep_critic=rep_critic_def,
            target_rep_critic=copy.deepcopy(rep_critic_def),
            spatial_cross_attention=spatial_cross_attention_def,
            value=value_def,
            target_value=copy.deepcopy(value_def),
            actor=actor_def,
        )
        
        # Define initialization arguments
        # Use dict for kwargs, tuple for args
        network_args = dict(
            # Encoders use kwargs with return_both=True to initialize spatial params
            state_encoder={'x': ex_observations, 'return_both': True},
            goal_encoder={'x': ex_observations, 'return_both': True},
            # Other networks use tuple args
            rep_value=(ex_encoder_outputs, ex_encoder_outputs),
            rep_critic=(ex_encoder_outputs, ex_encoder_outputs, ex_actions),
            target_rep_critic=(ex_encoder_outputs, ex_encoder_outputs, ex_actions),
            spatial_cross_attention=(ex_goal_reps, ex_spatial_tokens),
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
            # Agent name
            agent_name='gcivl_dual_crossattn_vis',
            
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
            
            # Encoder
            encoder='impala_small',
            spatial_dim=64,  # Dimension of spatial tokens
            
            # SPATIAL Cross-Attention Config
            cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_head_dim=64,
            cross_attn_ffn_dim=256,
            cross_attn_dropout=0.0,
            cross_attn_gate_init=-5.0,
            
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