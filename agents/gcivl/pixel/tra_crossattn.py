"""
GCIVL Visual TRA Agent with Cross-Attention Enhancement.

For pixel-based (visual) observations using IMPALA encoder.
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
from utils.networks import GCActor, GCDiscreteActor, GCValue, GCBilinearValue


# =============================================================================
# Cross-Attention Module (Embedded directly)
# =============================================================================

class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with spatial token decomposition."""
    num_heads: int = 4
    head_dim: int = 64
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, query, key_value, deterministic=True):
        squeeze_output = False
        if query.ndim == 1:
            query = query[None, :]
            squeeze_output = True
        if key_value.ndim == 1:
            key_value = key_value[None, :]
            
        batch_size = query.shape[0]
        d_model = self.num_heads * self.head_dim
        
        state_proj = nn.Dense(self.num_state_tokens * d_model, use_bias=True, name='state_to_tokens')(key_value)
        state_tokens = state_proj.reshape(batch_size, self.num_state_tokens, d_model)
        
        Q = nn.Dense(d_model, use_bias=False, name='query_proj')(query)
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        
        K = nn.Dense(d_model, use_bias=False, name='key_proj')(state_tokens)
        K = K.reshape(batch_size, self.num_state_tokens, self.num_heads, self.head_dim)
        K = jnp.transpose(K, (0, 2, 1, 3))
        
        V = nn.Dense(d_model, use_bias=False, name='value_proj')(state_tokens)
        V = V.reshape(batch_size, self.num_state_tokens, self.num_heads, self.head_dim)
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        scale = jnp.sqrt(self.head_dim).astype(query.dtype)
        attn_weights = jnp.einsum('bhqd,bhtd->bhqt', Q, K) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        attn_output = jnp.einsum('bhqt,bhtd->bhqd', attn_weights, V)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, d_model)
        
        output = nn.Dense(query.shape[-1], use_bias=True, name='output_proj')(attn_output)
        
        if squeeze_output:
            output = output.squeeze(0)
        return output


class CrossAttentionBlock(nn.Module):
    """Cross-Attention block with residual connection and FFN."""
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    layer_norm: bool = True
    
    @nn.compact
    def __call__(self, goal_rep, state_rep, deterministic=True):
        squeeze_output = False
        if goal_rep.ndim == 1:
            goal_rep = goal_rep[None, :]
            squeeze_output = True
        if state_rep.ndim == 1:
            state_rep = state_rep[None, :]
            
        residual = goal_rep
        attn_output = MultiHeadCrossAttention(
            num_heads=self.num_heads, head_dim=self.head_dim,
            num_state_tokens=self.num_state_tokens, dropout_rate=self.dropout_rate, name='cross_attn',
        )(goal_rep, state_rep, deterministic=deterministic)
        
        gate = self.param('attn_gate', nn.initializers.zeros, (goal_rep.shape[-1],))
        goal_rep = residual + jax.nn.sigmoid(gate) * attn_output
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='attn_ln')(goal_rep)
        
        residual = goal_rep
        ffn_output = nn.Dense(self.ffn_dim, name='ffn_up')(goal_rep)
        ffn_output = jax.nn.gelu(ffn_output)
        ffn_output = nn.Dense(goal_rep.shape[-1], name='ffn_down')(ffn_output)
        
        ffn_gate = self.param('ffn_gate', nn.initializers.zeros, (goal_rep.shape[-1],))
        goal_rep = residual + jax.nn.sigmoid(ffn_gate) * ffn_output
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='ffn_ln')(goal_rep)
        
        if squeeze_output:
            goal_rep = goal_rep.squeeze(0)
        return goal_rep


class StateAwareGoalEncoder(nn.Module):
    """Transform goal representation to be state-aware via Cross-Attention."""
    num_layers: int = 1
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    layer_norm: bool = True
    
    @nn.compact
    def __call__(self, goal_rep, state_rep, deterministic=True):
        x = goal_rep
        for i in range(self.num_layers):
            x = CrossAttentionBlock(
                num_heads=self.num_heads, head_dim=self.head_dim,
                ffn_dim=self.ffn_dim, num_state_tokens=self.num_state_tokens,
                dropout_rate=self.dropout_rate, layer_norm=self.layer_norm,
                name=f'cross_attn_block_{i}',
            )(x, state_rep, deterministic=deterministic)
        return x


class GCIVLVisualTRACrossAttnAgent(flax.struct.PyTreeNode):
    """GCIVL Visual agent with TRA goal representation enhanced by Cross-Attention."""
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_rep_loss(self, batch, grad_params):
        """Compute the contrastive value loss for the representation value function."""
        grad_obs = self.network.select('state_encoder')(batch['observations'], params=grad_params)
        grad_goals = self.network.select('goal_encoder')(batch['rep_goals'], params=grad_params)
        obs = jax.lax.stop_gradient(grad_obs)
        goals = jax.lax.stop_gradient(grad_goals)
        
        batch_size = obs.shape[0]

        v, phi, psi = self.network.select('contrastive')(
            grad_obs, grad_goals, actions=None, info=True, params=grad_params
        )
        if len(phi.shape) == 2:
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])

        I = jnp.eye(batch_size)
        contrastive_loss = -(
            jax.nn.log_softmax(logits, axis=0) * I[..., None] + jax.nn.log_softmax(logits, axis=1) * I[..., None]
        )
        contrastive_loss = jnp.mean(contrastive_loss)

        v = jnp.exp(v)
        logits = jnp.mean(logits, axis=-1)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        return contrastive_loss, {
            'contrastive_loss': contrastive_loss,
            'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos, 'logits_neg': logits_neg, 'logits': logits.mean(),
        }

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss with Cross-Attention enhanced goal representation."""
        grad_obs = self.network.select('state_encoder')(batch['observations'], params=grad_params)
        grad_next_obs = self.network.select('state_encoder')(batch['next_observations'], params=grad_params)
        grad_goals = self.network.select('goal_encoder')(batch['value_goals'], params=grad_params)
        obs = jax.lax.stop_gradient(grad_obs)
        next_obs = jax.lax.stop_gradient(grad_next_obs)
        goals = jax.lax.stop_gradient(grad_goals)

        _, _, goal_reps = self.network.select('contrastive')(obs, goals, actions=None, info=True)
        
        goal_reps_obs = self.network.select('cross_attention')(goal_reps, obs)
        goal_reps_next = self.network.select('cross_attention')(goal_reps, next_obs)

        (next_v1_t, next_v2_t) = self.network.select('target_value')(next_obs, goal_reps_next)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(obs, goal_reps_obs)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        
        _, _, grad_goal_reps = self.network.select('contrastive')(grad_obs, grad_goals, actions=None, info=True)
        grad_goal_reps_aware = self.network.select('cross_attention')(
            grad_goal_reps, grad_obs, params=grad_params
        )
        
        (v1, v2) = self.network.select('value')(grad_obs, grad_goal_reps_aware, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss, 'v_mean': v.mean(), 'v_max': v.max(), 'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss with Cross-Attention enhanced goal representation."""
        grad_obs = self.network.select('state_encoder')(batch['observations'], params=grad_params)
        grad_next_obs = self.network.select('state_encoder')(batch['next_observations'], params=grad_params)
        grad_goals = self.network.select('goal_encoder')(batch['actor_goals'], params=grad_params)
        obs = jax.lax.stop_gradient(grad_obs)
        next_obs = jax.lax.stop_gradient(grad_next_obs)
        goals = jax.lax.stop_gradient(grad_goals)

        _, _, goal_reps = self.network.select('contrastive')(obs, goals, actions=None, info=True)
        
        goal_reps_obs = self.network.select('cross_attention')(goal_reps, obs)
        goal_reps_next = self.network.select('cross_attention')(goal_reps, next_obs)

        v1, v2 = self.network.select('value')(obs, goal_reps_obs)
        nv1, nv2 = self.network.select('value')(next_obs, goal_reps_next)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        _, _, grad_goal_reps = self.network.select('contrastive')(grad_obs, grad_goals, actions=None, info=True)
        grad_goal_reps_aware = self.network.select('cross_attention')(
            grad_goal_reps, grad_obs, params=grad_params
        )
        
        dist = self.network.select('actor')(grad_obs, grad_goal_reps_aware, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {'actor_loss': actor_loss, 'adv': adv.mean(), 'bc_log_prob': log_prob.mean()}
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

        contrastive_loss, contrastive_info = self.contrastive_rep_loss(batch, grad_params)
        for k, v in contrastive_info.items():
            info[f'contrastive/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss + contrastive_loss
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

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        observations = self.network.select('state_encoder')(observations)
        goals = self.network.select('goal_encoder')(goals)
        
        _, _, goal_reps = self.network.select('contrastive')(observations, goals, actions=None, info=True)
        goal_reps_aware = self.network.select('cross_attention')(goal_reps, observations)
        
        dist = self.network.select('actor')(observations, goal_reps_aware, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = jnp.zeros(shape=(1, config['goalrep_dim']))
        if config['encoder'] in ['impala_debug', 'impala_small']:
            encoder_output_dim = 512
        else:
            encoder_output_dim = 512
        ex_encoder_outputs = jnp.zeros(shape=(1, encoder_output_dim))
        
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        encoder_module = encoder_modules[config['encoder']]
        state_encoder_def = encoder_module()
        goal_encoder_def = encoder_module()

        value_def = GCValue(hidden_dims=config['value_hidden_dims'], layer_norm=config['layer_norm'], ensemble=True)

        if config['discrete']:
            actor_def = GCDiscreteActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim)
        else:
            actor_def = GCActor(hidden_dims=config['actor_hidden_dims'], action_dim=action_dim,
                               state_dependent_std=False, const_std=config['const_std'])

        contrastive_def = GCBilinearValue(
            hidden_dims=config['rep_hidden_dims'], latent_dim=config['goalrep_dim'],
            layer_norm=config['layer_norm'], ensemble=True, value_exp=True, ret_mean=True,
        )

        cross_attention_def = StateAwareGoalEncoder(
            num_layers=config['cross_attn_layers'], num_heads=config['cross_attn_heads'],
            head_dim=config['cross_attn_head_dim'], ffn_dim=config['cross_attn_ffn_dim'],
            num_state_tokens=config['cross_attn_state_tokens'], dropout_rate=config['cross_attn_dropout'],
            layer_norm=config['layer_norm'],
        )

        network_info = dict(
            state_encoder=(state_encoder_def, (ex_observations,)),
            goal_encoder=(goal_encoder_def, (ex_observations,)),
            contrastive=(contrastive_def, (ex_encoder_outputs, ex_encoder_outputs)),
            cross_attention=(cross_attention_def, (ex_goals, ex_encoder_outputs)),
            value=(value_def, (ex_encoder_outputs, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_encoder_outputs, ex_goals)),
            actor=(actor_def, (ex_encoder_outputs, ex_goals)),
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
            agent_name='gcivl_tra_cross_vis',
            lr=3e-4, batch_size=256,
            rep_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True, discount=0.99, tau=0.005, expectile=0.9, alpha=10.0,
            const_std=True, discrete=False, goalrep_dim=256,
            encoder='impala_small',
            cross_attn_layers=1, cross_attn_heads=4, cross_attn_head_dim=64,
            cross_attn_ffn_dim=256, cross_attn_state_tokens=8, cross_attn_dropout=0.0,
            dataset_class='GCDataset', oraclerep=False, norm=False,
            value_p_curgoal=0.2, value_p_trajgoal=0.5, value_p_randomgoal=0.3, value_geom_sample=True,
            actor_p_curgoal=0.0, actor_p_trajgoal=1.0, actor_p_randomgoal=0.0, actor_geom_sample=False,
            rep_p_curgoal=0.0, rep_p_trajgoal=1.0, rep_p_randomgoal=0.0, rep_geom_sample=True,
            gc_negative=True, p_aug=0.5,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
