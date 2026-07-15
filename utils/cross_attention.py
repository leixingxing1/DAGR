"""
Cross-Attention Module for State-Aware Goal Representations.

This module implements the core Cross-Attention mechanism that transforms
state-independent goal representations φ(g) into state-aware representations φ(g|s):

    φ(g|s) = φ(g) + CrossAttn(φ(g), ψ(s))

Key Features:
1. Flash Attention style computation for memory efficiency
2. Gated residual connections (initialized to 0 for stable training)
3. Multi-token state representation for spatial attention
4. Handles both batch and single-sample inference
"""

from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp


class CrossAttention(nn.Module):
    """Basic Cross-Attention layer.
    
    Computes attention from goal (query) to state (key/value):
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        dropout_rate: Dropout rate for attention weights
    """
    num_heads: int = 4
    head_dim: int = 64
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,          # Goal representation: [B, D_g] or [D_g]
        key_value: jnp.ndarray,      # State representation: [B, D_s] or [D_s]
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply cross-attention from goal to state.
        
        Args:
            query: Goal representation
            key_value: State representation (used as both key and value)
            deterministic: Whether to apply dropout
            
        Returns:
            Attended representation with same shape as query
        """
        # Handle single sample vs batch
        squeeze_output = False
        if query.ndim == 1:
            query = query[None, :]
            squeeze_output = True
        if key_value.ndim == 1:
            key_value = key_value[None, :]
            
        batch_size = query.shape[0]
        d_model = self.num_heads * self.head_dim
        
        # Project query, key, value
        Q = nn.Dense(d_model, use_bias=False, name='query_proj')(query)
        K = nn.Dense(d_model, use_bias=False, name='key_proj')(key_value)
        V = nn.Dense(d_model, use_bias=False, name='value_proj')(key_value)
        
        # Reshape for multi-head attention: [B, H, 1, D] for query, [B, H, 1, D] for K/V
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))  # [B, H, 1, D]
        
        K = K.reshape(batch_size, 1, self.num_heads, self.head_dim)
        K = jnp.transpose(K, (0, 2, 1, 3))  # [B, H, 1, D]
        
        V = V.reshape(batch_size, 1, self.num_heads, self.head_dim)
        V = jnp.transpose(V, (0, 2, 1, 3))  # [B, H, 1, D]
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(query.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', Q, K) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        if not deterministic and self.dropout_rate > 0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, V)
        
        # Reshape back: [B, H, 1, D] -> [B, H*D]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, d_model)
        
        # Output projection
        output = nn.Dense(query.shape[-1], use_bias=True, name='output_proj')(attn_output)
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output


class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention with spatial token decomposition.
    
    Decomposes the state representation into multiple tokens to enable
    spatial attention, which is critical for puzzle-like tasks.
    
    Args:
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        num_state_tokens: Number of tokens to decompose state into
        dropout_rate: Dropout rate
    """
    num_heads: int = 4
    head_dim: int = 64
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,          # [B, D_g] or [D_g]
        key_value: jnp.ndarray,      # [B, D_s] or [D_s]
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply multi-head cross-attention with token decomposition."""
        # Handle single sample vs batch
        squeeze_output = False
        if query.ndim == 1:
            query = query[None, :]
            squeeze_output = True
        if key_value.ndim == 1:
            key_value = key_value[None, :]
            
        batch_size = query.shape[0]
        d_model = self.num_heads * self.head_dim
        
        # Project state to tokens: [B, D_s] -> [B, num_tokens, token_dim]
        token_dim = d_model
        state_proj = nn.Dense(
            self.num_state_tokens * token_dim, 
            use_bias=True, 
            name='state_to_tokens'
        )(key_value)
        state_tokens = state_proj.reshape(batch_size, self.num_state_tokens, token_dim)
        
        # Project query for attention
        Q = nn.Dense(d_model, use_bias=False, name='query_proj')(query)
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))  # [B, H, 1, D]
        
        # Project state tokens for keys and values
        K = nn.Dense(d_model, use_bias=False, name='key_proj')(state_tokens)
        K = K.reshape(batch_size, self.num_state_tokens, self.num_heads, self.head_dim)
        K = jnp.transpose(K, (0, 2, 1, 3))  # [B, H, T, D]
        
        V = nn.Dense(d_model, use_bias=False, name='value_proj')(state_tokens)
        V = V.reshape(batch_size, self.num_state_tokens, self.num_heads, self.head_dim)
        V = jnp.transpose(V, (0, 2, 1, 3))  # [B, H, T, D]
        
        # Scaled dot-product attention
        scale = jnp.sqrt(self.head_dim).astype(query.dtype)
        attn_weights = jnp.einsum('bhqd,bhtd->bhqt', Q, K) / scale
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        if not deterministic and self.dropout_rate > 0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )
        
        # Apply attention
        attn_output = jnp.einsum('bhqt,bhtd->bhqd', attn_weights, V)
        
        # Reshape: [B, H, 1, D] -> [B, d_model]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, d_model)
        
        # Output projection
        output = nn.Dense(query.shape[-1], use_bias=True, name='output_proj')(attn_output)
        
        if squeeze_output:
            output = output.squeeze(0)
            
        return output


class CrossAttentionBlock(nn.Module):
    """Full Cross-Attention block with residual connection and FFN.
    
    Structure:
        1. Multi-Head Cross-Attention
        2. Gated Residual Connection (gate initialized to 0)
        3. LayerNorm
        4. Feed-Forward Network
        5. Gated Residual Connection
        6. LayerNorm
    
    The gated residual allows the network to start as identity mapping
    and gradually learn to use the cross-attention signal.
    """
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    layer_norm: bool = True
    
    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,       # [B, D] or [D]
        state_rep: jnp.ndarray,      # [B, D] or [D]
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply cross-attention block."""
        # Handle dimensions
        squeeze_output = False
        if goal_rep.ndim == 1:
            goal_rep = goal_rep[None, :]
            squeeze_output = True
        if state_rep.ndim == 1:
            state_rep = state_rep[None, :]
            
        residual = goal_rep
        
        # Cross-Attention
        attn_output = MultiHeadCrossAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_state_tokens=self.num_state_tokens,
            dropout_rate=self.dropout_rate,
            name='cross_attn',
        )(goal_rep, state_rep, deterministic=deterministic)
        
        # Gated residual (gate initialized to 0 for stable training)
        gate = self.param(
            'attn_gate',
            nn.initializers.zeros,
            (goal_rep.shape[-1],)
        )
        goal_rep = residual + jax.nn.sigmoid(gate) * attn_output
        
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='attn_ln')(goal_rep)
        
        # FFN
        residual = goal_rep
        ffn_output = nn.Dense(self.ffn_dim, name='ffn_up')(goal_rep)
        ffn_output = jax.nn.gelu(ffn_output)
        if not deterministic and self.dropout_rate > 0:
            ffn_output = nn.Dropout(rate=self.dropout_rate)(
                ffn_output, deterministic=deterministic
            )
        ffn_output = nn.Dense(goal_rep.shape[-1], name='ffn_down')(ffn_output)
        
        # Gated residual for FFN
        ffn_gate = self.param(
            'ffn_gate',
            nn.initializers.zeros,
            (goal_rep.shape[-1],)
        )
        goal_rep = residual + jax.nn.sigmoid(ffn_gate) * ffn_output
        
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='ffn_ln')(goal_rep)
        
        if squeeze_output:
            goal_rep = goal_rep.squeeze(0)
            
        return goal_rep


class StateAwareGoalEncoder(nn.Module):
    """Main module: Transform goal representation to be state-aware.
    
    Implements: φ(g|s) = φ(g) + CrossAttn(φ(g), ψ(s))
    
    This is the key component for solving the Late Fusion problem in
    visual goal-conditioned RL.
    
    Args:
        num_layers: Number of cross-attention layers
        num_heads: Number of attention heads
        head_dim: Dimension of each head
        ffn_dim: FFN hidden dimension
        num_state_tokens: Number of tokens for state decomposition
        dropout_rate: Dropout rate
        layer_norm: Whether to use layer normalization
    """
    num_layers: int = 1
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    num_state_tokens: int = 8
    dropout_rate: float = 0.0
    layer_norm: bool = True
    
    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,       # [B, D] or [D]
        state_rep: jnp.ndarray,      # [B, D] or [D]
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Transform goal representation to be state-aware.
        
        Args:
            goal_rep: Base goal representation φ(g)
            state_rep: State representation ψ(s)
            deterministic: Whether in evaluation mode
            
        Returns:
            State-aware goal representation φ(g|s)
        """
        x = goal_rep
        
        for i in range(self.num_layers):
            x = CrossAttentionBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                ffn_dim=self.ffn_dim,
                num_state_tokens=self.num_state_tokens,
                dropout_rate=self.dropout_rate,
                layer_norm=self.layer_norm,
                name=f'cross_attn_block_{i}',
            )(x, state_rep, deterministic=deterministic)
        
        return x


# =============================================================================
# Goal-Conditioned Networks with Cross-Attention
# =============================================================================

class CrossAttentionGCValue(nn.Module):
    """Goal-Conditioned Value function with Cross-Attention.
    
    V(s, g) where g = φ(g|s) is the state-aware goal representation.
    """
    hidden_dims: tuple = (512, 512, 512)
    cross_attn_heads: int = 4
    cross_attn_head_dim: int = 64
    cross_attn_state_tokens: int = 8
    layer_norm: bool = True
    ensemble: bool = True
    
    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Make goal state-aware
        goal_aware = StateAwareGoalEncoder(
            num_layers=1,
            num_heads=self.cross_attn_heads,
            head_dim=self.cross_attn_head_dim,
            num_state_tokens=self.cross_attn_state_tokens,
            layer_norm=self.layer_norm,
            name='state_aware_encoder',
        )(goals, observations)
        
        # Concatenate inputs
        if actions is None:
            x = jnp.concatenate([observations, goal_aware], axis=-1)
        else:
            x = jnp.concatenate([observations, goal_aware, actions], axis=-1)
        
        # MLP
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'fc_{i}')(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = jax.nn.relu(x)
        
        # Output
        if self.ensemble:
            v1 = nn.Dense(1, name='value_1')(x).squeeze(-1)
            v2 = nn.Dense(1, name='value_2')(x).squeeze(-1)
            return v1, v2
        else:
            return nn.Dense(1, name='value')(x).squeeze(-1)


class CrossAttentionGCActor(nn.Module):
    """Goal-Conditioned Actor with Cross-Attention."""
    hidden_dims: tuple = (512, 512, 512)
    action_dim: int = 2
    cross_attn_heads: int = 4
    cross_attn_head_dim: int = 64
    cross_attn_state_tokens: int = 8
    layer_norm: bool = True
    const_std: bool = True
    
    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        temperature: float = 1.0,
    ):
        from utils.networks import TanhNormal
        
        # Make goal state-aware
        goal_aware = StateAwareGoalEncoder(
            num_layers=1,
            num_heads=self.cross_attn_heads,
            head_dim=self.cross_attn_head_dim,
            num_state_tokens=self.cross_attn_state_tokens,
            layer_norm=self.layer_norm,
            name='state_aware_encoder',
        )(goals, observations)
        
        x = jnp.concatenate([observations, goal_aware], axis=-1)
        
        for i, dim in enumerate(self.hidden_dims):
            x = nn.Dense(dim, name=f'fc_{i}')(x)
            if self.layer_norm:
                x = nn.LayerNorm(name=f'ln_{i}')(x)
            x = jax.nn.relu(x)
        
        mean = nn.Dense(self.action_dim, name='mean')(x)
        
        if self.const_std:
            log_std = self.param(
                'log_std',
                nn.initializers.zeros,
                (self.action_dim,)
            )
        else:
            log_std = nn.Dense(self.action_dim, name='log_std')(x)
        
        log_std = jnp.clip(log_std, -5.0, 2.0)
        std = jnp.exp(log_std) * temperature
        
        return TanhNormal(mean, std)


class CrossAttentionGCBilinearValue(nn.Module):
    """Goal-Conditioned Bilinear Value for contrastive learning with Cross-Attention."""
    hidden_dims: tuple = (512, 512, 512)
    latent_dim: int = 256
    cross_attn_heads: int = 4
    cross_attn_head_dim: int = 64
    cross_attn_state_tokens: int = 8
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    ret_mean: bool = False
    
    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        actions: Optional[jnp.ndarray] = None,
        info: bool = False,
    ):
        # State encoder
        state_x = observations
        for i, dim in enumerate(self.hidden_dims):
            state_x = nn.Dense(dim, name=f'state_fc_{i}')(state_x)
            if self.layer_norm:
                state_x = nn.LayerNorm(name=f'state_ln_{i}')(state_x)
            state_x = jax.nn.relu(state_x)
        
        phi = nn.Dense(self.latent_dim, name='state_out')(state_x)
        if self.layer_norm:
            phi = nn.LayerNorm(name='state_out_ln')(phi)
        
        # Goal encoder
        goal_x = goals
        for i, dim in enumerate(self.hidden_dims):
            goal_x = nn.Dense(dim, name=f'goal_fc_{i}')(goal_x)
            if self.layer_norm:
                goal_x = nn.LayerNorm(name=f'goal_ln_{i}')(goal_x)
            goal_x = jax.nn.relu(goal_x)
        
        psi = nn.Dense(self.latent_dim, name='goal_out')(goal_x)
        if self.layer_norm:
            psi = nn.LayerNorm(name='goal_out_ln')(psi)
        
        # Apply Cross-Attention to goal representation
        psi_aware = StateAwareGoalEncoder(
            num_layers=1,
            num_heads=self.cross_attn_heads,
            head_dim=self.cross_attn_head_dim,
            num_state_tokens=self.cross_attn_state_tokens,
            layer_norm=self.layer_norm,
            name='state_aware_encoder',
        )(psi, phi)
        
        # Bilinear value
        v = jnp.sum(phi * psi_aware, axis=-1)
        
        if self.value_exp:
            v = jnp.exp(v)
        
        if info:
            return v, phi, psi_aware
        return v
