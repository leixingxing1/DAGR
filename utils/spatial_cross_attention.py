"""
Spatial Cross-Attention Module for State-Aware Goal Representations.

This module implements cross-attention that operates on ACTUAL spatial tokens
from the CNN feature map, not pseudo-tokens from flattened vectors.

Key difference from original implementation:
- Original: state_encoder → 512-dim flat vector → Dense → pseudo-tokens (NO spatial info)
- New: state_encoder → spatial feature map → real spatial tokens (8x8=64 tokens with spatial structure)

This is CRITICAL for puzzle tasks where spatial correspondence matters.
Each token corresponds to a specific 8x8 region of the original image.

Architecture:
    φ(g|s) = φ(g) + SpatialCrossAttn(φ(g), spatial_tokens(s))
"""

from typing import Tuple, Optional, List
import flax.linen as nn
import jax
import jax.numpy as jnp


class SpatialCrossAttention(nn.Module):
    """Cross-attention from goal representation to state spatial tokens.
    
    This allows the goal to "look at" specific spatial positions in the state,
    which is essential for:
    - Puzzle tasks: identify which tiles are misplaced
    - Manipulation: identify object positions
    - Navigation: identify obstacles/paths relative to goal
    """
    num_heads: int = 4
    head_dim: int = 64
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,           # Goal representation: [B, D_q]
        spatial_tokens: jnp.ndarray,  # State spatial tokens: [B, T, D_t]
        deterministic: bool = True,
        return_attention: bool = False,
    ) -> jnp.ndarray:
        """
        Args:
            query: Goal representation [B, D_q] or [D_q]
            spatial_tokens: State spatial tokens [B, T, D_t] or [T, D_t]
            deterministic: Whether to apply dropout
            return_attention: Whether to return attention weights
            
        Returns:
            Attended output [B, D_q] (same shape as query)
            Optional: attention weights [B, H, 1, T]
        """
        # Handle single sample vs batch
        squeeze_output = False
        if query.ndim == 1:
            query = query[None, :]
            squeeze_output = True
        if spatial_tokens.ndim == 2:
            spatial_tokens = spatial_tokens[None, :, :]
            
        batch_size = query.shape[0]
        query_dim = query.shape[-1]
        num_tokens = spatial_tokens.shape[1]
        d_model = self.num_heads * self.head_dim
        
        # Project query: [B, D_q] -> [B, d_model]
        Q = nn.Dense(d_model, use_bias=False, name='query_proj')(query)
        Q = Q.reshape(batch_size, 1, self.num_heads, self.head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))  # [B, H, 1, head_dim]
        
        # Project spatial tokens: [B, T, D_t] -> [B, T, d_model]
        K = nn.Dense(d_model, use_bias=False, name='key_proj')(spatial_tokens)
        V = nn.Dense(d_model, use_bias=False, name='value_proj')(spatial_tokens)
        
        K = K.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = jnp.transpose(K, (0, 2, 1, 3))  # [B, H, T, head_dim]
        
        V = V.reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = jnp.transpose(V, (0, 2, 1, 3))  # [B, H, T, head_dim]
        
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
        
        # Reshape: [B, H, 1, head_dim] -> [B, d_model]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, d_model)
        
        # Output projection back to query dimension
        output = nn.Dense(query_dim, use_bias=True, name='output_proj')(attn_output)
        
        if squeeze_output:
            output = output.squeeze(0)
            if return_attention:
                attn_weights = attn_weights.squeeze(0)
        
        if return_attention:
            return output, attn_weights
        return output


class SpatialCrossAttentionBlock(nn.Module):
    """Complete cross-attention block with gated residual and FFN.
    
    Structure:
    1. Multi-Head Spatial Cross-Attention (goal → state spatial tokens)
    2. Gated Residual Connection (gate initialized near 0)
    3. LayerNorm
    4. Feed-Forward Network
    5. Gated Residual Connection
    6. LayerNorm
    
    The gates start near zero, so initially φ(g|s) ≈ φ(g).
    """
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    dropout_rate: float = 0.0
    layer_norm: bool = True
    gate_init: float = -5.0  # sigmoid(-5) ≈ 0.007
    
    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,        # [B, D] or [D]
        state_spatial: jnp.ndarray,   # [B, T, D_t] or [T, D_t]
        deterministic: bool = True,
        return_attention: bool = False,
    ):
        """Apply spatial cross-attention block."""
        squeeze_output = False
        if goal_rep.ndim == 1:
            goal_rep = goal_rep[None, :]
            squeeze_output = True
        if state_spatial.ndim == 2:
            state_spatial = state_spatial[None, :, :]
        
        rep_dim = goal_rep.shape[-1]
        residual = goal_rep
        
        # Spatial Cross-Attention
        if return_attention:
            attn_output, attn_weights = SpatialCrossAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                name='spatial_cross_attn',
            )(goal_rep, state_spatial, deterministic=deterministic, return_attention=True)
        else:
            attn_output = SpatialCrossAttention(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                name='spatial_cross_attn',
            )(goal_rep, state_spatial, deterministic=deterministic, return_attention=False)
            attn_weights = None
        
        # Gated residual
        attn_gate = self.param(
            'attn_gate',
            nn.initializers.constant(self.gate_init),
            (rep_dim,)
        )
        goal_rep = residual + jax.nn.sigmoid(attn_gate) * attn_output
        
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='attn_ln')(goal_rep)
        
        # FFN
        residual = goal_rep
        ffn_out = nn.Dense(self.ffn_dim, name='ffn_up')(goal_rep)
        ffn_out = jax.nn.gelu(ffn_out)
        if not deterministic and self.dropout_rate > 0:
            ffn_out = nn.Dropout(rate=self.dropout_rate)(
                ffn_out, deterministic=deterministic
            )
        ffn_out = nn.Dense(rep_dim, name='ffn_down')(ffn_out)
        
        # Gated residual for FFN
        ffn_gate = self.param(
            'ffn_gate',
            nn.initializers.constant(self.gate_init),
            (rep_dim,)
        )
        goal_rep = residual + jax.nn.sigmoid(ffn_gate) * ffn_out
        
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='ffn_ln')(goal_rep)
        
        if squeeze_output:
            goal_rep = goal_rep.squeeze(0)
        
        if return_attention:
            return goal_rep, attn_weights
        return goal_rep


class StateAwareSpatialGoalEncoder(nn.Module):
    """Transform goal representation to be state-aware using SPATIAL cross-attention.
    
    This is the KEY module that solves the late fusion problem for puzzle tasks.
    
    Key insight: Instead of cross-attending to pseudo-tokens created from a flat
    512-dim vector, we cross-attend to ACTUAL spatial tokens from the CNN feature map.
    
    For a 64x64 image processed by IMPALA:
    - CNN output: [B, 8, 8, 32] 
    - Spatial tokens: [B, 64, D] where each of 64 tokens corresponds to an 8x8 spatial region
    
    This preserves spatial correspondence that is CRITICAL for puzzle tasks.
    """
    num_layers: int = 1
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    dropout_rate: float = 0.0
    layer_norm: bool = True
    gate_init: float = -5.0
    
    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,        # [B, D] - goal representation
        state_spatial: jnp.ndarray,   # [B, T, D_t] - state spatial tokens
        deterministic: bool = True,
        return_attention: bool = False,
    ):
        """
        Transform goal representation to be state-aware.
        
        Args:
            goal_rep: Base goal representation φ(g) [B, D]
            state_spatial: Spatial tokens from state encoder [B, T, D_t]
            
        Returns:
            State-aware goal representation φ(g|s) [B, D]
        """
        x = goal_rep
        all_attn_weights = []
        
        for i in range(self.num_layers):
            if return_attention:
                x, attn_w = SpatialCrossAttentionBlock(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    ffn_dim=self.ffn_dim,
                    dropout_rate=self.dropout_rate,
                    layer_norm=self.layer_norm,
                    gate_init=self.gate_init,
                    name=f'spatial_cross_attn_block_{i}',
                )(x, state_spatial, deterministic=deterministic, return_attention=True)
                all_attn_weights.append(attn_w)
            else:
                x = SpatialCrossAttentionBlock(
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    ffn_dim=self.ffn_dim,
                    dropout_rate=self.dropout_rate,
                    layer_norm=self.layer_norm,
                    gate_init=self.gate_init,
                    name=f'spatial_cross_attn_block_{i}',
                )(x, state_spatial, deterministic=deterministic, return_attention=False)
        
        if return_attention:
            return x, all_attn_weights
        return x


def visualize_spatial_attention(
    attn_weights: jnp.ndarray,
    spatial_size: Tuple[int, int] = (8, 8),
) -> jnp.ndarray:
    """Reshape attention weights to spatial grid for visualization."""
    h, w = spatial_size
    if attn_weights.ndim == 4:
        return attn_weights.squeeze(2).reshape(
            attn_weights.shape[0], attn_weights.shape[1], h, w
        )
    else:
        return attn_weights.squeeze(1).reshape(-1, h, w)