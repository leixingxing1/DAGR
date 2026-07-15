"""
Multi-Scale Difference-Aware Cross-Attention (MS-SAGE).

Drop-in replacement for StateAwareGoalEncoder (SAGE).
Works with the ORIGINAL Dual encoder -- NO encoder modifications needed.

=== Interface ===

Identical to StateAwareGoalEncoder:
    __call__(goal_rep, state_rep, [goal_flat], deterministic) -> phi_MS(g|s)

All inputs are FLAT vectors from the unmodified ImpalaEncoder:
    goal_rep:   [B, D_goal]  -- from DualRepValue (typically 256-d)
    state_rep:  [B, D_enc]   -- from ImpalaEncoder (typically 512-d)
    goal_flat:  [B, D_enc]   -- from ImpalaEncoder on goal image (optional)

=== What SAGE does (for reference) ===

    state_flat [B, 512] -> Dense -> [B, 8, 256] pseudo-tokens
    phi(g|s) = phi(g) + sigmoid(gate) * CrossAttn(phi(g), pseudo-tokens)

=== What MS-SAGE does differently ===

1. Multi-scale pseudo-tokens:
    state_flat [B, 512] -> Dense_fine   -> [B, 16, D] (fine-grained, 16 tokens)
    state_flat [B, 512] -> Dense_medium -> [B, 8, D]  (medium, 8 tokens = SAGE)
    state_flat [B, 512] -> Dense_coarse -> [B, 4, D]  (coarse, 4 tokens)

2. Difference-aware attention bias:
    For each level, also project goal_flat to the same token structure:
        goal_flat [B, 512] -> Dense -> [B, T_l, D]
    Compute per-token difference:
        delta^(l) = |state_tokens^(l) - goal_tokens^(l)|  ->  [B, T_l]
    Add bias to attention:
        softmax(QK^T/sqrt(d) + softplus(lambda)*delta^(l))

3. Per-level gated residual + learnable fusion weights.

=== Why this is fair ===

- Encoder: IDENTICAL (original Dual ImpalaEncoder, flat 512-d output only)
- Downstream networks (value, actor, rep_value, rep_critic): IDENTICAL
- Data flow: IDENTICAL (same batch, same encoding)
- Only difference: the cross-attention module's internal structure

=== Why this helps puzzle but preserves other tasks ===

- All gates init to ~0, diff_lambda init to ~0, fusion weights init to equal
  -> at t=0, phi_MS(g|s) ~ phi(g), same starting point as SAGE
- Can degenerate to single-scale (SAGE) by learning fusion_logits -> [0, +inf, 0]
- Difference bias helps puzzle: attend to WHERE state differs from goal
- Multi-scale helps puzzle: different token counts capture different granularities
  of tile-level vs region-level vs global-level differences
"""

import flax.linen as nn
import jax
import jax.numpy as jnp


# ============================================================================
# Core: Difference-Aware Cross-Attention
# ============================================================================

class DiffAwareCrossAttention(nn.Module):
    """Cross-attention from goal (query) to state pseudo-tokens (key/value),
    with optional difference-aware attention bias.

    Standard SAGE:   softmax(QK^T / sqrt(d))             -> attends to similar
    Diff-aware:      softmax(QK^T / sqrt(d) + lambda*delta) -> can also attend to different
    """
    num_heads: int = 4
    head_dim: int = 64
    num_tokens: int = 8
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,            # [B, D_goal] goal representation
        state_rep: jnp.ndarray,         # [B, D_enc] flat state from encoder
        goal_flat: jnp.ndarray = None,  # [B, D_enc] flat goal from encoder (for diff)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Returns attended output [B, D_goal]."""
        squeeze = False
        if query.ndim == 1:
            query = query[None, :]
            squeeze = True
        if state_rep.ndim == 1:
            state_rep = state_rep[None, :]
        if goal_flat is not None and goal_flat.ndim == 1:
            goal_flat = goal_flat[None, :]

        B = query.shape[0]
        d_model = self.num_heads * self.head_dim

        # Project state to pseudo-tokens: [B, D_enc] -> [B, T, d_model]
        state_tokens = nn.Dense(
            self.num_tokens * d_model, use_bias=True, name='state_to_tokens'
        )(state_rep).reshape(B, self.num_tokens, d_model)

        # Compute diff map if goal_flat provided
        diff_map = None
        if goal_flat is not None:
            goal_tokens = nn.Dense(
                self.num_tokens * d_model, use_bias=True, name='goal_to_tokens'
            )(goal_flat).reshape(B, self.num_tokens, d_model)
            # Per-token L2 difference: [B, T]
            raw_diff = jnp.sqrt(
                jnp.sum(jnp.square(state_tokens - goal_tokens), axis=-1) + 1e-6
            )
            # Normalize per sample
            diff_max = jnp.max(raw_diff, axis=-1, keepdims=True) + 1e-6
            diff_map = raw_diff / diff_max

        # Q from goal_rep, K/V from state tokens
        Q = nn.Dense(d_model, use_bias=False, name='q_proj')(query)
        Q = Q.reshape(B, 1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        K = nn.Dense(d_model, use_bias=False, name='k_proj')(state_tokens)
        V = nn.Dense(d_model, use_bias=False, name='v_proj')(state_tokens)
        K = K.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(B, self.num_tokens, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        scale = jnp.sqrt(jnp.float32(self.head_dim))
        attn_logits = jnp.einsum('bhqd,bhtd->bhqt', Q, K) / scale

        # Difference-aware bias
        if diff_map is not None:
            # Init to -5.0 so softplus(-5) ~ 0.007, near-zero bias at start
            diff_lambda = self.param(
                'diff_lambda',
                nn.initializers.constant(-5.0),
                (self.num_heads,)
            )
            # [B, T] -> [B, 1, 1, T] * [1, H, 1, 1]
            bias = diff_map[:, None, None, :] * jax.nn.softplus(diff_lambda)[None, :, None, None]
            attn_logits = attn_logits + bias

        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        if not deterministic and self.dropout_rate > 0:
            attn_weights = nn.Dropout(rate=self.dropout_rate)(
                attn_weights, deterministic=deterministic
            )

        attn_out = jnp.einsum('bhqt,bhtd->bhqd', attn_weights, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, d_model)

        output = nn.Dense(query.shape[-1], use_bias=True, name='out_proj')(attn_out)

        if squeeze:
            output = output.squeeze(0)
        return output


class DiffAwareCrossAttentionBlock(nn.Module):
    """Cross-attention block with gated residual + FFN.

    Gate init to sigmoid(-5) ~ 0.007, so initially phi(g|s) ~ phi(g).
    Identical structure to original SAGE's CrossAttentionBlock.
    """
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    num_tokens: int = 8
    dropout_rate: float = 0.0
    layer_norm: bool = True
    gate_init: float = -5.0

    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,           # [B, D]
        state_rep: jnp.ndarray,           # [B, D_enc]
        goal_flat: jnp.ndarray = None,    # [B, D_enc]
        deterministic: bool = True,
    ) -> jnp.ndarray:
        squeeze = False
        if goal_rep.ndim == 1:
            goal_rep = goal_rep[None, :]
            squeeze = True
        if state_rep.ndim == 1:
            state_rep = state_rep[None, :]
        if goal_flat is not None and goal_flat.ndim == 1:
            goal_flat = goal_flat[None, :]

        D = goal_rep.shape[-1]
        residual = goal_rep

        # Cross-attention
        attn_out = DiffAwareCrossAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_tokens=self.num_tokens,
            dropout_rate=self.dropout_rate,
            name='diff_cross_attn',
        )(goal_rep, state_rep, goal_flat=goal_flat, deterministic=deterministic)

        # Gated residual (identical to SAGE)
        attn_gate = self.param('attn_gate', nn.initializers.constant(self.gate_init), (D,))
        goal_rep = residual + jax.nn.sigmoid(attn_gate) * attn_out
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='attn_ln')(goal_rep)

        # FFN (identical to SAGE)
        residual = goal_rep
        ffn_out = nn.Dense(self.ffn_dim, name='ffn_up')(goal_rep)
        ffn_out = jax.nn.gelu(ffn_out)
        if not deterministic and self.dropout_rate > 0:
            ffn_out = nn.Dropout(rate=self.dropout_rate)(ffn_out, deterministic=deterministic)
        ffn_out = nn.Dense(D, name='ffn_down')(ffn_out)

        ffn_gate = self.param('ffn_gate', nn.initializers.constant(self.gate_init), (D,))
        goal_rep = residual + jax.nn.sigmoid(ffn_gate) * ffn_out
        if self.layer_norm:
            goal_rep = nn.LayerNorm(name='ffn_ln')(goal_rep)

        if squeeze:
            goal_rep = goal_rep.squeeze(0)
        return goal_rep


# ============================================================================
# Main Module: Multi-Scale State-Aware Goal Encoder (MS-SAGE)
# ============================================================================

class MultiScaleStateAwareGoalEncoder(nn.Module):
    """Multi-Scale Difference-Aware State-Aware Goal Encoder.

    Drop-in replacement for StateAwareGoalEncoder.
    Works with FLAT encoder outputs only -- no encoder changes needed.

    SAGE does:
        state_flat -> 8 pseudo-tokens -> CrossAttn(goal_rep, tokens) -> phi(g|s)

    MS-SAGE does:
        state_flat -> 16 tokens (fine)   -> DiffCA(goal_rep, tokens, delta) -> level_0
        state_flat ->  8 tokens (medium) -> DiffCA(goal_rep, tokens, delta) -> level_1
        state_flat ->  4 tokens (coarse) -> DiffCA(goal_rep, tokens, delta) -> level_2
        phi_MS(g|s) = sum softmax(w_l) * level_l

    Args:
        num_heads: Attention heads per level.
        head_dim: Dimension per head.
        ffn_dim: FFN hidden dimension.
        ms_token_counts: Tuple of token counts for each scale level.
        dropout_rate: Dropout rate.
        layer_norm: Whether to use LayerNorm.
        gate_init: Initial value for gating (controls warmup behavior).
    """
    num_heads: int = 4
    head_dim: int = 64
    ffn_dim: int = 256
    ms_token_counts: tuple = (16, 8, 4)  # fine -> coarse
    dropout_rate: float = 0.0
    layer_norm: bool = True
    gate_init: float = -5.0

    @nn.compact
    def __call__(
        self,
        goal_rep: jnp.ndarray,           # [B, D_goal] or [D_goal]
        state_rep: jnp.ndarray,           # [B, D_enc] or [D_enc]
        goal_flat: jnp.ndarray = None,    # [B, D_enc] or [D_enc] (optional, for diff)
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            goal_rep:  phi(g) from DualRepValue [B, D_goal]
            state_rep: flat encoder output for state [B, D_enc]
            goal_flat: flat encoder output for goal [B, D_enc] (optional)
                       If provided, enables difference-aware attention.
                       If None, falls back to standard cross-attention (~ SAGE).
            deterministic: eval mode flag

        Returns:
            phi_MS(g|s) [B, D_goal] -- state-aware goal representation
        """
        squeeze = False
        if goal_rep.ndim == 1:
            goal_rep = goal_rep[None, :]
            squeeze = True
        if state_rep.ndim == 1:
            state_rep = state_rep[None, :]
        if goal_flat is not None and goal_flat.ndim == 1:
            goal_flat = goal_flat[None, :]

        num_levels = len(self.ms_token_counts)
        level_outputs = []

        for ell, num_tokens in enumerate(self.ms_token_counts):
            level_out = DiffAwareCrossAttentionBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                ffn_dim=self.ffn_dim,
                num_tokens=num_tokens,
                dropout_rate=self.dropout_rate,
                layer_norm=self.layer_norm,
                gate_init=self.gate_init,
                name=f'level_{num_tokens}tok',
            )(goal_rep, state_rep, goal_flat=goal_flat, deterministic=deterministic)

            level_outputs.append(level_out)

        # Learnable per-level fusion weights (initialized equal)
        fusion_logits = self.param(
            'fusion_logits',
            nn.initializers.zeros,
            (num_levels,)
        )
        fusion_weights = jax.nn.softmax(fusion_logits)

        # Weighted sum of level outputs
        fused = jnp.zeros_like(goal_rep)
        for ell in range(num_levels):
            fused = fused + fusion_weights[ell] * level_outputs[ell]

        if squeeze:
            fused = fused.squeeze(0)
        return fused
