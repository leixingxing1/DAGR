#!/bin/bash
# =============================================================================
# MS-SAGE Experiments
# Encoder: ORIGINAL (unmodified). Only cross-attention module differs.
# =============================================================================

set -e

STATE_SEEDS=(0 1 2 3 4 5 6 7)
VISUAL_SEEDS=(0 1 2 3)
STATE_TRAIN_STEPS=1000000
VISUAL_TRAIN_STEPS=500000
GPU_ID=2

# MS Cross-Attention Config
CROSS_ATTN_HEADS=4
CROSS_ATTN_HEAD_DIM=64
CROSS_ATTN_FFN_DIM=256
GOALREP_DIM=256
GATE_INIT=-5.0

STATE_ENVS=(
    #"pointmaze-medium-navigate-v0" "pointmaze-large-navigate-v0"
    #"antmaze-medium-navigate-v0" "antmaze-large-navigate-v0" "antmaze-giant-navigate-v0"
    "humanoidmaze-medium-navigate-v0" "humanoidmaze-large-navigate-v0"
    "antsoccer-arena-navigate-v0"
    "cube-single-play-v0" "cube-double-play-v0" "scene-play-v0"
    "puzzle-3x3-play-v0" "puzzle-4x4-play-v0"
)

VISUAL_ENVS=(
    "visual-antmaze-medium-navigate-v0" "visual-antmaze-large-navigate-v0"
    "visual-cube-single-play-v0" "visual-cube-double-play-v0" "visual-scene-play-v0"
    "visual-puzzle-3x3-play-v0" "visual-puzzle-4x4-play-v0"
)

run_ms_state() {
    local env=$1 seed=$2
    echo "[MS-SAGE State] $env seed=$seed"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --env_name=$env \
        --agent=agents/gcivl/state/dual_ms_crossattn.py \
        --seed=$seed \
        --train_steps=$STATE_TRAIN_STEPS \
        --agent.goalrep_dim=$GOALREP_DIM \
        --agent.cross_attn_heads=$CROSS_ATTN_HEADS \
        --agent.cross_attn_head_dim=$CROSS_ATTN_HEAD_DIM \
        --agent.cross_attn_ffn_dim=$CROSS_ATTN_FFN_DIM \
        --agent.cross_attn_gate_init=$GATE_INIT \
        --agent.alpha=10.0 \
        --agent.rep_type=bilinear \
        --agent.rep_expectile=0.9
}

run_ms_visual() {
    local env=$1 seed=$2
    echo "[MS-SAGE Visual] $env seed=$seed"
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
        --env_name=$env \
        --agent=agents/gcivl/pixel/dual_ms_crossattn.py \
        --seed=$seed \
        --train_steps=$VISUAL_TRAIN_STEPS \
        --agent.batch_size=256 \
        --agent.encoder=impala_small \
        --agent.p_aug=0.5 \
        --agent.goalrep_dim=$GOALREP_DIM \
        --agent.cross_attn_heads=$CROSS_ATTN_HEADS \
        --agent.cross_attn_head_dim=$CROSS_ATTN_HEAD_DIM \
        --agent.cross_attn_ffn_dim=$CROSS_ATTN_FFN_DIM \
        --agent.cross_attn_gate_init=$GATE_INIT \
        --agent.alpha=10.0 \
        --agent.rep_type=bilinear \
        --agent.rep_expectile=0.7 \
        --eval_episodes=15
}

case ${1:-"all"} in
    "state")
        for env in "${STATE_ENVS[@]}"; do
            for seed in "${STATE_SEEDS[@]}"; do run_ms_state "$env" "$seed"; done
        done ;;
    "visual")
        for env in "${VISUAL_ENVS[@]}"; do
            for seed in "${VISUAL_SEEDS[@]}"; do run_ms_visual "$env" "$seed"; done
        done ;;
    "puzzle")
        for env in "visual-puzzle-3x3-play-v0" "visual-puzzle-4x4-play-v0"; do
            for seed in "${VISUAL_SEEDS[@]}"; do run_ms_visual "$env" "$seed"; done
        done ;;
    "test")
        CUDA_VISIBLE_DEVICES=$GPU_ID python main.py \
            --env_name=visual-cube-single-play-v0 \
            --agent=agents/gcivl/pixel/dual_ms_crossattn.py \
            --seed=0 --train_steps=10000 \
            --agent.batch_size=64 --agent.encoder=impala_small --agent.p_aug=0.5 \
            --agent.goalrep_dim=$GOALREP_DIM \
            --agent.cross_attn_heads=$CROSS_ATTN_HEADS \
            --agent.cross_attn_head_dim=$CROSS_ATTN_HEAD_DIM \
            --agent.cross_attn_ffn_dim=$CROSS_ATTN_FFN_DIM \
            --agent.cross_attn_gate_init=$GATE_INIT \
            --agent.alpha=10.0 --agent.rep_type=bilinear --agent.rep_expectile=0.7 \
            --eval_episodes=5 --eval_interval=5000 --log_interval=1000 ;;
    "all")
        for env in "${STATE_ENVS[@]}"; do
            for seed in "${STATE_SEEDS[@]}"; do run_ms_state "$env" "$seed"; done
        done
        for env in "${VISUAL_ENVS[@]}"; do
            for seed in "${VISUAL_SEEDS[@]}"; do run_ms_visual "$env" "$seed"; done
        done ;;
    *)
        echo "Usage: $0 [state|visual|puzzle|test|all]"; exit 1 ;;
esac
echo "Done!"
