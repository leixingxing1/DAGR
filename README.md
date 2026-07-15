# DAGR: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention

Official implementation of **DAGR** (Difference-Aware Goal Representations), a plug-in module that
refines a *state-independent* goal embedding `φ(g)` into a *state-conditioned* one `φ(g | s)` through
**multi-scale gated cross-attention** with a learnable **difference-aware bias**.

DAGR is layered on top of the **Dual** goal representation and trained end-to-end with **GCIVL** as the
downstream offline GCRL algorithm, and is evaluated on **[OGBench](https://github.com/seohongpark/ogbench)**.

> **TL;DR** — Late-fusion goal encoders never see the current state, so `φ(g)` gives every state the same
> hint ("goal is over there"). DAGR turns it into a per-state hint ("go this way next") by cross-attending
> the goal query to state pseudo-tokens, biased by a per-token state–goal discrepancy map. A gated residual
> initialized near the identity keeps the base representation intact at the start of training and opens only
> where it helps.

---

## Table of Contents

- [Key Idea](#key-idea)
- [What DAGR is (and is not)](#what-dagr-is-and-is-not)
- [Repository Structure](#repository-structure)
- [Code ↔ Paper Naming](#code--paper-naming)
- [Installation](#installation)
- [Dataset Setup (read this first)](#dataset-setup-read-this-first)
- [Running Experiments](#running-experiments)
- [Aggregating Results](#aggregating-results)
- [Main Hyperparameters](#main-hyperparameters)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Key Idea

The Dual representation characterizes a goal `g` by the set of optimal temporal distances from every state:
`φ∨(g)(s') = d*(s', g)`, which depends on `g` alone. DAGR augments this with a state-dependent weighting:

```
φ∨_DAGR(g | s)(s') = d*(s', g) · Δ_{s,g}(s')
```

where `Δ_{s,g} ∈ [0, 1]` is large where `s'` is informative about the mismatch between `s` and `g`, and
small where it is not. In continuous environments, `Δ_{s,g}` is approximated by the per-token difference
map of **Multi-Scale Difference-Aware Goal Cross-Attention (MS-DGCA)**.

**Data flow (late fusion → state-conditioned refinement):**

```
   ψ(s)              φ(g)                g
     │                 │                 │
     │        ┌────────┴───────┐         │
     │        │  Dual Goal Rep │         │
     │        └────────┬───────┘         │
     │                 │  φ(g)           │
     │     ┌───────────▼───────────┐     │
     └────►│   MS-DGCA (this work) │◄────┘   goal-image encoding ψ_g(g), for the difference map
           │  fine T=16 / med T=8  │
           │  / coarse T=4         │
           └───────────┬───────────┘
                       │  φ(g | s)
              ┌────────▼────────┐
              │  Concat [s ; ·] │ ──► Policy π(a | h)  and  Value V(h)
              └─────────────────┘
```

Each scale level runs one **Difference-Aware Goal Cross-Attention** block:

1. **Token projection** — `ψ(s)` and `ψ_g(g)` are projected into `T` aligned pseudo-tokens.
2. **Difference map** — a normalized per-token L2 discrepancy `Δ_t ∈ [0, 1]`.
3. **DGCA attention** — `softmax(QKᵀ/√d + ζ(λ)·Δ) V`, with `ζ = softplus` and a per-head learnable `λ`
   (init `λ₀ = −5`, so the bias is ~0 at initialization).
4. **Gated residual + FFN** — vector gates `σ(α)` init at `α₀ = −5` (`σ(−5) ≈ 0.0067`), so
   `φ(g | s) ≈ φ(g)` at step 0.

The three scale outputs are combined with **learnable fusion weights** (`softmax(w)`, `w₀ = 0`, i.e. uniform
at the start).

---

## What DAGR is (and is not)

DAGR is a **structured refinement, not a universal improvement.** The paper is deliberate about the regime
where it helps and the regime where it does not:

- ✅ **Navigation** (position offset determines the action): DAGR is best or tied-best on every state-based
  navigation task except `pointmaze-large`, with the largest gains where Dual is weakest. On visual
  navigation it is the strongest of the six goal-representation baselines compared.
- ➖ **Manipulation / puzzles**: on `cube-double` and `scene` (tasks whose optimal action does *not* factor
  through a state–goal difference map) DAGR matches or falls below the base. Visual-Puzzle stays at zero for
  **every** late-fusion method — an *encoder-level* bottleneck (the IMPALA CNN pools away pixel-level
  correspondence), not a goal-representation one.
- 🔬 **Honest ablation**: the gains come from the **gated residual**, *not* from the difference bias the
  method is named after. The learned `ζ(λ)` stays near its initialization on 5 of 6 tasks, so the difference
  bias acts as an **inductive nudge at initialization** rather than a converged contribution. This is
  reported as a finding, not hidden.

---

## Repository Structure

The layout below follows the import paths in `main.py` and the `--agent=...` flags in the launcher script.

```
.
├── main.py                              # Training entry point (agent-agnostic)
├── hyperparameters_ms_crossattn.sh      # Experiment launcher (state / visual / puzzle / test / all)
├── attention_results.py                 # Aggregate eval CSVs across seeds/envs (OGBench-style, mean ± std)
│
├── agents/
│   └── gcivl/
│       ├── state/                       # State-based (flat-vector) agents
│       │   ├── dual.py                  # GCIVLDualAgent               — Dual baseline
│       │   ├── dual_crossattn.py        # GCIVLDualCrossAttnAgent      — +CA (standard cross-attention)
│       │   └── dual_ms_crossattn.py     # GCIVLDualMSCrossAttnAgent    — DAGR (ours)
│       └── pixel/                       # Visual (image) agents — same three variants + other baselines
│           ├── dual.py                  # GCIVLVisualDualAgent
│           ├── dual_crossattn.py        # GCIVLVisualDualCrossAttnAgent
│           ├── dual_ms_crossattn.py     # GCIVLVisualDualMSCrossAttnAgent — DAGR (visual)
│           ├── vib.py / tra.py / byol.py / vip.py           # other rep-learning baselines
│           └── *_crossattn.py                                # their cross-attention variants
│
└── utils/
    ├── datasets.py                      # Dataset, GCDataset, HGCDataset, VIPDataset (goal sampling, aug)
    ├── dual.py                          # DualRepresentationValue (bilinear / hilbert / mrn / iqe / asymmetric)
    ├── networks.py                      # MLP, GCActor, GCDiscreteActor, GCValue, GCBilinearValue, ... *
    ├── cross_attention.py               # StateAwareGoalEncoder (SAGE)      — the +CA module
    ├── ms_cross_attention.py            # MultiScaleStateAwareGoalEncoder   — the DAGR / MS-DGCA module ★
    ├── spatial_cross_attention.py       # SpatialCrossAttention (experimental spatial-token variant)
    ├── encoders.py                      # ImpalaEncoder, GCEncoder, encoder_modules
    ├── vib.py                           # Variational information bottleneck layer
    ├── env_utils.py                     # make_env_and_datasets, EpisodeMonitor, FrameStackWrapper
    ├── evaluation.py                    # evaluate()
    ├── flax_utils.py                    # TrainState, ModuleDict, save/restore
    └── log_utils.py                     # CsvLogger, wandb helpers, video utils
```

`★` The core contribution.  `*` = `utils/networks.py` is inherited from the base Dual / OGBench code and is a
required dependency of every agent.

---

## Code ↔ Paper Naming

The code was written under the working name **MS-SAGE** ("Multi-Scale State-Aware Goal Encoder"); the paper
calls the method **DAGR** and the module **MS-DGCA**. They are the same thing.

| Paper term                          | Code symbol / agent name                              | File |
|-------------------------------------|-------------------------------------------------------|------|
| DAGR module (MS-DGCA)               | `MultiScaleStateAwareGoalEncoder`                     | `utils/ms_cross_attention.py` |
| Single-scale DGCA block             | `DiffAwareCrossAttentionBlock` / `DiffAwareCrossAttention` | `utils/ms_cross_attention.py` |
| DAGR agent (state)                  | `GCIVLDualMSCrossAttnAgent` — `gcivl_dual_ms_crossattn`      | `agents/gcivl/state/dual_ms_crossattn.py` |
| DAGR agent (visual)                 | `GCIVLVisualDualMSCrossAttnAgent` — `gcivl_dual_ms_crossattn_vis` | `agents/gcivl/pixel/dual_ms_crossattn.py` |
| `+CA` baseline (standard attention) | `StateAwareGoalEncoder` — `gcivl_dual_crossattn`            | `utils/cross_attention.py`, `agents/.../dual_crossattn.py` |
| Dual base representation            | `GCIVLDualAgent` — `gcivl_dual`                              | `agents/gcivl/state/dual.py` |

---

## Installation

DAGR is built on **JAX / Flax**. A typical setup:

```bash
# 1. Create an environment (Python 3.9–3.11 recommended)
conda create -n dagr python=3.10 -y
conda activate dagr

# 2. Install JAX with CUDA support (match your CUDA version — see the JAX install guide)
pip install --upgrade "jax[cuda12]"   # or jax[cuda11] / CPU-only jax

# 3. Install the remaining dependencies
pip install flax optax distrax ml_collections numpy tqdm wandb absl-py \
            gymnasium pillow pandas

# 4. Install OGBench (environments + datasets)
pip install ogbench
```

Core dependencies used by the code: `jax`, `flax`, `optax`, `distrax`, `ml_collections`, `numpy`, `ogbench`,
`gymnasium`, `wandb`, `absl-py`, `tqdm`, `pillow` (logging/video), `pandas` (result aggregation).

---

## Dataset Setup (read this first)

> ⚠️ **The OGBench data directory is currently hard-coded.** In `utils/env_utils.py`:
> ```python
> env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
>     dataset_name, dataset_dir='/cvlabdata2/lx/ogbench_data', compact_dataset=True)
> ```
> Change `dataset_dir` to your own path (or remove it to use OGBench's default download location) before
> running. Datasets are downloaded automatically by OGBench on first use.

Logging uses **Weights & Biases in `offline` mode** by default (see `utils/log_utils.py::setup_wandb`), so no
account or network access is required; runs are written locally. Metrics are also written to `train.csv` and
`eval.csv` under the save directory.

---

## Running Experiments

All experiments are driven by **`hyperparameters_ms_crossattn.sh`**. It defines the environment lists, seeds,
training budgets, and the fixed DAGR hyperparameters, then dispatches by mode.

```bash
chmod +x hyperparameters_ms_crossattn.sh

# Quick smoke test (visual-cube-single, 1 seed, 10k steps)
./hyperparameters_ms_crossattn.sh test

# State-based tasks — 8 seeds each, 1e6 steps
./hyperparameters_ms_crossattn.sh state

# Visual tasks — 4 seeds each, 5e5 steps
./hyperparameters_ms_crossattn.sh visual

# Only the visual puzzle tasks
./hyperparameters_ms_crossattn.sh puzzle

# Everything (state, then visual)
./hyperparameters_ms_crossattn.sh all
```

The GPU is selected inside the script via `GPU_ID` (default `2`) → `CUDA_VISIBLE_DEVICES`. Edit `STATE_SEEDS`,
`VISUAL_SEEDS`, `STATE_ENVS`, and `VISUAL_ENVS` at the top of the script to change the sweep.

### Running a single configuration directly

The script simply wraps `main.py`. To launch one run yourself (state-based DAGR on AntMaze-Large):

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name=antmaze-large-navigate-v0 \
    --agent=agents/gcivl/state/dual_ms_crossattn.py \
    --seed=0 \
    --train_steps=1000000 \
    --agent.goalrep_dim=256 \
    --agent.cross_attn_heads=4 \
    --agent.cross_attn_head_dim=64 \
    --agent.cross_attn_ffn_dim=256 \
    --agent.cross_attn_gate_init=-5.0 \
    --agent.alpha=10.0 \
    --agent.rep_type=bilinear \
    --agent.rep_expectile=0.9
```

Visual DAGR on Visual-AntMaze-Medium:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --env_name=visual-antmaze-medium-navigate-v0 \
    --agent=agents/gcivl/pixel/dual_ms_crossattn.py \
    --seed=0 \
    --train_steps=500000 \
    --agent.batch_size=256 \
    --agent.encoder=impala_small \
    --agent.p_aug=0.5 \
    --agent.rep_expectile=0.7 \
    --eval_episodes=15
```

**Switching methods** is just a matter of swapping `--agent`:

| Method | `--agent` (state) | `--agent` (visual) |
|--------|-------------------|--------------------|
| Dual (baseline) | `agents/gcivl/state/dual.py` | `agents/gcivl/pixel/dual.py` |
| `+CA` (standard cross-attention) | `agents/gcivl/state/dual_crossattn.py` | `agents/gcivl/pixel/dual_crossattn.py` |
| **DAGR (ours)** | `agents/gcivl/state/dual_ms_crossattn.py` | `agents/gcivl/pixel/dual_ms_crossattn.py` |

`main.py` prints a per-task success-rate summary at every evaluation and saves `train.csv` / `eval.csv` under
`exp/<agent_name>/<env_name>/goal_representation/<run_group>/<exp_name>/`.

---

## Aggregating Results

After a sweep, `attention_results.py` aggregates the per-seed `eval.csv` files following OGBench's protocol
(average over the **last 3 evaluation epochs**, then average across seeds, reporting mean ± std):

```bash
python attention_results.py --exp_dir exp/ --output results_crossattn.csv
```

---

## Main Hyperparameters

DAGR uses a **single fixed configuration across all tasks** (no per-task tuning). Defaults live in each
agent's `get_config()`; the launcher overrides a handful.

| Group | Hyperparameter | Value |
|-------|----------------|-------|
| **MS-DGCA** | Scale levels `L` | 3 |
| | Token counts `(T₁, T₂, T₃)` | `(16, 8, 4)` |
| | Attention heads `H` | 4 |
| | Head dim `d_k` | 64 |
| | Model dim `d_m = H·d_k` | 256 |
| | FFN hidden dim | 256 |
| | Gate init `α₀` | −5 |
| | Difference-scale init `λ₀` | −5 |
| | Fusion-logit init `w₀` | 0 |
| | DGCA blocks per scale | 1 |
| **Dual** | Representation type | bilinear (inner product) |
| | Goal-rep dim | 256 |
| | Rep hidden dims | (512, 512, 512) |
| | Rep expectile (state / visual) | 0.9 / 0.7 |
| **GCIVL** | Learning rate | 3e-4 (Adam) |
| | Batch size (state / visual) | 1024 / 256 |
| | Discount `γ` | 0.99 |
| | Target update `τ` | 0.005 |
| | Value expectile | 0.9 |
| | AWR temperature `α` (`alpha`) | 10.0 |
| | Training steps (state / visual) | 1e6 / 5e5 |
| | Seeds (state / visual) | 8 / 4 |
| **Visual encoder** | Architecture | IMPALA-small |
| | Image augmentation prob. | 0.5 |

---

## Results

Success rates (%), mean ± std. `Dual` is the base representation; `DAGR` is Dual + MS-DGCA (ours). Baseline
columns (Orig, VIB, VIP, TRA, BYOL-γ, Dual) are reproduced from Park et al. (2026); see the paper for the
full comparison. **Bold** marks where DAGR is best or tied-best.

### State-based OGBench (8 seeds)

| Task | Dual | DAGR (ours) |
|------|:----:|:-----------:|
| *Navigation* | | |
| pointmaze-medium-navigate | 76 ± 7 | **87 ± 8** |
| pointmaze-large-navigate | 46 ± 6 | 41 ± 7 |
| antmaze-medium-navigate | 75 ± 4 | **95 ± 1** |
| antmaze-large-navigate | 28 ± 11 | **82 ± 3** |
| antmaze-giant-navigate | 0 ± 0 | **4 ± 2** |
| humanoidmaze-medium-navigate | 29 ± 3 | **83 ± 3** |
| humanoidmaze-large-navigate | 3 ± 2 | **62 ± 4** |
| antsoccer-arena-navigate | 31 ± 3 | **58 ± 4** |
| *Manipulation* | | |
| cube-single-play | 89 ± 3 | 87 ± 2 |
| cube-double-play | 60 ± 4 | 35 ± 2 |
| scene-play | 72 ± 6 | 59 ± 2 |
| *Discrete Reasoning* | | |
| puzzle-3x3-play | 5 ± 1 | 5 ± 1 |
| puzzle-4x4-play | 23 ± 3 | 14 ± 3 |
| **Average** | 41 | **55** |

### Visual OGBench (4 seeds)

| Task | Dual | DAGR (ours) |
|------|:----:|:-----------:|
| *Navigation* | | |
| visual-antmaze-medium-navigate | 78 ± 4 | **90 ± 2** |
| visual-antmaze-large-navigate | 40 ± 4 | **52 ± 2** |
| *Manipulation* | | |
| visual-cube-single-play | 58 ± 5 | 44 ± 2 |
| visual-cube-double-play | 9 ± 2 | **11 ± 4** |
| visual-scene-play | 26 ± 5 | **27 ± 3** |
| *Discrete Reasoning* | | |
| visual-puzzle-3x3-play | 0 ± 0 | 0 ± 0 |
| visual-puzzle-4x4-play | 0 ± 0 | 0 ± 0 |
| **Avg. (all 7)** | 30 | **32** |
| **Avg. (excl. puzzle)** | 42 | **45** |

### Standard cross-attention vs. DGCA

The `+CA` ablation isolates *why* DAGR helps: on navigation, DGCA clearly beats standard cross-attention, and
the gap is widest where Dual is weakest — but the separation comes from the **gated residual**, not the
difference bias (removing the bias leaves AntMaze-Large within one std of the full model).

| Task | Dual | +CA | +DGCA (ours) |
|------|:----:|:---:|:------------:|
| antmaze-medium | 75 ± 4 | 84 ± 6 | **95 ± 1** |
| antmaze-large | 28 ± 11 | 33 ± 4 | **82 ± 3** |
| humanoidmaze-medium | 29 ± 3 | 41 ± 7 | **83 ± 3** |
| humanoidmaze-large | 3 ± 2 | 8 ± 2 | **62 ± 4** |
| antsoccer-arena | 31 ± 3 | 44 ± 2 | **58 ± 4** |
| visual-cube-single | 58 ± 5 | **80 ± 3** | 44 ± 2 |

`visual-cube-single` is the sharpest counterexample in the paper: `+CA` gains +22 over Dual (the largest
single-task gain any module produces), yet the two ingredients that separate DGCA from `+CA` are jointly
harmful there. State-conditioning helps; the discrepancy-maximizing bias does not.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{lei2026dagr,
  title         = {{DAGR}: State-Conditioned Goal Representations via Difference-Aware Goal Cross-Attention},
  author        = {Lei, Xing and Yang, Wenyan and Zhang, Xuetao and Wang, Donglin},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

*(Update the entry with the final author list and venue once available.)*

---

## Acknowledgments

DAGR builds directly on prior work and reuses its experimental scaffolding:

- **OGBench** — Park et al., *Benchmarking Offline Goal-Conditioned RL*, ICLR 2025.
- **Dual goal representation** — Park et al., *Dual Goal Representations*, ICLR 2026 (the base `φ(g)` DAGR refines).
- **GCIVL** — the downstream offline GCRL algorithm (implicit V-learning), Kostrikov et al., 2022 / Park et al., 2025.

DAGR is **orthogonal and composable**: it operates on the output `φ(g)` of any late-fusion goal encoder rather
than on the objective that produces it.
