"""
OGBench-style Result Aggregation for Cross-Attention Enhanced Goal Representation.

This script aggregates results across seeds and environments following OGBench's methodology:
- Takes the average performance over the last 3 evaluation epochs
- Averages across seeds (8 for state, 4 for visual)
- Reports mean ± std

Usage:
    python aggregate_results.py --exp_dir exp/ --output results_crossattn.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse


# ============================================================================
# Configuration: Define environments and algorithms
# ============================================================================

STATE_ENVS = [
    "pointmaze-medium-navigate-v0",
    "pointmaze-large-navigate-v0",
    "antmaze-medium-navigate-v0",
    "antmaze-large-navigate-v0",
    "antmaze-giant-navigate-v0",
    "humanoidmaze-medium-navigate-v0",
    "humanoidmaze-large-navigate-v0",
    "antsoccer-arena-navigate-v0",
    "cube-single-play-v0",
    "cube-double-play-v0",
    "scene-play-v0",
    "puzzle-3x3-play-v0",
    "puzzle-4x4-play-v0",
]

VISUAL_ENVS = [
    "visual-antmaze-medium-navigate-v0",
    "visual-antmaze-large-navigate-v0",
    "visual-cube-single-play-v0",
    "visual-cube-double-play-v0",
    "visual-scene-play-v0",
    "visual-puzzle-3x3-play-v0",
    "visual-puzzle-4x4-play-v0",
]

# Algorithm name mappings: (folder_name_state, folder_name_visual, display_name)
ALGORITHMS = {
    "dual": ("gcivl_dual_ms_crossattn", "gcivl_dual_ms_crossattn_vis", "Dual+CrossAttn"),
    #"vib": ("gcivl_vib_cross", "gcivl_vib_cross_vis", "VIB+CrossAttn"),
    #"tra": ("gcivl_tra_cross", "gcivl_tra_cross_vis", "TRA+CrossAttn"),
    #"byol": ("gcivl_byol_cross", "gcivl_byol_cross_vis", "BYOL+CrossAttn"),
    #"vip": ("gcivl_vip_cross", "gcivl_vip_cross_vis", "VIP+CrossAttn"),
}

STATE_SEEDS = 8
VISUAL_SEEDS = 4


def find_experiment_dirs(exp_dir, agent_name, env_name):
    """Find all seed directories for a given agent and environment."""
    pattern = os.path.join(
        exp_dir, agent_name, env_name, "goal_representation", "*", "sd*"
    )
    dirs = glob.glob(pattern)
    return sorted(dirs)


def load_eval_csv(csv_path, last_n_epochs=3):
    """
    Load eval.csv and return the average success rate over the last n epochs.
    
    Following OGBench methodology: average over last 3 evaluation epochs.
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Get the overall_success column (last column before 'step')
        if 'evaluation/overall_success' in df.columns:
            success_col = 'evaluation/overall_success'
        elif 'overall_success' in df.columns:
            success_col = 'overall_success'
        else:
            # Find column containing 'overall_success'
            success_cols = [c for c in df.columns if 'overall_success' in c]
            if success_cols:
                success_col = success_cols[0]
            else:
                print(f"Warning: No overall_success column found in {csv_path}")
                return None
        
        # Get last n epochs
        values = df[success_col].values
        if len(values) >= last_n_epochs:
            avg_success = np.mean(values[-last_n_epochs:])
        else:
            avg_success = np.mean(values)
        
        return avg_success * 100  # Convert to percentage
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def aggregate_results(exp_dir, algorithms, state_envs, visual_envs, 
                      state_seeds=8, visual_seeds=4):
    """
    Aggregate results across all algorithms and environments.
    
    Returns:
        results: dict[env][algo] = (mean, std, n_seeds)
        missing: list of (algo, env, missing_seeds)
    """
    results = defaultdict(dict)
    missing = []
    
    # Process state environments
    print("\n" + "="*60)
    print("Processing STATE environments")
    print("="*60)
    
    for env in state_envs:
        for algo_key, (state_folder, _, display_name) in algorithms.items():
            exp_dirs = find_experiment_dirs(exp_dir, state_folder, env)
            
            # Extract seed numbers from directories
            seed_results = []
            found_seeds = set()
            
            for d in exp_dirs:
                # Extract seed number from directory name (e.g., sd000_xxx -> 0)
                dir_name = os.path.basename(d)
                if dir_name.startswith('sd'):
                    try:
                        seed_num = int(dir_name[2:5])
                        found_seeds.add(seed_num)
                    except ValueError:
                        continue
                
                csv_path = os.path.join(d, 'eval.csv')
                if os.path.exists(csv_path):
                    success_rate = load_eval_csv(csv_path)
                    if success_rate is not None:
                        seed_results.append(success_rate)
            
            if seed_results:
                mean_success = np.mean(seed_results)
                std_success = np.std(seed_results)
                results[env][algo_key] = (mean_success, std_success, len(seed_results))
                
                if len(seed_results) < state_seeds:
                    expected_seeds = set(range(state_seeds))
                    missing_seeds = expected_seeds - found_seeds
                    missing.append((display_name, env, "state", 
                                   len(seed_results), state_seeds, list(missing_seeds)))
            else:
                results[env][algo_key] = (None, None, 0)
                missing.append((display_name, env, "state", 0, state_seeds, list(range(state_seeds))))
    
    # Process visual environments
    print("\n" + "="*60)
    print("Processing VISUAL environments")
    print("="*60)
    
    for env in visual_envs:
        for algo_key, (_, visual_folder, display_name) in algorithms.items():
            exp_dirs = find_experiment_dirs(exp_dir, visual_folder, env)
            
            seed_results = []
            found_seeds = set()
            
            for d in exp_dirs:
                dir_name = os.path.basename(d)
                if dir_name.startswith('sd'):
                    try:
                        seed_num = int(dir_name[2:5])
                        found_seeds.add(seed_num)
                    except ValueError:
                        continue
                
                csv_path = os.path.join(d, 'eval.csv')
                if os.path.exists(csv_path):
                    success_rate = load_eval_csv(csv_path)
                    if success_rate is not None:
                        seed_results.append(success_rate)
            
            if seed_results:
                mean_success = np.mean(seed_results)
                std_success = np.std(seed_results)
                results[env][algo_key] = (mean_success, std_success, len(seed_results))
                
                if len(seed_results) < visual_seeds:
                    expected_seeds = set(range(visual_seeds))
                    missing_seeds = expected_seeds - found_seeds
                    missing.append((display_name, env, "visual",
                                   len(seed_results), visual_seeds, list(missing_seeds)))
            else:
                results[env][algo_key] = (None, None, 0)
                missing.append((display_name, env, "visual", 0, visual_seeds, list(range(visual_seeds))))
    
    return results, missing


def print_results_table(results, algorithms, envs, title):
    """Print results in a formatted table."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    
    # Header
    header = f"{'Environment':<45}"
    for algo_key in algorithms.keys():
        header += f" {algorithms[algo_key][2]:>15}"
    print(header)
    print("-"*100)
    
    # Results
    for env in envs:
        row = f"{env:<45}"
        for algo_key in algorithms.keys():
            if env in results and algo_key in results[env]:
                mean, std, n = results[env][algo_key]
                if mean is not None:
                    row += f" {mean:>6.1f}±{std:<5.1f} "
                else:
                    row += f" {'N/A':>13} "
            else:
                row += f" {'N/A':>13} "
        print(row)
    
    # Average
    print("-"*100)
    avg_row = f"{'Average':<45}"
    for algo_key in algorithms.keys():
        values = []
        for env in envs:
            if env in results and algo_key in results[env]:
                mean, _, _ = results[env][algo_key]
                if mean is not None:
                    values.append(mean)
        if values:
            avg_mean = np.mean(values)
            avg_std = np.std(values) / np.sqrt(len(values))  # Standard error
            avg_row += f" {avg_mean:>6.1f}±{avg_std:<5.1f} "
        else:
            avg_row += f" {'N/A':>13} "
    print(avg_row)


def print_missing_experiments(missing):
    """Print missing experiments."""
    if not missing:
        print("\n✓ All experiments are complete!")
        return
    
    print("\n" + "="*80)
    print("MISSING EXPERIMENTS")
    print("="*80)
    print(f"{'Algorithm':<20} {'Environment':<40} {'Type':<8} {'Found/Expected':<15} {'Missing Seeds'}")
    print("-"*80)
    
    for algo, env, env_type, found, expected, missing_seeds in missing:
        seeds_str = str(missing_seeds) if missing_seeds else "[]"
        print(f"{algo:<20} {env:<40} {env_type:<8} {found}/{expected:<14} {seeds_str}")


def generate_latex_table(results, algorithms, envs, caption, label):
    """Generate LaTeX table similar to the paper format."""
    print(f"\n% LaTeX Table: {label}")
    print("\\begin{table}[t!]")
    print("\\caption{\\footnotesize " + caption + "}")
    print(f"\\label{{{label}}}")
    print("\\centering")
    print("\\scalebox{0.9}{")
    
    # Determine number of algorithm columns
    n_algos = len(algorithms)
    col_spec = "l" + "*{" + str(n_algos) + "}{>{\\spew{.5}{+1}}r@{\\,}l}"
    
    print(f"\\begin{{tabularew}}{{{col_spec}}}")
    print("\\toprule")
    
    # Header row
    header = "\\multicolumn{1}{l}{\\tt{Environment}}"
    for algo_key, (_, _, display_name) in algorithms.items():
        header += f" & \\multicolumn{{2}}{{c}}{{\\tt{{{display_name}}}}}"
    header += " \\\\"
    print(header)
    print("\\midrule")
    
    # Data rows
    for env in envs:
        row = f"\\tt{{{env}}}"
        best_val = -1
        best_algos = []
        
        # Find best algorithm for this environment
        for algo_key in algorithms.keys():
            if env in results and algo_key in results[env]:
                mean, _, _ = results[env][algo_key]
                if mean is not None and mean > best_val:
                    best_val = mean
                    best_algos = [algo_key]
                elif mean is not None and abs(mean - best_val) < 0.95 * best_val:  # Within 95%
                    best_algos.append(algo_key)
        
        for algo_key in algorithms.keys():
            if env in results and algo_key in results[env]:
                mean, std, _ = results[env][algo_key]
                if mean is not None:
                    mean_int = int(round(mean))
                    std_int = int(round(std))
                    if algo_key in best_algos or mean >= 0.95 * best_val:
                        row += f" & \\tt{{\\color{{myblue}}{mean_int}}} &{{\\tiny $\\pm$\\tt{{{std_int}}}}}"
                    else:
                        row += f" & \\tt{{{mean_int}}} &{{\\tiny $\\pm$\\tt{{{std_int}}}}}"
                else:
                    row += " & \\tt{-} &{\\tiny $\\pm$\\tt{-}}"
            else:
                row += " & \\tt{-} &{\\tiny $\\pm$\\tt{-}}"
        row += " \\\\"
        print(row)
    
    # Average row
    print("\\midrule")
    avg_row = "\\tt{Average}"
    best_avg = -1
    avg_values = {}
    
    for algo_key in algorithms.keys():
        values = []
        for env in envs:
            if env in results and algo_key in results[env]:
                mean, _, _ = results[env][algo_key]
                if mean is not None:
                    values.append(mean)
        if values:
            avg_values[algo_key] = np.mean(values)
            if avg_values[algo_key] > best_avg:
                best_avg = avg_values[algo_key]
        else:
            avg_values[algo_key] = None
    
    for algo_key in algorithms.keys():
        if avg_values[algo_key] is not None:
            avg_mean = int(round(avg_values[algo_key]))
            std_err = int(round(np.std([results[env][algo_key][0] for env in envs 
                                        if env in results and algo_key in results[env] 
                                        and results[env][algo_key][0] is not None]) / np.sqrt(len(envs))))
            if avg_values[algo_key] >= 0.95 * best_avg:
                avg_row += f" & \\tt{{\\color{{myblue}}{avg_mean}}} &{{\\tiny $\\pm$\\tt{{{std_err}}}}}"
            else:
                avg_row += f" & \\tt{{{avg_mean}}} &{{\\tiny $\\pm$\\tt{{{std_err}}}}}"
        else:
            avg_row += " & \\tt{-} &{\\tiny $\\pm$\\tt{-}}"
    avg_row += " \\\\"
    print(avg_row)
    
    print("\\bottomrule")
    print("\\end{tabularew}")
    print("}")
    print("\\end{table}")


def save_to_csv(results, algorithms, envs, output_path):
    """Save results to CSV file."""
    rows = []
    for env in envs:
        row = {'Environment': env}
        for algo_key, (_, _, display_name) in algorithms.items():
            if env in results and algo_key in results[env]:
                mean, std, n = results[env][algo_key]
                if mean is not None:
                    row[f'{display_name}_mean'] = mean
                    row[f'{display_name}_std'] = std
                    row[f'{display_name}_n_seeds'] = n
                else:
                    row[f'{display_name}_mean'] = None
                    row[f'{display_name}_std'] = None
                    row[f'{display_name}_n_seeds'] = 0
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate OGBench results for Cross-Attention experiments')
    parser.add_argument('--exp_dir', type=str, default='exp/', help='Experiment directory')
    parser.add_argument('--output', type=str, default='results_crossattn.csv', help='Output CSV file')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX tables')
    args = parser.parse_args()
    
    print("="*80)
    print("OGBench Result Aggregation for Cross-Attention Enhanced Goal Representation")
    print("="*80)
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Output file: {args.output}")
    
    # Aggregate results
    results, missing = aggregate_results(
        args.exp_dir, 
        ALGORITHMS,
        STATE_ENVS, 
        VISUAL_ENVS,
        STATE_SEEDS,
        VISUAL_SEEDS
    )
    
    # Print results tables
    print_results_table(results, ALGORITHMS, STATE_ENVS, "State-based Tasks (GCIVL + CrossAttn)")
    print_results_table(results, ALGORITHMS, VISUAL_ENVS, "Pixel-based Tasks (GCIVL + CrossAttn)")
    
    # Print missing experiments
    print_missing_experiments(missing)
    
    # Generate LaTeX tables if requested
    if args.latex:
        generate_latex_table(
            results, ALGORITHMS, STATE_ENVS,
            "Results on state-based tasks with GCIVL + Cross-Attention.",
            "table:state_crossattn"
        )
        generate_latex_table(
            results, ALGORITHMS, VISUAL_ENVS,
            "Results on pixel-based tasks with GCIVL + Cross-Attention.",
            "table:pixel_crossattn"
        )
    
    # Save to CSV
    all_envs = STATE_ENVS + VISUAL_ENVS
    save_to_csv(results, ALGORITHMS, all_envs, args.output)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_experiments = len(STATE_ENVS) * len(ALGORITHMS) * STATE_SEEDS + \
                       len(VISUAL_ENVS) * len(ALGORITHMS) * VISUAL_SEEDS
    missing_count = sum(expected - found for _, _, _, found, expected, _ in missing)
    completed_count = total_experiments - missing_count
    
    print(f"Total expected experiments: {total_experiments}")
    print(f"Completed: {completed_count}")
    print(f"Missing: {missing_count}")
    print(f"Completion rate: {completed_count/total_experiments*100:.1f}%")


if __name__ == '__main__':
    main()