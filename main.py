"""
Modified main.py for goal representation algorithms.

Changes from original:
1. Added support for cross-attention enhanced agents
2. Modified evaluation to output 5 tasks' success rates and overall success rate
3. Broader exception handling in get_agents() to prevent cascading import failures
"""

import json
import os
import random
import time
from collections import defaultdict

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

# Import datasets and utilities
from utils.datasets import Dataset, GCDataset, HGCDataset, VIPDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 50, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0.0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_float('eval_goal_gaussian', None, 'Goal Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 0, 'Whether to evaluate on CPU.')

# New flag for controlling output verbosity
flags.DEFINE_integer('num_display_tasks', 5, 'Number of tasks to display in evaluation output.')

config_flags.DEFINE_config_file('agent', 'agents/crl/id.py', lock_config=False)


def get_agents():
    """Import all available agents.

    Uses broad exception handling (except Exception) to prevent cascading
    failures when some agent modules are missing or have import errors.
    """
    agents = {}

    # Try to import standard pixel agents
    try:
        from agents.gcivl.pixel.dual import GCIVLVisualDualAgent
        agents['gcivl_dual_vis'] = GCIVLVisualDualAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.vib import GCIVLVisualVIBAgent
        agents['gcivl_vib_vis'] = GCIVLVisualVIBAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.tra import GCIVLVisualTRAAgent
        agents['gcivl_tra_vis'] = GCIVLVisualTRAAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.byol import GCIVLVisualBYOLAgent
        agents['gcivl_byol_vis'] = GCIVLVisualBYOLAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.vip import GCIVLVisualVIPAgent
        agents['gcivl_vip_vis'] = GCIVLVisualVIPAgent
    except Exception:
        pass

    # Cross-attention pixel agents
    try:
        from agents.gcivl.pixel.dual_crossattn import GCIVLVisualDualCrossAttnAgent
        agents['gcivl_dual_crossattn_vis'] = GCIVLVisualDualCrossAttnAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.dual_ms_crossattn import GCIVLVisualDualMSCrossAttnAgent
        agents['gcivl_dual_ms_crossattn_vis'] = GCIVLVisualDualMSCrossAttnAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.vib_crossattn import GCIVLVisualVIBCrossAttnAgent
        agents['gcivl_vib_crossattn_vis'] = GCIVLVisualVIBCrossAttnAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.tra_crossattn import GCIVLVisualTRACrossAttnAgent
        agents['gcivl_tra_crossattn_vis'] = GCIVLVisualTRACrossAttnAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.byol_crossattn import GCIVLVisualBYOLCrossAttnAgent
        agents['gcivl_byol_crossattn_vis'] = GCIVLVisualBYOLCrossAttnAgent
    except Exception:
        pass

    try:
        from agents.gcivl.pixel.vip_crossattn import GCIVLVisualVIPCrossAttnAgent
        agents['gcivl_vip_crossattn_vis'] = GCIVLVisualVIPCrossAttnAgent
    except Exception:
        pass

    # Fallback: try importing from agents module (uses the __init__.py dict)
    try:
        from agents import agents as agent_module
        agents.update(agent_module)
    except Exception:
        pass

    return agents


def print_eval_summary(eval_metrics, task_names, num_display=5):
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Collect success rates for each task
    task_success_rates = []
    for task_name in task_names:
        key = f'evaluation/{task_name}_success'
        if key in eval_metrics:
            success_rate = eval_metrics[key]
            task_success_rates.append((task_name, success_rate))

    # Display top N tasks
    display_count = min(num_display, len(task_success_rates))
    print(f"\nTop {display_count} Task Success Rates:")
    print("-" * 40)

    for i, (task_name, success_rate) in enumerate(task_success_rates[:display_count]):
        print(f"  Task {i+1}: {task_name}")
        print(f"          Success Rate: {success_rate * 100:.2f}%")

    # Display overall success rate
    if 'evaluation/overall_success' in eval_metrics:
        overall_success = eval_metrics['evaluation/overall_success']
        print("\n" + "-" * 40)
        print(f"OVERALL SUCCESS RATE: {overall_success * 100:.2f}%")

    print("=" * 60 + "\n")


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='goal_representation', group=FLAGS.run_group, name=exp_name)

    config = FLAGS.agent
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, config['agent_name'], FLAGS.env_name, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Set up environment and dataset.
    env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    if 'oraclerep' in FLAGS.env_name and config['oraclerep'] == False:
        raise ValueError('Must enable oracle representation in config dictionary to use this environment!')

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
        'VIPDataset': VIPDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(norm=config['norm'], **train_dataset), config)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(norm=config['norm'], **val_dataset), config)
    # Need to pass into evaluation functions
    diff = train_dataset.get_diff()

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    if config['discrete']:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    # Get available agents
    agents = get_agents()

    if config['agent_name'] not in agents:
        raise ValueError(f"Unknown agent: {config['agent_name']}. Available agents: {sorted(list(agents.keys()))}")

    agent_class = agents[config['agent_name']]
    ex_goals = example_batch['value_goals'] if config['oraclerep'] else None
    agent = agent_class.create(
        FLAGS.seed, example_batch['observations'], example_batch['actions'], config, ex_goals=ex_goals
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    # Get task info for display
    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    task_names = [task_infos[i]['task_name'] for i in range(num_tasks)]

    print(f"\n{'='*60}")
    print(f"Training {config['agent_name']} on {FLAGS.env_name}")
    print(f"Total training steps: {FLAGS.train_steps}")
    print(f"Number of evaluation tasks: {num_tasks}")
    print(f"{'='*60}\n")

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            print(f"\n[Step {i}] Starting evaluation...")

            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            renders = []
            eval_metrics = {}
            overall_metrics = defaultdict(list)

            for task_id in tqdm.trange(1, num_tasks + 1, desc="Evaluating tasks"):
                task_name = task_infos[task_id - 1]['task_name']
                eval_info, trajs, cur_renders = evaluate(
                    agent=eval_agent,
                    env=env,
                    task_id=task_id,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                    eval_temperature=FLAGS.eval_temperature,
                    eval_gaussian=FLAGS.eval_gaussian,
                    eval_goal_gaussian=FLAGS.eval_goal_gaussian,
                    diff=diff,
                )
                renders.extend(cur_renders)
                metric_names = ['success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

            # Print formatted evaluation summary
            print_eval_summary(eval_metrics, task_names, num_display=FLAGS.num_display_tasks)

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders, n_cols=num_tasks)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Total time: {time.time() - first_time:.2f}s")
    print(f"Results saved to: {FLAGS.save_dir}")
    print("=" * 60 + "\n")

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
