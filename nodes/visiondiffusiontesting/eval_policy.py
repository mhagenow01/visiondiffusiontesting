#!/usr/bin/env python3
"""
Script to evaluate a diffusion policy on recorded episodes.
Loads episodes matching a file pattern, runs the policy to predict each action,
and reports mean squared error (MSE) per episode and overall.
"""
import os
import glob
import pickle
import argparse
import numpy as np

# adjust import to your policy implementation
from policies.diffusion_with_images import Diffusion


def load_episodes(dir_path: str, pattern: str = 'episode_*.pkl'):
    """
    Load all episodes from dir_path matching pattern.
    Returns lists of (states, actions, images) arrays.
    Each:
      states:  np.ndarray of shape (7, T)
      actions: np.ndarray of shape (7, T)
      images:  np.ndarray of shape (T, H, W, C)
    """
    states_list, actions_list, images_list = [], [], []
    search = os.path.join(dir_path, pattern)
    for fname in sorted(glob.glob(search)):
        with open(fname, 'rb') as f:
            states, actions, images = pickle.load(f)
        states_list.append(states)
        actions_list.append(actions)
        images_list.append(images)
    return states_list, actions_list, images_list


def evaluate_policy(policy, states_list, actions_list, images_list):
    """
    Run policy.getAction on each step and compute MSE.
    Returns per-episode MSE list and overall MSE.
    """
    episode_mse = []
    total_sq_error = 0.0
    total_count = 0

    for idx, (states, actions, images) in enumerate(zip(states_list, actions_list, images_list)):
        T = states.shape[1]
        sq_error = 0.0
        count = 0
        for t in range(T):
            state_vec = states[:, t]
            img = images[t]
            # predict next action
            pred = policy.getAction(state_vec, img, forecast=False)
            true = actions[:, t]
            err = (pred - true) ** 2
            sq_error += err.sum()
            count += err.size
        mse = sq_error / count if count > 0 else np.nan
        episode_mse.append(mse)
        total_sq_error += sq_error
        total_count += count
        print(f"Episode {idx:2d}: steps={T}  MSE={mse:.6f}")

    overall_mse = total_sq_error / total_count if total_count > 0 else np.nan
    print(f"\nOverall: episodes={len(episode_mse)}  total_steps={total_count}  MSE={overall_mse:.6f}")
    return episode_mse, overall_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate diffusion policy on recorded episodes')
    parser.add_argument('model_path', help='Path to trained .pkl policy model')
    parser.add_argument('--data_dir', default='.', help='Directory containing episode_*.pkl files')
    parser.add_argument('--pattern',  default='episode_*.pkl', help='Filename pattern for episodes')
    args = parser.parse_args()

    # load episodes
    states_list, actions_list, images_list = load_episodes(args.data_dir, args.pattern)
    print(f"Loaded {len(states_list)} episodes from {args.data_dir}\n")

    # load policy
    policy = Diffusion()
    policy.load(args.model_path)

    # evaluate