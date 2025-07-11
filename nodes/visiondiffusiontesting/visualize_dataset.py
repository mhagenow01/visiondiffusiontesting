#!/usr/bin/env python3
import glob
import pickle
import numpy as np
import plotly.graph_objs as go
import argparse
import os

def load_trajectories(pattern):
    trajs = []
    total_points = 0

    for fname in sorted(glob.glob(pattern)):
        with open(fname, 'rb') as f:
            states, actions, images = pickle.load(f)
            data = np.array(states)
            if data.shape[0] < 3:
                raise ValueError(f"{fname} has fewer than 3 state dimensions")
            xyz = data[:3, :].T  # shape: (T, 3)
            trajs.append((os.path.basename(fname), xyz))
            total_points += xyz.shape[0]

    print(f"Loaded {len(trajs)} episodes with {total_points} total datapoints.")
    return trajs

def equal_axis_range(fig):
    x_vals, y_vals, z_vals = [], [], []
    for trace in fig.data:
        x_vals.extend(trace.x)
        y_vals.extend(trace.y)
        z_vals.extend(trace.z)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    x_range = [np.min(x_vals), np.max(x_vals)]
    y_range = [np.min(y_vals), np.max(y_vals)]
    z_range = [np.min(z_vals), np.max(z_vals)]

    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )

    def center_range(r):
        mid = sum(r) / 2
        return [mid - max_range/2, mid + max_range/2]

    return dict(
        xaxis=dict(range=center_range(x_range)),
        yaxis=dict(range=center_range(y_range)),
        zaxis=dict(range=center_range(z_range)),
    )

def plot_3d_trajectories(trajs):
    fig = go.Figure()

    for name, data in trajs:
        fig.add_trace(go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode='lines+markers',
            name=name,
            marker=dict(size=2),  # 👈 smaller marker size
            line=dict(width=2)    # optional: keep lines visible
        ))
    fig.update_layout(
        scene=equal_axis_range(fig),
        scene_aspectmode='cube',
        title="3D Trajectories",
        margin=dict(l=0, r=0, b=0, t=30),
        legend=dict(x=0.01, y=0.99)
    )
    fig.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='./../../data_downsampled/tapping*.pkl',
                        help='Glob pattern for input .pkl files (default: "episode_*.pkl")')
    args = parser.parse_args()

    trajectories = load_trajectories(args.pattern)
    plot_3d_trajectories(trajectories)