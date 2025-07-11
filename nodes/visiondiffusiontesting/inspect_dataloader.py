#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

from vision_diffusion_unet import TrajectoryDatasetDiffusion
from vision_encoding import getVisionModel

def load_episode_dataset_no_encode(dir_path, pattern):
    states, actions, images = [], [], []
    for fname in sorted(os.listdir(dir_path)):
        if pattern in fname and fname.endswith('.pkl'):
            with open(os.path.join(dir_path, fname), 'rb') as f:
                s, a, i = pickle.load(f)
            states.append(s)
            actions.append(a)
            images.append(i)
    print("Loaded", len(states), "episodes.")
    return states, actions, images

def show_image_comparison(img1, img2, title1="Before", title2="After"):
    """
    Displays two images side-by-side.

    Args:
        img1 (Tensor or np.ndarray): Image with shape (3, H, W) or (H, W, 3)
        img2 (Tensor or np.ndarray): Same as img1
        title1 (str): Title for the first image
        title2 (str): Title for the second image
    """
    def prepare_img(img):
        if isinstance(img, torch.Tensor):
            if img.dim() == 3 and img.shape[0] == 3:  # (3, H, W)
                img = img.permute(1, 2, 0)  # (H, W, 3)
            img = img.cpu().numpy()
        return img.clip(0, 1)

    img1 = prepare_img(img1)
    img2 = prepare_img(img2)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[0].set_title(title1)
    axes[0].axis('off')
    axes[1].imshow(img2)
    axes[1].set_title(title2)
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()


def inspect_dataset(dataset, vision_model, distort=True, num_samples=4):
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    for i in range(min(num_samples, len(batch['obs']))):
        print(f"\n=== Sample {i} ===")
        print("Obs:\n", batch['obs'][i])
        print("Action:\n", batch['action'][i])

        # Get correct raw image
        flat_index = dataset.indices[i][0] + dataset.obs_horizon - 1
        ep_idx = np.searchsorted(dataset.ends, flat_index, side='right')
        local_idx = flat_index - dataset.ends[ep_idx - 1] if ep_idx > 0 else flat_index

        raw_img = dataset.raw_images[ep_idx][local_idx]

        # Apply distortion (returns torch.Tensor: (3, H, W))
        distorted_tensor = vision_model.transform([raw_img], distort=distort)[0]

        # Undo normalization for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = distorted_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean)
        img_np = np.clip(img_np, 0, 1)

        print(np.mean(raw_img))
        print(np.mean(img_np))
        show_image_comparison(raw_img.astype(np.float32) / 255.0,img_np)

if __name__ == '__main__':
    # === CONFIG ===
    data_dir = '../../data/'
    pattern = 'tapping_6-6'
    use_distort = True
    obs_horizon = 2
    pred_horizon = 32
    action_horizon = 4
    vision_model_name = 'resnet18'

    # === LOAD DATA ===
    states, actions, images = load_episode_dataset_no_encode(data_dir, pattern)
    model = getVisionModel(vision_model_name)

    dataset = TrajectoryDatasetDiffusion(
        model=model,
        states=states,
        actions=actions,
        images=images,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        distort=use_distort
    )

    inspect_dataset(dataset, model, distort=use_distort)