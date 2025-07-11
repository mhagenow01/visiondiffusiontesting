#!/usr/bin/env python3
import sys
import os
import glob
import pickle
import numpy as np
import time
import cv2

from vision_diffusion_unet import VisionDiffusionUNet

project_path = os.path.dirname(os.path.realpath(__file__))

def downsample_episode_dataset(
    dir_path: str = '.', 
    pattern: str = 'episode_*.pkl',
    target_size: tuple = (240, 320)
):
    """
    Loads all pickle episodes in `dir_path` matching `pattern`, downsamples images,
    and saves new pickle files to a sibling directory with '_downsampled' appended.

    Assumes each pickle contains (states_arr, actions_arr, images_arr), 
    where images_arr.shape = (T_i, H, W, C)
    """
    output_dir = dir_path.rstrip('/\\') + '_downsampled'
    os.makedirs(output_dir, exist_ok=True)

    glob_path = os.path.join(dir_path, pattern)
    print(f"Loading and downsampling from {dir_path} to {output_dir}...")

    for fname in sorted(glob.glob(glob_path)):
        with open(fname, 'rb') as f:
            states_arr, actions_arr, images_arr = pickle.load(f)

        if images_arr.ndim != 4:
            print(f"Skipping {fname}: unexpected image shape {images_arr.shape}")
            continue

        downsampled_images = np.array([
            cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
            for img in images_arr
        ])

        # Save to corresponding output path
        base_name = os.path.basename(fname)
        out_path = os.path.join(output_dir, base_name)
        with open(out_path, 'wb') as f:
            pickle.dump((states_arr, actions_arr, downsampled_images), f)

    print("Done.")


def load_episode_dataset_no_encode(dir_path: str = '.', pattern: str = 'episode_*.pkl'):
    """
    Loads all pickle episodes in `dir_path` matching `pattern` and returns
    three lists:
      - states_list:  [ array of shape (n, T_i), ... ]
      - actions_list: [ array of shape (m, T_i), ... ]
      - images_list:  [ array of shape (T_i, H, W, C), ... ]
    so you can call:
        train(states_list, actions_list, images_list)
    """
    states_list  = []
    actions_list = []
    images_list  = []

    # find and sort all matching files
    glob_path = os.path.join(dir_path, pattern)
    print("Loading data...")
    for fname in sorted(glob.glob(glob_path)):
        # print(fname)
        with open(fname, 'rb') as f:
            states_arr, actions_arr, images_arr = pickle.load(f)
        states_list.append(states_arr)
        actions_list.append(actions_arr)
        images_list.append(images_arr)
    
    print(".... ",len(states_list)," files loaded")
    return states_list, actions_list, images_list
    

def list_episode_paths(dir_path: str = '.', pattern: str = 'episode_*.pkl'):
    """
    Returns a sorted list of episode file paths that match the pattern.
    """
    glob_path = os.path.join(dir_path, pattern)
    paths = sorted(glob.glob(glob_path))
    print(f"Found {len(paths)} episodes in '{dir_path}'")
    return paths


if __name__ == '__main__':
    # only needs to be run once
    # downsample_episode_dataset(dir_path=project_path+'/../../data_6_30/', pattern="tapping_from_homed_6-30*.pkl")

    states, actions, images = load_episode_dataset_no_encode(dir_path=project_path+'/../../data_downsampled/', pattern="tapping_from_homed_6-30*.pkl")

    print(project_path)

    params = {
            'save_location': project_path+'/../../policies/visiondiffusion_7-07_obs_hor_2_pred_32.pkl',
            'save_epoch_freq': 1000,
            'distort': True,
            'num_epochs': 12000,
            'diffusion_iterations': 10,
            'obs_horizon': 2,
            'pred_horizon' : 32,
            'action_horizon' : 4,
            'vision_model_name': 'resnet18',
            'freeze_encoder': False,
        }
    
    policy = VisionDiffusionUNet(params)
    
    policy.train(states, actions, images, params['save_location'], save_epoch_freq=params['save_epoch_freq'], distort=True, freeze_encoder=params['freeze_encoder'])
    policy.save(params['save_location'])
    print("Finished saving ",policy.__class__.__name__," at ",params['save_epoch_freq'])


    # params2 = {
    #         'save_location': project_path+'/../../policies/visiondiffusion_6-17_obs_hor_4_pred_32.pkl',
    #         'save_epoch_freq': 1000,
    #         'distort': True,
    #         'num_epochs': 10000,
    #         'diffusion_iterations': 10,
    #         'obs_horizon': 4,
    #         'pred_horizon' : 32,
    #         'action_horizon' : 4,
    #         'vision_model_name': 'resnet18',
    #         'freeze_encoder': True
    #     }
    
    # policy2 = VisionDiffusionUNet(params2)
    # policy2.train(states, actions, images, params2['save_location'], save_epoch_freq=params2['save_epoch_freq'], distort=True, freeze_encoder=params2['freeze_encoder'])
    # policy2.save(params2['save_location'])
    # print("Finished saving ",policy.__class__.__name__," at ",params2['save_epoch_freq'])