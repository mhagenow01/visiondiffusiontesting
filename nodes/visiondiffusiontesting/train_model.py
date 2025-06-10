#!/usr/bin/env python3
import sys
import os
import glob
import pickle
import numpy as np
import time

from vision_diffusion_unet import VisionDiffusionUNet

project_path = os.path.dirname(os.path.realpath(__file__))

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
    

if __name__ == '__main__':
       
    states, actions, images = load_episode_dataset_no_encode(dir_path=project_path+'../../data/', pattern="tapping_6-6*.pkl")

    params = {
            'save_location': project_path+'../../policies/visiondiffusion_6-10_obs_hor_2_pred_32.pkl',
            'save_epoch_freq': 1000,
            'distort': True,
            'num_epochs': 2,
            'obs_horizon': 2,
            'pred_horizon' : 32,
            'action_horizon' : 4,
            'vision_model_name': 'none',
        }
    
    policy = VisionDiffusionUNet(params)
    policy.train(states, actions, images, params.savelocation, save_epoch_freq=params.save_epoch_freq, distort=True)
    policy.save(params.savelocation)
    print("Finished saving ",policy.__class__.__name__," at ",params.savelocation)