#!/usr/bin/env python3
import sys
import os
import glob
import pickle
import numpy as np
import time

project_path = os.path.dirname(os.path.realpath(__file__))

from vision_diffusion import VisionDiffusion, precompute_dino_feats
from vision_diffusion_resnet import VisionDiffusionResnet, precompute_resnet_feats


def load_episode_dataset(dir_path: str = '.', pattern: str = 'episode_*.pkl'):
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

    for fname in sorted(glob.glob(glob_path)):
        print(fname)
        with open(fname, 'rb') as f:
            states_arr, actions_arr, images_arr = pickle.load(f)
        states_list.append(states_arr)
        actions_list.append(actions_arr)
        print(np.shape(images_arr))
        startt = time.time()
        feats = precompute_resnet_feats(images_arr)
        print("Feature shape",np.shape(feats))
        # print(time.time()-startt)
        images_list.append(feats)

    return states_list, actions_list, images_list

def load_episode_dataset_dino(dir_path: str = '.', pattern: str = 'episode_*.pkl'):
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

    for fname in sorted(glob.glob(glob_path)):
        print(fname)
        with open(fname, 'rb') as f:
            states_arr, actions_arr, images_arr = pickle.load(f)
        states_list.append(states_arr)
        actions_list.append(actions_arr)
        print(np.shape(images_arr))
        startt = time.time()
        feats = precompute_dino_feats(images_arr)
        print("Feature shape",np.shape(feats))
        # print(time.time()-startt)
        images_list.append(feats)

    return states_list, actions_list, images_list

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
   
    # states, actions, images_resnet = load_episode_dataset(dir_path=".", pattern="tapping_6-6*.pkl")

    # savelocation1 = project_path+'/visiondiffusionresnet_ur5e_6-9_obs_hor_2_pred_32.pkl'

    # policy = VisionDiffusionResnet(num_epochs=8000, obs_horizon=2, pred_horizon=32)
    # policy.train(states, actions, images_resnet, savelocation1,save_epoch_freq=2000)
    # policy.save(savelocation1)
    # print("Finished saving ",policy.__class__.__name__," at",savelocation1)

    # states, actions, images_dino = load_episode_dataset_dino(dir_path=".", pattern="tapping_6-6*.pkl")

    # savelocation2 = project_path+'/visiondiffusion_ur5e_6-9_obs_hor_2_pred_32.pkl'

    # policy = VisionDiffusion(num_epochs=8000, obs_horizon=2, pred_horizon=32)
    # policy.train(states, actions, images_dino, savelocation2,save_epoch_freq=2000)
    # policy.save(savelocation2)
    # print("Finished saving ",policy.__class__.__name__," at",savelocation2)

        # states, actions, images_resnet = load_episode_dataset(dir_path=".", pattern="tapping_6-6*.pkl")

    # after bringing the encoding back in!!!
    savelocation = project_path+'/visiondiffusionresnet_distory_ur5e_6-10_obs_hor_2_pred_32.pkl'
    states, actions, images = load_episode_dataset_no_encode(dir_path=".", pattern="tapping_6-6*.pkl")
    policy = VisionDiffusionResnet(num_epochs=8000, obs_horizon=2, pred_horizon=32)
    policy.train(states, actions, images, savelocation,save_epoch_freq=1000,distort=True)
    policy.save(savelocation)
    print("Finished saving ",policy.__class__.__name__," at",savelocation)