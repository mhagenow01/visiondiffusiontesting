#!/usr/bin/env python3
import sys
import os
import glob
import pickle

project_path = os.path.dirname(os.path.realpath(__file__))

from vision_diffusion import VisionDiffusion


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
        with open(fname, 'rb') as f:
            states_arr, actions_arr, images_arr = pickle.load(f)
        states_list.append(states_arr)
        actions_list.append(actions_arr)
        images_list.append(images_arr)

    return states_list, actions_list, images_list
    


if __name__ == '__main__':
   
    savelocation = project_path+'/saved_policies/diffusion_ur5e_4-7-24.pkl'

    policy = VisionDiffusion(num_epochs=100, pred_horizon=32)

  
    states, actions, images = load_episode_dataset(dir_path=".", pattern="episode_*.pkl")

    policy.train(states, actions, images)

    policy.save(savelocation)

    print("Finished saving ",policy.__class__.__name__," at",savelocation)