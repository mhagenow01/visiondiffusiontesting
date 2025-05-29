#!/usr/bin/env python3
import sys
import os
import glob
import pickle
import numpy as np
import time

project_path = os.path.dirname(os.path.realpath(__file__))

from vision_diffusion import VisionDiffusion
from vision_diffusion import precompute_dino_feats


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
        print(np.shape(images_arr))
        startt = time.time()
        precompute_dino_feats(images_arr)
        # print(time.time()-startt)
        images_list.append(precompute_dino_feats(images_arr))

    return states_list, actions_list, images_list
    



if __name__ == '__main__':
   
    savelocation = project_path+'/visiondiffusion_ur5e.pkl'

    policy = VisionDiffusion(num_epochs=1, pred_horizon=16)
  
    states, actions, images = load_episode_dataset(dir_path=".", pattern="episode_*.pkl")

    print(np.shape(states))
    policy.train(states, actions, images)

    policy.save(savelocation)

    print("Finished saving ",policy.__class__.__name__," at",savelocation)