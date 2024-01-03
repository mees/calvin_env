from copy import deepcopy
import glob
import os
from pathlib import Path
import time

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p


@hydra.main(config_path="../../conf", config_name="config_data_collection")
def run_env(cfg):
    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True)

    root_dir = Path("/home/lukas/phd/repos/hulc/dataset/calvin_debug_dataset/training")

    ep_start_end_ids = np.sort(np.load(root_dir / "ep_start_end_ids.npy"), axis=0)

    for s, e in ep_start_end_ids:
        print("new_episode")
        for i in range(s, e + 1):
            file = root_dir / f"episode_{i:07d}.npz"
            data = np.load(file)
            img = data["rgb_static"]

            obs = env.reset(scene_obs=data["scene_obs"], robot_obs=data["robot_obs"])
            img_high = obs["rgb_obs"]["rgb_static"]

            cv2.imshow("win1", img[:, :, ::-1])
            cv2.imshow("win2", img_high[:, :, ::-1])
            cv2.waitKey(1)



if __name__ == "__main__":
    run_env()
