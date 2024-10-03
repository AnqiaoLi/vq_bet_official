import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf
import einops
import matplotlib.pyplot as plt
import time

import kitchen_env
import wandb
from video import VideoRecorder
import pickle
from Mock_env import MockEnv
import os

config_name = "train_oven"

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


def seed_everything(random_seed: int):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


@hydra.main(config_path="configs", config_name=config_name, version_base="1.2")
def main(cfg):
    # print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    # Initialize dataset, only for initialization of the model input
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    # Initialize a mock environment to roll out the model
    train_env = MockEnv(cfg, train_data, num_env = 50, history_stat_index = 0)
    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim += 1024
    # Initialize model
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )
    # Function for evaluation on the mock environment
    @ torch.no_grad()
    def eval_on_mockenv(
        cfg,
        eval_steps = 1000,
        receding_horizon = 1,
    ):
        """ Evaluate the model on the mock environment
        Args:
            eval_steps: the number of steps to evaluate
            receding_horizon: execute steps for each action prediction 
        """
        for step in tqdm.tqdm(range(eval_steps), desc="eval on mock env"):
            if step == 0 or ai >= receding_horizon: 
                ai = 0
                obs = train_env.get_obs()
                goal = torch.zeros((obs.shape[0], cfg.model.goal_dim)) # placeholder, goal_dim = 0
                # predict the action
                action_pred, _, _ = cbet_model(obs, goal, None)
                action = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cbet_model.obs_window_size)[:, -1, 0:1, :]
                actions = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cbet_model.obs_window_size)[
                    :, -1, :, :
                ]
            action = actions[:, ai:ai+1, :]
            # step the environment
            train_env.step(action, mode = cfg.action_mode)
            ai += 1
   
        return train_env.state_list, train_env.action_list
    # change the model to evaluation
    cbet_model.eval()
    ####################################################### 
    # set different inital angle for each trajectories file 
    # for a in np.arange(0, -1.6, -0.1):
    #     train_env.set_start_angle(a)
    #     state_list, action_list = eval_on_mockenv(cfg)
    #     train_env.save_data("/home/anqiao/MasterThesis/Data/OvenOpening/pt_vis/mangle_0-1.2/start_{:.2f}_rolling_from_0_16envs.pt".format(a))
    #######################################################
    ## set different initial angle in one trajectories
    # angle_list = np.arange(0, -1.6, -0.1)
    # angle_list.fill(-0.1)
    # train_env.set_angle_list(angle_list)
    #######################################################
    ## set to true is fix the oven angle
    # train_env.freeze_angle = True
    #######################################################
    # normal evaluation
    state_list, action_list = eval_on_mockenv(cfg, eval_steps = 2000)
    fig_1 = train_env.plot_different_state(plt_indicies = [2, 7, 8, 45, 46, 47], plt_time = 2000, separate_env = False, plot_env_num=50) # see Mock_env.py for indicies_name pairs 
    fig_2 = train_env.plot_different_state(plt_indicies = [2, 7, 8, 45, 46, 47], plt_time = 2000, separate_env = True)
    plt.show()
    # save the trajectory
    # train_env.save_data("/media/anqiao/AnqiaoT7/MasterThesis/Data/OvenOpening/pt_vis/task_8/res_receding_1.pt")
    # fig = train_env.plot_different_state(plt_indicies = [6, 7, 8, 30, 31], plt_time = 1500)

    return 0        


if __name__ == "__main__":
    main()
