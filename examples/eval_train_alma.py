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
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)
    train_data, test_data = hydra.utils.instantiate(cfg.data)
    train_env = MockEnv(cfg, train_data, num_env = 50, history_stat_index = 0)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, pin_memory=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, pin_memory=False
    )
    if "visual_input" in cfg and cfg.visual_input:
        print("use visual environment")
        cfg.model.gpt_model.config.input_dim = 1024
    cbet_model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    # cfg.load_path = "/home/anqiao/MasterThesis/Checkpoints/VQ-BeT/checkpoints/kitchen-v0/2024-04-02/20-56-20/sleek-pond-23"
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )
    env = hydra.utils.instantiate(cfg.env.gym)
    # goal_fn = hydra.utils.instantiate(cfg.goal_fn)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode="disabled"
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @ torch.no_grad()
    def eval_on_mockenv(
        cfg,
        eval_steps = 1000,
        batch = 2,
    ):
        receding_horizon = 20
        for step in tqdm.tqdm(range(eval_steps), desc="eval on mock env"):
            if step == 0 or ai >= receding_horizon: 
                ai = 0
                obs = train_env.get_obs()
                goal = torch.zeros((batch, cfg.model.goal_dim))
                # predict the action
                st = time.time()
                action_pred, _, _ = cbet_model(obs, goal, None)
                # print("time", time.time() - st)
                action = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cbet_model.obs_window_size)[:, -1, 0:1, :]
                # step the environment
                actions = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cbet_model.obs_window_size)[
                    :, -1, :, :
                ]
            action = actions[:, ai:ai+1, :]
            train_env.step(action, mode = cfg.action_mode)
            ai += 1


        # for step in tqdm.tqdm(range(eval_steps), desc="eval on mock env"):
        #     obs = train_env.get_obs()
        #     goal = torch.zeros((batch, cfg.model.goal_dim))
        #     # predict the action
        #     action_pred, _, _ = cbet_model(obs, goal, None)
        #     action = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cfg.model.obs_window_size)[
        #         :, -1, 0:1, :
        #     ]
        #     # step the environment
        #     train_env.step(action, mode = cfg.action_mode)
   
        return train_env.state_list, train_env.action_list
    # change the model to evaluation
    cbet_model.eval()
    # for a in np.arange(0, -1.6, -0.1):
    #     train_env.set_start_angle(a)
    #     state_list, action_list = eval_on_mockenv(cfg)
    #     train_env.save_data("/home/anqiao/MasterThesis/Data/OvenOpening/pt_vis/mangle_0-1.2/start_{:.2f}_rolling_from_0_16envs.pt".format(a))
    
    # angle_list = np.arange(0, -1.6, -0.1)
    # angle_list.fill(-0.1)
    # train_env.set_angle_list(angle_list)
    train_env.freeze_angle = False
    state_list, action_list = eval_on_mockenv(cfg, eval_steps = 2000)
    train_env.plot_different_state(plt_indicies = [30, 31, 42], plt_time = 2000, plot_env_num=2)
    train_env.save_data("/media/anqiao/AnqiaoT7/MasterThesis/Data/OvenOpening/pt_vis/task_6/res_receding_20.pt")
    fig = train_env.plot_different_state(plt_indicies = [6, 7, 8, 30, 31], plt_time = 1500)

    return 0        


if __name__ == "__main__":
    main()
