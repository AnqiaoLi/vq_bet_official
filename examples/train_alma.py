import os
import random
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import tqdm
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import einops

import kitchen_env
import wandb
from video import VideoRecorder
import pickle
from Mock_env import MockEnv


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
    train_env = MockEnv(cfg, train_data, num_env=2, history_stat_index=0)
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
    if cfg.load_path:
        cbet_model.load_model(Path(cfg.load_path))
    optimizer = cbet_model.configure_optimizers(
        weight_decay=cfg.optim.weight_decay,
        learning_rate=cfg.optim.lr,
        betas=cfg.optim.betas,
    )
    env = hydra.utils.instantiate(cfg.env.gym)
    goal_fn = hydra.utils.instantiate(cfg.goal_fn)
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        # mode="disabled"
    )
    run_name = run.name or "Offline"
    save_path = Path(cfg.save_path) / run_name
    save_path.mkdir(parents=True, exist_ok=False)
    video = VideoRecorder(dir_name=save_path)

    @torch.no_grad()
    def eval_on_mockenv(
        cfg,
        eval_steps = 1000,
        batch = 2,
    ):
        for step in tqdm.tqdm(range(eval_steps), desc="eval on mock env"):
            obs = train_env.get_obs()
            goal = torch.zeros((batch, cfg.model.goal_dim))
            # predict the action
            action_pred, _, _ = cbet_model(obs, goal, None)
            action = einops.rearrange(action_pred, "(N T) W A -> N T W A", T=cfg.model.obs_window_size)[
                :, -1, 0:1, :
            ]
            # step the environment
            train_env.step(action, mode=cfg.action_mode)
   
        return train_env.state_list, train_env.action_list


    for epoch in tqdm.trange(cfg.epochs):
        cbet_model.eval()
        if (epoch % cfg.eval_on_env_freq == 0):
            train_env.reset()
            eval_on_mockenv(cfg)
            fig = train_env.plot_different_state(plt_indicies = [6, 7, 8, 9, 10, 11], plt_time = 300)
            wandb.log({"eval_on_mockenv": wandb.Image(fig)})
            del fig

        if epoch % cfg.eval_freq == 0:
            total_loss = 0
            action_diff = 0
            action_diff_tot = 0
            action_diff_mean_res1 = 0
            action_diff_mean_res2 = 0
            action_diff_max = 0
            with torch.no_grad():
                for data in test_loader:
                    obs, act, goal = (x.to(cfg.device) for x in data)
                    predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
                    total_loss += loss.item()
                    wandb.log({"eval/{}".format(x): y for (x, y) in loss_dict.items()})
                    action_diff += loss_dict["action_diff"]
                    action_diff_tot += loss_dict["action_diff_tot"]
                    action_diff_mean_res1 += loss_dict["action_diff_mean_res1"]
                    action_diff_mean_res2 += loss_dict["action_diff_mean_res2"]
                    action_diff_max += loss_dict["action_diff_max"]
            print(f"Test loss: {total_loss / len(test_loader)}")
            wandb.log({"eval/epoch_wise_action_diff": action_diff})
            wandb.log({"eval/epoch_wise_action_diff_tot": action_diff_tot})
            wandb.log({"eval/epoch_wise_action_diff_mean_res1": action_diff_mean_res1})
            wandb.log({"eval/epoch_wise_action_diff_mean_res2": action_diff_mean_res2})
            wandb.log({"eval/epoch_wise_action_diff_max": action_diff_max})

        for data in tqdm.tqdm(train_loader):
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].zero_grad()
                optimizer["optimizer2"].zero_grad()
            else:
                optimizer["optimizer2"].zero_grad()
            obs, act, goal = (x.to(cfg.device) for x in data)
            # add noise to observation
            if cfg.noise_enhance_coef > 0:
                obs += torch.randn_like(obs) * cfg.noise_enhance_coef
            predicted_act, loss, loss_dict = cbet_model(obs, goal, act)
            wandb.log({"train/{}".format(x): y for (x, y) in loss_dict.items()})
            loss.backward()
            if epoch < (cfg.epochs * 0.5):
                optimizer["optimizer1"].step()
                optimizer["optimizer2"].step()
            else:
                optimizer["optimizer2"].step()

        if epoch % cfg.save_every == 0:
            cbet_model.save_model(save_path)



if __name__ == "__main__":
    main()
