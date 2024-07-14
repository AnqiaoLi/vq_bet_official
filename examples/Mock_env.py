import torch
import matplotlib.pyplot as plt
import os

# define a dict of {index: name}
index_to_name_dict = {
    0: "velocity_x",
    1: "velocity_y",
    2: "velocity_z",
    3: "angular_velocity_x",
    4: "angular_velocity_y",
    5: "angular_velocity_z",
    6: "position_x",
    7: "position_y",
    8: "position_z",
    9: "euler_z",
    10: "euler_y",
    11: "euler_x",
    30: "object_p",
    31: "object_v"
}

class MockEnv():
    """ Mock environment for rolling out the model"""
    def __init__(self, cfg, data, num_env = 2, history_stat_index = 0, freeze_angle = False):
        self.num_env = num_env
        self.data = data.dataset.dataset.observations
        self.obs_length = cfg.model.obs_window_size
        self.history_stat_index = history_stat_index
        self.freeze_angle = freeze_angle
        self.device = cfg.device

        self.env_indicies = torch.randperm(self.data.shape[0])[:num_env]
        # expend the env_indicies to the number of environments, if the dataset is too small
        while self.env_indicies.shape[0] < num_env:
            self.env_indicies = torch.cat([self.env_indicies, torch.randperm(self.data.shape[0])[:num_env - self.env_indicies.shape[0]]], dim = 0)
        self.init_state = self.data[self.env_indicies, self.history_stat_index:self.obs_length + self.history_stat_index]
        self.state = None
        self.reference_state = self.data[self.env_indicies, self.history_stat_index:]
        self.reset()

    ##########################################
    # Functions for setting the initial state
    def set_start_angle(self, start_state):
        """set start angle for the oven"""
        self.init_state[: ,:, -2 ] = start_state
        self.reset()

    def random_start_angle(self):
        """set random start angle for the oven from 0 to -1.5"""
        self.init_state[: ,:, -2 ] = - torch.rand((self.num_env, 1)).to(self.device) * 1.5
        self.reset()

    def set_angle_list(self, angle_list):
        """set the different ovens start from different angles"""
        # check if the length of the angle_list is the same as the number of environments
        assert len(angle_list) == self.num_env, "The length of the angle list should be the same as the number of environments"
        angle_list = torch.tensor(angle_list).to(self.device)
        angle_list = angle_list.view(-1, 1)
        if type(angle_list) != torch.Tensor:
            angle_list = torch.tensor(angle_list).to(self.device)
        self.init_state[: ,:, -2 ] = angle_list
        self.reset()
    ##########################################

    def reset(self):
        self.state = self.init_state
        self.state_list = torch.zeros((self.num_env, 0, self.data.shape[-1])).to(self.device)
        self.action_list = torch.zeros((self.num_env, 0, self.data.shape[-1])).to(self.device)
        self.state_list = torch.cat([self.state_list, self.init_state], dim = 1)        

    def step(self, action, mode = "r"):
        """ step the state with transformer output 
        Args:
            action: torch.tensor, shape (batch_size, action_dim)
            mode: str, "r" or "f", "r" for residual, "f" for full state
        """
        # if action includes contact, remove it
        if action.shape[-1] > self.state.shape[-1]:
            action = action[:, :, :self.state.shape[-1]]
        # Debug:keep the angle of the oven unchanged
        if self.freeze_angle:
                action[:, 0, -2] = self.state[:, -1, -2]
                action[:, 0, -1] = 0

        if mode == "r":
            self.state = torch.cat([self.state[:, 1:], 
                                    self.state[:, -1:] + action], dim = 1)
        elif mode == "f":
            self.state = torch.cat([self.state[:, 1:], 
                                    action], dim=1) 

        # append the state and action to the container
        self.state_list = torch.cat([self.state_list, self.state[:, -1:]], dim = 1)
        self.action_list = torch.cat([self.action_list, action], dim = 1)

    def get_obs(self):
        return self.state   

    def plot_different_state(self, plt_indicies = [6, 7, 8, 9, 10, 11], plt_time = 100, plot_env_num = 2, separate_env = False):
        """ plot the difference between the state and the reference state in each environment"""
        if separate_env:
            fig, axs = plt.subplots(plot_env_num, len(plt_indicies), figsize = (10*len(plt_indicies), 10))
            text_size = 20
            line_width = 3
            plt.rcParams['font.size'] = text_size
            for i in range(plot_env_num):
                for j in range(len(plt_indicies)):
                    axs[i, j].plot(self.state_list[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), label = "state", linewidth = line_width)
                    axs[i, j].plot(self.reference_state[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), label = "reference", linewidth = line_width)
                    axs[i, j].set_title(f"env {i}, {index_to_name_dict[plt_indicies[j]]}")
            # plt.show()
            plt.legend()
        else:
            fig, axs = plt.subplots(1, len(plt_indicies), figsize = (10*len(plt_indicies), 5))
            text_size = 20
            line_width = 3
            plt.rcParams['font.size'] = text_size
            for j in range(len(plt_indicies)):
                for i in range(plot_env_num):
                    axs[j].plot(self.state_list[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), linewidth = line_width)
                    # axs[j].plot(self.reference_state[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), label = "reference", linewidth = line_width)
                    axs[j].set_title(f"{index_to_name_dict[plt_indicies[j]]}")

            # plt.show()
            plt.legend()
        return fig
    
    def save_data(self, path):
        save_params = {
            "state_list": self.state_list,
            "action_list": self.action_list,
            "reference_state": self.reference_state,
            "history_stat_index": self.history_stat_index,
        }
        if ".pt" not in path:
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(save_params, path + "/rolling_out_data_from_{}.pt".format(self.history_stat_index))
        else:
            if not os.path.exists(path.rsplit("/", 1)[0]):
                os.makedirs(path.rsplit("/", 1)[0])
            torch.save(save_params, path)