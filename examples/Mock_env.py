import torch
import matplotlib.pyplot as plt

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
    def __init__(self, cfg, data, num_env = 2, history_stat_index = 0):
        self.num_env = num_env
        self.data = data.dataset.dataset.observations
        self.obs_length = cfg.model.obs_window_size
        self.history_stat_index = history_stat_index
        self.device = cfg.device

        self.env_indicies = torch.randperm(self.data.shape[0])[:num_env]
        self.init_state = self.data[self.env_indicies, self.history_stat_index:self.obs_length + self.history_stat_index]
        self.state = None
        self.reference_state = self.data[self.env_indicies, self.history_stat_index:]
        self.reset()
        # # init the state with the first obs_length observations
        # self.state = self.init_state
        # # state and action container
        # self.state_list = torch.zeros((batch, 0, cfg.model.obs_dim)).to(cfg.device)
        # self.action_list = torch.zeros((batch, 0, cfg.model.act_dim)).to(cfg.device)
        # self.state_list = torch.cat([self.state_list, self.init_state], dim = 1)


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
    
    # def append_state_action(self):
    #     self.state_list = torch.cat([self.state_list, self.state[:, -1:]], dim = 1)
    #     self.action_list = torch.cat([self.action_list, self.action], dim = 1)

    def plot_different_state(self, plt_indicies = [6, 7, 8, 9, 10, 11], plt_time = 100):
        """ plot the difference between the state and the reference state in each environment"""
        fig, axs = plt.subplots(self.num_env, len(plt_indicies), figsize = (10*len(plt_indicies), 10))
        text_size = 20
        line_width = 3
        plt.rcParams['font.size'] = text_size
        for i in range(self.num_env):
            for j in range(len(plt_indicies)):
                axs[i, j].plot(self.state_list[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), label = "state", linewidth = line_width)
                axs[i, j].plot(self.reference_state[i, :plt_time, plt_indicies[j]].cpu().detach().numpy(), label = "reference", linewidth = line_width)
                axs[i, j].set_title(f"env {i}, {index_to_name_dict[plt_indicies[j]]}")
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
        torch.save(save_params["state_list"], path + "/rolling_out_data_from_{}.pt".format(self.history_stat_index))
