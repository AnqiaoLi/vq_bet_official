from torch.nn import ParameterDict
import torch
import tqdm

# normlizer, normlize the data before passing into the encoder
class Normalizer:
    def __init__(self, mode = "limits", device = "cuda"):
        # mode: "limits" or "gaussian"
        self.mode = mode
        self.params_dict = {}
        self.params_dict["mode"] = mode
        self.device = device

    def _fit(self, data, output_max = 1, output_min = -1):
        # data: {key: (N, D)}
        if self.mode == "limits":
            for key, value in data.items():
                self.params_dict[key] = {
                    "min": data[key].min(axis = 0).values,
                    "max": data[key].max(axis = 0).values,
                    "output_max": output_max,
                    "output_min": output_min,
                }
        elif self.mode == "gaussian":
            for key, value in data.items():
                self.params_dict[key] = {
                    "mean": data[key].mean(axis = 0).values,
                    "std": data[key].std(axis = 0).values,
                }
        
    def fit(self, data, output_max = 1, output_min = -1):
        # data: each trajectory is a tuple of (obs, act). obs: (N, D), act: (N, D)
        # change it to the format of {obs: (N, D), act: (N, D)}
        obs_dim, act_dim = data[0][0].shape[1], data[0][1].shape[1]
        data_dict = {"obs": torch.zeros((0, obs_dim)).to(self.device), "act": torch.zeros((0, act_dim)).to(self.device)}

        for data_trajectory in tqdm.tqdm(data, desc="Fitting normalizer"):
            data_dict["obs"] = torch.cat((data_dict["obs"], data_trajectory[0]), dim = 0)
            data_dict["act"] = torch.cat((data_dict["act"], data_trajectory[1]), dim = 0)
        self._fit(data_dict, output_max, output_min)

    def normalize(self, data):
        # data: {key: (N, D)}
        ndata = {}
        for key, value in data.items():
            if self.mode == "limits":
                input_range = self.params_dict[key]["max"] - self.params_dict[key]["min"]
                output_range = self.params_dict[key]["output_max"] - self.params_dict[key]["output_min"]
                ndata[key] = (data[key] - self.params_dict[key]["min"]) / input_range * output_range + self.params_dict[key]["output_min"]
                ndata[key][:, :, input_range == 0] = data[key][:, :, input_range == 0] - self.params_dict[key]["min"][input_range == 0]
            elif self.mode == "gaussian":
                ndata[key] = (data[key] - self.params_dict[key]["mean"]) / self.params_dict[key]["std"]
        return ndata

    def denormalize(self, ndata):
        # ndata: {key: (N, D)}
        data = {}
        for key, value in ndata.items():
            if self.mode == "limits":
                input_range = self.params_dict[key]["max"] - self.params_dict[key]["min"]
                output_range = self.params_dict[key]["output_max"] - self.params_dict[key]["output_min"]
                data[key] = (ndata[key] - self.params_dict[key]["output_min"]) / output_range * input_range + self.params_dict[key]["min"]
                data[key][:, :, input_range == 0] = ndata[key][:, :, input_range == 0] + self.params_dict[key]["min"][input_range == 0]
            elif self.mode == "gaussian":
                data[key] = ndata[key] * self.params_dict[key]["std"] + self.params_dict[key]["mean"]
        return data
        

    def save_params(self, path):
        torch.save(self.params_dict, path)

    def load_params(self, params_dict):
        self.params_dict = params_dict
        self.mode = params_dict["mode"]

    def get_params(self):
        return self.params_dict


    
