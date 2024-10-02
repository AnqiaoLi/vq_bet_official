import torch
import pandas as pd
from tabulate import tabulate
from robot_state_indices import STATE_INPUT_INDICIES_DICT, VISUAL_INPUT_INDICIES_DICT


state_noise_dict = {
    "root_position": 0.05,
    "root_gso": 0.05,
    "projected_gravity": 0.05,
    "root_linear_velocity": 0.1,
    "root_angular_velocity": 0.2,
    "joint_position": 0.05,
    "ee_position": 0.05,
    "ee_gso": 0.05,
    "door_position": 0.05,
    "handle_ee_distance": 0.05,
    "phase": 0.05,
    }    

class ObsNoise():
    def __init__(self, input_mode = "state_input", obs_dim = None):
        # Choose which indices_dict to use
        self.input_mode = input_mode
        if input_mode == "state_input":
            self.indices_dict = STATE_INPUT_INDICIES_DICT
        elif input_mode == "visual_input":
            self.indices_dict = VISUAL_INPUT_INDICIES_DICT
            
        # Build state noise vector
        self.state_noise_vector = torch.empty(obs_dim)
        logging_table = {'Index': [], 'State Name': [], 'Noise Value': []}
        for index, state_name_element_wise in STATE_INPUT_INDICIES_DICT.items():
            found = False
            for state_name, noise in state_noise_dict.items():
                if state_name in state_name_element_wise:
                    self.state_noise_vector[index] = noise
                    # For logging
                    logging_table['Index'].append(index)
                    logging_table['State Name'].append(state_name_element_wise)
                    logging_table['Noise Value'].append(state_noise_dict[state_name])
                    found = True
                    break
            if not found:
                raise Exception(f"Cannot find the state noise for state {state_name_element_wise}. Please check the state_noise_dict")
            if index == obs_dim - 1:
                break
            
        self.df_logging_table = pd.DataFrame(logging_table)
    
    def log_noise(self):
        print("Noise Vector of the" + self.input_mode + " model:")
        print(tabulate(self.df_logging_table, headers='keys', tablefmt='pipe', showindex=False))
    
    def get_noise_vector(self):
        return self.state_noise_vector
    

if __name__ == "__main__":
    obs_noise = ObsNoise(input_mode = "state_input", obs_dim = 46)
    obs_noise.log_noise()
    obs_noise.get_noise_vector()
    
    obs_noise = ObsNoise(input_mode = "visual_input", obs_dim = 36)
    obs_noise.log_noise()
    obs_noise.get_noise_vector() 

        