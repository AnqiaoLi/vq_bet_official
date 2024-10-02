import torch
import pandas as pd


# dict of indiices and corresponding names of robot state for state-input model
STATE_INPUT_INDICIES_DICT = {
    0: "root_position_x_handle",
    1: "root_position_y_handle",
    2: "root_position_z_handle",
    3: "root_gso_x1",
    4: "root_gso_y1",
    5: "root_gso_z1",
    6: "root_gso_x2",
    7: "root_gso_y2",
    8: "root_gso_z2",
    9: "projected_gravity_x_base",
    10: "projected_gravity_y_base",
    11: "projected_gravity_z_base",
    12: "root_linear_velocity_x_base",
    13: "root_linear_velocity_y_base",
    14: "root_linear_velocity_z_base",
    15: "root_angular_velocity_x_base",
    16: "root_angular_velocity_y_base",
    17: "root_angular_velocity_z_base",
    18: "joint_position_1",
    19: "joint_position_2",
    20: "joint_position_3",
    21: "joint_position_4",
    22: "joint_position_5",
    23: "joint_position_6",
    24: "joint_position_7",
    25: "joint_position_8",
    26: "joint_position_9",
    27: "joint_position_10",
    28: "joint_position_11",
    29: "joint_position_12",
    30: "joint_position_13",
    31: "joint_position_14",
    32: "joint_position_15",
    33: "joint_position_16",
    34: "joint_position_17",
    35: "joint_position_18",
    36: "ee_position_x_base",
    37: "ee_position_y_base",
    38: "ee_position_z_base",
    39: "ee_gso_x1",
    40: "ee_gso_y1",
    41: "ee_gso_z1",
    42: "ee_gso_x2",
    43: "ee_gso_y2",
    44: "ee_gso_z2",
    45: "door_position",
    46: "handle_ee_distance",
    47: "phase"
}

# dict of indiices and corresponding names of robot state for visual-input model
# robot_state_mappings.py

VISUAL_INPUT_INDICIES_DICT = {
    0: "projected_gravity_x_base",
    1: "projected_gravity_y_base",
    2: "projected_gravity_z_base",
    3: "root_linear_velocity_x_base",
    4: "root_linear_velocity_y_base",
    5: "root_linear_velocity_z_base",
    6: "root_angular_velocity_x_base",
    7: "root_angular_velocity_y_base",
    8: "root_angular_velocity_z_base",
    9: "joint_position_1",
    10: "joint_position_2",
    11: "joint_position_3",
    12: "joint_position_4",
    13: "joint_position_5",
    14: "joint_position_6",
    15: "joint_position_7",
    16: "joint_position_8",
    17: "joint_position_9",
    18: "joint_position_10",
    19: "joint_position_11",
    20: "joint_position_12",
    21: "joint_position_13",
    22: "joint_position_14",
    23: "joint_position_15",
    24: "joint_position_16",
    25: "joint_position_17",
    26: "joint_position_18",
    27: "ee_position_x_base",
    28: "ee_position_y_base",
    29: "ee_position_z_base",
    30: "ee_gso_x1",
    31: "ee_gso_y1",
    32: "ee_gso_z1",
    33: "ee_gso_x2",
    34: "ee_gso_y2",
    35: "ee_gso_z2",
    36: "handle_ee_distance",
    37: "root_position_x_handle",
    38: "root_position_y_handle",
    39: "root_position_z_handle",
    40: "root_gso_x1",
    41: "root_gso_y1",
    42: "root_gso_z1",
    43: "root_gso_x2",
    44: "root_gso_y2",
    45: "root_gso_z2",
    46: "door_position",
    47: "phase"
}

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
    
# STATE_INPUT_DIM = 46
# VISUAL_INPUT_DIM = 36


# # Build state noise vector
# data = {'Index': [], 'State Name': [], 'Noise Value': []}
# state_noise_vector = torch.empty(len(STATE_INPUT_INDICIES_DICT))
# for index, state_name_element_wise in STATE_INPUT_INDICIES_DICT.items():
#     found = False
#     for state_name, noise in state_noise_dict.items():
#         if state_name in state_name_element_wise:
#             state_noise_vector[index] = state_noise_dict[state_name]
#             state_noise_vector[index] = noise
#             # For logging
#             data['Index'].append(index)
#             data['State Name'].append(state_name_element_wise)
#             data['Noise Value'].append(state_noise_dict[state_name])
#             found = True
#             break
#     if not found:
#         raise Exception(f"Cannot find the state noise for state {state_name_element_wise}. Please check the state_noise_dict")


# df = pd.DataFrame(data)
# print(df)