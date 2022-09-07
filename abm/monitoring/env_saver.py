"""
@author: mezdahun
@description: Helper functions to save all environmental variables/parameters
    of a simulation initialization as a json file, so that it can be loaded
    for simulation reproducibility.
"""

from dotenv import dotenv_values
import json
import os

from abm.contrib import ifdb_params as ifdbp


def save_env_vars(env_files_list, json_path, pop_num=None):
    """Reading env variables from multiple files (if necessary) and saving them into
    a single json file in the data folder"""
    all_env_vars_dict = {}
    for file_path in env_files_list:
        envconf = dotenv_values(file_path)
        for k, v in envconf.items():
            all_env_vars_dict[k] = v

    if pop_num is None:
        save_path = os.path.join(ifdbp.TIMESTAMP_SAVE_DIR, json_path)
    else:
        save_path = os.path.join(ifdbp.TIMESTAMP_SAVE_DIR+f"_pop{pop_num}", json_path)
    with open(save_path, 'w') as f:
        json.dump(all_env_vars_dict, f, indent=4)