"""
data_loader.py : including the main classes to load previously saved data (csv+json) into an initialized replayable simulation.
    The DataLoader class is only the data layer that loads data and then can create a LoadedSimulation instance accordingly.
"""

import json
import os
from abm.agent.agent import Agent
from abm.loader import helper as dh
import numpy as np


class LoadedAgent(Agent):
    """
    LoadedAgent class inheriting from Agent. Initialization now happens with timeseries of data and the update
        process takes timestep as argument so it updates the state of the agent to that time.
    """
    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion,
                 pos_data, or_data, rew_data):
        """Init method of LoadedAgent class. Parameters as for agents due to inheritance. Additional parameters are:
            :param pos_data: time serties of position data
            :param or_data: time series of orientation data
            :param rew_data: time series of reward consumtioon data"""
        # Initializing superclass with passed arguments
        super().__init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion)


class DataLoader:
    """
    DataLoader class that loads a csv and json file and initializes a LoadedSimulation instance with the loaded
    parameters
    """

    def __init__(self, data_folder_path):
        """
        Initalization method of main DataLoader class

        :param data_folder_path: path of the folder to be loaded. Inside there must be the following data files:
            - agent_data.csv: including time series of agent data
            - resource_data.csv: including time series of resource data
            - env_params.json: json file including all environmental variables that was used for the simulation with
                the given data folder.
        """
        # Initializing DataLoader paths according to convention
        self.data_folder_path = data_folder_path
        self.agent_csv_path = os.path.join(self.data_folder_path, "agent_data.csv")
        self.resource_csv_path = os.path.join(self.data_folder_path, "resource_data.csv")
        self.env_json_path = os.path.join(self.data_folder_path, "env_params.json")
        self.agent_data = {}
        self.resource_data = {}
        self.env_data = {}
        self.load_files()
        self.preprocess_data()

    def load_files(self):
        """loading agent and resource data files into memory and make post-processing on time series"""
        self.agent_data = dh.load_csv_file(self.agent_csv_path)
        self.resource_data = dh.load_csv_file(self.resource_csv_path)
        with open(self.env_json_path, "r") as file:
            self.env_data = json.load(file)

    def preprocess_data(self):
        """preprocessing loaded data structures"""
        time_len = len(self.agent_data["t"])
        new_time = [i for i in range(time_len)]
        self.agent_data["t"] = new_time
        self.resource_data["t"] = new_time

        # Change env data types
        self.env_data["N"] = int(self.env_data["N"]),
        self.env_data["T"] = int(self.env_data["T"]),
        self.env_data["VISUAL_FIELD_RESOLUTION"] = int(self.env_data["VISUAL_FIELD_RESOLUTION"]),
        self.env_data['AGENT_FOV'] = float(self.env_data['AGENT_FOV']),
        self.env_data["INIT_FRAMERATE"] = int(self.env_data["INIT_FRAMERATE"]),
        self.env_data["WITH_VISUALIZATION"] = bool(int(self.env_data["WITH_VISUALIZATION"])),
        self.env_data["ENV_WIDTH"] = int(self.env_data["ENV_WIDTH"]),
        self.env_data["ENV_HEIGHT"] = int(self.env_data["ENV_HEIGHT"]),
        self.env_data["SHOW_VISUAL_FIELDS"] = bool(int(self.env_data["SHOW_VISUAL_FIELDS"])),
        self.env_data["POOLING_TIME"] = int(self.env_data["POOLING_TIME"]),
        self.env_data["POOLING_PROBABILITY"] = float(self.env_data["POOLING_PROBABILITY"]),
        self.env_data["RADIUS_AGENT"] = int(self.env_data["RADIUS_AGENT"]),
        self.env_data["N_RESOURCES"] = int(self.env_data["N_RESOURCES"]),
        self.env_data["MIN_RESOURCE_PER_PATCH"] = int(self.env_data["MIN_RESOURCE_PER_PATCH"]),
        self.env_data["MAX_RESOURCE_PER_PATCH"] = int(self.env_data["MAX_RESOURCE_PER_PATCH"]),
        self.env_data["MIN_RESOURCE_QUALITY"] = float(self.env_data["MIN_RESOURCE_QUALITY"]),
        self.env_data["MAX_RESOURCE_QUALITY"] = float(self.env_data["MAX_RESOURCE_QUALITY"]),
        self.env_data["RADIUS_RESOURCE"] = int(self.env_data["RADIUS_RESOURCE"]),
        self.env_data["REGENERATE_PATCHES"] = bool(int(self.env_data["REGENERATE_PATCHES"])),
        self.env_data["AGENT_CONSUMPTION"] = int(self.env_data["AGENT_CONSUMPTION"]),
        self.env_data["GHOST_WHILE_EXPLOIT"] = bool(int(self.env_data["GHOST_WHILE_EXPLOIT"])),
        self.env_data["PATCHWISE_SOCIAL_EXCLUSION"] = bool(int(self.env_data["PATCHWISE_SOCIAL_EXCLUSION"])),
        self.env_data["TELEPORT_TO_MIDDLE"] = bool(int(self.env_data["TELEPORT_TO_MIDDLE"])),
        self.env_data["VISION_RANGE"] = int(self.env_data["VISION_RANGE"]),
        self.env_data["VISUAL_EXCLUSION"] = bool(int(self.env_data["VISUAL_EXCLUSION"])),
        self.env_data["SHOW_VISION_RANGE"] = bool(int(self.env_data["SHOW_VISION_RANGE"])),
        self.env_data["USE_IFDB_LOGGING"] = bool(int(self.env_data["USE_IFDB_LOGGING"])),
        self.env_data["SAVE_CSV_FILES"] = bool(int(self.env_data["SAVE_CSV_FILES"]))

        self.agent_data['vfield_up_agent-01']

        #Change time-series data types
        for k, v in self.agent_data.items():
            if k.find("vfield") == -1:
                self.agent_data[k] = np.array([float(i) for i in v])
            else:
                self.agent_data[k] = np.array([json.loads(i.replace(" ", ", ")) for i in v], dtype=object)

        for k, v in self.resource_data.items():
                self.resource_data[k] = np.array([float(i) for i in v])

    def get_loaded_data(self):
        """returning the loaded data upon request"""
        return self.agent_data, self.resource_data, self.env_data


class LoadedSimulation:
    def __init__(self, data_folder_path):
        """Init method of LadedSimulation class to initialize a simulation-like structure according to
        previously saved data"""
        self.agent_data, self.resource_data, self.env_data = DataLoader(data_folder_path).get_loaded_data()

