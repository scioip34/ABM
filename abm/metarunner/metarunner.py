"""
metarunner.py: including the main classes and methods to programatically ruzn metaprotocols of simulations, i.e.
     for some parameter search.
"""

import numpy as np
import os
import itertools
from dotenv import dotenv_values

envconf = dotenv_values("../.env")


def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")


class Constant:
    """A constant parameter value for a given parameter that shall be used for simulations"""

    def __init__(self, var_name, constant):
        """defines a single variable value with name"""
        self.tunable = Tunable(var_name, values_override=[constant])
        self.name = self.tunable.name

    def get_values(self):
        return self.tunable.values

    def print(self):
        """printing method"""
        print(f"Constant {self.tunable.name} = {self.tunable.values[0]}")


class Tunable:
    """A parameter range in which we want to loop through (explore)"""

    def __init__(self, var_name, min_v=None, max_v=None, num_data_points=None, values_override=None):
        """init method of the Tunable class. We want to loop through a parameter defined with var_name from the
        min value to the max value with num_data_points number of individual parameter values between

        In case we have specific values to loop through we can pass a list of values instead of borders and number
        of datapoints."""

        if min_v is None and values_override is None:
            raise Exception("Neither value borders nor override values have been given to create Tunable!")
        elif min_v is not None and values_override is not None:
            raise Warning("Both value borders and override values are defined when creating Tunable, using override"
                          "values as default!")

        self.name = var_name
        if values_override is None:
            self.min_val = min_v
            self.max_val = max_v
            self.n_data = num_data_points
            self.generated = True
            self.values = np.linspace(self.min_val, self.max_val, num=self.n_data, endpoint=True)
        else:
            self.min_val = min(values_override)
            self.max_val = max(values_override)
            self.n_data = len(values_override)
            self.generated = False
            self.values = values_override

    def print(self):
        """printing method"""
        print(f"Tunable: {self.name} = {self.min_val}  -  -  -n={self.n_data}-  -  -  {self.max_val}")
        print(f"Values : {self.values}")

    def get_values(self):
        return self.values


class MetaProtocol:
    """Metaprotocol class that is initialized with Tunables and runs through the desired simulations accordingly"""

    def __init__(self):
        self.default_envconf = envconf
        self.tunables = []

    def add_criterion(self, criterion):
        """Adding a criterion to the metaprotocol as a Tunable or Constant"""
        self.tunables.append(criterion)
        print("---Added new criterion to MetaProtocol: ")
        criterion.print()

    def generate_temp_env_files(self):
        """generating a batch of env files that will describe the metaprotocol in a temporary folder"""

        temp_dir = "abm/data/metaprotocol/temp"
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, temp_dir)

        tunable_names = [t.name for t in self.tunables]
        tunable_values = [t.get_values() for t in self.tunables]
        combos = list(itertools.product(*tunable_values))

        print(f"Generating {len(combos)} env files for simulations")

        for i, combo in enumerate(combos):
            new_envconf = self.default_envconf.copy()
            for j, value in enumerate(combo):
                name = tunable_names[j]
                if not isinstance(value, bool):
                    new_envconf[name] = value
                else:
                    new_envconf[name] = int(value)
            generate_env_file(new_envconf, f"{i}.env", temp_dir)

        print(f"Env files generated according to criterions!")

