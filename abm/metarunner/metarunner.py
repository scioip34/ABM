"""
metarunner.py: including the main classes and methods to programatically ruzn metaprotocols of simulations, i.e.
     for some parameter search.
"""
import shutil
import numpy as np
import os
import itertools
from dotenv import dotenv_values
import warnings
from abm import app
import glob
from time import sleep

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)


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
            warnings.warn("Both value borders and override values are defined when creating Tunable, using override"
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

    def __init__(self, experiment_name=None, num_batches=1, parallel=False, description=None):
        self.default_envconf = envconf
        self.tunables = []
        self.experiment_name = experiment_name
        self.num_batches = num_batches
        self.description = description
        # in case we want to run multiple experiemnts in different terminals set this to True
        if experiment_name is None and parallel==True:
            raise Exception("Can't run multiple experiments parallely without experiment name!")
        self.parallel_run = parallel
        if self.experiment_name is not None:
            self.temp_dir = f"abm/data/metaprotocol/temp/{self.experiment_name}"
        else:
            self.temp_dir = "abm/data/metaprotocol/temp"

    def add_criterion(self, criterion):
        """Adding a criterion to the metaprotocol as a Tunable or Constant"""
        self.tunables.append(criterion)
        print("---Added new criterion to MetaProtocol: ")
        criterion.print()

    def generate_temp_env_files(self):
        """generating a batch of env files that will describe the metaprotocol in a temporary folder"""
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)

        if os.path.isdir(temp_dir):
            warnings.warn("Temprary directory for env files is not empty and will be overwritten")
            shutil.rmtree(temp_dir)

        tunable_names = [t.name for t in self.tunables]
        tunable_values = [t.get_values() for t in self.tunables]
        combos = list(itertools.product(*tunable_values))

        print(f"Generating {len(combos)} env files for simulations")

        for nb in range(self.num_batches):
            for i, combo in enumerate(combos):
                new_envconf = self.default_envconf.copy()
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if not isinstance(value, bool):
                        new_envconf[name] = value
                    else:
                        new_envconf[name] = int(value)
                if self.experiment_name is None:
                    new_envconf["SAVE_ROOT_DIR"] = os.path.join("abm/data/simulation_data", "UnknownExp", f"batch_{nb}")
                else:
                    new_envconf["SAVE_ROOT_DIR"] = os.path.join("abm/data/simulation_data", self.experiment_name, f"batch_{nb}")

                generate_env_file(new_envconf, f"{i}_b{nb}.env", temp_dir)

        print(f"Env files generated according to criterions!")

    def save_description(self):
        """Saving description text as txt file in the experiment folder"""
        if self.description is not None:
            if self.experiment_name is None:
                experiment_folder = os.path.join("abm/data/simulation_data", "UnknownExp")
            else:
                experiment_folder = os.path.join("abm/data/simulation_data", self.experiment_name)
            description_path = os.path.join(experiment_folder, "README.txt")
            os.makedirs(experiment_folder, exist_ok=True)
            with open(description_path, "w") as readmefile:
                readmefile.write(self.description)

    def run_protocol(self, env_path):
        """Runs a single simulation run according to an env file given by the env path"""
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        default_env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
        backup_default_env = os.path.join(root_abm_dir, ".env-orig")
        if os.path.isfile(default_env_path) and not os.path.isfile(backup_default_env):
            shutil.copyfile(default_env_path, backup_default_env)
        os.remove(default_env_path)
        os.rename(env_path, default_env_path)
        # here we run the simulation
        app.start(self.parallel_run)
        os.remove(default_env_path)
        shutil.copyfile(backup_default_env, default_env_path)
        sleep(2)

    def run_protocols(self):
        """Running all remaining protocols in tep env folder"""
        self.save_description()

        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)

        glob_pattern = os.path.join(temp_dir, "*.env")
        print("found files: ", sorted(glob.iglob(glob_pattern)))
        i = 1
        for env_path in sorted(glob.iglob(glob_pattern)):
            print(f"Running protocol {i}/{len(sorted(glob.iglob(glob_pattern)))}")
            self.run_protocol(env_path)
            i+=1
