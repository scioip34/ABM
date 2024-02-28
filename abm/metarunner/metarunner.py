"""
metarunner.py: including the main classes and methods to programatically ruzn metaprotocols of simulations, i.e.
     for some parameter search.
"""
import importlib
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

class TunedPairRestrain:
    """Parameter pair to be restrained together with multiplication"""
    def __init__(self, var_name1, var_name2, restrained_product):
        self.var1 = var_name1
        self.var2 = var_name2
        self.product_restrain = restrained_product

    def get_vars(self):
        return [self.var1, self.var2]

    def print(self):
        """printing method"""
        print(f"Product of {self.var1} and {self.var2} should be {self.product_restrain}")

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

    def __init__(self, experiment_name=None, num_batches=1, parallel=False, description=None, headless=False):
        self.default_envconf = envconf
        self.tunables = []
        self.tuned_pairs = []
        self.q_tuned_pairs = []
        self.sum_tuned_pairs = []
        self.experiment_name = experiment_name
        self.num_batches = num_batches
        self.description = description
        self.headless = headless
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

    def add_tuned_pair(self, tuned_pair):
        self.tuned_pairs.append(tuned_pair)
        print("---Added new restrained pair: ")
        tuned_pair.print()

    def add_quadratic_tuned_pair(self, tuned_pair):
        self.q_tuned_pairs.append(tuned_pair)
        print("---Added new restrained *quadratic* pair: ")
        tuned_pair.print()

    def add_sum_tuned_pair(self, tuned_pair):
        self.sum_tuned_pairs.append(tuned_pair)
        print("---Added new restrained *sum* pair: ")
        tuned_pair.print()


    def consider_tuned_pairs(self, combos):
        """removing combinations from a list of combinations where a tuned pair criterion is not met"""
        tunable_names = [t.name for t in self.tunables]
        new_combos = combos.copy()
        for sum_tuned_pair in self.sum_tuned_pairs:
            for i, combo in enumerate(combos):
                print("combo", combo)
                sum = 0.0
                for value in combo[:2]:

                        sum += value

                if sum != 1.0:
                    print("POP")
                    new_combos.remove(combo)
        print("new_combos", new_combos)

        for tuned_pair in self.tuned_pairs:
            for i, combo in enumerate(combos):
                print("combo", combo)
                product = 1
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if name in tuned_pair.get_vars():
                        product *= value
                if product != tuned_pair.product_restrain:
                    print("POP")
                    try:
                        new_combos.remove(combo)
                    except ValueError:
                        print("combo already removed")
        for tuned_pair in self.q_tuned_pairs:
            for i, combo in enumerate(combos):
                print("combo", combo)
                product = 1
                for j, value in enumerate(combo):
                    name = tunable_names[j]
                    if name == tuned_pair.get_vars()[0]:
                        product *= value
                    elif name == tuned_pair.get_vars()[1]:
                        product *= value * value
                if not np.isclose(product,tuned_pair.product_restrain):
                    print("POP")
                    try:
                        new_combos.remove(combo)
                    except ValueError:
                        print("combo already removed")
        return new_combos

    def generate_temp_env_files(self, num_samples=None):
        """generating a batch of env files that will describe the metaprotocol in a temporary folder"""
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)

        if os.path.isdir(temp_dir):
            warnings.warn("Temprary directory for env files is not empty and will be overwritten")
            shutil.rmtree(temp_dir)

        tunable_names = [t.name for t in self.tunables]
        tunable_values = [t.get_values() for t in self.tunables]
        combos = list(itertools.product(*tunable_values))

        combos = self.consider_tuned_pairs(combos)
        #TODO: Remove this segment and the num_samples arg (I used it to for hyperparameter tuning and evaluation )
        # 1. add seed attribute in .env file to have different executions
        # 2. for hyperparameter tuning with random search, I should use optuna instead
        '''
        if num_samples is not None:
            if len(combos) == 1:
                tmp =[]
                #make duplicates
                while len(tmp) < num_samples:
                    tmp = tmp + combos
                combos = tmp
            else:
                combos = random.sample(combos, num_samples+1)
        '''


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
            root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
            if self.experiment_name is None:
                experiment_folder = os.path.join(root_abm_dir, "abm/data/simulation_data", "UnknownExp")
            else:
                experiment_folder = os.path.join(root_abm_dir, "abm/data/simulation_data", self.experiment_name)
            description_path = os.path.join(experiment_folder, "README.txt")
            os.makedirs(experiment_folder, exist_ok=True)
            with open(description_path, "w") as readmefile:
                readmefile.write(self.description)

    def run_protocol(self, env_path, project="Base"):



        """Runs a single simulation run according to an env file given by the env path"""
        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        default_env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")

        backup_default_env = os.path.join(root_abm_dir, ".env-orig")
        if os.path.isfile(default_env_path) and not os.path.isfile(backup_default_env):
            shutil.copyfile(default_env_path, backup_default_env)
        os.remove(default_env_path)
        shutil.copy(env_path, default_env_path)
        os.remove(env_path)

        import abm.contrib.ifdb_params as ifdbp
        importlib.reload(ifdbp)


        # here we run the simulation
        if project == "Base":
            app.start(parallel=self.parallel_run, headless=self.headless)
        if project == "MADRLForaging":
            from abm import app_madrl_foraging
            import abm.projects.madrl_foraging.madrl_contrib.madrl_learning_params as madrlp
            importlib.reload(madrlp)
            print("Running MADRLForaging")

            app_madrl_foraging .start(parallel=self.parallel_run, headless=self.headless)
        elif project == "CoopSignaling":
            from abm import app_collective_signaling
            app_collective_signaling.start(parallel=self.parallel_run, headless=self.headless)
        elif project == "VisualFlocking":
            from abm import app_visual_flocking
            app_visual_flocking.start(parallel=self.parallel_run, headless=self.headless)

        os.remove(default_env_path)
        shutil.copyfile(backup_default_env, default_env_path)
        sleep(2)

    def run_protocols(self, project="Base"):
        """Running all remaining protocols in tep env folder"""
        self.save_description()

        root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        temp_dir = os.path.join(root_abm_dir, self.temp_dir)

        glob_pattern = os.path.join(temp_dir, "*.env")
        print("found files: ", sorted(glob.iglob(glob_pattern)))

        i = 1
        for env_path in sorted(glob.iglob(glob_pattern)):
            print(f"Running protocol {i}/{len(sorted(glob.iglob(glob_pattern)))}")
            self.run_protocol(env_path, project=project)
            i += 1
