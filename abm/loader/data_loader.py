"""
data_loader.py : including the main classes to load previously saved data (csv+json) into an initialized replayable simulation.
    The DataLoader class is only the data layer that loads data and then can create a LoadedSimulation instance accordingly.
"""

import json
import os
import glob
import shutil
import sys

from abm.agent.agent import Agent, supcalc
from abm.loader import helper as dh
from abm.monitoring.ifdb import pad_to_n_digits
import numpy as np
from matplotlib import pyplot as plt


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
        super().__init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad,
                         pooling_time,
                         pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion)


class DataLoader:
    """
    DataLoader class that loads a csv and json file and initializes a LoadedSimulation instance with the loaded
    parameters
    """

    def __init__(self, data_folder_path, only_env=False, only_agent=False, undersample=1, t_start=0, t_end=5000):
        """
        Initalization method of main DataLoader class

        :param data_folder_path: path of the folder to be loaded. Inside there must be the following data files:
            - agent_data.csv/json: including time series of agent data
            - resource_data.csv/json: including time series of resource data
            - env_params.json: json file including all environmental variables that was used for the simulation with
                the given data folder.
        :param only_env: if true only env files are read in
        :param only_agent: if true only env and agent data is read
        :param undersample: factor of data undersampling during summary
        :param t_start: start time of summary (slice start)
        :param t_end: end time of summary (slice end)
        """
        # Initializing DataLoader paths according to convention
        self.undersample = undersample
        self.only_env = only_env
        self.only_agent = only_agent
        self.data_folder_path = data_folder_path
        self.t_start = t_start
        self.t_end = t_end
        self.patch_id_dict = None

        # Defining path for agent data
        self.agent_csv_path = os.path.join(self.data_folder_path, "agent_data.csv")
        if not os.path.isfile(self.agent_csv_path):
            self.agent_json_path = os.path.join(self.data_folder_path, "agent_data.json")
            self.agent_csv_path = None
            if not os.path.isfile(self.agent_json_path):
                print("Neither json nor csv data found for agent data!")
                self.agent_json_path = None
        else:
            self.agent_json_path = None

        # Defining path for resource data
        self.resource_csv_path = os.path.join(self.data_folder_path, "resource_data.csv")
        if not os.path.isfile(self.resource_csv_path):
            self.resource_json_path = os.path.join(self.data_folder_path, "resource_data.json")
            self.resource_csv_path = None
            self.res_json_format = True
            if not os.path.isfile(self.resource_json_path):
                print("Neither json nor csv data found for resource data!")
                self.resource_json_path = None
        else:
            self.resource_json_path = None
            self.res_json_format = False

        self.env_json_path = os.path.join(self.data_folder_path, "env_params.json")
        self.agent_data = {}
        self.resource_data = {}
        self.env_data = {}
        self.load_files()
        self.preprocess_data()

    def agent_json_to_csv_format(self):
        """transforming a read in json format dictionary to the standard data structure we use when read in a csv file"""
        new_dict = {}
        for agent_id, agent_dict in self.agent_data.items():
            agent_name = agent_dict['agent_name']
            for mes_name, mes in agent_dict.items():
                if mes_name != "agent_name":
                    new_dict_key = mes_name + "_" + agent_name
                    new_dict[new_dict_key] = mes
                    # undersampling agent data
                    if self.t_start is not None and self.t_end is not None:
                        new_dict[new_dict_key] = np.array(new_dict[new_dict_key])[
                                                 self.t_start:self.t_end:self.undersample]
                    else:
                        new_dict[new_dict_key] = np.array(new_dict[new_dict_key])[::self.undersample]
                    if mes_name == "posx" and "t" not in new_dict.keys():
                        new_dict["t"] = [i for i in range(len(mes))]
        return new_dict

    def resource_json_to_csv_format(self):
        """transforming a read in json format dictionary to the standard data structure we use when read in a csv file"""
        new_dict = {}
        time_len = len(self.agent_data["t"])
        for res_id, res_dict in self.resource_data.items():
            res_name = res_dict['res_name']
            start_time = res_dict["start_time"] - 1
            end_time = res_dict["end_time"]
            if end_time is None:
                end_time = time_len
            else:
                end_time -= 1
            for mes_name, mes in res_dict.items():
                if mes_name != "res_name":
                    if mes_name == "pos_x":
                        mes_name = "posx"
                    if mes_name == "pos_y":
                        mes_name = "posy"
                    new_dict_key = mes_name + "_" + res_name
                    data = np.zeros(time_len) - 1
                    data[start_time:end_time] = mes
                    # undersampling resource data
                    if self.t_start is not None and self.t_end is not None:
                        data = data[self.t_start:self.t_end:self.undersample]
                    else:
                        data = data[::self.undersample]
                    new_dict[new_dict_key] = data.copy()
        self.match_patch_ids()
        return new_dict

    def match_patch_ids(self):
        """"Every time a patch with id K is depleted a new patch with a new id will be created in the database. Later
        we want to compress data so that we want to match the newly created patch with the one that was depleted. For
        this we check the creation and depletion times of patches and we match accordingly. The result is a dictionary where
        each key is the id of a depleted patch and each corresponding value is the id of the newly created patch."""
        matched_patch_ids_path = os.path.join(self.data_folder_path, "matched_res_ids.json")
        if os.path.isfile(matched_patch_ids_path):
            with open(matched_patch_ids_path, "r") as f:
                patch_id_dict_r = json.load(f)
                self.patch_id_dict = {int(k): int(v) for k, v in patch_id_dict_r.items()}
        else:
            self.patch_id_dict = {}
            start_times = []
            end_times = []
            for res_id, res_dict in self.resource_data.items():
                start_time = res_dict["start_time"] - 1
                end_time = res_dict["end_time"]
                if end_time is not None:
                    end_time -= 1
                # if int(res_id) < 5:
                #     print(res_id)
                #     print("start ", start_time)
                #     print("end ", end_time)
                start_times.append(start_time)
                end_times.append(end_time)

            for id, etime in enumerate(end_times):
                if etime in start_times:
                    # the patch ends and regenerates as
                    matched_ids = [i for i, value in enumerate(start_times) if value == etime]
                    for matched_id in matched_ids:
                        if matched_id not in self.patch_id_dict.keys():  # if there are more than one matching ids
                            if not id in self.patch_id_dict.keys():
                                self.patch_id_dict[matched_id] = id
                            else:
                                self.patch_id_dict[matched_id] = self.patch_id_dict[id]
                            break
                else:
                    # the patch never ends
                    self.patch_id_dict[-id] = id
                # saving results
                with open(matched_patch_ids_path, "w") as f:
                    json.dump(self.patch_id_dict, f)

    def load_files(self):
        """loading agent and resource data files into memory and make post-processing on time series"""
        if not self.only_env:
            if self.agent_csv_path is not None:
                self.agent_data = dh.load_csv_file(self.agent_csv_path, undersample=self.undersample)
            elif self.agent_json_path is not None:
                with open(self.agent_json_path, "r") as f:
                    self.agent_data = json.load(f)
                self.agent_data = self.agent_json_to_csv_format()
            print("agent_data loaded")
            print(sys.getsizeof(self.agent_data))

            if not self.only_agent:
                if self.resource_csv_path is not None:
                    self.resource_data = dh.load_csv_file(self.resource_csv_path, undersample=self.undersample)
                    print("OLD")
                elif self.resource_json_path is not None:
                    with open(self.resource_json_path, "r") as f:
                        self.resource_data = json.load(f)
                    self.resource_data = self.resource_json_to_csv_format()
                print("resource data loaded")
                print(sys.getsizeof(self.agent_data))
            else:
                self.resource_data = None
        with open(self.env_json_path, "r") as file:
            self.env_data = json.load(file)

    def preprocess_data(self):
        """preprocessing loaded data structures"""
        if not self.only_env:
            time_len = len(self.agent_data["t"])
            new_time = [i for i in range(time_len)]
            self.agent_data["t"] = new_time
            if not self.only_agent:
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
        self.env_data["RADIUS_RESOURCE"] = float(self.env_data["RADIUS_RESOURCE"]),
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
        self.env_data["SUMMARY_UNDERSAMPLE"] = float(self.undersample)
        if self.res_json_format:
            self.env_data["RES_JSON_FORMAT"] = True

        for k, v in self.env_data.items():
            if isinstance(v, tuple):
                self.env_data[k] = v[0]

        # Change time-series data types
        if not self.only_env:
            for k, v in self.agent_data.items():
                if k.find("vfield") == -1:
                    self.agent_data[k] = np.array([float(i) for i in v])
                else:
                    self.agent_data[k] = np.array([i.replace("   ", " ").replace("  ", " ").replace("[  ", "[").replace(
                        "[ ", "[").replace(" ", ", ") for i in v], dtype=object)

            if not self.only_agent:
                if not self.res_json_format:
                    for k, v in self.resource_data.items():
                        # replacing empty strings with -1
                        self.resource_data[k] = np.array([float(i) if i != "" else -1.0 for i in v])

    def get_loaded_data(self):
        """returning the loaded data upon request"""
        return self.agent_data, self.resource_data, self.env_data

    def get_loaded_res_data_json(self):
        """returning resource data and patch depletion ids"""
        return self.agent_data, self.resource_data, self.env_data, self.patch_id_dict


class ExperimentLoader:
    """Loads and transforms a whole experiment folder with multiple batches and simulations"""

    def __init__(self, experiment_path, enforce_summary=False, undersample=1, with_plotting=False, collapse_plot=None,
                 t_start=None, t_end=None):
        # experiment data after summary
        self.undersample = int(undersample)
        self.env = None
        self.description = None
        self.efficiency = None
        self.eff_std = None
        self.mean_efficiency = None
        self.res_summary = None
        self.agent_summary = None
        self.varying_params = {}
        self.distances = None
        self.t_start = t_start
        self.t_end = t_end
        print(f"Experiment time window: t= {t_start} --- {t_end}")
        # COLLAPSE OF MULTIDIMENSIONAL PLOTS
        # in case 3 variables are present, and we kept a pair of variables changing together, we can collapse
        # the visualization into 2 dimensions by taking only non-zero elements into considerations.
        # this is equivalent defining a new variable that is a combination of 2 changed variables along simulations
        # The string encodes how the collision should work:
        # MIN/MAX/NONZERO-VARINDEXTHATISNOTCOLLAPSED
        # example: MIN-0: the 0th variable will kept as a single axis and the data will be collapsed along the 1st
        # and 2nd variables into a single axis, where each datapoint will be the Minimum of the collapsed datapoints
        self.collapse_plot = collapse_plot
        if self.collapse_plot is not None:
            self.collapse_method = self.collapse_plot.split('-')[0]
            if self.collapse_method == "MAX":
                self.collapse_method = np.max
            elif self.collapse_method == "MIN":
                self.collapse_method = np.min
            self.collapse_fixedvar_ind = int(self.collapse_plot.split('-')[1])

        # path variables
        self.experiment_path = experiment_path
        if not os.path.isdir(self.experiment_path):
            raise Exception(f"Can not find experiment folder {self.experiment_path}")
        self.experiment_name = os.path.basename(experiment_path)

        # collecting batch folders
        glob_pattern = os.path.join(experiment_path, "*")
        self.batch_folders = [path for path in glob.iglob(glob_pattern) if
                              path.find("summary") < 0 and path.find("README") < 0]
        self.num_batches = len(self.batch_folders)
        self.num_runs = None

        # reading and restructuring raw data into numpy arrays
        if not self.is_already_summarized() or enforce_summary:
            # check parameter variability
            self.get_changing_variables()
            # load and summarize data run by run
            self.read_all_data()

        # reloading previously saved numpy arrays
        self.reload_summarized_data()
        if with_plotting:
            self.plot_search_efficiency()
            self.plot_mean_relocation_time()
            # self.plot_mean_travelled_distances()

    def set_collapse_param(self, collapse_plot):
        # COLLAPSE OF MULTIDIMENSIONAL PLOTS
        # in case 3 variables are present and we kept a pair of variables changing together, we can collapse
        # the visualization into 2 dimensions by taking only non-zero elements into considerations.
        # this is equivalent defining a new variable that is a combination of 2 changed variables along simulations
        # The string encodes how the collision should work:
        # MIN/MAX/NONZERO-VARINDEXTHATISNOTCOLLAPSED
        # example: MIN-0: the 0th variable will kept as a single axis and the data will be collapsed along the 1st
        # and 2nd variables into a single axis, where each datapoint will be the Minimum of the collapsed datapoints
        self.collapse_plot = collapse_plot
        if self.collapse_plot is not None:
            self.collapse_method = self.collapse_plot.split('-')[0]
            if self.collapse_method == "MAX":
                self.collapse_method = np.max
            elif self.collapse_method == "MIN":
                self.collapse_method = np.min
            self.collapse_fixedvar_ind = int(self.collapse_plot.split('-')[1])

    def read_all_data(self):
        """reading all data in the experiment folder and saving them as numpy files"""
        print("Reading all experimental data first...")
        # calculating the maximum number of resource patches over all experiments
        max_r_in_runs = 0
        for i, batch_path in enumerate(self.batch_folders):
            glob_pattern = os.path.join(batch_path, "*")
            run_folders = [path for path in glob.iglob(glob_pattern)]
            if i == 0:
                self.num_runs = len(run_folders)

            for j, run in enumerate(run_folders):
                print(f"Reading agent data batch {i}, run {j}")
                agent_data, _, env_data = DataLoader(run, undersample=self.undersample, only_agent=True,
                                                     t_start=self.t_start, t_end=self.t_end).get_loaded_data()
                del _

                # finding out max depleted patches for next loop when we summarize
                # resource data
                num_in_run = int(float(
                    env_data['N_RESOURCES']))  # len([k for k in list(res_data.keys()) if k.find("posx_res") > -1])
                print(f"in this run we have {num_in_run} resources")
                if num_in_run > max_r_in_runs:
                    max_r_in_runs = num_in_run

                if i == 0 and j == 0:
                    if "N" not in list(self.varying_params.keys()):
                        num_agents = int(float(env_data['N']))
                    else:
                        print("Detected varying group size across runs, will use maximum agent number...")
                        num_agents = int(np.max(self.varying_params["N"]))
                    if self.t_start is not None and self.t_end is not None:
                        num_timesteps = int((self.t_end - self.t_start) / self.undersample)
                    else:
                        num_timesteps = int(float(env_data['T']) / self.undersample)
                    axes_lens = []
                    for k in sorted(list(self.varying_params.keys())):
                        axes_lens.append(len(self.varying_params[k]))
                    print("Initialize data arrays for agent data")
                    # num_batches x criterion1 x criterion2 x ... x criterionN x num_agents x time
                    # criteria as in self.varying_params and ALWAYS IN ALPHABETIC ORDER
                    posx_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    posy_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    rew_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    ori_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    vel_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    w_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    u_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    Ip_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    mode_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                    expl_patch_array = np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))

                index = [self.varying_params[k].index(float(env_data[k])) for k in
                         sorted(list(self.varying_params.keys()))]
                for ai in range(int(float(env_data["N"]))):
                    ind = (i,) + tuple(index) + (ai,)
                    posx_array[ind] = agent_data[f'posx_agent-{pad_to_n_digits(ai, n=2)}']
                    posy_array[ind] = agent_data[f'posy_agent-{pad_to_n_digits(ai, n=2)}']
                    rew_array[ind] = agent_data[f'collectedr_agent-{pad_to_n_digits(ai, n=2)}']
                    ori_array[ind] = agent_data[f'orientation_agent-{pad_to_n_digits(ai, n=2)}']
                    vel_array[ind] = agent_data[f'velocity_agent-{pad_to_n_digits(ai, n=2)}']
                    w_array[ind] = agent_data[f'w_agent-{pad_to_n_digits(ai, n=2)}']
                    u_array[ind] = agent_data[f'u_agent-{pad_to_n_digits(ai, n=2)}']
                    Ip_array[ind] = agent_data[f'Ipriv_agent-{pad_to_n_digits(ai, n=2)}']
                    mode_array[ind] = agent_data[f'mode_agent-{pad_to_n_digits(ai, n=2)}']
                    expl_patch_array[ind] = agent_data[f'expl_patch_id_agent-{pad_to_n_digits(ai, n=2)}']

                del agent_data

        print("Datastructures initialized according to loaded data!")
        print("Saving agent summary...")
        summary_path = os.path.join(self.experiment_path, "summary")
        os.makedirs(summary_path, exist_ok=True)
        np.savez(os.path.join(summary_path, "agent_summary.npz"),
                 posx=posx_array,
                 posy=posy_array,
                 orientation=ori_array,
                 velocity=vel_array,
                 Ipriv=Ip_array,
                 collresource=rew_array,
                 w=w_array,
                 u=u_array,
                 mode=mode_array,
                 explpatch=expl_patch_array)

        del posx_array, posy_array, rew_array, ori_array, vel_array, \
            w_array, u_array, Ip_array, mode_array, expl_patch_array

        for i, batch_path in enumerate(self.batch_folders):
            glob_pattern = os.path.join(batch_path, "*")

            run_folders = [path for path in glob.iglob(glob_pattern)]
            if i == 0:
                self.num_runs = len(run_folders)

            for j, run in enumerate(run_folders):
                print(f"Reading resource data batch {i}, run {j}")
                _, res_data, env_data, patch_id_dict = DataLoader(run, undersample=self.undersample,
                                                                  t_start=self.t_start,
                                                                  t_end=self.t_end).get_loaded_res_data_json()

                if i == 0 and j == 0:
                    print("\nInitializing resource data structures")
                    # num_batches x criterion1 x criterion2 x ... x criterionN x max_num_resources x time
                    # criteria as in self.varying_params and ALWAYS IN ALPHABETIC ORDER
                    # where the value is -1 the resource does not exist in time
                    r_posx_array = np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                    r_posy_array = np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                    r_qual_array = np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                    r_rescleft_array = np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))

                index = [self.varying_params[k].index(float(env_data[k])) for k in
                         sorted(list(self.varying_params.keys()))]
                # in the raw data we create a new column every time a new patch appears
                env_num_res_in_run = int(float(env_data['N_RESOURCES']))
                num_res_in_run = len([k for k in list(res_data.keys()) if k.find("posx_res") > -1])

                # recording times of patch depletion so that we can continously log data into a single
                # column instead of saving new patches in new columns
                depletion_times = np.zeros(env_num_res_in_run)
                for ri in range(num_res_in_run):
                    if patch_id_dict is None:
                        # we try to collapse data according to generated raw csv if data was not saved in json, otherwise
                        # already done with preprocessing
                        # in the numpy array we only have N_RESOURCES column
                        if ri < env_num_res_in_run:
                            # we check which timestep the patch is depleted and store these in switching time
                            data = res_data[f'posx_res-{pad_to_n_digits(ri + 1, n=3)}']
                            data = np.array([float(d) if d != "" else 0.0 for d in data])
                            data[data < 0] = 0
                            try:
                                depletion_times[ri] = np.nonzero(np.diff(data))[0]
                            except:
                                depletion_times[ri] = len(data)
                            collapsed_ri = ri
                        else:
                            # if the columnid is larger than how many resources we defined in the env file
                            # that will mean we have a regenereted patch. We need to find which previous patch disappeard
                            # when this one appeared so we can continously store data in less columns
                            # print(ri, "depletion_times: ", depletion_times)
                            data = res_data[f'posx_res-{pad_to_n_digits(ri + 1, n=3)}']
                            data = np.array([float(d) if d != "" else 0.0 for d in data])
                            data[data < 0] = 0
                            # finding which patch is continued by this
                            try:
                                appearance_time = np.nonzero(np.diff(data))[0][0]
                            except IndexError as e:
                                print("Patch didn't appear in this undersampling rate")
                                continue

                            # print("patch appeared @ ", appearance_time)
                            try:
                                depletion_time = np.nonzero(np.diff(data))[0][1]
                                # print("patch depleted @ ", depletion_time)
                            except IndexError as e:
                                # print("last patch")
                                depletion_time = -1
                            collapsed_ri = np.where(depletion_times == appearance_time)[0][0]
                            # print("patch continues prev. patch ", collapsed_ri)
                            depletion_times[collapsed_ri] = depletion_time
                            # print(ri, "new depletion_times: ", depletion_times)
                    else:
                        # switching keys and values so we get end:start
                        patch_id_dict_r = patch_id_dict  # {endt: startt for startt, endt in patch_id_dict.items()}
                        if ri in list(patch_id_dict_r.keys()):
                            # this patch is a regenerated patch, we need to find parents
                            collapsed_ri = patch_id_dict_r[ri]
                        else:
                            # This is an original patch
                            collapsed_ri = ri

                    ind = (i,) + tuple(index) + (collapsed_ri,)

                    data = res_data[f'posx_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = np.array([float(d) if d != "" else 0.0 for d in data])
                    data[data < 0] = 0
                    r_posx_array[ind] += data

                    data = res_data[f'posy_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = np.array([float(d) if d != "" else 0.0 for d in data])
                    data[data < 0] = 0
                    r_posy_array[ind] += data

                    data = res_data[f'quality_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = np.array([float(d) if d != "" else 0.0 for d in data])
                    data[data < 0] = 0
                    r_qual_array[ind] += data

                    data = res_data[f'resc_left_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = np.array([float(d) if d != "" else 0.0 for d in data])
                    data[data < 0] = 0
                    r_rescleft_array[ind] += data

        print("Saving resource summary...")
        np.savez(os.path.join(summary_path, "resource_summary.npz"),
                 posx=r_posx_array,
                 posy=r_posy_array,
                 quality=r_qual_array,
                 resc_left=r_rescleft_array)

        del r_posx_array, r_posy_array, r_qual_array, r_rescleft_array

        with open(os.path.join(summary_path, "fixed_env.json"), "w") as fenvf:
            fixed_env = env_data
            for k, v in fixed_env.items():
                if k in list(self.varying_params.keys()):
                    fixed_env[k] = "----TUNED----"
            json.dump(fixed_env, fenvf)

        with open(os.path.join(summary_path, "tuned_env.json"), "w") as tenvf:
            json.dump(self.varying_params, tenvf)

        raw_description_path = os.path.join(self.experiment_path, "README.txt")
        sum_description_path = os.path.join(summary_path, "README.txt")
        if os.path.isfile(raw_description_path):
            shutil.copyfile(raw_description_path, sum_description_path)

        print("Summary saved!")

    def get_changing_variables(self):
        """Collecting env variables along which the initialization has changed across runs"""
        print("Checking for changing parameters along runs...")
        all_env = {}
        for i, batch_path in enumerate(self.batch_folders):
            all_env[i] = {}
            glob_pattern = os.path.join(batch_path, "*")
            run_folders = [path for path in glob.iglob(glob_pattern)]
            if i == 0:
                self.num_runs = len(run_folders)

            for j, run in enumerate(run_folders):
                _, _, env_data = DataLoader(run, only_env=True).get_loaded_data()
                all_env[i][j] = env_data

        base_keys = list(all_env[0][0].keys())
        variability = {}

        for base_key in base_keys:
            variability[base_key] = []

        # here we assume that parameter ranges do not change from batch to batch
        for ke, env in all_env[0].items():
            for k, v in env.items():
                variability[k].append(v)

        for k, v in variability.items():
            if k != "SAVE_ROOT_DIR":
                if len(list(set(v))) > 1:
                    self.varying_params[k] = sorted([float(i) for i in list(set(v))])
                    print(f"Found tuned parameter {k} with values {self.varying_params[k]}")

    def is_already_summarized(self):
        """Deciding if the experiment was laready summarized before"""
        if os.path.isdir(os.path.join(self.experiment_path, "summary")):
            print("Experiment is already summarized!")
            return True
        else:
            print("experiment is not summarized yet!")
            return False

    def reload_summarized_data(self):
        """Loading an already summarized experiment to spare time and resources"""
        print("Reloading previous experiment summary!")
        self.agent_summary = np.load(os.path.join(self.experiment_path, "summary", "agent_summary.npz"))
        self.res_summary = np.load(os.path.join(self.experiment_path, "summary", "resource_summary.npz"))
        with open(os.path.join(self.experiment_path, "summary", "fixed_env.json"), "r") as fixf:
            self.env = json.loads(fixf.read())
        print("Overwriting undersample ratio with the one read from env file...")
        self.undersample = int(float(self.env.get("SUMMARY_UNDERSAMPLE", "1")))
        with open(os.path.join(self.experiment_path, "summary", "tuned_env.json"), "r") as tunedf:
            self.varying_params = json.loads(tunedf.read())
        print("Agent, resource and parameter data reloaded!")
        description_path = os.path.join(self.experiment_path, "summary", "README.txt")
        if os.path.isfile(description_path):
            with open(description_path, "r") as readmefile:
                data = readmefile.readlines()
                self.description = "".join(data)
            print("\n____README____")
            print(self.description)
            print("___END_README___\n")
        print("Experiment loaded")

    def calculate_search_efficiency(self, t_start_plot=0, t_end_plot=-1, used_batches=None):
        """Method to calculate search efficiency throughout the experiments as the sum of collected resorces normalized
        with the travelled distance. The timestep in which the efficiency is calculated. This might mismatch from
        the real time according to how much the data was undersampled during sammury"""
        # Caclulating length of time window for normalizing efficiency
        if t_end_plot == -1:
            if self.t_end is None:  # we processed the whole experiment
                T = int(self.env['T'])
            else:  # we have cut down some part at the end of the experiment
                T = self.t_end
        else:  # We calculate the efficiency until a given point of time
            T = t_end_plot * self.undersample

        T_start = t_start_plot * self.undersample
        dT = T - T_start
        print(f"Using T_start={T_start}, T_end={T} delta_T={dT} to normalize efficiency!")

        print("Calculating mean search efficiency...")
        # self.get_travelled_distances()

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if used_batches is None:
            collres = self.agent_summary["collresource"][..., t_end_plot] - self.agent_summary["collresource"][
                ..., t_start_plot]
        else:
            # limiting number of used batches (e.g. for quick prototyping)
            print(f"Using {used_batches} batches to calculate efficiency!")
            print(self.agent_summary["collresource"].shape)
            collres = self.agent_summary["collresource"][0:used_batches, ..., t_end_plot] - self.agent_summary[
                                                                                                "collresource"][
                                                                                            0:used_batches, ...,
                                                                                            t_start_plot]
        # normalizing with distances needs good temporal resolution when reading data back
        # using large downsampling factors will make it impossibly to calculate trajectory lengths
        # and thus makes distance measures impossible
        # sum_distances = np.sum(self.distances, axis=time_dim)
        self.efficiency = collres / dT

        self.mean_efficiency = np.mean(np.mean(self.efficiency, axis=agent_dim), axis=batch_dim)
        self.eff_std = np.std(np.mean(self.efficiency, axis=agent_dim), axis=batch_dim)

    def plot_search_efficiency(self, t_start=0, t_end=-1, from_script=False, used_batches=None):
        """Method to plot search efficiency irrespectively of how many parameters have been tuned during the
        experiments."""
        cbar = None
        self.calculate_search_efficiency(t_start_plot=t_start, t_end_plot=t_end, used_batches=used_batches)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Search Efficiency")
            plt.plot(self.mean_efficiency)
            plt.plot(self.mean_efficiency + self.eff_std)
            plt.plot(self.mean_efficiency - self.eff_std)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_efficiency)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_efficiency.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_efficiency = np.zeros((self.mean_efficiency.shape[0:3]))
                print(self.new_mean_efficiency.shape)
                for j in range(self.mean_efficiency.shape[0]):
                    for i in range(self.mean_efficiency.shape[1]):
                        self.new_mean_efficiency[j, i, :] = self.mean_efficiency[j, i, :, i]
                self.mean_efficiency = self.new_mean_efficiency
            if self.collapse_plot is None:
                num_plots = self.mean_efficiency.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_efficiency[i, :, :], vmin=0, vmax=2500)
                    ax[i].set_title(f"{keys[0]}={self.varying_params[keys[0]][i]}")

                    if i == 0:
                        ax[i].set_yticks(range(len(self.varying_params[keys[1]])))
                        ax[i].set_yticklabels(self.varying_params[keys[1]])
                        ax[i].set_ylabel(keys[1])

                    ax[i].set_xticks(range(len(self.varying_params[keys[2]])))
                    ax[i].set_xticklabels(self.varying_params[keys[2]])
                    ax[i].set_xlabel(keys[2])

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = fig.colorbar(img, cax=cbar_ax)
            else:
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                shape_along_fixed_ind = self.mean_efficiency.shape[self.collapse_fixedvar_ind]
                labels = []
                # collapsing data
                for i in range(shape_along_fixed_ind):
                    if self.collapse_fixedvar_ind == 0:
                        collapsed_data_row = self.collapse_method(self.mean_efficiency[i, :, :], axis=0)
                        ind = np.argmax(self.mean_efficiency[i, :, :], axis=0)
                        max1_ind = 1
                        max2_ind = 2
                    elif self.collapse_fixedvar_ind == 1:
                        collapsed_data_row = self.collapse_method(self.mean_efficiency[:, i, :], axis=0)
                        ind = np.argmax(self.mean_efficiency[:, i, :], axis=0)
                        max1_ind = 0
                        max2_ind = 2
                    elif self.collapse_fixedvar_ind == 2:
                        collapsed_data_row = self.collapse_method(self.mean_efficiency[:, :, i], axis=0)
                        ind = np.argmax(self.mean_efficiency[:, :, i], axis=0)
                        max1_ind = 0
                        max2_ind = 1

                    if i == 0:
                        collapsed_data = collapsed_data_row
                    else:
                        collapsed_data = np.vstack((collapsed_data, collapsed_data_row))

                for j in range(len(ind)):
                    label = f"{keys[max1_ind]}={self.varying_params[keys[max1_ind]][ind[j]]}\n" \
                            f"{keys[max2_ind]}={self.varying_params[keys[max2_ind]][j]}"
                    labels.append(label)

                img = ax.imshow(collapsed_data)
                ax.set_yticks(range(len(self.varying_params[keys[self.collapse_fixedvar_ind]])))
                ax.set_yticklabels(self.varying_params[keys[self.collapse_fixedvar_ind]])
                ax.set_ylabel(keys[self.collapse_fixedvar_ind])

                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45)
                ax.set_xlabel("Combined Parameters")

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = fig.colorbar(img, cax=cbar_ax)
                # ax.set_xlabel(f"Combined Parameters\n"
                #               f"{keys[max1_ind]}={self.varying_params[keys[max1_ind]]}\n"
                #               f"{keys[max2_ind]}={self.varying_params[keys[max2_ind]]}")

        num_agents = self.agent_summary["collresource"].shape[agent_dim]
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"of total collected resource units normalized with the mean total\n" \
                           f"distance travelled by agents over the experiments.\n"
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

    def plot_mean_relocation_time(self):
        """Plotting the mean relative relocation time over agents and batches"""

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        rel_reloc_matrix = np.mean((self.agent_summary["mode"] == 2).astype(int), axis=time_dim)
        mean_rel_reloc = np.mean(np.mean(rel_reloc_matrix, axis=agent_dim), axis=batch_dim)
        std_rel_reloc = np.std(np.mean(rel_reloc_matrix, axis=agent_dim), axis=batch_dim)

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Mean (over agents and batches) relative relocation time")
            plt.plot(mean_rel_reloc)
            plt.plot(mean_rel_reloc + std_rel_reloc)
            plt.plot(mean_rel_reloc - std_rel_reloc)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(rel_reloc_matrix, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(mean_rel_reloc)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3:
            if self.collapse_plot is None:
                num_plots = mean_rel_reloc.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(mean_rel_reloc[i, :, :], vmin=0, vmax=np.max(mean_rel_reloc))
                    ax[i].set_title(f"{keys[0]}={self.varying_params[keys[0]][i]}")

                    if i == 0:
                        ax[i].set_yticks(range(len(self.varying_params[keys[1]])))
                        ax[i].set_yticklabels(self.varying_params[keys[1]])
                        ax[i].set_ylabel(keys[1])

                    ax[i].set_xticks(range(len(self.varying_params[keys[2]])))
                    ax[i].set_xticklabels(self.varying_params[keys[2]])
                    ax[i].set_xlabel(keys[2])

                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(img, cax=cbar_ax)
            else:
                fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                shape_along_fixed_ind = mean_rel_reloc.shape[self.collapse_fixedvar_ind]
                # collapsing data
                labels = []
                for i in range(shape_along_fixed_ind):
                    if self.collapse_fixedvar_ind == 0:
                        collapsed_data_row = self.collapse_method(mean_rel_reloc[i, :, :], axis=0)
                        ind = np.argmax(mean_rel_reloc[i, :, :], axis=0)
                        max1_ind = 1
                        max2_ind = 2
                    elif self.collapse_fixedvar_ind == 1:
                        collapsed_data_row = self.collapse_method(mean_rel_reloc[:, i, :], axis=0)
                        ind = np.argmax(mean_rel_reloc[:, i, :], axis=0)
                        max1_ind = 0
                        max2_ind = 2
                    elif self.collapse_fixedvar_ind == 2:
                        collapsed_data_row = self.collapse_method(mean_rel_reloc[:, :, i], axis=0)
                        ind = np.argmax(mean_rel_reloc[:, :, i], axis=0)
                        max1_ind = 0
                        max2_ind = 1

                    if i == 0:
                        collapsed_data = collapsed_data_row
                    else:
                        collapsed_data = np.vstack((collapsed_data, collapsed_data_row))

                for j in range(len(ind)):
                    label = f"{keys[max1_ind]}={self.varying_params[keys[max1_ind]][ind[j]]}\n" \
                            f"{keys[max2_ind]}={self.varying_params[keys[max2_ind]][j]}"
                    labels.append(label)

                ax.imshow(collapsed_data)
                ax.set_yticks(range(len(self.varying_params[keys[self.collapse_fixedvar_ind]])))
                ax.set_yticklabels(self.varying_params[keys[self.collapse_fixedvar_ind]])
                ax.set_ylabel(keys[self.collapse_fixedvar_ind])

                ax.set_xticks(range(len(labels)))
                # ax.set_xlabel(f"Combined Parameters\n"
                #               f"{keys[max1_ind]}={self.varying_params[keys[max1_ind]]}\n"
                #               f"{keys[max2_ind]}={self.varying_params[keys[max2_ind]]}")
                ax.set_xticklabels(labels, rotation=45)
                ax.set_xlabel("Combined Parameters")

        num_agents = self.agent_summary["collresource"].shape[agent_dim]
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"of relative relocation time, i.e. ratio of time spent in relocation\n"
        self.add_plot_interaction(description_text, fig, ax, show=True)

    def plot_mean_travelled_distances(self):

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        fig, ax = plt.subplots(1, 1)
        plt.title("Mean (over agents and batches) travelled distance")
        self.get_travelled_distances()

        sum_distances = np.sum(self.distances, axis=time_dim)
        mean_dist = np.mean(np.mean(sum_distances, axis=agent_dim), axis=batch_dim)
        std_dist = np.std(np.mean(sum_distances, axis=agent_dim), axis=batch_dim)

        if num_var_params == 1:
            plt.plot(mean_dist)
            plt.plot(mean_dist + std_dist)
            plt.plot(mean_dist - std_dist)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(sum_distances, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])
        elif num_var_params == 2:
            im = ax.imshow(mean_dist)

            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])
            ax.set_yticks(range(len(self.varying_params[list(self.varying_params.keys())[1]])))
            ax.set_yticklabels(self.varying_params[list(self.varying_params.keys())[1]])
            plt.ylabel(list(self.varying_params.keys())[1])

        num_agents = self.agent_summary["collresource"].shape[agent_dim]
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"travelled distance\n"
        self.add_plot_interaction(description_text, fig, ax, show=True)

    def add_plot_interaction(self, description_text, fig, ax, show=True, from_script=False):
        """Adding plot description to figure with interaction and showing if requested"""
        num_runs = 1
        for k, v in self.varying_params.items():
            num_runs *= len(v)
        description_text = f"{description_text}" \
                           f"Varied parameters: {list(self.varying_params.keys())}\n" \
                           f"Simulation time per run: {self.env['T']}\n" \
                           f"Number of par.combinations per batch: {num_runs}\n" \
                           f"Number of resource patches: {self.env['N_RESOURCES']}\n" \
                           f"Resource Quality and Contained units: " \
                           f"Q{self.env['MIN_RESOURCE_QUALITY']}-{self.env['MAX_RESOURCE_QUALITY']}, " \
                           f"U{self.env['MIN_RESOURCE_PER_PATCH']}-{self.env['MAX_RESOURCE_PER_PATCH']}"
        bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
        try:
            annot = ax.annotate(description_text, xy=(0.05, 0.95), xycoords='axes fraction', horizontalalignment='left',
                                verticalalignment='top', bbox=bbox_props)
            annot.set_visible(False)
            fig.canvas.mpl_connect('button_press_event', lambda event: show_plot_description(event, fig, annot))
            fig.canvas.mpl_connect('button_release_event', lambda event: hide_plot_description(event, fig, annot))
        except AttributeError:
            for axi in ax:
                annot = axi.annotate(description_text, xy=(0.05, 0.95), xycoords='axes fraction',
                                     horizontalalignment='left',
                                     verticalalignment='top', bbox=bbox_props)
                annot.set_visible(False)
                fig.canvas.mpl_connect('button_press_event', lambda event: show_plot_description(event, fig, annot))
                fig.canvas.mpl_connect('button_release_event', lambda event: hide_plot_description(event, fig, annot))

        if show:
            if not from_script:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.1)

    def get_travelled_distances(self):
        """calculating the travelled distance for all agents in all runs and batches in an experiment.
        Returns a numpy array that has dimensions of
            (batches, *[var_1, var_2, ..., var_N], agents) where each value is the total distance travelled
            in a specific batch for a specific parameter combination (var1...N) and agent"""
        if self.distances is None:
            posx = self.agent_summary["posx"]
            posy = self.agent_summary["posy"]
            T = posx.shape[-1]
            x1s = posx[..., 1::]
            y1s = posy[..., 1::]
            x2s = posx[..., 0:T - 1]
            y2s = posy[..., 0:T - 1]
            self.distances = supcalc.distance_coords(x1s, y1s, x2s, y2s, vectorized=True)


def show_plot_description(event, fig, annotation_box):
    """Callback function for matplotlib figure.canvas.mpl_connect function to show some
    description text for the user about what exactly is on the plot and how the data was
    summarized"""
    annotation_box.set_visible(True)
    fig.canvas.draw_idle()


def hide_plot_description(event, fig, annotation_box):
    """Callback function for matplotlib figure.canvas.mpl_connect function to show some
    description text for the user about what exactly is on the plot and how the data was
    summarized"""
    annotation_box.set_visible(False)
    fig.canvas.draw_idle()
