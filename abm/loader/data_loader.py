"""
data_loader.py : including the main classes to load previously saved data (csv+json) into an initialized replayable simulation.
    The DataLoader class is only the data layer that loads data and then can create a LoadedSimulation instance accordingly.
"""
import itertools
import json
import os
import glob
import shutil
import sys

from scipy.spatial import ConvexHull

from abm.agent.agent import Agent, supcalc
from abm.agent.supcalc import angle_between
from abm.loader import helper as dh
from abm.monitoring.ifdb import pad_to_n_digits
import numpy as np
from matplotlib import pyplot as plt
import zarr
from fastcluster import linkage
from scipy.cluster.hierarchy import dendrogram


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

    def __init__(self, data_folder_path, only_env=False, only_agent=False, undersample=1, t_start=0, t_end=5000,
                 project_version="Base"):
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
        :param project_version: project specific version such as Base or CooperativeSignaling
        """
        # Initializing DataLoader paths according to convention
        self.undersample = undersample
        self.only_env = only_env
        self.only_agent = only_agent
        self.data_folder_path = data_folder_path
        self.t_start = t_start
        self.t_end = t_end
        self.patch_id_dict = None
        self.project_version = project_version

        self.zarr_compressed_runs = False
        self.zarr_extension = None

        # Defining path for agent data
        self.agent_csv_path = os.path.join(self.data_folder_path, "agent_data.csv")
        if not os.path.isfile(self.agent_csv_path):
            self.agent_json_path = os.path.join(self.data_folder_path, "agent_data.json")
            self.agent_csv_path = None
            if not os.path.isfile(self.agent_json_path):
                print("Neither json nor csv data found for agent data, looking for zarr!")
                self.agent_json_path = None
                if os.path.isdir(os.path.join(self.data_folder_path, "ag_posx.zarr")) or \
                        os.path.isfile(os.path.join(self.data_folder_path, "ag_posx.zip")):
                    print("Found zarr archives, using them!")
                    self.zarr_compressed_runs = True
                    if os.path.isdir(os.path.join(self.data_folder_path, "ag_posx.zarr")):
                        self.zarr_extension = ".zarr"
                    else:
                        self.zarr_extension = ".zip"
                    print(f"Using zarr format with filetype {self.zarr_extension}")
                else:
                    print("No zarr archives found!")
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
                if self.zarr_compressed_runs:
                    print("But zarr archives found!")
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
            else:
                if self.zarr_compressed_runs:
                    self.agent_data = {}
                    self.agent_data['posx'] = zarr.open(
                        os.path.join(self.data_folder_path, f"ag_posx{self.zarr_extension}"), mode='r')
                    self.agent_data['posy'] = zarr.open(
                        os.path.join(self.data_folder_path, f"ag_posy{self.zarr_extension}"), mode='r')
                    self.agent_data['orientation'] = zarr.open(
                        os.path.join(self.data_folder_path, f"ag_ori{self.zarr_extension}"),
                        mode='r')
                    self.agent_data['mode'] = zarr.open(
                        os.path.join(self.data_folder_path, f"ag_mode{self.zarr_extension}"), mode='r')
                    self.agent_data['velocity'] = zarr.open(
                        os.path.join(self.data_folder_path, f"ag_vel{self.zarr_extension}"),
                        mode='r')
                    if self.project_version == "Base":
                        self.agent_data['w'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_w{self.zarr_extension}"), mode='r')
                        self.agent_data['u'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_u{self.zarr_extension}"), mode='r')
                        self.agent_data['Ipriv'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_ipriv{self.zarr_extension}"), mode='r')
                        self.agent_data['collresource'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_collr{self.zarr_extension}"),
                            mode='r')
                        self.agent_data['expl_patch_id'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_explr{self.zarr_extension}"),
                            mode='r')
                    elif self.project_version == "CooperativeSignaling":
                        self.agent_data['meter'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_meter{self.zarr_extension}"), mode='r')
                        self.agent_data['signalling'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_sig{self.zarr_extension}"), mode='r')
                        self.agent_data['collresource'] = zarr.open(
                            os.path.join(self.data_folder_path, f"ag_collr{self.zarr_extension}"),
                            mode='r')
                else:
                    raise Exception("No json, csv or zarr archive found for agent data!")
            print("agent_data loaded")

            if not self.only_agent:
                try:
                    if self.resource_csv_path is not None:
                        self.resource_data = dh.load_csv_file(self.resource_csv_path, undersample=self.undersample)
                        print("OLD")
                    elif self.resource_json_path is not None:
                        with open(self.resource_json_path, "r") as f:
                            self.resource_data = json.load(f)
                        self.resource_data = self.resource_json_to_csv_format()
                    else:
                        if self.zarr_compressed_runs:
                            self.resource_data = {}
                            self.resource_data['posx'] = zarr.open(
                                os.path.join(self.data_folder_path, f"res_posx{self.zarr_extension}"),
                                mode='r')
                            self.resource_data['posy'] = zarr.open(
                                os.path.join(self.data_folder_path, f"res_posy{self.zarr_extension}"),
                                mode='r')
                            self.resource_data['radius'] = zarr.open(
                                os.path.join(self.data_folder_path, f"res_rad{self.zarr_extension}"),
                                mode='r')
                            if self.project_version == "Base":
                                self.resource_data['resc_left'] = zarr.open(
                                    os.path.join(self.data_folder_path, f"res_left{self.zarr_extension}"),
                                    mode='r')
                                self.resource_data['quality'] = zarr.open(
                                    os.path.join(self.data_folder_path, f"res_qual{self.zarr_extension}"),
                                    mode='r')
                except:
                    pass

                print("resource data loaded")
                print(sys.getsizeof(self.agent_data))
            else:
                self.resource_data = None
        with open(self.env_json_path, "r") as file:
            self.env_data = json.load(file)

    def preprocess_data(self):
        """preprocessing loaded data structures"""
        if not self.only_env:
            if not self.zarr_compressed_runs:
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
        self.env_data["RADIUS_AGENT"] = int(float(self.env_data["RADIUS_AGENT"])),
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
        if self.zarr_compressed_runs:
            self.env_data["SUMMARY_ZARR_FORMAT"] = True

        for k, v in self.env_data.items():
            if isinstance(v, tuple):
                self.env_data[k] = v[0]

        # Change time-series data types
        if not self.only_env:
            if not self.zarr_compressed_runs:
                for k, v in self.agent_data.items():
                    if k.find("vfield") == -1:
                        self.agent_data[k] = np.array([float(i) for i in v])
                    else:
                        self.agent_data[k] = np.array(
                            [i.replace("   ", " ").replace("  ", " ").replace("[  ", "[").replace(
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
        self.clustering_data = None
        self.clustering_ids = None
        self.project_version = None
        self.zarr_extension = ".zarr"
        self.mean_iid = None
        self.mean_nn_dist = None
        self.iid_matrix = None
        self.elong_matrix = None
        self.mean_elong = None
        self.hull_points_array = None
        self.undersample = int(undersample)
        self.chunksize = None  # chunk size in zarr array
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
            self.read_all_data(project_version=self.project_version)

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

    def read_all_data(self, only_res=False, project_version="Base"):
        """reading all data in the experiment folder and saving them as zarr summary archives"""
        max_r_in_runs = None
        if not only_res:
            print("Reading all experimental data first...")
            # calculating the maximum number of resource patches over all experiments
            # and reading agent data at the same time
            max_r_in_runs = 0
            for i, batch_path in enumerate(self.batch_folders):
                glob_pattern = os.path.join(batch_path, "*")
                run_folders = [path for path in glob.iglob(glob_pattern) if path.find(".json") < 0]
                if i == 0:
                    self.num_runs = len(run_folders)

                for j, run in enumerate(run_folders):
                    print(f"Reading agent data batch {i}, run {j}, {run}")
                    agent_data, _, env_data = DataLoader(run, undersample=self.undersample, only_agent=True,
                                                         t_start=self.t_start, t_end=self.t_end,
                                                         project_version=self.project_version).get_loaded_data()
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

                        # Calculating number of timesteps to create data structures
                        if self.t_start is not None and self.t_end is not None:
                            num_timesteps = int((self.t_end - self.t_start) / self.undersample)
                        else:
                            num_timesteps = int(float(env_data['T']) / self.undersample)
                        self.chunksize = num_timesteps

                        # Calculating axes length along varying parameter dimensions
                        axes_lens = []
                        for k in sorted(list(self.varying_params.keys())):
                            axes_lens.append(len(self.varying_params[k]))

                        print("Initialize data arrays for agent data")
                        # num_batches x criterion1 x criterion2 x ... x criterionN x num_agents x time
                        # criteria as in self.varying_params and ALWAYS IN ALPHABETIC ORDER
                        summary_path = os.path.join(self.experiment_path, "summary")
                        ax_chunk = [1 for i in range(len(axes_lens))]
                        os.makedirs(summary_path, exist_ok=True)
                        posx_array = zarr.open(os.path.join(summary_path, f"agent_posx{self.zarr_extension}"), mode='w',
                                               shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                               chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                        posy_array = zarr.open(os.path.join(summary_path, f"agent_posy{self.zarr_extension}"), mode='w',
                                               shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                               chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                        ori_array = zarr.open(os.path.join(summary_path, f"agent_ori{self.zarr_extension}"), mode='w',
                                              shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                              chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                        vel_array = zarr.open(os.path.join(summary_path, f"agent_vel{self.zarr_extension}"), mode='w',
                                              shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                              chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                        mode_array = zarr.open(os.path.join(summary_path, f"agent_mode{self.zarr_extension}"), mode='w',
                                               shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                               chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        if project_version == "Base":
                            # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                            rew_array = zarr.open(os.path.join(summary_path, f"agent_rew{self.zarr_extension}"),
                                                  mode='w',
                                                  shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                  chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                            w_array = zarr.open(os.path.join(summary_path, f"agent_w{self.zarr_extension}"), mode='w',
                                                shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                            u_array = zarr.open(os.path.join(summary_path, f"agent_u{self.zarr_extension}"), mode='w',
                                                shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                            Ip_array = zarr.open(os.path.join(summary_path, f"agent_Ip{self.zarr_extension}"), mode='w',
                                                 shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                 chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            # np.zeros((self.num_batches, *axes_lens, num_agents, num_timesteps))
                            expl_patch_array = zarr.open(
                                os.path.join(summary_path, f"agent_explpatch{self.zarr_extension}"), mode='w',
                                shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        elif project_version == "CooperativeSignaling":
                            meter_array = zarr.open(os.path.join(summary_path, f"agent_meter{self.zarr_extension}"),
                                                    mode='w',
                                                    shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                    chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            sig_array = zarr.open(os.path.join(summary_path, f"agent_sig{self.zarr_extension}"),
                                                  mode='w',
                                                  shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                  chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                            rew_array = zarr.open(os.path.join(summary_path, f"agent_rew{self.zarr_extension}"),
                                                  mode='w',
                                                  shape=(self.num_batches, *axes_lens, num_agents, num_timesteps),
                                                  chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')

                    index = [self.varying_params[k].index(float(env_data[k])) for k in
                             sorted(list(self.varying_params.keys()))]

                    if not env_data.get("SUMMARY_ZARR_FORMAT", False):
                        for ai in range(int(float(env_data["N"]))):
                            ind = (i,) + tuple(index) + (ai,)
                            posx_array[ind] = agent_data[f'posx_agent-{pad_to_n_digits(ai, n=2)}']
                            posy_array[ind] = agent_data[f'posy_agent-{pad_to_n_digits(ai, n=2)}']
                            ori_array[ind] = agent_data[f'orientation_agent-{pad_to_n_digits(ai, n=2)}']
                            vel_array[ind] = agent_data[f'velocity_agent-{pad_to_n_digits(ai, n=2)}']
                            mode_array[ind] = agent_data[f'mode_agent-{pad_to_n_digits(ai, n=2)}']
                            if project_version == "Base":
                                rew_array[ind] = agent_data[f'collectedr_agent-{pad_to_n_digits(ai, n=2)}']
                                w_array[ind] = agent_data[f'w_agent-{pad_to_n_digits(ai, n=2)}']
                                u_array[ind] = agent_data[f'u_agent-{pad_to_n_digits(ai, n=2)}']
                                Ip_array[ind] = agent_data[f'Ipriv_agent-{pad_to_n_digits(ai, n=2)}']
                                expl_patch_array[ind] = agent_data[f'expl_patch_id_agent-{pad_to_n_digits(ai, n=2)}']
                            elif project_version == "CooperativeSignaling":
                                meter_array[ind] = agent_data[f'meter_agent-{pad_to_n_digits(ai, n=2)}']
                                sig_array[ind] = agent_data[f'sig_agent-{pad_to_n_digits(ai, n=2)}']
                                rew_array[ind] = agent_data[f'collectedr_agent-{pad_to_n_digits(ai, n=2)}']
                    else:
                        ind = (i,) + tuple(index)
                        posx_array[ind] = agent_data['posx'][..., self.t_start:self.t_end:self.undersample]
                        posy_array[ind] = agent_data['posy'][..., self.t_start:self.t_end:self.undersample]
                        ori_array[ind] = agent_data['orientation'][..., self.t_start:self.t_end:self.undersample]
                        mode_array[ind] = agent_data['mode'][..., self.t_start:self.t_end:self.undersample]
                        vel_array[ind] = agent_data['velocity'][..., self.t_start:self.t_end:self.undersample]
                        if project_version == "Base":
                            w_array[ind] = agent_data['w'][..., self.t_start:self.t_end:self.undersample]
                            u_array[ind] = agent_data['u'][..., self.t_start:self.t_end:self.undersample]
                            Ip_array[ind] = agent_data['Ipriv'][..., self.t_start:self.t_end:self.undersample]
                            rew_array[ind] = agent_data['collresource'][..., self.t_start:self.t_end:self.undersample]
                            expl_patch_array[ind] = agent_data['expl_patch_id'][...,
                                                    self.t_start:self.t_end:self.undersample]
                        elif project_version == "CooperativeSignaling":
                            meter_array[ind] = agent_data['meter'][..., self.t_start:self.t_end:self.undersample]
                            sig_array[ind] = agent_data['signalling'][..., self.t_start:self.t_end:self.undersample]
                            rew_array[ind] = agent_data['collresource'][..., self.t_start:self.t_end:self.undersample]

                    del agent_data

            print("Datastructures initialized according to loaded data!")
            print("Saving agent summary...")
            summary_path = os.path.join(self.experiment_path, "summary")
            os.makedirs(summary_path, exist_ok=True)
            # legacy npz saving is not supported anymore
            # np.savez(os.path.join(summary_path, "agent_summary.npz"),
            #          # posx=posx_array,
            #          posy=posy_array,
            #          orientation=ori_array,
            #          velocity=vel_array,
            #          Ipriv=Ip_array,
            #          collresource=rew_array,
            #          w=w_array,
            #          u=u_array,
            #          mode=mode_array,
            #          explpatch=expl_patch_array)

            del posx_array, posy_array, ori_array, vel_array, mode_array
            if project_version == "Base":
                del rew_array, w_array, u_array, Ip_array, expl_patch_array
            elif project_version == "CooperativeSignaling":
                del meter_array, sig_array, rew_array

            # Saving max patch number for further calc
            env_data['SUMMARY_MAX_PATCHES'] = int(max_r_in_runs)
            if self.t_start is not None:
                env_data['SUMMARY_TSTART'] = int(self.t_start)
            if self.t_end is not None:
                env_data['SUMMARY_TEND'] = int(self.t_end)

            with open(os.path.join(summary_path, "fixed_env.json"), "w") as fenvf:
                fixed_env = env_data
                for k, v in fixed_env.items():
                    if k in list(self.varying_params.keys()):
                        fixed_env[k] = "----TUNED----"
                json.dump(fixed_env, fenvf)

            with open(os.path.join(summary_path, "tuned_env.json"), "w") as tenvf:
                json.dump(self.varying_params, tenvf)

        if only_res:
            from pprint import pprint
            print("Agent and env data has been already summarized, continuing from there!")
            # reading back saved env variables from previous interrupted summary
            summary_path = os.path.join(self.experiment_path, "summary")
            with open(os.path.join(summary_path, "fixed_env.json"), "r") as fenvf:
                self.env = json.loads(fenvf.read())
            print("Found fixed env parameters")
            pprint(self.env)

            with open(os.path.join(summary_path, "tuned_env.json"), "r") as tenvf:
                self.varying_params = json.loads(tenvf.read())
            print("Found Varying parameters")
            pprint(self.varying_params)

            max_r_in_runs = int(float(self.env.get('SUMMARY_MAX_PATCHES')))
            self.undersample = int(float(self.env.get("SUMMARY_UNDERSAMPLE", "1")))
            self.t_start = self.env.get('SUMMARY_TSTART', 0)
            self.t_end = self.env.get('SUMMARY_TEND', self.env['T'])
            num_timesteps = int((self.t_end - self.t_start) / self.undersample)
            axes_lens = []

            for k in sorted(list(self.varying_params.keys())):
                axes_lens.append(len(self.varying_params[k]))
            print(
                f"Previous summary had parameters t_start={self.t_start} : us-{self.undersample}-us : {self.t_end}=t_end")
            print(f"Previous summary will have varying axes dimensions: {axes_lens}")

        # Calculating res data
        for i, batch_path in enumerate(self.batch_folders):
            glob_pattern = os.path.join(batch_path, "*")

            run_folders = [path for path in glob.iglob(glob_pattern) if path.find(".json") < 0]
            if i == 0:
                self.num_runs = len(run_folders)

            for j, run in enumerate(run_folders):
                print(f"Reading resource data batch {i}, run {j}")
                _, res_data, env_data, patch_id_dict = DataLoader(run, undersample=self.undersample,
                                                                  t_start=self.t_start,
                                                                  t_end=self.t_end,
                                                                  project_version=self.project_version).get_loaded_res_data_json()

                if i == 0 and j == 0:
                    print("\nInitializing resource data structures")
                    # num_batches x criterion1 x criterion2 x ... x criterionN x max_num_resources x time
                    # criteria as in self.varying_params and ALWAYS IN ALPHABETIC ORDER
                    # where the value is -1 the resource does not exist in time
                    ax_chunk = [1 for i in range(len(axes_lens))]
                    r_posx_array = zarr.open(os.path.join(summary_path, f"res_posx{self.zarr_extension}"), mode='w',
                                             shape=(self.num_batches, *axes_lens, max_r_in_runs, num_timesteps),
                                             chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                    # legacy noz: np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                    r_posy_array = zarr.open(os.path.join(summary_path, f"res_posy{self.zarr_extension}"), mode='w',
                                             shape=(self.num_batches, *axes_lens, max_r_in_runs, num_timesteps),
                                             chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                    if project_version == "Base":
                        # legacy npz: np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                        r_qual_array = zarr.open(os.path.join(summary_path, f"res_qual{self.zarr_extension}"), mode='w',
                                                 shape=(self.num_batches, *axes_lens, max_r_in_runs, num_timesteps),
                                                 chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # legacy npz: np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))
                        r_rescleft_array = zarr.open(os.path.join(summary_path, f"res_rescleft{self.zarr_extension}"),
                                                     mode='w',
                                                     shape=(self.num_batches, *axes_lens, max_r_in_runs, num_timesteps),
                                                     chunks=(1, *ax_chunk, 1, num_timesteps), dtype='float')
                        # legacy npz: np.zeros((self.num_batches, *axes_lens, max_r_in_runs, num_timesteps))

                index = [self.varying_params[k].index(float(env_data[k])) for k in
                         sorted(list(self.varying_params.keys()))]

                if not env_data.get("SUMMARY_ZARR_FORMAT", False):
                    # in the raw data we create a new column every time a new patch appears
                    env_num_res_in_run = int(float(env_data['N_RESOURCES']))
                    num_res_in_run = len([k for k in list(res_data.keys()) if k.find("posx_res") > -1])

                    # recording times of patch depletion so that we can continuously log data into a single
                    # column instead of saving new patches in new columns
                    depletion_times = np.zeros(env_num_res_in_run)
                    for ri in range(num_res_in_run):
                        print(f"Processing patch {ri}/{num_res_in_run}")
                        if patch_id_dict is None:
                            # we try to collapse data according to generated raw csv if data was not saved in json,
                            # otherwise already done with preprocessing in the numpy array we only have N_RESOURCES
                            # column
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
                                # if the columnid is larger than how many resources we defined in the env file that
                                # will mean we have a regenerated patch. We need to find which previous patch
                                # disappeared when this one appeared so we can continuously store data in less columns
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

                        if project_version == "Base":
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

                else:
                    try:
                        num_res_in_run = res_data['posx'].shape[0]
                    except:
                        num_res_in_run = 0
                    for pid in range(num_res_in_run):
                        ind = (i,) + tuple(index) + (pid,)
                        r_posx_array[ind] = res_data['posx'][pid, self.t_start:self.t_end:self.undersample]
                        r_posy_array[ind] = res_data['posy'][pid, self.t_start:self.t_end:self.undersample]
                        if project_version == "Base":
                            r_qual_array[ind] = res_data['quality'][pid, self.t_start:self.t_end:self.undersample]
                            r_rescleft_array[ind] = res_data['resc_left'][pid, self.t_start:self.t_end:self.undersample]

        # print("Saving resource summary...")
        # legacy npz saving is not used anymore due to memory issues and data handling
        # np.savez(os.path.join(summary_path, "resource_summary.npz"),
        #          posx=r_posx_array,
        #          posy=r_posy_array,
        #          quality=r_qual_array,
        #          resc_left=r_rescleft_array)

        del r_posx_array, r_posy_array
        if project_version == "Base":
            del r_qual_array, r_rescleft_array

        raw_description_path = os.path.join(self.experiment_path, "README.txt")
        sum_description_path = os.path.join(summary_path, "README.txt")
        if os.path.isfile(raw_description_path):
            shutil.copyfile(raw_description_path, sum_description_path)

        print("Summary saved!")

    def get_changing_variables(self):
        """Collecting env variables along which the initialization has changed across runs in experiment"""
        print("Checking for changing parameters along runs...")
        all_env = {}
        for i, batch_path in enumerate(self.batch_folders):
            all_env[i] = {}
            glob_pattern = os.path.join(batch_path, "*")
            run_folders = [path for path in glob.iglob(glob_pattern) if path.find(".json") < 0]
            if i == 0:
                self.num_runs = len(run_folders)

            for j, run in enumerate(run_folders):
                _, _, env_data = DataLoader(run, only_env=True).get_loaded_data()
                if self.project_version is None:
                    self.project_version = env_data.get("APP_VERSION", "Base")
                    print(f"Found Project Version: {self.project_version}")
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
            print("Experiment is not summarized yet!")
            return False

    def reload_summarized_data(self):
        """Loading an already summarized experiment to spare time and resources"""
        print("Reloading previous experiment summary!")
        with open(os.path.join(self.experiment_path, "summary", "fixed_env.json"), "r") as fixf:
            self.env = json.loads(fixf.read())
        self.project_version = self.env.get("APP_VERSION", "Base")
        if os.path.isfile(os.path.join(self.experiment_path, "summary", "agent_summary.npz")):
            # found npz summary for agent data
            self.agent_summary = np.load(os.path.join(self.experiment_path, "summary", "agent_summary.npz"))
        else:
            # no npz summary available for agent data
            if os.path.isdir(os.path.join(self.experiment_path, "summary", "agent_posx.zarr")):
                extension = ".zarr"
            elif os.path.isfile(os.path.join(self.experiment_path, "summary", "agent_posx.zip")):
                extension = ".zip"
            else:
                print("No npz or zarr format summary has been found for agent data!")
                self.agent_summary = None
                extension = None

            if extension is not None:
                # found zarr summary for agent data
                self.agent_summary = {}
                self.agent_summary['posx'] = zarr.open(
                    os.path.join(self.experiment_path, "summary", f"agent_posx{extension}"),
                    mode='r')
                self.chunksize = int(self.agent_summary['posx'].shape[-1])
                self.num_batches = self.agent_summary['posx'].shape[0]
                self.agent_summary['posy'] = zarr.open(
                    os.path.join(self.experiment_path, "summary", f"agent_posy{extension}"),
                    mode='r')
                self.agent_summary['orientation'] = zarr.open(
                    os.path.join(self.experiment_path, "summary", f"agent_ori{extension}"),
                    mode='r')  # self.experiment.agent_summary['orientation']
                self.agent_summary['mode'] = zarr.open(
                    os.path.join(self.experiment_path, "summary", f"agent_mode{extension}"),
                    mode='r')  # self.experiment.agent_summary['mode']
                if self.project_version == "Base":
                    # u and w are not crucial for replay, so we can skip them if they are not available
                    try:
                        self.agent_summary['u'] = zarr.open(
                            os.path.join(self.experiment_path, "summary", f"agent_u{extension}"),
                            mode='r')
                    except zarr.errors.PathNotFoundError:
                        print("No agent_u found!")
                    try:
                        self.agent_summary['w'] = zarr.open(
                            os.path.join(self.experiment_path, "summary", f"agent_w{extension}"),
                            mode='r')
                    except zarr.errors.PathNotFoundError:
                        print("No agent_w found!")
                    self.agent_summary['explpatch'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", f"agent_explpatch{extension}"),
                        mode='r')
                    self.agent_summary['collresource'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", f"agent_rew{extension}"),
                        mode='r')
                elif self.project_version == "CooperativeSignaling":
                    self.agent_summary['meter'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", f"agent_meter{extension}"),
                        mode='r')
                    self.agent_summary['signalling'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", f"agent_sig{extension}"),
                        mode='r')
                    self.agent_summary['collresource'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", f"agent_rew{extension}"),
                        mode='r')
        if not os.path.isfile(os.path.join(self.experiment_path, "summary", "resource_summary.npz")):
            # no npz summary found for resources
            if os.path.isdir(os.path.join(self.experiment_path, "summary", "res_posx.zarr")):
                extension = ".zarr"
            elif os.path.isfile(os.path.join(self.experiment_path, "summary", "res_posx.zip")):
                extension = ".zip"
            else:
                print(
                    "Previous summary folder has been found but does not contain resource data. Summarizing resource data!")
                self.read_all_data(only_res=True)
                extension = None
            if extension is not None:
                # we found zarr format summary for resources
                self.res_summary = {}
                self.res_summary['posx'] = zarr.open(os.path.join(self.experiment_path, "summary", "res_posx.zarr"),
                                                     mode='r')
                self.res_summary['posy'] = zarr.open(os.path.join(self.experiment_path, "summary", "res_posy.zarr"),
                                                     mode='r')
                if self.project_version == "Base":
                    self.res_summary['resc_left'] = zarr.open(
                        os.path.join(self.experiment_path, "summary", "res_rescleft.zarr"),
                        mode='r')
                    try:
                        self.res_summary['quality'] = zarr.open(
                            os.path.join(self.experiment_path, "summary", "res_qual.zarr"),
                            mode='r')
                    except zarr.errors.PathNotFoundError:
                        print("No res_qual found!")


        else:
            # found npz summary for resources
            self.res_summary = np.load(os.path.join(self.experiment_path, "summary", "resource_summary.npz"),
                                       mmap_mode="r+")

        print("Overwriting undersample ratio with the one read from env file...")
        self.t_start = self.env.get('SUMMARY_TSTART')
        self.t_end = self.env.get('SUMMARY_TEND')
        self.undersample = int(float(self.env.get("SUMMARY_UNDERSAMPLE", "1")))
        print(f"Previous summary had parameters t_start={self.t_start} : us-{self.undersample}-us : {self.t_end}=t_end")

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

    def calculate_pairwise_pol_matrix_supervect(self, t, undersample=1, batchi=0):
        """
        Calculating NxN matrix of pairwise polarizations between agents in a given condition.
        The condition idx is a tuple of the varying parameter indices.
        t is the time index.
        """
        # Defining dimesnions in the orientation array
        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        num_agents = self.agent_summary["orientation"].shape[agent_dim]

        # reshape the orientation array to have the first dimensions batch_dim and num_var_params
        print("Loading Orientation data!")
        ori_data = self.agent_summary["orientation"][batchi, ..., ::undersample]
        ori_shape = ori_data.shape
        num_timesteps = ori_shape[-1]
        new_shape = ori_shape[0:agent_dim] + (num_agents, num_timesteps)
        print(ori_shape, new_shape)
        ori_reshaped = np.reshape(ori_data, new_shape)

        # calculate pairwise polarizations
        ag_uni = np.stack((np.cos(ori_reshaped), np.sin(ori_reshaped)), axis=-1)
        ag_uni = ag_uni.reshape((-1, num_agents, num_timesteps, 2))
        normed_sum = np.linalg.norm(ag_uni[:, :, None, :] + ag_uni[:, None, :, :], axis=-1) / 2
        pol_matrix = normed_sum.reshape((new_shape[:-1] + (num_agents, num_agents, num_timesteps)))

        # reshape pol_matrix back to the desired shape
        pol_matrix = np.moveaxis(pol_matrix, -1, agent_dim)
        return pol_matrix

    def return_clustering_distnace(self, condition_idx, t_real, undersample, pm=None):
        """Returns the clustering sidtance calculated from orientation and iid"""
        t_closest = int(t_real / undersample)
        # print("Using t_closest=", t_closest, "for t_real=", t_real)
        max_dist = (self.env.get("ENV_WIDTH") ** 2 + self.env.get("ENV_HEIGHT") ** 2) ** 0.5 / 2
        niidm = self.iid_matrix[condition_idx + tuple([slice(None), slice(None), t_closest])]  # /max_dist
        niidm = np.abs(np.median(niidm) - niidm) / max_dist
        # niidm = (niidm - mean_iidm) / std_iidm
        # niidm = (niidm - np.mean(niidm)) / max_dist
        # # punishing large distances more
        # niidm = niidm ** 2
        # make lower triangle elements the same as upper triangle elements
        niidm = niidm + niidm.T

        if pm is None:
            pm = self.calculate_pairwise_pol_matrix_vectorized(condition_idx, int(t_closest * undersample))

        # standardizing pm and normiidm so that their mean is 0 and their std is 1
        # todo: here we normalize with the mean and std of the slice and not the whole matrix
        # pm = (pm - np.mean(pm)) / np.std(pm)
        dist_pm = 1 - pm.astype('float')
        # # squaring distances to punish large distances more
        # dist_pm = dist_pm ** 2

        # calculating the final distance matrix as a weighted average of the two
        dist = (dist_pm + niidm) / 2

        return niidm, dist_pm, dist

    def return_dendogram_data(self, linkage, labels, thr=0.275, show_leaf_counts=True, no_plot=True):
        """Returning the clustering data according to the linkage matrix"""
        ret = dendrogram(linkage, color_threshold=thr, labels=labels,
                         show_leaf_counts=show_leaf_counts, no_plot=no_plot)
        return ret


    def plot_convex_hull_in_current_t(self, idx, agposx=None, agposy=None, with_plotting=True, on_torus=False,
                                      calc_longest_d=False, calc_orthogonal=False):
        """Plotting the estimated convex hull in a given time step"""

        # positions are not passed, we read it according to the passed index
        if agposx is None or agposy is None:
            if idx is not None:
                num_agents = int(self.env.get("N"))
                # The shape will depend on the IID matrix which is costly.
                # If we use undersample there, we also use undersample here
                num_timesteps_orig = self.env.get("T")
                num_timesteps = self.agent_summary["posx"].shape[-1]
                undersample = int(num_timesteps_orig // num_timesteps)
                print(f"Num varying params: {len(self.varying_params)}")
                num_varying_params = len(self.varying_params)

                condition_idx = idx[0:-1]
                t = idx[-1]

                t_closest = int(t / undersample)
                agposx = self.agent_summary["posx"][condition_idx + tuple([slice(None), t_closest])]
                agposy = self.agent_summary["posy"][condition_idx + tuple([slice(None), t_closest])]
            else:
                raise ValueError("If no positions are passed, idx must be passed!")
        else:
            num_agents = agposx.shape[0]

        if with_plotting:
            plt.figure()
            # plt.scatter(agposy, agposx)
            # draw arena as a black rectangle
            plt.plot([0, 0, self.env.get("ENV_HEIGHT"), self.env.get("ENV_HEIGHT"), 0],
                     [0, self.env.get("ENV_WIDTH"), self.env.get("ENV_WIDTH"), 0, 0], 'k-', label="Arena")
            # restricting the plot from -arena size to + arena size
            plt.xlim(-self.env.get("ENV_HEIGHT"), 2 * self.env.get("ENV_HEIGHT"))
            plt.ylim(-self.env.get("ENV_WIDTH"), 2 * self.env.get("ENV_WIDTH"))

        if on_torus:
            # plotting agent copies
            # defining 9 colors for different copy sets
            agent_copies = np.zeros((num_agents, 2, 9))
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'yellow', 'orange']

            # doing the same with list iteration
            for i, (wh, hh) in enumerate(itertools.product([-1, 0, 1], repeat=2)):
                agent_copies[:, 0, i] = agposx + wh * self.env.get("ENV_HEIGHT")
                agent_copies[:, 1, i] = agposy + hh * self.env.get("ENV_WIDTH")
                if with_plotting:
                    color = colors.pop(0)
                    if i == 0:
                        plt.scatter(agent_copies[:, 0, i], agent_copies[:, 1, i], c=color, label=f"Agent Copies")
                    else:
                        plt.scatter(agent_copies[:, 0, i], agent_copies[:, 1, i], c=color)

            agent_pos = np.zeros((num_agents, 2))
            for agi in range(num_agents):
                # print(f"Finding closest agent copy on torus tiling for agent {agi}")
                # if first agent we take the real coordinates
                if agi == 0:
                    agent_pos[agi, 0] = agposx[agi]
                    agent_pos[agi, 1] = agposy[agi]
                # now we loop through the agents and we select the closest copy of the agent for every timestep
                else:
                    prev_ag_posx = agent_pos[0, 0]
                    prev_ag_posy = agent_pos[0, 1]

                    # Vectorized distance calculation
                    differences_x = agent_copies[agi, 0, :] - prev_ag_posx
                    differences_y = agent_copies[agi, 1, :] - prev_ag_posy
                    distances = np.sqrt(differences_x ** 2 + differences_y ** 2)

                    # Find the index of the closest copy
                    closest_copy = np.argmin(distances)

                    agent_pos[agi, 0] = agent_copies[agi, 0, closest_copy]
                    agent_pos[agi, 1] = agent_copies[agi, 1, closest_copy]

                    # plotting line from previous point to new one with light grey
                    if with_plotting:
                        if agi == 1:
                            plt.plot([prev_ag_posx, agent_pos[agi, 0]], [prev_ag_posy, agent_pos[agi, 1]], 'lightgrey', label="connection route")
                        else:
                            plt.plot([prev_ag_posx, agent_pos[agi, 0]], [prev_ag_posy, agent_pos[agi, 1]], 'lightgrey')

        else:
            agent_pos = np.zeros((num_agents, 2))
            agent_pos[:, 0] = agposx
            agent_pos[:, 1] = agposy

        # calculating the convex hull
        hull = ConvexHull(agent_pos)

        if with_plotting:
            # plotting the convex hull
            plt.scatter(hull.points[:, 0], hull.points[:, 1], c='k')
            # plotting agent_pos with x markers to see if this is correct
            plt.scatter(agent_pos[:, 0], agent_pos[:, 1], c='r', marker='x')
            for si, simplex in enumerate(hull.simplices):
                if si == 0:
                    plt.plot(agent_pos[simplex, 0], agent_pos[simplex, 1], 'k--', label="Convex hull")
                else:
                    plt.plot(agent_pos[simplex, 0], agent_pos[simplex, 1], 'k--')

        if calc_longest_d:
            hull_points = hull.points[hull.vertices]
            differences = hull_points[:, np.newaxis, :] - hull_points[np.newaxis, :, :]
            distances = np.sqrt(np.sum(differences ** 2, axis=-1))
            # forbidding consecutive indices, i.e. (i, i+1) indices should be 0
            distances[[i for i in range(len(hull_points) - 1)], [i + 1 for i in range(len(hull_points) - 1)]] = 0
            # forbidding other way around
            distances[[i + 1 for i in range(len(hull_points) - 1)], [i for i in range(len(hull_points) - 1)]] = 0
            # finding the index of the longest radius from distances
            max_inds = np.unravel_index(np.argmax(distances), distances.shape)
            # calculate distance between max inds
            max_dist = distances[max_inds]

            if calc_orthogonal:
                # angles of new radii to longest one
                angles = []
                shifts = []
                if len(hull_points)==4:
                    min_inds = []
                    # we simply take the 2 indices which are not in max_inds
                    for i in range(len(hull_points)):
                        if i not in max_inds:
                            min_inds.append(i)
                    min_inds = tuple(min_inds)
                else:
                    for hi in range(int(len(hull_points))):
                        for ki in range(int(len(hull_points))):
                            hii = (max_inds[0] + hi) % len(hull_points)
                            kii = (max_inds[1] + ki) % len(hull_points)
                            if len(hull_points) > 3:
                                forbidden_indices = (np.abs(hii - kii) != 1  # not consecutive
                                                     and np.abs(hii - kii) != len(hull_points) - 1  # not consecutive on circle
                                                     and hii not in [max_inds[0], max_inds[1]]  # not the same as max_inds
                                                     and kii not in [max_inds[0], max_inds[1]])
                            else:
                                forbidden_indices = True  # if the hull is a triangle we take any 2 points that is most orthogonal

                            if forbidden_indices:
                                # calculating a gle between line created with points max_inds and new shifted lines
                                new_indices = hii, kii
                                # calculating angle defined by 2 set of points
                                longest_radius_vector = hull_points[max_inds[1]] - hull_points[max_inds[0]]
                                new_radius_vector = hull_points[new_indices[1]] - hull_points[new_indices[0]]
                                angle = angle_between(longest_radius_vector, new_radius_vector)
                                if not np.isnan(angle):
                                    shifts.append((hii, kii))
                                    angles.append(np.abs(angle))

                    # finding index of angles where item is closest to 90 degrees
                    if len(angles) > 0:
                        best_shift_index = np.argmin(np.abs(np.array(angles) - np.pi / 2))
                        best_shift = shifts[best_shift_index]
                    else:
                        best_shift = max_inds
                        print("No orthogonal radius found, taking longest radius as orthogonal radius")
                        plt.figure()
                        plt.scatter(hull_points[:, 0], hull_points[:, 1])
                        plt.show()

                    # print("Best shift: ", best_shift)

                    min_inds = best_shift

                min_dist = distances[min_inds]
            else:
                min_inds = None
                min_dist = None
        else:
            max_inds = None
            max_dist = None
            min_inds = None
            min_dist = None

        # print(max_inds, min_inds, min_dist, max_inds, max_dist)

        # plotting the shortest/longest radius
        if with_plotting:
            if calc_longest_d:
                plt.plot([hull_points[max_inds[0], 0], hull_points[max_inds[1], 0]],
                         [hull_points[max_inds[0], 1], hull_points[max_inds[1], 1]], 'r-', label="Longest diameter")
                # draw circle around longest diameter
                circle = plt.Circle(((hull_points[max_inds[0], 0] + hull_points[max_inds[1], 0]) / 2,
                                     (hull_points[max_inds[0], 1] + hull_points[max_inds[1], 1]) / 2),
                                    max_dist / 2, color='r', fill=False, label="Circle around longest diameter")
                plt.gca().add_artist(circle)

            if calc_orthogonal:
                plt.plot([hull_points[min_inds[0], 0], hull_points[min_inds[1], 0]],
                         [hull_points[min_inds[0], 1], hull_points[min_inds[1], 1]], 'b-', label="approx. 'Orthogonal' diameter")
            # setting axis to preserve aspect ratio
            plt.axis('equal')
            # reversing y axis
            plt.gca().invert_yaxis()
            plt.legend()
            plt.xlabel("Virtual X with tiling")
            plt.ylabel("Virtual Y with tiling")
            plt.title(f"Convex Hull and Longest Diameter at t={t_closest} on torus tiling")
            if calc_longest_d:
                # adding text with hull area
                plt.text(0, 0, f"Hull Area: {hull.volume:.2f}\nCircle Area: {np.pi * (max_dist / 2) ** 2:.2f},\nRatio: {hull.volume / (np.pi * (max_dist / 2) ** 2):.2f}",
                         fontsize=12, verticalalignment='bottom', horizontalalignment='right')
            plt.show()

        return hull, max_inds, max_dist, min_inds, min_dist

    def plot_clustering_in_current_t(self, idx, with_plotting=True):
        """Calculating clustering in a given time step
        param: idx: index tuple of the currently viewed slice from the replay tool
            idx = (batchi, <dims along varying_params>, slice(None), slice(None), t)
            idx will be used to index both the IID and polarization matrices to get an NxN matrix of pairwise distances
            for both the interindividual distance and the pairwise polarization."""
        if self.iid_matrix is None:
            self.calculate_interindividual_distance()

        num_agents = int(self.env.get("N"))
        # The shape will depend on the IID matrix which is costly.
        # If we use undersample there, we also use undersample here
        num_timesteps_orig = self.env.get("T")
        num_timesteps = self.iid_matrix.shape[-1]
        undersample = int(num_timesteps_orig // num_timesteps)
        print(f"Num varying params: {len(self.varying_params)}")
        num_varying_params = len(self.varying_params)

        condition_idx = idx[0:-1]
        t = idx[-1]
        t_closest = int(t / undersample)

        # calculating the clustering distance
        niidm, dist_pm, dist = self.return_clustering_distnace(condition_idx, t, undersample)

        # clustering
        linkage_matrix = linkage(dist, "ward")
        # print(linkage_matrix.shape, linkage_matrix)
        ret = self.return_dendogram_data(linkage_matrix, [i for i in range(num_agents)], no_plot=not with_plotting)
        colors = [color for _, color in sorted(zip(ret['leaves'], ret['leaves_color_list']))]
        # print(colors)
        # print(len(list(set(colors))))
        group_ids = np.array([int(a.split("C")[1]) for a in colors])
        for i, gid in enumerate(group_ids):
            # when gid is zero we make it the maximum element
            # because it means it is an independent leaf
            if gid == 0:
                group_ids[i] = np.max(group_ids) + 1
        group_ids -= 1
        # print(group_ids)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(dist_pm, cmap="viridis")
        ax[0].set_title("Pairwise polarization")
        ax[1].imshow(niidm, cmap="viridis")
        ax[1].set_title("Interindividual distance")
        plt.show()

    def calculate_clustering(self):
        """Using hierarhical clustering according to inter-individual distance and order scores to get number of
        subgroups"""
        # Checking if in the previous run a cluster_dict has already been saved and reloading it if exists
        clustering_data_path = os.path.join(self.experiment_path, "summary", "clustering_data.npy")
        largest_clustering_data_path = os.path.join(self.experiment_path, "summary", "largest_clustering_data.npy")
        clustering_id_path = os.path.join(self.experiment_path, "summary", "clustering_id.npy")
        if self.iid_matrix is None:
            self.calculate_interindividual_distance()

        if os.path.isfile(clustering_data_path):
            print("Cluster dict reloaded!")
            # loading cluster data numpy array
            self.clustering_data = np.load(clustering_data_path, mmap_mode="r+")
            self.largest_clustering_data = np.load(largest_clustering_data_path, mmap_mode="r+")
            self.clustering_ids = np.load(clustering_id_path, mmap_mode="r+")
            return

        from fastcluster import linkage
        from scipy.cluster.hierarchy import dendrogram
        clustering_dict = {}
        num_agents = int(self.env.get("N"))
        # calculating the maximum distance on the torus as the half diameter of the arena
        max_dist = (self.env.get("ENV_WIDTH") ** 2 + self.env.get("ENV_HEIGHT") ** 2) ** 0.5 / 2
        # The shape will depend on the IID matrix which is costly.
        # If we use undersample there, we also use undersample here
        num_timesteps_orig = self.env.get("T")
        num_timesteps = self.iid_matrix.shape[-1]
        undersample = int(num_timesteps_orig // num_timesteps)
        clustering_dict['num_subgroups'] = []
        print(f"Num varying params: {len(self.varying_params)}")
        num_varying_params = len(self.varying_params)
        clustering_data = np.zeros(tuple(list(self.iid_matrix.shape[:num_varying_params + 1]) + [num_timesteps]))
        largest_clustering_data = np.zeros(
            tuple(list(self.iid_matrix.shape[:num_varying_params + 1]) + [num_timesteps]))
        clustering_ids = np.zeros(
            tuple(list(self.iid_matrix.shape[:num_varying_params + 1]) + [num_agents, num_timesteps]))
        # calculating the number of parameter combinations along iidm.shape[:num_varying_params+1]
        num_combinations = np.prod(self.iid_matrix.shape[:num_varying_params + 1])
        curr_param_comb = 0
        print("Calculating polarizations first")
        t_slice = slice(0, num_timesteps_orig, undersample)

        for batchi in range(self.agent_summary["orientation"].shape[0]):
            print("Batchi: ", batchi)
            condition_idx = (batchi, slice(None), slice(None), slice(None))
            pol_m_large = self.calculate_pairwise_pol_matrix_vectorized(condition_idx, t_slice)
            for idx_base_ in np.ndindex(*self.iid_matrix.shape[1:num_varying_params + 1]):
                print("Progress: ", curr_param_comb, "/", num_combinations)
                idx_base = tuple([batchi] + list(idx_base_))
                for t in range(num_timesteps):
                    idx = tuple(list(idx_base) + [slice(None), slice(None), t])
                    idx_ = tuple(list(idx_base_) + [slice(None), slice(None), t])

                    # calculating the clustering distance
                    normiidm, dist_pm, dist = self.return_clustering_distnace(idx_base, t * undersample, undersample,
                                                                              pm=pol_m_large[idx_])

                    if idx_base == (0, 0, 0, 0) and 5 < t < 3:
                        with_plotting = True
                        print(dist_pm)
                        print(normiidm)
                    else:
                        with_plotting = False

                    # clustering
                    linkage_matrix = linkage(dist, "ward")
                    ret = self.return_dendogram_data(linkage_matrix, [i for i in range(num_agents)],
                                                     no_plot=not with_plotting)
                    colors = [color for _, color in sorted(zip(ret['leaves'], ret['leaves_color_list']))]
                    group_ids = np.array([int(a.split("C")[1]) for a in colors])
                    for i, gid in enumerate(group_ids):
                        # when gid is zero we make it the maximum element
                        # because it means it is an independent leaf
                        if gid == 0:
                            group_ids[i] = np.max(group_ids) + 1

                    group_ids -= 1
                    clustering_data[tuple(list(idx_base) + [t])] = len(list(set(group_ids)))
                    largest_clustering_data[tuple(list(idx_base) + [t])] = self.calculate_largest_subcluster_size(
                        group_ids)
                    clustering_ids[tuple(list(idx_base) + [slice(None), t])] = group_ids
                    if with_plotting:
                        plt.show()
                        input(group_ids)
                curr_param_comb += 1

        self.clustering_data = clustering_data
        self.largest_clustering_data = largest_clustering_data
        self.clustering_ids = clustering_ids
        print(f"Saving clustering data...")
        # save clustering_data as numpy array
        np.save(clustering_data_path, clustering_data)
        np.save(largest_clustering_data_path, largest_clustering_data)
        np.save(clustering_id_path, clustering_ids)

        return clustering_dict

    def calculate_largest_subcluster_size(self, cluster_id_list):
        """Calculating the size (i.e. number of agents) of largest subcluster of the group"""
        cluster_sizes = []
        for cluster_id in set(cluster_id_list):
            cluster_sizes.append(np.sum(cluster_id_list == cluster_id))
        return np.max(cluster_sizes)


    def plot_elongation(self):
        """Method to plot elongation of the agents in the environment"""
        cbar = None
        T = self.agent_summary["posx"].shape[-1]
        if T > 1000:
            undersample = int(T / 1000)
        else:
            undersample = 1
        self.calculate_group_elongation(undersample=undersample)
        min_data = np.min(self.mean_elong)
        max_data = np.max(self.mean_elong)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            # fig, ax = plt.subplots(1, 1)
            # plt.title("Number of Subclusters")
            # plt.plot(self.mean_clusters)
            # for run_i in range(self.efficiency.shape[0]):
            #     plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            # ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            # ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            # plt.xlabel(list(self.varying_params.keys())[0])
            pass

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_elong)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_elong.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_elong = np.zeros((self.mean_elong.shape[0:3]))
                print(self.new_mean_elong.shape)
                for j in range(self.mean_elong.shape[0]):
                    for i in range(self.mean_elong.shape[1]):
                        self.new_mean_elong[j, i, :] = self.mean_elong[j, i, :, i]
                self.mean_elong = self.new_mean_elong
            if self.collapse_plot is None:
                num_plots = self.mean_elong.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_elong[i, :, :], vmin=min_data, vmax=max_data)
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

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True)
        return fig, ax, cbar


    def plot_largest_subclusters(self):
        """Method to plot size of largest subclusters irrespectively of how many parameters have been tuned during the
                experiments."""
        cbar = None
        self.calculate_clustering()
        self.mean_largest_clusters = np.mean(np.mean(self.largest_clustering_data, axis=0), axis=-1)
        min_data = np.min(self.mean_largest_clusters)
        max_data = np.max(self.mean_largest_clusters)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            # fig, ax = plt.subplots(1, 1)
            # plt.title("Number of Subclusters")
            # plt.plot(self.mean_clusters)
            # for run_i in range(self.efficiency.shape[0]):
            #     plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            # ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            # ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            # plt.xlabel(list(self.varying_params.keys())[0])
            pass

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_largest_clusters)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_largest_clusters.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_largest_clusters = np.zeros((self.mean_largest_clusters.shape[0:3]))
                print(self.new_mean_largest_clusters.shape)
                for j in range(self.mean_largest_clusters.shape[0]):
                    for i in range(self.mean_largest_clusters.shape[1]):
                        self.new_mean_largest_clusters[j, i, :] = self.mean_largest_clusters[j, i, :, i]
                self.mean_largest_clusters = self.new_mean_largest_clusters
            if self.collapse_plot is None:
                num_plots = self.mean_largest_clusters.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_largest_clusters[i, :, :], vmin=min_data, vmax=max_data)
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

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True)
        return fig, ax, cbar

    def plot_clustering(self):
        """Method to plot clustering irrespectively of how many parameters have been tuned during the
                experiments."""
        cbar = None
        self.calculate_clustering()
        self.mean_clusters = np.mean(np.mean(self.clustering_data, axis=0), axis=-1)
        min_data = np.min(self.mean_clusters)
        max_data = np.max(self.mean_clusters)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            # fig, ax = plt.subplots(1, 1)
            # plt.title("Number of Subclusters")
            # plt.plot(self.mean_clusters)
            # for run_i in range(self.efficiency.shape[0]):
            #     plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            # ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            # ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            # plt.xlabel(list(self.varying_params.keys())[0])
            pass

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_clusters)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_clusters.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_clusters = np.zeros((self.mean_clusters.shape[0:3]))
                print(self.new_mean_clusters.shape)
                for j in range(self.mean_clusters.shape[0]):
                    for i in range(self.mean_clusters.shape[1]):
                        self.new_mean_clusters[j, i, :] = self.mean_clusters[j, i, :, i]
                self.mean_clusters = self.new_mean_clusters
            if self.collapse_plot is None:
                num_plots = self.mean_clusters.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_clusters[i, :, :], vmin=min_data, vmax=max_data)
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

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True)
        return fig, ax, cbar

    def calculate_search_efficiency(self, t_start_plot=0, t_end_plot=-1, used_batches=None):
        """Method to calculate search efficiency throughout the experiments as the sum of collected resorces normalized
        with the travelled distance. The timestep in which the efficiency is calculated. This might mismatch from
        the real time according to how much the data was undersampled during sammury"""
        summary_path = os.path.join(self.experiment_path, "summary")
        effpath = os.path.join(summary_path, "eff.npy")
        if os.path.isfile(effpath):
            print("Found saved efficiency array in summary, reloading it...")
            self.efficiency = np.load(effpath)
            batch_dim = 0
            num_var_params = len(list(self.varying_params.keys()))
            agent_dim = batch_dim + num_var_params + 1
            time_dim = agent_dim + 1
            self.mean_efficiency = np.mean(np.mean(self.efficiency, axis=agent_dim), axis=batch_dim)
            self.eff_std = np.std(np.mean(self.efficiency, axis=agent_dim), axis=batch_dim)
        else:
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

            # Saving calculated efficiency for future use
            print("Saving efficiency arrays into summary!")
            np.save(effpath, self.efficiency)

    def calculate_interindividual_distance_slice(self, posx, posy):
        """Calculating iid just in a point of time and for just a specific parameter combination.
        In this case posx and posy is already indexed/sliced, so that both has a shape of (num_agents)"""
        num_agents = len(posx)
        iid = np.zeros((num_agents, num_agents))
        for agi in range(num_agents):
            distance = supcalc.distance_coords(posx, posy, np.roll(posx, -agi), np.roll(posy, -agi), vectorized=True)
            for i in range(len(distance)):
                j = (i + agi) % num_agents
                iid[i, j] = distance[i]
        return iid

    def calculate_group_elongation(self, undersample=1, avg_over_time=False):
        """Calculating the group elongation as the effective radius of the surrounding polygon of agents"""
        # reloading data if already calculated
        summary_path = os.path.join(self.experiment_path, "summary")
        elongpath = os.path.join(summary_path, "elong.npy")
        meanelong_path = os.path.join(summary_path, "meanelong.npy")
        hullpoints_path = os.path.join(summary_path, "hullpoints.npy")
        # First we only implement on torus
        if os.path.isfile(elongpath):
            print("Found saved Elongation array in summary, reloading it...")
            self.elong_matrix = np.load(elongpath)
            self.mean_elong = np.load(meanelong_path)
            print("Found saved hull points array in summary, reloading it...")
            self.hull_points_array = np.load(hullpoints_path)
        else:
            # calculating elongation by tiling the arena and finding agent copies that are closest to previous
            # agent/or copy.
            # Method Summary: Start with Agent 1: Select the original position of the first agent as the starting point.
            # Iterate Over Agents: For each subsequent agent: Consider all possible copies of this agent within the 3x3
            # duplicated arena space (including the original position and the positions shifted by the arena's width
            # and/or height in all directions). Calculate the Euclidean distance from this agent's copies to the
            # previously selected agent (or agent copy). Select the copy that minimizes this distance.
            # Construct Final List: Compile the selected positions into a final list that represents all agents in a
            # modified Euclidean space that accounts for the toroidal wrapping. Apply Euclidean Metrics: Now that you've
            # effectively translated the problem into a Euclidean space, you can apply standard geometrical or
            # statistical analyses, such as calculating the convex hull to determine the group's elongation or the
            # effective radius of the surrounding polygon.

            agposx = self.agent_summary['posx']
            agposy = self.agent_summary['posy']

            t_idx = -1
            num_batches = agposx.shape[0]
            num_agents = agposx.shape[-2]

            new_shape = list(agposx.shape)

            # collapsing shape along agent dimension as we will calculate 1 number per timestep and not per agent
            if avg_over_time:
                # collapsing along time dimension as we will average here
                new_shape[t_idx] = 1
            else:
                new_shape[t_idx] = int(new_shape[t_idx] / undersample)

            new_shape.pop(-2)
            new_shape = tuple(new_shape)

            # ## ----Elongation matrix---- will have dim (num_batches, *[dim of varying params], time): and
            # includes the elongation of the group at time t at the index: elong[..., t] where the first
            # dimensions will be the same as in our convention according to varying parameters. As an example if
            # we changed the parameter1 along 3 different cases and parameter2 along 5 different cases,
            # and we had 20 batches with 10 agents we can get the elongation at time 100 as elong[...,
            # 100] which has the shape of (20, 3, 5)

            elong = np.zeros(new_shape)
            elong_rs = elong.reshape((elong.shape[0], -1, elong.shape[-1]))

            hull_points_array = np.zeros(elong_rs.shape[0:2] + (num_agents, 2, elong.shape[-1]))
            hull_points_array[:] = np.nan

            for batchi in range(num_batches):
                if self.env.get("BOUNDARY") == "infinite":
                    on_torus = True
                else:
                    on_torus = False
                print(f"Calculating group elongation for batch {batchi}, torus={on_torus}...")
                agposx_rs = np.array(agposx[batchi, ..., ::undersample]).reshape((-1, agposx.shape[-2], elong.shape[-1]))
                agposy_rs = np.array(agposy[batchi, ..., ::undersample]).reshape((-1, agposy.shape[-2], elong.shape[-1]))
                for i in range(agposx_rs.shape[0]):
                    print(f"Batch {batchi}/{num_batches} : {i/agposx_rs.shape[0]*100:.2f}%", end="\r")
                    for t in range(agposx_rs.shape[-1]):
                        posx = agposx_rs[i, :, t]
                        posy = agposy_rs[i, :, t]
                        try:
                            hull, max_inds, max_dist, min_inds, min_dist = self.plot_convex_hull_in_current_t(None, agposx=posx,
                                                                                          agposy=posy,
                                                                                          with_plotting=False,
                                                                                          on_torus=on_torus,
                                                                                          calc_longest_d=True,
                                                                                          calc_orthogonal=False)
                        except:
                            print("Error in convex hull calculation! Noting with None in the elongation matrix.")
                            max_dist = None
                            elong_rs[batchi, i, t] = max_dist
                            continue
                        else:
                            pass

                        # calculating circle area diameter max_dist
                        if max_dist is not None:
                            circle_area = np.pi * (max_dist / 2) ** 2

                            # calculating the elongation as the ratio of the convex hull area to the circle area
                            elong_rs[batchi, i, t] = hull.volume / circle_area
                        else:
                            elong_rs[batchi, i, t] = None

                        # saving convex hull into hull_points_array
                        hull_points = hull.points[hull.vertices]
                        for agi in range(num_agents):
                            if agi < len(hull_points):
                                hull_points_array[batchi, i, agi, :, t] = hull_points[agi]
                            else:
                                # concatenating None until the array is filled
                                hull_points_array[batchi, i, agi, :, t] = np.array([None, None])

            # reshaping back to original
            self.elong_matrix = elong_rs.reshape(elong.shape)
            print("Saving elongation arrays into summary!")
            np.save(elongpath, self.elong_matrix)

            # calculating mean elongation and saving it
            self.mean_elong = np.nanmean(np.nanmean(self.elong_matrix, axis=-1), axis=0)
            print("Saving mean elongation arrays into summary!")
            np.save(meanelong_path, self.mean_elong)

            # saving hull points array
            # as the new shape we take the shape of the elong until the last dimension, then we add the number of
            # agents and the number of spatial dimensions (x, y) and the time dimension
            hull_points_array = hull_points_array.reshape(elong.shape[:-1] + (num_agents, 2, elong.shape[-1]))
            self.hull_points_array = hull_points_array
            print("Saving hull points arrays into summary!")
            np.save(hullpoints_path, self.hull_points_array)


    def calculate_interindividual_distance(self, undersample=1, avg_over_time=False, periodic_boundary=False):
        """Method to calculate inter-individual distance array from posx and posy arrays of agents. The final
        array has the same dimension as any of the input arrays, i.e.:
        (num_batches, *[dims of varying params], num_agents, time)
        and defines the mean (across group members) inter-individual distance for a given agent i in timestep t.
        If periodic boundary is true, we calculate distances on a torus.
        """
        summary_path = os.path.join(self.experiment_path, "summary")
        iidpath = os.path.join(summary_path, "iid.npy")
        meaniid_path = os.path.join(summary_path, "meaniid.npy")
        if os.path.isfile(iidpath):
            print("Found saved I.I.D array in summary, reloading it...")
            self.iid_matrix = np.load(iidpath)
        else:
            if self.env.get("BOUNDARY") == "infinite":
                print(
                    "Dataset was generated with periodic boundary conditions. Do you want to calculate I.I.D. according"
                    "to torus environment? (Y/N)")
                periodic_boundary = input().lower() == "y"

            agposx = self.agent_summary['posx']
            agposy = self.agent_summary['posy']

            t_idx = -1
            num_batches = agposx.shape[0]
            num_agents = agposx.shape[-2]

            new_shape = list(agposx.shape)
            new_shape.insert(-1, num_agents)
            if avg_over_time:
                # collapsing along time dimension as we will average here
                new_shape[t_idx] = 1
            else:
                new_shape[t_idx] = int(new_shape[t_idx] / undersample)
            new_shape = tuple(new_shape)

            # ----IID matrix---- will have dim (num_batches, *[dim of varying params], num_agents, num_agents,
            # t) and includes the inter individual distance between agent i and j in time t at the index: iid[..., i, j,
            # t] where the first dimensions will be the same as in our convention according to varying parameters. As an
            # example if we changed the batch radius along 3 different cases and the agent radius along 5 different
            # cases, and we had 20 batches with 10 agents we can get the iid between agent 2 and 7 at time 100 as
            # iid[..., 2, 7, 100] which has the shape of (20, 3, 5) according to the different scenarios and number of
            # batches The time dimension can vary according to undersampling rate!
            iid = np.zeros(new_shape)

            for batchi in range(num_batches):
                print(f"Calculating iid for batch {batchi}, torus={periodic_boundary}")
                for agi in range(num_agents):
                    for agj in range(num_agents):
                        if agj > agi:
                            x1s = agposx[batchi, ..., agi, ::undersample]
                            y1s = agposy[batchi, ..., agi, ::undersample]
                            x2s = agposx[batchi, ..., agj, ::undersample]
                            y2s = agposy[batchi, ..., agj, ::undersample]
                            if not periodic_boundary:
                                distance_matrix = supcalc.distance_coords(x1s, y1s, x2s, y2s, vectorized=True)
                            else:
                                p1 = np.array([[x1s[ii], y1s[ii]] for ii in range(len(x1s))])
                                p2 = np.array([[x2s[ii], y2s[ii]] for ii in range(len(x2s))])
                                dim = np.array([int(self.env["ENV_WIDTH"]), int(self.env["ENV_HEIGHT"])])
                                dima = np.zeros_like(p1)
                                dima[:, 0, ...] = dim[0]
                                dima[:, 1, ...] = dim[1]
                                distance_matrix = supcalc.distance_torus(p1, p2, dima)
                            if not avg_over_time:
                                iid[batchi, ..., agi, agj, :] = distance_matrix
                            else:
                                iid[batchi, ..., agi, agj, 0] = np.mean(distance_matrix, axis=-1)

            self.iid_matrix = iid

        if os.path.isfile(meaniid_path):
            print("Found mean I.I.D matrix in summary. Reloading it...")
            self.mean_iid = np.load(meaniid_path)
        else:
            # for the mean we restrict the iid matrix as upper triangular in the agent dimensions so that we
            # don't calculate IIDs in mean twice (as IID is symmetric, i.e. the distance between agent i and j
            # is the same as between j and i)
            num_agents = self.agent_summary['posx'].shape[-2]
            restr_m = self.iid_matrix[..., np.triu_indices(num_agents, k=1)[0], np.triu_indices(num_agents, k=1)[1], :]
            # Then we take the mean along the flattened dimension in which we defined the previous restriction, and we also
            # take the mean along all the repeated batches (0dim)
            if avg_over_time:
                # in this case the last dimension (time) is unnecessary as we have shape 1 along this
                self.mean_iid = np.mean(np.mean(restr_m, axis=-2)[..., 0], axis=0)
                self.iid_matrix = self.iid_matrix[..., 0]
            else:
                self.mean_iid = np.mean(np.mean(restr_m, axis=-2), axis=0)

            # Saving calculated arrays for future use
            print("Saving I.I.D and mean I.I.D arrays into summary!")
            np.save(iidpath, self.iid_matrix)
            np.save(meaniid_path, self.mean_iid)

    def calculate_mean_NN_dist(self, undersample=1, avg_over_time=False):
        """Calculating the mean nearest neighbor metric over time. We take the average over
        agents of the min neighbor distance per agent."""
        summary_path = os.path.join(self.experiment_path, "summary")
        iidpath = os.path.join(summary_path, "iid.npy")
        meanNNd_path = os.path.join(summary_path, "meanNNd.npy")
        if os.path.isfile(meanNNd_path):
            print("Found saved mean NND array in summary, reloading it...")

            self.mean_nn_dist = np.load(meanNNd_path)
        if self.iid_matrix is None:
            if os.path.isfile(iidpath):
                print("Found saved IID array in summary, reloading it...")
                self.iid_matrix = np.load(iidpath)
            else:
                print("Didn't find saved IID array in summary, calculating them...")
                self.calculate_interindividual_distance(undersample=undersample, avg_over_time=False)
        iid = self.iid_matrix.copy()
        num_agents = iid.shape[-2]
        for i in range(num_agents):
            iid[..., i, i, :] = np.nan
            # setting lower triangle equal to upper triangular
            iid[..., i:, i, :] = iid[..., i, i:, :]

        nearest_per_agents = np.nanmin(iid, axis=-2)
        mean_nearest = np.mean(nearest_per_agents, axis=-2)
        self.mean_nn_dist = np.mean(mean_nearest, axis=0)
        print("Saving mean NNd array under ", meanNNd_path)
        np.save(meanNNd_path, self.mean_nn_dist)

    def plot_mean_rotational_order(self, t_start=0, t_end=-1, from_script=False, used_batches=None):
        """Method to plot rotational order irrespectively of how many parameters have been tuned during the
        experiments."""
        cbar = None
        self.calculate_rotational_order()

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Polarization")
            plt.plot(self.mean_rotord)
            plt.plot(self.mean_rotord + self.rotord_std)
            plt.plot(self.mean_rotord - self.rotord_std)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_rotord)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_rotord.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_rotord = np.zeros((self.mean_rotord.shape[0:3]))
                print(self.new_mean_rotord.shape)
                for j in range(self.mean_rotord.shape[0]):
                    for i in range(self.mean_rotord.shape[1]):
                        self.new_mean_rotord[j, i, :] = self.mean_rotord[j, i, :, i]
                self.mean_rotord = self.new_mean_rotord
            if self.collapse_plot is None:
                num_plots = self.mean_rotord.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_rotord[i, :, :])
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

                collapsed_data, labels = self.collapse_mean_data(self.mean_rotord, save_name="coll_rotord.npy")
                coll_std, _ = self.collapse_mean_data(self.rotord_std, save_name="coll_rotordstd.npy")

                # # column-wise normalization
                # for coli in range(collapsed_data.shape[1]):
                #     print(f"Normalizing column {coli}")
                #     minval = np.min(collapsed_data[:, coli])
                #     maxval = np.max(collapsed_data[:, coli])
                #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)

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
                fig.set_tight_layout(True)

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

    def plot_mean_polarization(self, t_start=0, t_end=-1, from_script=False, used_batches=None):
        """Method to plot polarization irrespectively of how many parameters have been tuned during the
        experiments."""
        cbar = None
        self.calculate_polarization()

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Polarization")
            plt.plot(self.mean_pol)
            plt.plot(self.mean_pol + self.pol_std)
            plt.plot(self.mean_pol - self.pol_std)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_pol)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_pol.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_pol = np.zeros((self.mean_pol.shape[0:3]))
                print(self.new_mean_pol.shape)
                for j in range(self.mean_pol.shape[0]):
                    for i in range(self.mean_pol.shape[1]):
                        self.new_mean_pol[j, i, :] = self.mean_pol[j, i, :, i]
                self.mean_pol = self.new_mean_pol
            if self.collapse_plot is None:
                num_plots = self.mean_pol.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_pol[i, :, :], vmin=0, vmax=1)
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

                collapsed_data, labels = self.collapse_mean_data(self.mean_pol, save_name="coll_pol.npy")
                coll_std, _ = self.collapse_mean_data(self.pol_std, save_name="coll_polstd.npy")

                # # column-wise normalization
                # for coli in range(collapsed_data.shape[1]):
                #     print(f"Normalizing column {coli}")
                #     minval = np.min(collapsed_data[:, coli])
                #     maxval = np.max(collapsed_data[:, coli])
                #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)

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
                fig.set_tight_layout(True)

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

    def plot_mean_collision_time(self, t_start=0, t_end=-1, from_script=False, used_batches=None, undersample=1):
        cbar = None
        self.calculate_collision_time(undersample=undersample)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Collision Time")
            plt.plot(self.mean_aacoll)
            plt.plot(self.mean_aacoll + self.aacoll_std)
            plt.plot(self.mean_aacoll - self.aacoll_std)
            for run_i in range(self.efficiency.shape[0]):
                plt.plot(np.mean(self.efficiency, axis=agent_dim)[run_i, ...], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(self.mean_aacoll)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(self.mean_aacoll.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_pol = np.zeros((self.mean_aacoll.shape[0:3]))
                print(self.new_mean_pol.shape)
                for j in range(self.mean_aacoll.shape[0]):
                    for i in range(self.mean_aacoll.shape[1]):
                        self.new_mean_pol[j, i, :] = self.mean_aacoll[j, i, :, i]
                self.mean_aacoll = self.new_mean_pol
            if self.collapse_plot is None:
                num_plots = self.mean_aacoll.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(self.mean_aacoll[i, :, :], vmin=0, vmax=np.max(self.mean_aacoll))
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

                collapsed_data, labels = self.collapse_mean_data(self.mean_aacoll, save_name="coll_aacoll.npy")
                coll_std, _ = self.collapse_mean_data(self.aacoll_std, save_name="coll_polstd.npy")

                # # column-wise normalization
                # for coli in range(collapsed_data.shape[1]):
                #     print(f"Normalizing column {coli}")
                #     minval = np.min(collapsed_data[:, coli])
                #     maxval = np.max(collapsed_data[:, coli])
                #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)

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
                fig.set_tight_layout(True)

        num_agents = self.agent_summary["orientation"].shape[agent_dim]
        description_text = ""
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

    def calculate_pairwise_pol_matrix_vectorized(self, condition_idx, t):
        """t is also slice"""
        # Defining dimesnions in the orientation array
        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        num_agents = self.agent_summary["orientation"].shape[agent_dim]

        # Get the orientations of all agents at time t
        agent_ori = self.agent_summary["orientation"][condition_idx + (slice(None), t)]
        # plt.figure()
        # plt.plot(agent_ori)
        # plt.show()
        # print(agent_ori.shape)
        # input()

        # Calculate the univectors of all agent pairs
        agent_uni = np.stack([np.cos(agent_ori), np.sin(agent_ori)], axis=-1)
        # print(agent_uni.shape)
        # input()
        if isinstance(t, slice):
            # print("t is slice")
            agi_uni = np.expand_dims(agent_uni, axis=-3)
            # print(agi_uni.shape)
            agj_uni = np.expand_dims(agent_uni, axis=-4)
            # print(agj_uni.shape)
            # input()
        else:
            # print("t is not slice")
            agi_uni = np.expand_dims(agent_uni, axis=-2)
            # print(agi_uni.shape)
            agj_uni = np.expand_dims(agent_uni, axis=-3)
            # print(agj_uni.shape)
            # input()

        # Calculate the pairwise polarizations using matrix multiplication
        pol_matrix = np.linalg.norm(agi_uni + agj_uni, axis=-1) / 2

        # print(pol_matrix.shape)
        # input()
        # print(pol_matrix[..., 0])

        return pol_matrix

    def calculate_pairwise_pol_matrix(self, condition_idx, t):
        """
        Calculating NxN matrix of pairwise polarizations between agents in a given condition.
        The condition idx is a tuple of the varying parameter indices.
        t is the time index.
        """
        # Defining dimesnions in the orientation array
        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        num_agents = self.agent_summary["orientation"].shape[agent_dim]

        # calculate pairwise polarizations
        pol_matrix = np.zeros((num_agents, num_agents))
        for i in range(num_agents):
            for j in range(num_agents):
                # getting orientation of 2 agents i and j
                agi_ori = self.agent_summary["orientation"][condition_idx + (i, t)]
                agj_ori = self.agent_summary["orientation"][condition_idx + (j, t)]

                # calculating univectors with orientations of agent i and j
                agi_uni = np.array([np.cos(agi_ori), np.sin(agi_ori)])
                agj_uni = np.array([np.cos(agj_ori), np.sin(agj_ori)])

                # calculating the absolute sum of the univectors normed by the number of agents
                # contributing to the sum (=2, pairwise)
                normed_sum = np.linalg.norm(agi_uni + agj_uni) / 2

                pol_matrix[i, j] = normed_sum

        return pol_matrix

    def calculate_rotational_order(self, undersample=1, filtered_by_wallcoll=0, filtering_window=50):
        """Calculating the rotational order of the simulated agents as defined in:
        https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002915#s2
        The rotational order is the sum of cross products between
        1.) ri: the vector pointing from COM to agent i and
        2.) ui: the unit vector of the orientation of agent i
        """
        summary_path = os.path.join(self.experiment_path, "summary")
        if filtered_by_wallcoll:
            self.find_wall_collisions()
            rotordpath = os.path.join(summary_path, f"rotord_wallcoll.npy")
        else:
            rotordpath = os.path.join(summary_path, f"rotord_us{undersample}.npy")

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if os.path.isfile(rotordpath):
            print("Found saved rotational order array in summary, reloading it...")
            self.rotord_matrix = np.abs(np.load(rotordpath))
            self.mean_rotord = np.nanmean(np.nanmean(self.rotord_matrix, axis=batch_dim), axis=-1)
            self.rotord_std = np.nanmean(np.nanstd(self.rotord_matrix, axis=batch_dim), axis=-1)
        else:
            num_agents = self.agent_summary["orientation"].shape[agent_dim]
            num_timesteps = self.agent_summary["orientation"].shape[time_dim] / undersample
            ori_shape = list(self.agent_summary["orientation"].shape)
            new_shape = ori_shape[0:num_var_params + 1] + [int(num_timesteps / undersample)]

            self.rotord_matrix = np.zeros(new_shape)
            unitvec_shape = ori_shape[1:-2] + [2] + [num_agents, int(num_timesteps / undersample)]

            for runi in range(self.num_batches):
                print(f"Calculating rotational order for batch {runi}")
                if filtered_by_wallcoll:
                    orif = self.agent_summary["orientation"][runi, ...]
                    print(f"Extending collision filtering with time window {filtering_window}")
                    wrefs = self.wrefs[runi][:]
                    for wi in range(filtering_window):
                        print(wi)
                        new_times = wrefs[-1] + 1
                        new_times[new_times > self.env["T"] - 1] = self.env["T"] - 1
                        wrefs[-1] = new_times
                        new_wrefs = tuple(wrefs)
                        orif[new_wrefs] = np.nan
                else:
                    orif = self.agent_summary["orientation"][runi, ..., ::undersample]

                unitvecs = np.zeros(unitvec_shape)
                for robi in range(num_agents):
                    if filtered_by_wallcoll:
                        ori = orif[..., robi, :]
                    else:
                        ori = self.agent_summary["orientation"][runi, ..., robi, ::undersample]
                    unitvecs[..., 0, robi, :] = np.array([np.cos(ang) for ang in ori])
                    unitvecs[..., 1, robi, :] = np.array([np.sin(ang) for ang in ori])

                # calculating the center of mass of the agents in all timesteps for current run alomg agent dimension
                comx = np.nanmean(self.agent_summary["posx"][runi, ..., ::undersample], axis=-2)
                comy = np.nanmean(self.agent_summary["posy"][runi, ..., ::undersample], axis=-2)

                # calculating the vector pointing from COM to agent i
                rvec = np.zeros(unitvec_shape)
                for robi in range(num_agents):
                    rvec[..., 0, robi, :] = self.agent_summary["posx"][runi, ..., robi, ::undersample] - comx
                    rvec[..., 1, robi, :] = self.agent_summary["posy"][runi, ..., robi, ::undersample] - comy

                # calculating the absolute value of cross product between rvec and unitvec
                crossprod = np.abs(np.cross(rvec, unitvecs, axis=-3))

                # summing over all agents
                self.rotord_matrix[runi, ...] = np.nansum(crossprod, axis=-2)

            self.mean_rotord = np.nanmean(np.nanmean(self.rotord_matrix, axis=batch_dim), axis=-1)
            self.rotord_std = np.nanmean(np.nanstd(self.rotord_matrix, axis=batch_dim), axis=-1)
            print("Saving calculated rotational order time matrix!")
            np.save(rotordpath, self.rotord_matrix)

        return self.rotord_matrix, self.mean_rotord

    def calculate_polarization(self, undersample=1, filtered_by_wallcoll=0, filtering_window=50):
        """Calculating polarization of agents in the environment used to
        quantify e.g. flocking models"""
        summary_path = os.path.join(self.experiment_path, "summary")
        if filtered_by_wallcoll:
            self.find_wall_collisions()
            polpath = os.path.join(summary_path, f"polarization_wallcoll.npy")
        else:
            polpath = os.path.join(summary_path, f"polarization_us{undersample}.npy")

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if os.path.isfile(polpath):
            print("Found saved polarization array in summary, reloading it...")
            self.pol_matrix = np.load(polpath)
            self.mean_pol = np.nanmean(np.nanmean(self.pol_matrix, axis=batch_dim), axis=-1)
            self.pol_std = np.nanmean(np.nanstd(self.pol_matrix, axis=batch_dim), axis=-1)

        else:
            num_agents = self.agent_summary["orientation"].shape[agent_dim]
            num_timesteps = self.agent_summary["orientation"].shape[time_dim] / undersample
            ori_shape = list(self.agent_summary["orientation"].shape)
            new_shape = ori_shape[0:num_var_params + 1] + [int(num_timesteps / undersample)]

            self.pol_matrix = np.zeros(new_shape)
            unitvec_shape = ori_shape[1:-2] + [2] + [num_agents, int(num_timesteps / undersample)]

            for runi in range(self.num_batches):
                print(f"Calculating polarization for batch {runi}")
                if filtered_by_wallcoll:

                    # wrefs = list(tuple(self.wrefs[runi][:]))
                    # for ti, t in enumerate(wrefs_old[-1]):
                    #     print(ti)
                    #     wrefs[-1] = np.append(wrefs[1], np.array([tk for tk in range(t, t+100)]))
                    #     for wi in range(len(wrefs)-1):
                    #         wrefs[wi] = np.append(wrefs[wi], np.array([wrefs[wi][t] for tk in range(100)]))

                    orif = self.agent_summary["orientation"][runi, ...]
                    print(f"Extending collision filtering with time window {filtering_window}")
                    wrefs = self.wrefs[runi][:]
                    for wi in range(filtering_window):
                        print(wi)
                        new_times = wrefs[-1] + 1
                        new_times[new_times > self.env["T"] - 1] = self.env["T"] - 1
                        wrefs[-1] = new_times
                        new_wrefs = tuple(wrefs)
                        orif[new_wrefs] = np.nan

                unitvecs = np.zeros(unitvec_shape)
                for robi in range(num_agents):
                    if filtered_by_wallcoll:
                        ori = orif[..., robi, :]
                    else:
                        ori = self.agent_summary["orientation"][runi, ..., robi, ::undersample]
                    unitvecs[..., 0, robi, :] = np.array([np.cos(ang) for ang in ori])
                    unitvecs[..., 1, robi, :] = np.array([np.sin(ang) for ang in ori])

                unitsum = np.nansum(unitvecs, axis=-2)  # summing for all robots
                unitsum_norm = np.linalg.norm(unitsum, axis=-2) / num_agents  # getting norm in x and y
                self.pol_matrix[runi, ...] = unitsum_norm  # np.nanmean(unitsum_norm, axis=-1)

            # for runi in range(self.num_batches):
            #     pol_matrix[runi, ...] = np.mean(np.array(
            #         [np.linalg.norm([unitsum[runi, 0, t], unitsum[runi, 1, t]]) / num_agents for t in
            #          range(num_timesteps)]), axis=-1)

            self.mean_pol = np.nanmean(np.nanmean(self.pol_matrix, axis=batch_dim), axis=-1)
            self.pol_std = np.nanmean(np.nanstd(self.pol_matrix, axis=batch_dim), axis=-1)
            print("Saving calculated polarization time matrix!")
            np.save(polpath, self.pol_matrix)

        return self.pol_matrix, self.mean_pol

    def calculate_collision_time(self, undersample=1):
        summary_path = os.path.join(self.experiment_path, "summary")
        iidpath = os.path.join(summary_path, "iid.npy")
        aacollpath = os.path.join(summary_path, "aacoll.npy")
        batch_dim = 0

        if os.path.isfile(aacollpath):
            print("Found saved collision time array in summary, reloading it...")
            self.aacoll_matrix = np.load(aacollpath)
        else:
            if self.iid_matrix is None and not os.path.isfile(iidpath):
                self.calculate_interindividual_distance(undersample=undersample)
            elif os.path.isfile(iidpath):
                self.iid_matrix = np.load(iidpath)
            aacoll = np.zeros(list(self.agent_summary['orientation'].shape)[0:-2])
            num_timesteps = self.iid_matrix.shape[-1]
            if self.iid_matrix.shape[-1] < 1000:
                raise Exception(
                    f"The IID matrix has been collapsed on the time axis to len {self.iid_matrix.shape[-1]} when was calculated, can not"
                    " calculate collision matrix. Please delete iid.npy from the summary folder and"
                    "try again!")
            else:
                iid_individual_coll = np.any(self.iid_matrix < (2 * self.env["RADIUS_AGENT"]),
                                             axis=-2)  # containing info about which agent has been collided
                iid_sum_coll = np.any(
                    np.logical_and(self.iid_matrix > 0, self.iid_matrix < (2 * self.env["RADIUS_AGENT"])),
                    axis=-2)  # is there collision or not in every timestep
                aacoll = np.count_nonzero(iid_sum_coll, axis=-1) / num_timesteps
                self.aacoll_matrix = aacoll

        self.mean_aacoll = np.mean(np.mean(self.aacoll_matrix, axis=batch_dim), axis=-1)
        self.aacoll_std = np.std(np.mean(self.aacoll_matrix, axis=batch_dim), axis=-1)
        print("Saving calculated mean aacoll time matrix!")
        np.save(aacollpath, self.aacoll_matrix)

        return self.aacoll_matrix, self.mean_aacoll

    def find_wall_collisions(self, undersample=1):
        """Finding the timepoints of the data where ANY of the agents have been reflected from ANY of the walls"""
        summary_path = os.path.join(self.experiment_path, "summary")
        wrefpaths = [os.path.join(summary_path, f"wallref_b{bi}.zarr") for bi in range(self.num_batches)]
        self.wrefs = {}
        if os.path.isdir(wrefpaths[0]):
            print("Found saved wall reflection array in summary, reloading it...")
            for bi in range(self.num_batches):
                self.wrefs[bi] = zarr.open(wrefpaths[bi], mode='r', dtype='int')
        else:
            boundaries_x = [0, int(float(self.env.get("ENV_WIDTH")))]
            boundaries_y = [0, int(float(self.env.get("ENV_HEIGHT")))]
            for bi in range(self.num_batches):
                print(f"Calculating wall reflection times to batch {bi}")
                agposx = self.agent_summary['posx'][bi, ...]
                agposy = self.agent_summary['posy'][bi, ...]

                wrefs = np.array(np.where(np.logical_or(np.logical_or(
                    agposx < boundaries_x[0] - self.env.get("RADIUS_AGENT"),
                    agposx > boundaries_x[1] - self.env.get("RADIUS_AGENT")),
                    np.logical_or(
                        agposy < boundaries_y[0] - self.env.get("RADIUS_AGENT"),
                        agposy > boundaries_y[1] - self.env.get("RADIUS_AGENT")))))

                wrefszarr = zarr.open(wrefpaths[bi], mode='w',
                                      shape=wrefs.shape, dtype='int')
                wrefszarr[...] = wrefs[...]
                self.wrefs[bi] = wrefszarr

    def calculate_relocation_time(self, undersample=1):
        """Calculating relocation time matrix over the given data"""
        summary_path = os.path.join(self.experiment_path, "summary")
        rtimepath = os.path.join(summary_path, "reloctime.npy")
        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if os.path.isfile(rtimepath):
            print("Found saved relocation time array in summary, reloading it...")
            rel_reloc_matrix = np.load(rtimepath)
            mean_rel_reloc = np.mean(np.mean(rel_reloc_matrix, axis=agent_dim), axis=batch_dim)

        else:
            rel_reloc_matrix = np.zeros(self.agent_summary["mode"].shape[0:-1])
            for i in range(self.num_batches):
                print(f"Calculating relocation time in batch {i}")
                a = np.mean((self.agent_summary["mode"][i, ...] == 2).astype(int), axis=time_dim - 1)
                rel_reloc_matrix[i] = a.copy()
                del a
            mean_rel_reloc = np.mean(np.mean(rel_reloc_matrix, axis=agent_dim), axis=batch_dim)
            print("Saving calculated relocation time matrix!")
            np.save(rtimepath, rel_reloc_matrix)

        return rel_reloc_matrix, mean_rel_reloc

    def collapse_mean_data(self, mean_data, save_name=None):
        """Collapsing a high dimensional mean data array according to chosen axis and method via UI.
        The generated collapsed adataframe can be saved in the summary folder when save_name set to
        a .npy file"""
        keys = sorted(list(self.varying_params.keys()))
        shape_along_fixed_ind = mean_data.shape[self.collapse_fixedvar_ind]
        labels = []
        # collapsing data
        for i in range(shape_along_fixed_ind):
            if self.collapse_fixedvar_ind == 0:
                collapsed_data_row = self.collapse_method(mean_data[i, :, :], axis=0)
                ind = np.argmax(mean_data[i, :, :], axis=0)
                max1_ind = 1
                max2_ind = 2
            elif self.collapse_fixedvar_ind == 1:
                collapsed_data_row = self.collapse_method(mean_data[:, i, :], axis=0)
                ind = np.argmax(mean_data[:, i, :], axis=0)
                max1_ind = 0
                max2_ind = 2
            elif self.collapse_fixedvar_ind == 2:
                collapsed_data_row = self.collapse_method(mean_data[:, :, i], axis=0)
                ind = np.argmax(mean_data[:, :, i], axis=0)
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

        if save_name is not None:
            if save_name.endswith(".npy"):
                save_path_data = os.path.join(self.experiment_path, "summary", save_name)
                save_path_labels = save_path_data.split(".")[0] + ".txt"
                np.save(save_path_data, collapsed_data)
                np.savetxt(save_path_labels, labels, newline=",\n", fmt="%s")
            else:
                raise Exception("Given filename is not npy file")

        return collapsed_data, labels

    def plot_mean_NN_dist(self, from_script=False, undersample=1):
        """Method to plot mean inter-individual distance irrespectively of how many parameters have been tuned during the
        experiments."""
        cbar = None
        if self.mean_nn_dist is None:
            # self.calculate_interindividual_distance(undersample=undersample)
            self.calculate_mean_NN_dist(undersample=undersample)

        print(self.mean_nn_dist.shape)
        _mean_nn_dist = np.mean(self.mean_nn_dist, axis=-1)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            raise Exception("Not implemented yet!")

        elif num_var_params == 2:
            raise Exception("Not implemented yet!")

        elif num_var_params == 3 or num_var_params == 4:
            if len(_mean_nn_dist.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_nn_dist = np.zeros((_mean_nn_dist.shape[0:3]))
                print(self.new_mean_nn_dist.shape)
                for j in range(_mean_nn_dist.shape[0]):
                    for i in range(_mean_nn_dist.shape[1]):
                        self.new_mean_nn_dist[j, i, :] = _mean_nn_dist[j, i, :, i]
                _mean_nn_dist = self.new_mean_nn_dist

            if self.collapse_plot is None:
                num_plots = _mean_nn_dist.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(_mean_nn_dist[i, :, :])
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

                collapsed_data, labels = self.collapse_mean_data(_mean_nn_dist, save_name="coll_iid.npy")

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

                fig.set_tight_layout(True)

        num_agents = self.agent_summary["posx"].shape[agent_dim]
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"of inter-individual distance between agents.\n"
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

    def plot_mean_iid(self, from_script=False, undersample=1):
        """Method to plot mean inter-individual distance irrespectively of how many parameters have been tuned during the
        experiments."""
        cbar = None
        if self.iid_matrix is None:
            self.calculate_interindividual_distance(undersample=undersample)

        _mean_iid = np.mean(self.mean_iid, axis=-1)
        print("shape:", _mean_iid.shape)

        batch_dim = 0
        num_var_params = len(list(self.varying_params.keys()))
        agent_dim = batch_dim + num_var_params + 1
        time_dim = agent_dim + 1

        if num_var_params == 1:
            fig, ax = plt.subplots(1, 1)
            plt.title("Inter-individual distance (mean)")
            plt.plot(_mean_iid)
            num_agents = self.iid_matrix.shape[-2]
            restr_m = self.iid_matrix[..., np.triu_indices(num_agents, k=1)[0], np.triu_indices(num_agents, k=1)[1]]
            for run_i in range(self.iid_matrix.shape[0]):
                plt.plot(restr_m[run_i, :, :], marker=".", linestyle='None')
            ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
            ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
            plt.xlabel(list(self.varying_params.keys())[0])

        elif num_var_params == 2:
            fig, ax = plt.subplots(1, 1)
            plt.title("Inter-individual distance (mean)")
            keys = sorted(list(self.varying_params.keys()))
            im = ax.imshow(_mean_iid)

            ax.set_yticks(range(len(self.varying_params[keys[0]])))
            ax.set_yticklabels(self.varying_params[keys[0]])
            ax.set_ylabel(keys[0])

            ax.set_xticks(range(len(self.varying_params[keys[1]])))
            ax.set_xticklabels(self.varying_params[keys[1]])
            ax.set_xlabel(keys[1])

        elif num_var_params == 3 or num_var_params == 4:
            if len(_mean_iid.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                self.new_mean_iid = np.zeros((_mean_iid.shape[0:3]))
                print(self.new_mean_iid.shape)
                for j in range(_mean_iid.shape[0]):
                    for i in range(_mean_iid.shape[1]):
                        self.new_mean_iid[j, i, :] = _mean_iid[j, i, :, i]
                _mean_iid = self.new_mean_iid

            if self.collapse_plot is None:
                num_plots = _mean_iid.shape[0]
                fig, ax = plt.subplots(1, num_plots, sharex=True, sharey=True)
                keys = sorted(list(self.varying_params.keys()))
                for i in range(num_plots):
                    img = ax[i].imshow(_mean_iid[i, :, :], vmin=np.min(_mean_iid), vmax=np.max(_mean_iid))
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

                collapsed_data, labels = self.collapse_mean_data(_mean_iid, save_name="coll_iid.npy")

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

                fig.set_tight_layout(True)

        num_agents = self.agent_summary["posx"].shape[agent_dim]
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"of inter-individual distance between agents.\n"
        self.add_plot_interaction(description_text, fig, ax, show=True, from_script=from_script)
        return fig, ax, cbar

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
                    img = ax[i].imshow(self.mean_efficiency[i, :, :], vmin=0, vmax=0.1)
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

                collapsed_data, labels = self.collapse_mean_data(self.mean_efficiency, save_name="coll_eff.npy")
                coll_std, _ = self.collapse_mean_data(self.eff_std, save_name="coll_effstd.npy")

                # # column-wise normalization
                # for coli in range(collapsed_data.shape[1]):
                #     print(f"Normalizing column {coli}")
                #     minval = np.min(collapsed_data[:, coli])
                #     maxval = np.max(collapsed_data[:, coli])
                #     collapsed_data[:, coli] = (collapsed_data[:, coli] - minval) / (maxval - minval)

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
                fig.set_tight_layout(True)

        num_agents = self.agent_summary["posx"].shape[agent_dim]
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

        rel_reloc_matrix, mean_rel_reloc = self.calculate_relocation_time()
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

        elif num_var_params == 3 or num_var_params == 4:
            if len(mean_rel_reloc.shape) == 4:
                # reducing the number of variables to 3 by connecting 2 of the dimensions
                new_mean_rel_reloc = np.zeros((mean_rel_reloc.shape[0:3]))
                print(mean_rel_reloc.shape)
                for j in range(mean_rel_reloc.shape[0]):
                    for i in range(mean_rel_reloc.shape[1]):
                        new_mean_rel_reloc[j, i, :] = mean_rel_reloc[j, i, :, i]
                mean_rel_reloc = new_mean_rel_reloc

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

                collapsed_data, labels = self.collapse_mean_data(mean_rel_reloc, save_name="coll_reloctime.npy")

                img = ax.imshow(collapsed_data)
                ax.set_yticks(range(len(self.varying_params[keys[self.collapse_fixedvar_ind]])))
                ax.set_yticklabels(self.varying_params[keys[self.collapse_fixedvar_ind]])
                ax.set_ylabel(keys[self.collapse_fixedvar_ind])

                ax.set_xticks(range(len(labels)))
                # ax.set_xlabel(f"Combined Parameters\n"
                #               f"{keys[max1_ind]}={self.varying_params[keys[max1_ind]]}\n"
                #               f"{keys[max2_ind]}={self.varying_params[keys[max2_ind]]}")
                ax.set_xticklabels(labels, rotation=45)
                ax.set_xlabel("Combined Parameters")

                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                cbar = fig.colorbar(img, cax=cbar_ax)

                fig.set_tight_layout(True)

        num_agents = self.agent_summary["posx"].shape[agent_dim]
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

        num_agents = self.agent_summary["posx"].shape[agent_dim]
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
