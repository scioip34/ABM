"""
data_loader.py : including the main classes to load previously saved data (csv+json) into an initialized replayable simulation.
    The DataLoader class is only the data layer that loads data and then can create a LoadedSimulation instance accordingly.
"""

import json
import os
import glob
from abm.agent.agent import Agent
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

        for k, v in self.env_data.items():
            if isinstance(v, tuple):
                self.env_data[k] = v[0]

        # Change time-series data types
        for k, v in self.agent_data.items():
            if k.find("vfield") == -1:
                self.agent_data[k] = np.array([float(i) for i in v])
            else:
                self.agent_data[k] = np.array([json.loads(i.replace(" ", ", ")) for i in v], dtype=object)

        for k, v in self.resource_data.items():
            self.resource_data[k] = np.array([float(i) if i != "" else -1.0 for i in v])

    def get_loaded_data(self):
        """returning the loaded data upon request"""
        return self.agent_data, self.resource_data, self.env_data


class ExperimentLoader:
    """Loads and transforms a whole experiment folder with multiple batches and simulations"""

    def __init__(self, experiment_path, enforce_summary=False):
        # experiment data after summary
        self.res_summary = None
        self.agent_summary = None
        self.varying_params = {}

        # raw temporary experiment data
        self.all_env = {}
        self.all_agdata = {}
        self.all_rdata = {}

        # path variables
        self.experiment_path = experiment_path
        self.experiment_name = os.path.basename(experiment_path)

        # collecting batch folders
        glob_pattern = os.path.join(experiment_path, "*")
        self.batch_folders = [path for path in glob.iglob(glob_pattern) if path.find("summary") < 0]

        self.num_batches = len(self.batch_folders)
        self.num_runs = None

        # reading and restructuring raw data into numpy arrays
        if not self.is_already_summarized() or enforce_summary:
            self.read_all_data()
            # check parameter variability
            self.get_changing_variables()
            # summarizing loaded data into arrays
            self.summarize_data()

        # reloading previously saved numpy arrays
        self.reload_summarized_data()

        self.mean_collected_resources()

    def read_all_data(self):
        """reading all data in the experiment folder and storing them in the memory"""
        print("Reading all experimental data first...")
        for i, batch_path in enumerate(self.batch_folders):
            glob_pattern = os.path.join(batch_path, "*")
            run_folders = [path for path in glob.iglob(glob_pattern)]
            if i == 0:
                self.num_runs = len(run_folders)
            batch_env = {}
            batch_agdata = {}
            batch_rdata = {}
            for j, run in enumerate(run_folders):
                print(f"Reading batch {i}, run {j}")
                agent_data, res_data, env_data = DataLoader(run).get_loaded_data()
                batch_env[j] = env_data
                batch_agdata[j] = agent_data
                batch_rdata[j] = res_data
            self.all_env[i] = batch_env
            self.all_agdata[i] = batch_agdata
            self.all_rdata[i] = batch_rdata
        print("Datastructures initialized according to loaded data!")

    def get_changing_variables(self):
        """Collecting env variables along which the initialization has changed across runs"""
        print("Checking for changing parameters along runs...")
        base_keys = list(self.all_env[0][0].keys())
        variability = {}

        for base_key in base_keys:
            variability[base_key] = []

        # here we assume that parameter ranges do not change from batch to batch
        for ke, env in self.all_env[0].items():
            for k, v in env.items():
                variability[k].append(v)

        for k, v in variability.items():
            if len(list(set(v))) > 1:
                self.varying_params[k] = sorted([float(i) for i in list(set(v))])
                print(f"Found tuned parameter {k} with values {self.varying_params[k]}")

    def find_max_num_resources(self):
        """finding the maximum number of resources in an experiment through batches and runs"""
        max_num = 0
        for i in range(self.num_batches):
            for j in range(self.num_runs):
                num_in_run = len([k for k in list(self.all_rdata[i][j].keys()) if k.find("posx_res") > -1])
                if num_in_run > max_num:
                    max_num = num_in_run
        return max_num

    def summarize_data(self):
        """summarizing loaded data into numpy arrays and metadata"""

        print("summarizing experiment data into NumPy arrays...")
        num_agents = int(float(self.all_env[0][0]['N']))
        num_timesteps = int(float(self.all_env[0][0]['T']))
        axes_lens = []
        for k, v in self.varying_params.items():
            axes_lens.append(len(v))
        max_num_resources = self.find_max_num_resources()

        print("Summarizing collected agent data")
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

        for i in range(self.num_batches):
            print(f"*", end="")
            for j in range(self.num_runs):
                print(f".", end="")
                # indexing along the varying axes happens according to the extracted varying parameters
                index = [self.varying_params[k].index(float(self.all_env[i][j][k])) for k in
                         sorted(list(self.varying_params.keys()))]
                for ai in range(num_agents):
                    ind = (i,) + tuple(index) + (ai,)
                    posx_array[ind] = self.all_agdata[i][j][f'posx_agent-{pad_to_n_digits(ai, n=2)}']
                    posy_array[ind] = self.all_agdata[i][j][f'posy_agent-{pad_to_n_digits(ai, n=2)}']
                    rew_array[ind] = self.all_agdata[i][j][f'collectedr_agent-{pad_to_n_digits(ai, n=2)}']
                    ori_array[ind] = self.all_agdata[i][j][f'orientation_agent-{pad_to_n_digits(ai, n=2)}']
                    vel_array[ind] = self.all_agdata[i][j][f'velocity_agent-{pad_to_n_digits(ai, n=2)}']
                    w_array[ind] = self.all_agdata[i][j][f'w_agent-{pad_to_n_digits(ai, n=2)}']
                    u_array[ind] = self.all_agdata[i][j][f'u_agent-{pad_to_n_digits(ai, n=2)}']
                    Ip_array[ind] = self.all_agdata[i][j][f'Ipriv_agent-{pad_to_n_digits(ai, n=2)}']
                    mode_array[ind] = self.all_agdata[i][j][f'mode_agent-{pad_to_n_digits(ai, n=2)}']
                    expl_patch_array[ind] = self.all_agdata[i][j][f'expl_patch_id_agent-{pad_to_n_digits(ai, n=2)}']

                # remove dict data from memory
                self.all_agdata[i][j] = None

        print("\nSummarizing collected resource data")
        # num_batches x criterion1 x criterion2 x ... x criterionN x max_num_resources x time
        # criteria as in self.varying_params and ALWAYS IN ALPHABETIC ORDER
        # where the value is -1 the resource does not exist in time
        r_posx_array = -1 * np.ones((self.num_batches, *axes_lens, max_num_resources, num_timesteps))
        r_posy_array = -1 + np.ones((self.num_batches, *axes_lens, max_num_resources, num_timesteps))
        r_qual_array = -1 + np.ones((self.num_batches, *axes_lens, max_num_resources, num_timesteps))
        r_rescleft_array = -1 + np.ones((self.num_batches, *axes_lens, max_num_resources, num_timesteps))

        for i in range(self.num_batches):
            print(f"*", end="")
            for j in range(self.num_runs):
                print(f".", end="")
                # indexing along the varying axes happens according to the extracted varying parameters
                index = [self.varying_params[k].index(float(self.all_env[i][j][k])) for k in
                         sorted(list(self.varying_params.keys()))]
                num_res_in_run = len([k for k in list(self.all_rdata[i][j].keys()) if k.find("posx_res") > -1])
                for ri in range(num_res_in_run):
                    ind = (i,) + tuple(index) + (ri,)
                    data = self.all_rdata[i][j][f'posx_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = [float(d) if d != "" else -1.0 for d in data]
                    # clean empty strings as -1s
                    r_posx_array[ind] = data

                    data = self.all_rdata[i][j][f'posy_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = [float(d) if d != "" else -1.0 for d in data]
                    # clean empty strings as -1s
                    r_posy_array[ind] = data

                    data = self.all_rdata[i][j][f'quality_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = [float(d) if d != "" else -1.0 for d in data]
                    # clean empty strings as -1s
                    r_qual_array[ind] = data

                    data = self.all_rdata[i][j][f'resc_left_res-{pad_to_n_digits(ri + 1, n=3)}']
                    # clean empty strings
                    data = [float(d) if d != "" else -1.0 for d in data]
                    # clean empty strings as -1s
                    r_rescleft_array[ind] = data

            # remove dict data from memory
            self.all_rdata[i][j] = None

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

        np.savez(os.path.join(summary_path, "resource_summary.npz"),
                 posx=r_posx_array,
                 posy=r_posy_array,
                 quality=r_qual_array,
                 resc_left=r_rescleft_array)

        with open(os.path.join(summary_path, "fixed_env.json"), "w") as fenvf:
            fixed_env = self.all_env[0][0]
            for k, v in fixed_env.items():
                if k in list(self.varying_params.keys()):
                    fixed_env[k] = "----TUNED----"
            json.dump(fixed_env, fenvf)

        with open(os.path.join(summary_path, "tuned_env.json"), "w") as tenvf:
            json.dump(self.varying_params, tenvf)

        # cleaning initial data structures from memory
        self.all_agdata = None
        self.all_rdata = None
        self.all_env = None
        print("\nData summarized and saved under experiment folder!")

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
        with open(os.path.join(self.experiment_path, "summary", "tuned_env.json"), "r") as tunedf:
            self.varying_params = json.loads(tunedf.read())

        print("Experiment loaded")

    def mean_collected_resources(self):

        fig, ax = plt.subplots(1, 1)
        plt.title("Mean (over agents and batches) total collected resource units")
        im = ax.imshow(np.mean(np.mean(self.agent_summary["collresource"][:, :, :, :, -1], axis=3), axis=0))
        ax.set_xticks(range(len(self.varying_params[list(self.varying_params.keys())[0]])))
        ax.set_xticklabels(self.varying_params[list(self.varying_params.keys())[0]])
        plt.xlabel(list(self.varying_params.keys())[0])
        ax.set_yticks(range(len(self.varying_params[list(self.varying_params.keys())[1]])))
        ax.set_yticklabels(self.varying_params[list(self.varying_params.keys())[1]])
        plt.ylabel(list(self.varying_params.keys())[1])

        num_agents = self.agent_summary["collresource"].shape[3]
        num_runs = 1
        for k, v in self.varying_params.items():
            num_runs *= len(v)
        description_text = f"Showing the mean (over {self.num_batches} batches and {num_agents} agents)\n" \
                           f"of total collected resource units over the experiments.\n\n" \
                           f"Varied parameters: {list(self.varying_params.keys())}\n" \
                           f"Simulation time per run: {self.env['T']}\n" \
                           f"Number of runs per batch: {num_runs}\n" \
                           f"Number of resource patches: {self.env['N_RESOURCES']}\n" \
                           f"Resource Quality and Contained units: " \
                           f"Q{self.env['MIN_RESOURCE_QUALITY']}-{self.env['MAX_RESOURCE_QUALITY']}, " \
                           f"U{self.env['MIN_RESOURCE_PER_PATCH']}-{self.env['MAX_RESOURCE_PER_PATCH']}"
        bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
        annot = ax.annotate(description_text, xy=(0.1, 0.9), xycoords='axes fraction', horizontalalignment='left',
                            verticalalignment='top', bbox=bbox_props)
        annot.set_visible(False)

        fig.canvas.mpl_connect('button_press_event', lambda event: show_plot_description(event, fig, annot))
        fig.canvas.mpl_connect('button_release_event', lambda event: hide_plot_description(event, fig, annot))

        print(self.varying_params)
        plt.show()


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



# class LoadedSimulation:
#     def __init__(self, data_folder_path):
#         """Init method of LadedSimulation class to initialize a simulation-like structure according to
#         previously saved data"""
#         self.agent_data, self.resource_data, self.env_data = DataLoader(data_folder_path).get_loaded_data()
