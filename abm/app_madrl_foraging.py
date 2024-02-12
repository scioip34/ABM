import importlib
import json
import shutil
from contextlib import ExitStack
from pathlib import Path


import optuna as optuna

import os
# loading env variables from dotenv file
from dotenv import dotenv_values

import abm


def setup_environment():
    EXP_NAME = os.getenv("EXPERIMENT_NAME", "madrl_foraging")
    EXP_NAME_COPY = f"{EXP_NAME}_copy"
    os.path.dirname(os.path.realpath(__file__))
    root_abm_dir = Path(__file__).parent.parent

    env_file_dir = root_abm_dir / "abm" / "projects" / "madrl_foraging"  # Path(__file__).parent
    env_path = env_file_dir / f"{EXP_NAME}.env"
    env_path_copy = env_file_dir / f"{EXP_NAME_COPY}.env"
    # make a duplicate of the env file to be used by the playground
    shutil.copyfile(env_path, env_path_copy)
    envconf = dotenv_values(env_path)
    return env_file_dir, EXP_NAME_COPY, envconf



def transform_envconf(envconf):
    key_mapping = {
        "N": ("N", int),
        "T": ("T", int),
        "TRAIN": ("train", lambda x: bool(int(x))),
        "WINDOW_PAD": ("window_pad", int),
        "AGENT_TYPE": ("agent_type", str),
        "VISUAL_FIELD_RESOLUTION": ("v_field_res", int),
        "AGENT_FOV": ("agent_fov", float),
        "INIT_FRAMERATE": ("framerate", int),
        "WITH_VISUALIZATION": ("with_visualization", lambda x: bool(int(x))),
        "ENV_WIDTH": ("width", int),
        "ENV_HEIGHT": ("height", int),
        "SHOW_VISUAL_FIELDS": ("show_vis_field", lambda x: bool(int(x))),
        "SHOW_VISUAL_FIELDS_RETURN": ("show_vis_field_return", lambda x: bool(int(x))),
        "POOLING_TIME": ("pooling_time", int),
        "POOLING_PROBABILITY": ("pooling_prob", float),
        "RADIUS_AGENT": ("agent_radius", int),
        "N_RESOURCES": ("N_resc", int),
        "PATCH_BORDER_OVERLAP": ("allow_border_patch_overlap", lambda x: bool(int(x))),
        "MIN_RESOURCE_PER_PATCH": ("min_resc_perpatch", int),
        "MAX_RESOURCE_PER_PATCH": ("max_resc_perpatch", int),
        "MIN_RESOURCE_QUALITY": ("min_resc_quality", float),
        "MAX_RESOURCE_QUALITY": ("max_resc_quality", float),
        "RADIUS_RESOURCE": ("patch_radius", int),
        "REGENERATE_PATCHES": ("regenerate_patches", lambda x: bool(int(x))),
        "AGENT_CONSUMPTION": ("agent_consumption", int),
        "GHOST_WHILE_EXPLOIT": ("ghost_mode", lambda x: bool(int(x))),
        "PATCHWISE_SOCIAL_EXCLUSION": ("patchwise_exclusion", lambda x: bool(int(x))),
        "TELEPORT_TO_MIDDLE": ("teleport_exploit", lambda x: bool(int(x))),
        "VISION_RANGE": ("vision_range", int),
        "VISUAL_EXCLUSION": ("visual_exclusion", lambda x: bool(int(x))),
        "SHOW_VISION_RANGE": ("show_vision_range", lambda x: bool(int(x))),
        "AGENT_AGENT_COLLISION": ("collide_agents", lambda x: bool(int(x))),
        "TRAIN_EVERY": ("train_every", int),


        "USE_RAM_LOGGING":("use_ram_logging", lambda x: bool(int(x))),
        "USE_ZARR_FORMAT": ("use_zarr", lambda x: bool(int(x))),
        "SAVE_CSV_FILES": ("save_csv_files", lambda x: bool(int(x))),



    }
    transformed_dict = {}

    for env_key, (new_key, data_type) in key_mapping.items():
        if env_key in envconf:
            transformed_dict[new_key] = data_type(envconf[env_key])

    return transformed_dict

def start_playground():
    # changing env file according to playground default parameters before
    # running any component of the SW
    save_isims_env(root_abm_dir, EXP_NAME, pgt, envconf)
    # Start interactive simulation
    sim = PlaygroundSimulation()
    sim.start()


def start(parallel=True, headless=False):
    # Define root abm directory from which env file is read out
    root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Finding env file
    EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
    env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
    if os.path.isfile(env_path):
        print(f"Read env vars from {env_path}")
        envconf = dotenv_values(env_path)
        app_version = envconf.get("APP_VERSION", "Base")
        if app_version != "MADRLForaging":
            raise Exception(".env file was not created for madrl foraging")
    else:
        raise Exception(f"Could not find .env file under path {env_path}")

    from abm.projects.madrl_foraging.madrl_simulation.madrl_sims import MADRLSimulation as Simulation



    vscreen_width = int(envconf["ENV_WIDTH"]) + 2 * int(envconf["WINDOW_PAD"]) + 10
    vscreen_height = int(envconf["ENV_HEIGHT"]) + 2 * int(envconf["WINDOW_PAD"]) + 10
    sim_params = transform_envconf(envconf)

    #TODO: Headless mode does not work
    if headless:
        # required to start pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from xvfbwrapper import Xvfb


    with ExitStack() if not headless else Xvfb(width=vscreen_width, height=vscreen_height) as xvfb:
        sim = Simulation(parallel=parallel,**sim_params)

        #sim.write_batch_size = 100
        if sim_params["train"]:
            print("Starting training")

            _ = sim.start_madqn_train()
        else:
            print("Starting evaluation")
            _ = sim.start_madqn_eval()

if __name__ == '__main__':
    start()



###################################################################################
##################################OPTUNE FUNCTIONS#################################
###################################################################################

"""def objective(trial):
    parallel=False
    headless=False
    agent_behave_param_list=None
    root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    envconf = dotenv_values(os.path.join(root_abm_dir, f"{EXP_NAME}.env"))
    window_pad = 30
    vscreen_width = int(float(envconf["ENV_WIDTH"])) + 2 * window_pad + 10
    vscreen_height = int(float(envconf["ENV_HEIGHT"])) + 2 * window_pad + 10
    print("Headless {}".format(headless))
    if headless:
        # required to start pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from xvfbwrapper import Xvfb
    with ExitStack() if not headless else Xvfb(width=vscreen_width, height=vscreen_height) as xvfb:
        sim = Simulation(N=int(float(envconf["N"])),
                         agent_type=envconf["AGENT_TYPE"],
                         T=int(float(envconf["T"])),
                         v_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
                         agent_fov=float(envconf['AGENT_FOV']),
                         framerate=int(float(envconf["INIT_FRAMERATE"])),
                         with_visualization=bool(int(float(envconf["WITH_VISUALIZATION"]))),
                         width=int(float(envconf["ENV_WIDTH"])),
                         height=int(float(envconf["ENV_HEIGHT"])),
                         show_vis_field=bool(int(float(envconf["SHOW_VISUAL_FIELDS"]))),
                         show_vis_field_return=bool(int(envconf['SHOW_VISUAL_FIELDS_RETURN'])),
                         pooling_time=int(float(envconf["POOLING_TIME"])),
                         pooling_prob=float(envconf["POOLING_PROBABILITY"]),
                         agent_radius=int(float(envconf["RADIUS_AGENT"])),
                         N_resc=int(float(envconf["N_RESOURCES"])),
                         allow_border_patch_overlap=bool(int(float(envconf["PATCH_BORDER_OVERLAP"]))),
                         min_resc_perpatch=int(float(envconf["MIN_RESOURCE_PER_PATCH"])),
                         max_resc_perpatch=int(float(envconf["MAX_RESOURCE_PER_PATCH"])),
                         min_resc_quality=float(envconf["MIN_RESOURCE_QUALITY"]),
                         max_resc_quality=float(envconf["MAX_RESOURCE_QUALITY"]),
                         patch_radius=int(float(envconf["RADIUS_RESOURCE"])),
                         regenerate_patches=bool(int(float(envconf["REGENERATE_PATCHES"]))),
                         agent_consumption=int(float(envconf["AGENT_CONSUMPTION"])),
                         ghost_mode=bool(int(float(envconf["GHOST_WHILE_EXPLOIT"]))),
                         patchwise_exclusion=bool(int(float(envconf["PATCHWISE_SOCIAL_EXCLUSION"]))),
                         teleport_exploit=bool(int(float(envconf["TELEPORT_TO_MIDDLE"]))),
                         vision_range=int(float(envconf["VISION_RANGE"])),
                         visual_exclusion=bool(int(float(envconf["VISUAL_EXCLUSION"]))),
                         show_vision_range=bool(int(float(envconf["SHOW_VISION_RANGE"]))),
                         use_ifdb_logging=False,
                         use_ram_logging=False,
                         save_csv_files=False,
                         use_zarr=bool(int(float(envconf["USE_ZARR_FORMAT"]))),
                         parallel=parallel,
                         window_pad=window_pad,
                         agent_behave_param_list=agent_behave_param_list,
                         collide_agents=bool(int(float(envconf["AGENT_AGENT_COLLISION"]))),
                         train=bool(int(float(envconf["TRAIN"]))),
                         train_every=int(float(envconf["TRAIN_EVERY"])),
                         replay_memory_capacity=int(float(envconf["REPLAY_MEMORY_CAPACITY"])),
                         batch_size = int(float(envconf["BATCH_SIZE"])),
                         gamma = trial.suggest_float("gamma", 0.9, 0.99),
                         lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                         epsilon_start = trial.suggest_float("epsilon_start", 0.1, 1.0),
                         epsilon_end = trial.suggest_float("epsilon_end", 0.001, 0.1),
                         epsilon_decay = trial.suggest_float("epsilon_decay", 0.9, 0.99),
                         tau = trial.suggest_float("tau", 0.0001, 0.01),
                         optimizer = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
                         models_dir=envconf["MODELS_DIR"]
                         )
        sim.write_batch_size = 100
        avg_search_efficiency = sim.start_opt(trial)
        return avg_search_efficiency
"""

"""
def generate_env_file(env_data, file_name, save_folder):
    #TODO: Delete this and import it from metaprotocol if needed 
    '''Generating a single env file under save_folder with file_name including env_data as env format'''
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")
"""

"""
def save_trial_info(trial, index):
    trial_params = trial.params
    trial_score = trial.value

    filename = f"best_trial_{index + 1}.json"
    with open(filename, "w") as file:
        json.dump({"Hyperparameters": trial_params, "Score": trial_score}, file)
"""
"""
def tune_hyperparams():
    # Set up Optuna study
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    # Run the optimization
    print(f"Sampler is {study.sampler.__class__.__name__}")
    print(f"Pruner is {study.pruner.__class__.__name__}")
    study.optimize(objective, n_trials=100)  # You can adjust the number of trials

    # Print the best hyperparameters
    best_params = study.best_params
    print("Best Hyperparameters:", best_params)
    print("Best Score:", study.best_value)

    # Save best hyperparameters and scores for trials in Pareto front
    pareto_front_trials = study.best_trials
    for i, trial in enumerate(pareto_front_trials):
        save_trial_info(trial, i)
"""

