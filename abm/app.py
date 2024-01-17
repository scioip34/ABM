from contextlib import ExitStack

from abm.simulation.sims import Simulation
from abm.simulation.isims import PlaygroundSimulation
import abm.contrib.playgroundtool as pgt

import os
# loading env variables from dotenv file
from dotenv import dotenv_values

print("HELLO")
print(os.getenv("ENV_PATH"))
print(dotenv_values(os.getenv("ENV_PATH")))
EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
env_path = os.getenv("ENV_PATH", os.path.join(root_abm_dir, f"{EXP_NAME}.env"))
print(env_path)
envconf = dotenv_values(env_path)
print(envconf)

def start(parallel=False, headless=False, agent_behave_param_list=None):
    root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    envconf = dotenv_values(os.path.join(root_abm_dir, f"{EXP_NAME}.env"))
    window_pad = 30
    vscreen_width = int(float(envconf["ENV_WIDTH"])) + 2 * window_pad + 10
    vscreen_height = int(float(envconf["ENV_HEIGHT"])) + 2 * window_pad + 10
    if headless:
        # required to start pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from xvfbwrapper import Xvfb
    with ExitStack() if not headless else Xvfb(width=vscreen_width, height=vscreen_height) as xvfb:
        sim = Simulation(N=int(float(envconf["N"])),
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
                         use_ifdb_logging=bool(int(float(envconf["USE_IFDB_LOGGING"]))),
                         use_ram_logging=bool(int(float(envconf["USE_RAM_LOGGING"]))),
                         save_csv_files=bool(int(float(envconf["SAVE_CSV_FILES"]))),
                         use_zarr=bool(int(float(envconf["USE_ZARR_FORMAT"]))),
                         parallel=parallel,
                         window_pad=window_pad,
                         agent_behave_param_list=agent_behave_param_list,
                         collide_agents=bool(int(float(envconf["AGENT_AGENT_COLLISION"])))
                         )
        sim.write_batch_size = 100
        sim.start()


def start_headless():
    print("Start ABM in Headless Mode...")
    start(headless=True)


def start_playground():
    # changing env file according to playground default parameters before
    # running any component of the SW
    save_isims_env(root_abm_dir, EXP_NAME, pgt, envconf)
    # Start interactive simulation
    sim = PlaygroundSimulation()
    sim.start()


def save_isims_env(env_dir, _EXP_NAME, _pgt, _envconf):
    """translating a default parameters dictionary to an environment
    file and using env variable keys instead of class attribute names
    :param env_dir: directory path of environemnt file"""
    def_params = _pgt.default_params
    def_env_vars = _pgt.def_env_vars
    translator_dict = _pgt.def_params_to_env_vars
    translated_dict = _envconf

    for k in def_params.keys():
        if k in list(translator_dict.keys()):
            v = def_params[k]
            if v == "True" or v is True:
                v = "1"
            elif v == "False" or v is False:
                v = "0"
            translated_dict[translator_dict[k]] = v
    for def_env_name, def_env_val in def_env_vars.items():
        translated_dict[def_env_name] = def_env_val

    print("Saving playground default params in env file under path ", env_dir)
    if os.path.isfile(os.path.join(env_dir, f"{_EXP_NAME}.env")):
        os.remove(os.path.join(env_dir, f"{_EXP_NAME}.env"))
    generate_env_file(translated_dict, f"{_EXP_NAME}.env", env_dir)

def generate_env_file(env_data, file_name, save_folder):
    """Generating a single env file under save_folder with file_name including env_data as env format"""
    os.makedirs(save_folder, exist_ok=True)
    file_path = os.path.join(save_folder, file_name)
    with open(file_path, "a") as file:
        for k, v in env_data.items():
            file.write(f"{k}={v}\n")
