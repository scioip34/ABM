import shutil
from contextlib import ExitStack
from pathlib import Path

import os
# loading env variables from dotenv file
from dotenv import dotenv_values


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


        "USE_RAM_LOGGING":("use_ram_logging", lambda x: bool(int(x))),
        "USE_ZARR_FORMAT": ("use_zarr", lambda x: bool(int(x))),
        "SAVE_CSV_FILES": ("save_csv_files", lambda x: bool(int(x))),



    }
    transformed_dict = {}

    for env_key, (new_key, data_type) in key_mapping.items():
        if env_key in envconf:
            transformed_dict[new_key] = data_type(envconf[env_key])

    return transformed_dict


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

    if envconf["BRAIN_TYPE"] == "ideal":
        from abm.projects.madrl_foraging.madrl_simulation.heuristic_sims import HeuristicSimulation as Simulation
    else:
        from abm.projects.madrl_foraging.madrl_simulation.madrl_sims import MADRLSimulation as Simulation

    vscreen_width = int(envconf["ENV_WIDTH"]) + 2 * int(envconf["WINDOW_PAD"]) + 10
    vscreen_height = int(envconf["ENV_HEIGHT"]) + 2 * int(envconf["WINDOW_PAD"]) + 10
    sim_params = transform_envconf(envconf)

    # TODO: Headless mode does not work
    if headless:
        # required to start pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from xvfbwrapper import Xvfb


    with ExitStack() if not headless else Xvfb(width=vscreen_width, height=vscreen_height) as xvfb:
        sim = Simulation(parallel=parallel,**sim_params)

        #sim.write_batch_size = 100
        if envconf["BRAIN_TYPE"] == "ideal":
            _ = sim.start_heuristic()
        else:
            _ = sim.start_madqn()


if __name__ == '__main__':
    start()

