import os
import shutil
from pathlib import Path

from contextlib import ExitStack
from dotenv import dotenv_values

from abm.app import save_isims_env
from abm.projects.visual_flocking.vf_simulation.vf_isims import VFPlaygroundSimulation
from abm.projects.visual_flocking.vf_contrib.vf_playgroundtool import setup_visflock_playground
from abm.projects.visual_flocking.vf_simulation.vf_sims import VFSimulation


def setup_environment():
    EXP_NAME = os.getenv("EXPERIMENT_NAME", "visual_flocking")
    EXP_NAME_COPY = f"{EXP_NAME}_copy"
    os.path.dirname(os.path.realpath(__file__))
    root_abm_dir = Path(__file__).parent.parent
    env_file_dir = root_abm_dir / "abm" / "projects" / "visual_flocking"  # Path(__file__).parent
    env_path = env_file_dir / f"{EXP_NAME}.env"
    env_path_copy = env_file_dir / f"{EXP_NAME_COPY}.env"
    # make a duplicate of the env file to be used by the playground
    shutil.copyfile(env_path, env_path_copy)
    envconf = dotenv_values(env_path)
    return env_file_dir, EXP_NAME_COPY, envconf


def start_playground():
    """starting simulation with interactive interface"""
    env_file_dir, EXP_NAME_COPY, envconf = setup_environment()
    # changing env file according to playground default parameters before
    # running any component of the SW
    pgt = setup_visflock_playground()
    save_isims_env(env_file_dir, EXP_NAME_COPY, pgt, envconf)
    # Start interactive simulation
    sim = VFPlaygroundSimulation()
    sim.start()


def start(parallel=False, headless=False, agent_behave_param_list=None):
    """starting simulation without interactive interface"""
    # Define root abm directory from which env file is read out
    root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Finding env file
    EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
    env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
    if os.path.isfile(env_path):
        print(f"Read env vars from {env_path}")
        envconf = dotenv_values(env_path)
        app_version = envconf.get("APP_VERSION", "Base")
        if app_version != "VisualFlocking":
            raise Exception(".env file was not created for project visual "
                            "flocking or no APP_VERSION parameter found!")
    else:
        raise Exception(f"Could not find .env file under path {env_path}")

    # Initializing virtual display
    window_pad = 30
    vscreen_width = int(float(envconf["ENV_WIDTH"])) + 2 * window_pad + 10
    vscreen_height = int(float(envconf["ENV_HEIGHT"])) + 2 * window_pad + 10

    # Running headless simulation on virtual display
    if headless:
        # required to start pygame in headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        from xvfbwrapper import Xvfb

    with ExitStack() if not headless else Xvfb(width=vscreen_width, height=vscreen_height) as xvfb:
        sim = VFSimulation(N=int(float(envconf["N"])),
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
