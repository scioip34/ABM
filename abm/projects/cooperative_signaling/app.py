import os
import shutil
from pathlib import Path

from dotenv import dotenv_values

import abm.contrib.playgroundtool as pgt
from abm.app import save_isims_env
from abm.simulation.isims import PlaygroundSimulation


def setup_environment():
    EXP_NAME = os.getenv("EXPERIMENT_NAME", "cooperative_signaling")
    EXP_NAME_COPY = f"{EXP_NAME}_copy"
    os.path.dirname(os.path.realpath(__file__))
    # root_abm_dir = Path(__file__).parent.parent.parent.parent
    env_file_dir = Path(__file__).parent
    env_path = env_file_dir / f"{EXP_NAME}.env"
    env_path_copy = env_file_dir / f"{EXP_NAME_COPY}.env"
    # make a duplicate of the env file to be used by the playground
    shutil.copyfile(env_path, env_path_copy)
    envconf = dotenv_values(env_path)
    return env_file_dir, EXP_NAME_COPY, envconf


def start_playground():
    env_file_dir, EXP_NAME_COPY, envconf = setup_environment()
    # changing env file according to playground default parameters before
    # running any component of the SW
    save_isims_env(env_file_dir, EXP_NAME_COPY, pgt, envconf)
    # Start interactive simulation
    sim = PlaygroundSimulation()
    sim.start()


if __name__ == "__main__":
    start_playground()
