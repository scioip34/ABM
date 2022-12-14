# loading env variables from dotenv file
from dotenv import dotenv_values
import os

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

# Crowding parameters
max_proj_size_percentage = float(envconf.get("MAX_PROJ_SIZE_PERCENTAGE", 0.5))
crowd_density_threshold = float(envconf.get("CROWD_DENSITY_THRESHOLD", 0.45))

# Signaling parameters
memory_depth = int(envconf.get("MEMORY_DEPTH", 5))

# Max agent speed
max_speed = float(envconf.get("MAX_SPEED", 1.5))

