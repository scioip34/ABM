import os
import datetime
from dotenv import dotenv_values

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

INFLUX_HOST = "127.0.0.1"
INFLUX_PORT = "8086"
INFLUX_USER = "monitoring"
INFLUX_PSWD = "password"
INFLUX_DB_NAME = "home"
write_batch_size = 1000

# SAVE_DIR is counted from the ABM parent directory.
SAVE_DIR = envconf.get("SAVE_ROOT_DIR", "abm/data/simulation_data")

# create base folder in data
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TIMESTAMP_SAVE_DIR = os.path.join(root_abm_dir, SAVE_DIR, save_timestamp)