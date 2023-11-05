import os
import datetime
from dotenv import dotenv_values

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
WRITE_EACH_POINT = os.getenv("WRITE_EACH_POINT")

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

INFLUX_HOST = "localhost"
INFLUX_PORT = "8086"
INFLUX_USER = "monitoring"
INFLUX_PSWD = "password"
INFLUX_DB_NAME = "home"
INFLUX_TIMEOUT = 30  # timeout for requests to wait for client in seconds
INFLUX_RETRIES = 3  # number of retries before fail when timeout reached

print("Using IFDB long timeout and retries!")

if WRITE_EACH_POINT is not None:
    write_batch_size = 1
else:
    T = float(int(envconf.get("T", 1000)))
    if T <= 1000:
        write_batch_size = T
    else:
        if T % 1000 != 0:
            raise Exception("Simulation time (T) must be dividable by 1000 or smaller than 1000!")
        write_batch_size = 1000

# SAVE_DIR is counted from the ABM parent directory.
SAVE_DIR = envconf.get("SAVE_ROOT_DIR", "abm/data/simulation_data")

# create base folder in data
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TIMESTAMP_SAVE_DIR = os.path.join(root_abm_dir, SAVE_DIR, save_timestamp)