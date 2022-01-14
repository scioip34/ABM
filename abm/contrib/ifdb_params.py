import os
import datetime

INFLUX_HOST = "127.0.0.1"
INFLUX_PORT = "8086"
INFLUX_USER = "monitoring"
INFLUX_PSWD = "password"
INFLUX_DB_NAME = "home"

# SAVE_DIR is counted from the ABM parent directory.
SAVE_DIR = "abm/data/simulation_data"

# create base folder in data
root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
save_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
TIMESTAMP_SAVE_DIR = os.path.join(root_abm_dir, SAVE_DIR, save_timestamp)