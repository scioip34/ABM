# loading env variables from dotenv file
from dotenv import dotenv_values
import os

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

GAM  = float(envconf.get("VF_GAMMA", 0.1))
V0   = float(envconf.get("VF_V0", 1))
ALP0 = float(envconf.get("VF_ALP0", 0.25))
ALP1 = float(envconf.get("VF_ALP1", 0.0008))
ALP2 = float(envconf.get("VF_ALP2", 0))
BET0 = float(envconf.get("VF_BET0", 5))
BET1 = float(envconf.get("VF_BET1", 0.0008))
BET2 = float(envconf.get("VF_BET2", 0))
BOUNDARY = envconf.get("BOUNDARY", "infinite")
