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
ALP0 = float(envconf.get("VF_ALP0", 1))
ALP1 = float(envconf.get("VF_ALP1", 0.09))
ALP2 = float(envconf.get("VF_ALP2", 0))
BET0 = float(envconf.get("VF_BET0", 1))
BET1 = float(envconf.get("VF_BET1", 0.09))
BET2 = float(envconf.get("VF_BET2", 0))

# Boundary conditions, either walls or infinite
BOUNDARY = envconf.get("BOUNDARY", "walls")

# Limiting movement of agents to maximum velocity and turning rate
LIMIT_MOVEMENT = bool(float(envconf.get("VF_LIMIT_MOVEMENT", "0")))
MAX_VEL = float(envconf.get("VF_MAX_VEL", "3"))
MAX_TH = float(envconf.get("VF_MAX_TH", "0.1"))

# Using sigmnoid masks instead of cos and sin
USE_SIN_SIGMOID = bool(float(envconf.get("VF_USE_SIN_SIGMOID", "0")))
USE_COS_SIGMOID = bool(float(envconf.get("VF_USE_COS_SIGMOID", "0")))
