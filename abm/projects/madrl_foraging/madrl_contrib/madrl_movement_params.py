"""parameters for individual exploration, relocation and exploitation movements"""

from dotenv import dotenv_values
import os

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, f"{EXP_NAME}.env")
envconf = dotenv_values(env_path)

# Exploration movement parameters
exp_vel_min = float(envconf.get("MOV_EXP_VEL_MIN", 3))
exp_vel_max = float(envconf.get("MOV_EXP_VEL_MAX", 3))
exp_theta_min = float(envconf.get("MOV_EXP_TH_MIN", -0.5))
exp_theta_max = float(envconf.get("MOV_EXP_TH_MAX", 0.5))

# Relocation movement parameters
reloc_des_vel = float(envconf.get("MOV_REL_DES_VEL", 1))
reloc_theta_max = float(envconf.get("MOV_REL_TH_MAX", 0.5))

# Exploitation params
# deceleration when a patch is reached
exp_stop_ratio = float(envconf.get("CONS_STOP_RATIO", 0.175))
