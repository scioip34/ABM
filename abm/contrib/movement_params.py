"""parameters for individual exploration, relocation and exploitation movements"""

from dotenv import dotenv_values

envconf = dotenv_values(".env")

# Exploration movement parameters
exp_vel_min = float(envconf.get("MOV_EXP_VEL_MIN", 1))
exp_vel_max = float(envconf.get("MOV_EXP_VEL_MAX", 1))
exp_theta_min = float(envconf.get("MOV_EXP_TH_MIN", -0.3))
exp_theta_max = float(envconf.get("MOV_EXP_TH_MAX", 0.3))

# Relocation movement parameters
reloc_des_vel = float(envconf.get("MOV_REL_DES_VEL", 1))
reloc_theta_max = float(envconf.get("MOV_REL_TH_MAX", 0.5))