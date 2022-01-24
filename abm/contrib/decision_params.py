# loading env variables from dotenv file
from dotenv import dotenv_values
import os

root_abm_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
env_path = os.path.join(root_abm_dir, ".env")
envconf = dotenv_values(env_path)

#### W #####
# Excitatory threshold above which a relocation is initiated
T_w = float(envconf.get("DEC_TW", 0.5))
# Social excitability
Eps_w = float(envconf.get("DEC_EPSW", 3))
# w decay time constant
g_w = float(envconf.get("DEC_GW", 0.085))
# Baseline of decision process
B_w = float(envconf.get("DEC_BW", 0))
# max value for w
w_max = float(envconf.get("DEC_WMAX", 1))

#### U #####
# Refractory threshold above which u resets decision w
T_u = float(envconf.get("DEC_TU", 0.5))
# Sensitivity of u to nearby agents
Eps_u = float(envconf.get("DEC_EPSU", 3))
# Timeconstant of u decay
g_u = float(envconf.get("DEC_GU", 0.085))
# Baseline of refractory variable u
B_u = float(envconf.get("DEC_BU", 0))
# max value for u
u_max = float(envconf.get("DEC_UMAX", 1))

##### Inhibition ####
S_wu = float(envconf.get("DEC_SWU", 0.25))  # strength from w to u
S_uw = float(envconf.get("DEC_SUW", 0.01))  # strength from u to w

##### Calculating Private Information #####
Tau = int(float(envconf.get("DEC_TAU", 10)))
F_N = float(envconf.get("DEC_FN", 2))  # novelty multiplier
F_R = float(envconf.get("DEC_FR", 1))  # quality multiplier
