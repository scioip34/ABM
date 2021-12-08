### Sectioning visual field
D_near_proc = 0.1

#### W #####
# Excitatory threshold above which a relocation is initiated
T_w = 1
# Social excitability
Eps_w = 2
# w decay time constant
g_w = 0.085
# Baseline of decision process
B_w = 0
# max value for w
w_max = 2

#### U #####
# Refractory threshold above which u resets decision w
T_u = 1
# Sensitivity of u to nearby agents
Eps_u = 1
# Timeconstant of u decay
g_u = 0.085
# Baseline of refractory variable u
B_u = 0
# max value for u
u_max = 2

##### Inhibition ####
S_wu = 0.02  # strength from w to u
S_uw = 0.2  # strength from u to w
