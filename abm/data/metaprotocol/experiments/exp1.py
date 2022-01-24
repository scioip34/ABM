"""
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment 1
Goal:       Understand the balance of social vs individual excitablility in fixed environment
Defined by: mezdahun and DominikDeffner @ github
"""
from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol

# Defining fixed criteria for all automized simulations/experiments
fixed_criteria = [
    Constant("USE_IFDB_LOGGING", 1),
    Constant("SAVE_CSV_FILES", 1),
    Constant("WITH_VISUALIZATION", 0),  # how does the simulation speed scale with N
    Constant("TELEPORT_TO_MIDDLE", 0),
    Constant("GHOST_WHILE_EXPLOIT", 1),
    Constant("PATCHWISE_SOCIAL_EXCLUSION", 1),
    Constant("POOLING_TIME", 0),
    Constant("MOV_EXP_VEL_MIN", 1),
    Constant("MOV_EXP_VEL_MAX", 1),
    Constant("MOV_REL_DES_VEL", 1),
    Constant("SHOW_VISUAL_FIELDS", 0),
    Constant("SHOW_VISUAL_FIELDS_RETURN", 0),
    Constant("SHOW_VISION_RANGE", 0),
    Constant("ENV_WIDTH", 600),
    Constant("ENV_HEIGHT", 600),
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("VISUAL_EXCLUSION", 1),
    Constant("VISION_RANGE", 1000),
    Constant("AGENT_FOV", 0.5),  # what is in the VR experiment and how we can control this?
    Constant("AGENT_CONSUMPTION", 1),
    Constant("RADIUS_AGENT", 10),
    Constant("RADIUS_RESOURCE", 40),
    Constant("MIN_RESOURCE_QUALITY", 0.1),
    Constant("MAX_RESOURCE_QUALITY", 0.1),
    Constant("MIN_RESOURCE_PER_PATCH", 100),
    Constant("MAX_RESOURCE_PER_PATCH", 101),
    Constant("MOV_EXP_TH_MIN", -0.25),
    Constant("MOV_EXP_TH_MAX", 0.25),
    Constant("MOV_REL_TH_MAX", 0.5),
    Constant("CONS_STOP_RATIO", 0.1),
    Constant("REGENERATE_PATCHES", 1),
    Constant("REGENERATE_PATCHES", 5000),
    Constant("DEC_FN", 0.5),
    Constant("DEC_FR", 0.5),
    Constant("DEC_TAU", 10),
    Constant("DEC_BW", 0),
    Constant("DEC_WMAX", 0),
    Constant("DEC_BU", 0),
    Constant("DEC_UMAX", 1),
    Constant("DEC_GW", 0.085),
    Constant("DEC_GU", 0.085),
    Constant("DEC_TW", 0.5),
    Constant("DEC_TU", 0.5),
    Constant("T", 5000),
]

# Defining decision param
criteria_exp1 = [
    Constant("N", 1),
    Tunable("DEC_EPSW", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Tunable("DEC_EPSU", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("N_RESOURCES", 5),
]

criteria_exp2 = [
    Constant("N", 3),
    Tunable("DEC_EPSW", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Tunable("DEC_EPSU", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("N_RESOURCES", 5),
]

criteria_exp3 = [
    Constant("N", 5),
    Tunable("DEC_EPSW", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Tunable("DEC_EPSU", values_override=[0, 0.01, 0.1, 0.5, 1, 2, 5]),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("N_RESOURCES", 5),
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol()
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp2:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()