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
    Constant("AGENT_CONSUMPTION", 1),
    Constant("RADIUS_AGENT", 10),
    Constant("RADIUS_RESOURCE", 40),
    Constant("MAX_RESOURCE_QUALITY", -10),
    Constant("MOV_EXP_TH_MIN", -0.25),
    Constant("MOV_EXP_TH_MAX", 0.25),
    Constant("MOV_REL_TH_MAX", 0.5),
    Constant("CONS_STOP_RATIO", 0.1),
    Constant("REGENERATE_PATCHES", 1),
    Constant("DEC_FN", 0.5),
    Constant("DEC_FR", 0.5),
    Constant("DEC_TAU", 10),
    Constant("DEC_BW", 0),
    Constant("DEC_WMAX", 1),
    Constant("DEC_BU", 0),
    Constant("DEC_UMAX", 1),
    Constant("DEC_GW", 0.085),
    Constant("DEC_GU", 0.085),
    Constant("DEC_TW", 0.5),
    Constant("DEC_TU", 0.5)
]

# Defining decision param
criteria_exp = [
    Tunable("N", values_override=[3, 5, 7, 10]),
    Constant("AGENT_FOV", 0.45),  # limited
    Tunable("DEC_EPSW", values_override=[0, 0.75, 1, 1.5, 2, 3]),
    Constant("DEC_EPSU", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),
    Constant("MIN_RESOURCE_PER_PATCH", 250),
    Constant("MAX_RESOURCE_PER_PATCH", 251),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("N_RESOURCES", 7),
    Constant("T", 10000)
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name="Experiment3_limitedFOV", num_batches=2, parallel=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()

