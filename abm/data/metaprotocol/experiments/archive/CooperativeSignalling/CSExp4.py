from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np
import os

# On HPC the filename will be used as experiment name
# otherwise we will need to pass it for the script, such as
# EXPERIMENT_NAME=template python ExperimentCoopSigTemplate.py
EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
if EXP_NAME == "":
    raise Exception("No experiment name has been passed")

description_text = f"""
Title:      Experiment : {EXP_NAME}
Date:       03.02.2023
Parameters: 
        Generating dataset to replicate the upcoming human study.
        - resource speed (RES_VEL): hard (equal to the agent speed) and easy (equal to the agent speed/2)
        - signaling condition (SIGNALLING_PROB): 0 (no signaling), 0.5 (to mimic the free choice condition) and 
          1 (all agents signaling)
        - number of agents (N): 1 (individual condition) and 5 (group condition)
        - the total time of the experiment (TOTAL_TIME):  15 minutes = 15 * 60 * 2 (sample frequency) = 1800 time steps
                
Project Maintainers (CoopSignalling): mezdahun & vagechrikov  
"""

# Defining fixed criteria for all automized simulations/experiments
arena_w = 1200  # 1m=3px, 1sec=2ts
arena_h = 1200
fixed_criteria = [
    Constant("ENV_WIDTH", arena_w),
    Constant("ENV_HEIGHT", arena_h),
    Constant("USE_IFDB_LOGGING", 0),
    Constant("USE_RAM_LOGGING", 1),  # as we have plenty of resources we don't have to deal with IFDB on HPC
    Constant("USE_ZARR_FORMAT", 1),
    Constant("SAVE_CSV_FILES", 1),
    Constant("WITH_VISUALIZATION", 1),
    Constant("SHOW_VISUAL_FIELDS", 0),
    Constant("SHOW_VISUAL_FIELDS_RETURN", 0),
    Constant("SHOW_VISION_RANGE", 0),
    Constant("TELEPORT_TO_MIDDLE", 0),
    Constant("PATCHWISE_SOCIAL_EXCLUSION", 1),
    Constant("POOLING_TIME", 0),
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("MAX_RESOURCE_QUALITY", -1),  # so that the minimum value will be used as definite
    Constant("MAX_RESOURCE_PER_PATCH", -1),  # so that the minimum value will be used as definite
    Constant("MOV_EXP_VEL_MIN", 1),
    Constant("MOV_EXP_VEL_MAX", 1),
    Constant("MOV_REL_DES_VEL", 1),
    Constant("CONS_STOP_RATIO", 0.175),
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
    Constant("DEC_TU", 0.5),
    Constant("PATCH_BORDER_OVERLAP", 1),
    Constant("DEC_EPSW", 0),
    Constant("DEC_EPSU", 1),
    Constant("AGENT_AGENT_COLLISION", 0),
    Constant("MIN_RESOURCE_PER_PATCH", 100),
    Constant("N_RESOURCES", 1),
    Constant("VISUAL_EXCLUSION", 0),  # no visual occlusion
    Constant("VISION_RANGE", 2000),  # unlimited visual range
    Constant("GHOST_WHILE_EXPLOIT", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),
    Constant("RADIUS_RESOURCE", 20),
    Constant("DEC_SWU", 0),  # no cross-inhibition
    Constant("DEC_SUW", 0),  # no cross-inhibition
]

# Defining decision param
arena_size = arena_w * arena_h
agent_vel = 10  # px/ts, 1sec=2ts
agent_radius = 8  # shall be 3 but won't make a qualitative difference
# 10% of the arena similar to 40 resource radius, which is 10% of the arena diameter of 400 meters
detection_range = arena_w * 0.1
signaling_probs = [0, 0.5, 1]
n_agents = [1, 5]
resource_vels = [i*agent_vel for i in [0.5, 1]]
criteria_exp = [
    Constant("APP_VERSION", "CooperativeSignaling"),  # Enabling project specific run in headless mode
    Tunable("N", values_override=n_agents),
    Constant("RADIUS_AGENT", agent_radius),
    Constant("MAX_SPEED", agent_vel),
    # Turning off crowding
    Constant("MAX_PROJ_SIZE_PERCENTAGE", 0.05),
    Constant("CROWD_DENSITY_THRESHOLD", 1),
    # Taxis
    Constant("PHOTOTAX_THETA_FAC", 0.42),
    # Signaling
    Constant("SIGNALLING_COST", 0),
    Tunable("SIGNALLING_PROB", values_override=signaling_probs),
    Constant("SIGNAL_PROB_UPDATE_FREQ", 10),
    # Movement
    Constant("MOV_EXP_TH_MIN", -0.21),
    Constant("MOV_EXP_TH_MAX", 0.21),
    Constant("MOV_REL_TH_MAX", 0.21),
    # Memory
    Constant("MEMORY_DEPTH", 1),
    # Vision
    Constant("AGENT_FOV", 0.25),
    # Resource
    Constant("RES_THETA", 0.2),
    Tunable("RES_VEL", values_override=resource_vels),
    Constant("METER_TO_RES_MULTI", 1),
    Constant("DETECTION_RANGE", detection_range),
    Constant("T", 1800),
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=1, parallel=True,
                  description=description_text, headless=False)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations with project specific application
mp.run_protocols(project="CoopSignaling")
