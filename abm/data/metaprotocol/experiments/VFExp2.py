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
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations.
The filename shall not include underscore characters when running on HPC. This description will be included
as a txt file in the generated experiment data folder.

Title:      Experiment : {EXP_NAME}
Date:       27.02.2023
Parameters: 
        Comparing different conditions regarding FOV, alpha0 vs. beta0 with Wall reflections 
                
Project Maintainers (VisualFlockig): mezdahun  
"""

# Defining fixed criteria for all automized simulations/experiments
arena_w = 900
arena_h = 900
fixed_criteria = [
    Constant("ENV_WIDTH", arena_w),
    Constant("ENV_HEIGHT", arena_h),
    Constant("USE_IFDB_LOGGING", 1),
    Constant("USE_RAM_LOGGING", 1),  # as we have plenty of resources we don't have to deal with IFDB on HPC
    Constant("USE_ZARR_FORMAT", 1),
    Constant("SAVE_CSV_FILES", 1),
    Constant("WITH_VISUALIZATION", 0),
    Constant("SHOW_VISUAL_FIELDS", 0),
    Constant("SHOW_VISUAL_FIELDS_RETURN", 0),
    Constant("SHOW_VISION_RANGE", 0),
    Constant("TELEPORT_TO_MIDDLE", 0),
    Constant("PATCHWISE_SOCIAL_EXCLUSION", 1),
    Constant("POOLING_TIME", 0),
    Constant("VISUAL_FIELD_RESOLUTION", 320),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("MAX_RESOURCE_QUALITY", -1),  # so that the minimum value will be used as definite
    Constant("MAX_RESOURCE_PER_PATCH", -1),  # so that the minimum value will be used as definite
    Constant("MOV_EXP_VEL_MIN", 3),
    Constant("MOV_EXP_VEL_MAX", 3),
    Constant("MOV_REL_DES_VEL", 3),
    Constant("MOV_EXP_TH_MIN", -0.5),
    Constant("MOV_EXP_TH_MAX", 0.5),
    Constant("MOV_REL_TH_MAX", 1.8),
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
criteria_exp = [
    Constant("APP_VERSION", "VisualFlocking"),  # Enabling project specific run in headless mode
    # Constant("INIT_FRAMERATE", 100),
    Constant("RADIUS_AGENT", 5.5),
    Constant("VF_GAM", 0.1),
    Constant("VF_V0", 1),
    Constant("VF_ALP1", 0.0014),
    Constant("VF_ALP2", 0),
    Constant("VF_BET1", 0.0014),
    Constant("VF_BET2", 0),
    Tunable("AGENT_FOV", values_override=[0.25, 0.5, 0.75, 1]),  # unlimited FOV
    Constant("BOUNDARY", "walls"),
    Constant("T", 25000),
    Constant("N", 10),  # Can not be tuned, must be fixed for Replay tool
    Tunable("VF_ALP0", values_override=[0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 3]),
    Tunable("VF_BET0", values_override=[0, 0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 3])
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=1, parallel=True,
                  description=description_text, headless=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations with project specific application
mp.run_protocols(project="VisualFlocking")
