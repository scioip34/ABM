# script for doing runs with multiple simulations/parameters
# run via `python path_to_exp_file.py`

from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np
import os
EXP_NAME = os.getenv("EXPERIMENT_NAME", "patch_place_distr")
if EXP_NAME == "":
    raise Exception("No experiment name has been passed")

description_text = f"""
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment : {EXP_NAME}
Date:       28.04.2022
Goal:       When placing the recources in the arena we want to get a uniform
            distribution of their locations. In this experiment, we want to
            compare two different cases. In the first case, the resource
            patches are allowed to cross the borders of the arena. In the
            second case they are not. We want to check how the distribution of
            locations changes when inhibiting overlap.
            We use different patch radii.

Defined by: nb
"""
# parameters can kept mostly untouched
# Defining fixed criteria for all automized simulations/experiments
arena_w = 500
arena_h = 500
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
    Constant("ENV_WIDTH", arena_w),
    Constant("ENV_HEIGHT", arena_h),
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("VISION_RANGE", 2000),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("RADIUS_AGENT", 10),
    Constant("MAX_RESOURCE_QUALITY", -1),  # so that the minimum value will be used as definite
    Constant("MAX_RESOURCE_PER_PATCH", -1),  # so that the minimum value will be used as definite
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

# # quick check that restricted_overall_res_area is not larger than 20% of restricted_arena_size
# radius = 30
# restricted_arena_w = arena_w - 2 * radius
# restricted_arena_h = arena_h - 2 * radius
# restricted_arena_size = restricted_arena_w * restricted_arena_h
# restricted_overall_res_area = int(restricted_arena_size * 0.2)
# restricted_overall_res_area
# nup = restricted_overall_res_area/(np.pi*radius*radius)
# nup

# Defining decision param
sum_resources = 3000
arena_size = arena_w * arena_h
# keeping the covered area on 20% on overall area
overall_res_area = int(arena_size * 0.2)
num_patches = 10
different_radii = [10, 20, 30]
criteria_exp = [
    # arbitrarily chosen parameters not making a difference for the experiment
    Constant("N", 3),
    Constant("VISUAL_EXCLUSION", 0),
    Constant("AGENT_FOV", 1),  # unlimited
    Constant("DEC_EPSW", 0.25), # social excitability
    Constant("DEC_EPSU", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),
    Constant("MIN_RESOURCE_PER_PATCH", 30),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    # relevant parameters
    Tunable("RADIUS_RESOURCE", values_override=different_radii),
    # Tunable("RADIUS_RESOURCE", values_override=[np.sqrt(overall_res_area/(np.pi*radius)) for radius in different_radii]),
    Constant("N_RESOURCES", num_patches),
    Constant("T", 1)
]

# restrain pair of tuned parameters
# one can also change number of batches num_batches

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=10, parallel=True,
                  description=description_text, headless=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# # Locking the overall resource units in environment
# constant_runits = TunedPairRestrain("N_RESOURCES", "MIN_RESOURCE_PER_PATCH", sum_resources)
# mp.add_tuned_pair(constant_runits)
#
# # keeping the covered area on 20% on overall area
# constant_r_area = TunedPairRestrain("N_RESOURCES", "RADIUS_RESOURCE", overall_res_area/np.pi)
# mp.add_quadratic_tuned_pair(constant_r_area)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()
