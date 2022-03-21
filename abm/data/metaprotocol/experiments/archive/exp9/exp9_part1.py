description_text = """
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment 9 (part1)
Date:       18.03.2022
Goal:       In this experiment we restrict the total amount of resources as before, but when we change the number
            of resource patches we restrict the size of the patches together. This way the covered resource area 
            will remain the same (20%, or similar) over the experiment that was growing with the number of
            patches in Experiment 5-7. Visual exclusion is now off. Number of patches are changed in
            more extreme ways than in #5-6 as now the arena does not get saturated with patches. The goal is to
            compare results from #8 (with visual cclusion) with this. Partitioned to run parallely on HPC and merge
            later (part1).
Defined by: mezdahun
"""
from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np

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

# Defining decision param
sum_resources = 3000
arena_size = arena_w * arena_h
# keeping the covered area on 20% on overall area
overall_res_area = int(arena_size * 0.2)
num_patches = [1, 3, 5, 10, 30, 50, 100]
criteria_exp = [
    Constant("N", 10),
    Constant("VISUAL_EXCLUSION", 0),
    Constant("AGENT_FOV", 1),  # unlimited
    Tunable("DEC_EPSW", values_override=[0, 0.5, 0.75, 1, 2, 3]),
    Constant("DEC_EPSU", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),
    Tunable("MIN_RESOURCE_PER_PATCH", values_override=[int(sum_resources/nup) for nup in num_patches]),
    Tunable("RADIUS_RESOURCE", values_override=[np.sqrt(overall_res_area/(np.pi*nup)) for nup in num_patches]),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Tunable("N_RESOURCES", values_override=num_patches),
    Constant("T", 15000)
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name="Experiment9_part1", num_batches=2, parallel=False,
                  description=description_text, headless=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Locking the overall resource units in environment
constant_runits = TunedPairRestrain("N_RESOURCES", "MIN_RESOURCE_PER_PATCH", sum_resources)
mp.add_tuned_pair(constant_runits)

# keeping the covered area on 20% on overall area
constant_r_area = TunedPairRestrain("N_RESOURCES", "RADIUS_RESOURCE", overall_res_area/np.pi)
mp.add_quadratic_tuned_pair(constant_r_area)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()