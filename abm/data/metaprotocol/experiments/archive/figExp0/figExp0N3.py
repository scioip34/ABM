from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np
import os
EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
if EXP_NAME == "":
    raise Exception("No experiment name has been passed")

description_text = f"""
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment : {EXP_NAME}
Date:       08.09.2022
Parameters: figExp0N3: 
                Base experiment with fully idealistic agents to show the effect of changing environment and
                social excitability on collective efficiency. N=3 agents case.

                After the 17 preliminary experiments we decided on the following experimental setting:

                ----Patch Sizes and Resources----
                - We use unscaled patch sizes so that the covered area by patches (and by that the search difficulty)
                varies across conditions. We chose this setup as for multiple agents there is a stronger effect
                of social information use, and due to similarity with human behavioral experiments. We argued that
                a normalized patch size (i.e. keeping covered area by patches on 20% of the overall arena) might make
                sense for smaller number of agents (especially for edge cases with 2-3 agents) to increase the chance of
                individual discovery (keeping search difficulty constant). We further considered interpretational points,
                i.e. how we can interpret patches that are the same size but richer or less rich.
                - We fix the total amount of resource units distributed in patches to make batches comparable with each 
                other.
                - The amount of resource units has been heuristically determined so that the depletion timescale
                (even with small groups of 5 agents) can be compared with the simulation time, i.e. we can observe
                "enough" depletion events to analyze the effect of social excitability. If patches are too rich, agents
                get "trapped" in them to the whole simulation and further analysis will be biased.
                - The number of patches has been fixed to a golden standard to 1, 3, 5, 8, 10, 20, 30, 50, 100
                
                ----Number of Agents----
                - We fixed the golden standard for the whole article as the set of group sizes as follows:
                - 3, 4, 5, 7, 10, 25, 50, 75, 100
                - lower number of agents might be impractical as we need longer simulation sizes and effect of social
                information might be less relevant especially in larger arenas.
                - larger group sizes are currently out of scope for our study and might introduce physical cluttering of
                the environment.
                
                ----Social Excitability----
                - We programmatically change the social excitability parameter across batches to shed light on the effect
                of varying strength of social information use in different environments (w.r.t resource distribution)
                - We are aware of the unnormed nature of this parameter and that it depends on the resolution of the 
                visual field of the agents as well as the physical size of the agents. (i.e. on all parameters that might
                change the projection of socially relevant information on the retina of agents). Therefore absolute values
                of social excitability parameter will not be comparable when changing the above mentioned parameters
                
                ----Visual Occlusion and FOV----
                - We study the effect of physical constraints on the system starting with visual occlusions and FOV. 
                - We study the difference between the system with (and without) visual occlusion
                
                ----Ghost Mode----
                - If "Ghost Mode" is turned on, agents can overlap during exploitation
                - We plan to show the effect of such idealization that is often made in other computational models on 
                the system
                
                ----Simulation time and number of batches----
                - We have run multiple preliminary experiments to see the desired number of simulation timesteps to
                show reasonable effects in our stochastic system
                - We try to minimize the amount of simulation needed for our study to reduce the carbon footprint
                of our research
                - For small group sizes a larger amount of data might be necessary due to stochastic effects, 
                i.e. no or few discovery events during some of the simulations
                - Movement speed is also optimized to decrease simulation steps          
                
Defined by: David Mezey
"""

# Defining fixed criteria for all automized simulations/experiments
arena_w = 500
arena_h = 500
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
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("RADIUS_AGENT", 10),
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
    Constant("PATCH_BORDER_OVERLAP", 1)
]

# Defining decision param
sum_resources = 2400
arena_size = arena_w * arena_h
# keeping the covered area on 20% on overall area
keep_covered_ratio = 0.2
overall_res_area = int(arena_size * keep_covered_ratio)
num_patches = [1, 3, 5, 8, 10, 20, 30, 50, 100]
criteria_exp = [
    Constant("N", 3),
    Constant("VISUAL_EXCLUSION", 0),  # no visual occlusion
    Constant("VISION_RANGE", 2000),  # unlimited visual range
    Constant("AGENT_FOV", 1),  # unlimited FOV
    Constant("GHOST_WHILE_EXPLOIT", 1),
    Tunable("DEC_EPSW", values_override=[0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]),
    Constant("DEC_EPSU", 1),
    Constant("MIN_RESOURCE_QUALITY", 0.25),  # we fixed the max quality to negative so can control the value with MIN
    Tunable("MIN_RESOURCE_PER_PATCH", values_override=[int(sum_resources/nup) for nup in num_patches]),  # same here
    Constant("RADIUS_RESOURCE", 15),
    Constant("DEC_SWU", 0),  # no cross-inhibition
    Constant("DEC_SUW", 0),  # no cross-inhibition
    Tunable("N_RESOURCES", values_override=num_patches),
    Constant("T", 25000)
]

# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=1, parallel=True,
                  description=description_text, headless=True)
for crit in fixed_criteria:
    mp.add_criterion(crit)
for crit in criteria_exp:
    mp.add_criterion(crit)

# Locking the overall resource units in environment
constant_runits = TunedPairRestrain("N_RESOURCES", "MIN_RESOURCE_PER_PATCH", sum_resources)
mp.add_tuned_pair(constant_runits)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()

# Running the simulations
mp.run_protocols()
