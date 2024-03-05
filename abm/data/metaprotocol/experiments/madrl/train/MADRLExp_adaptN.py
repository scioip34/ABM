from abm.metarunner.metarunner import Tunable, Constant, MetaProtocol, TunedPairRestrain
import numpy as np
import os

EXP_NAME = os.getenv("EXPERIMENT_NAME", "")
if EXP_NAME == "":
   raise Exception("No experiment name has been passed")

description_text = f"""
Experiment file using the MetaRunner interfacing language to define a set of criteria for batch simulations

Title:      Experiment : {EXP_NAME}

Experiment 1: Impact of Reward Function

Each time an agent consumes a resource unit, it receives a reward. The reward function is proportional to his contribution to either
individual or collective search efficiency. The reward function is defined as follows:
    - Individual Search Efficiency (ISE) is the ratio of the resource units found by the agent to the total resource units found by all agents.
    - Collective Search Efficiency (CSE) is the ratio of the resource units found by the agent to the total resource units found by all agents.
    
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
Parameters:
    Fixed Parameters:
        Time Penalty = False
        BINARY_ENV_STATUS: False
    Tunable Parameters:
        1. ISE_W = 0.8, CSE_W = 0.2
        2. ISE_W = 0.0, CSE_W = 1.0
        3. ISE_W = 0.5, CSE_W = 0.5
        4. ISE_W = 1.0, CSE_W = 0.0

-	Description: Examining the impact of different reward functions:
    1. Fully individualistic (4) 
    2. Hybrid (individual and collective search efficiencies) (3 and 1)
    3. Fully cooperative (2)

-	Hypothesis:  Optimal agent performance is expected when balancing individual and collective search efficiencies.

Defined by: Feriel Amira
"""
sum_resources = 2400
num_patches = [100]
print("Running 1 experiment")
criteria_exp = [
    Constant("APP_VERSION", "MADRLForaging"),
    Constant("ENV_WIDTH", 500),
    Constant("ENV_HEIGHT", 500),
    Constant("WINDOW_PAD", 30),
    Constant("N", 3),
    Constant("T", 20000),
    Constant("SEED", 0),
    Constant("TRAIN", 1),
    Constant("TRAIN_EVERY", 1),
    Constant("PRETRAINED", 0),
    Constant("BATCH_SIZE", 128),
    Constant("REPLAY_MEMORY_CAPACITY", 20000),
    Constant("GAMMA", 0.99),
    Constant("LR", 1e-05),
    Constant("EPSILON_START", 1.0),
    Constant("EPSILON_END", 0.01),
    Constant("EPSILON_DECAY", 50000),
    Constant("TAU", 0.01),
    Constant("OPTIMIZER", "RSMprop"),
    Constant("PRETRAINED_MODELS_DIR", ""),
    Constant("BRAIN_TYPE", "DQN"),
    Constant("ISE_W", 1.0),
    Constant("CSE_W", 0.0),
    Constant("TP", 0),
    Constant("BINARY_ENV_STATUS", 0),
    Constant("WITH_VISUALIZATION", 0),
    Constant("INIT_FRAMERATE", 100),
    Constant("SHOW_VISUAL_FIELDS", 0),
    Constant("SHOW_VISUAL_FIELDS_RETURN", 0),
    Constant("SHOW_VISION_RANGE", 0),
    Constant("USE_IFDB_LOGGING", 0),
    Constant("USE_RAM_LOGGING", 0),
    Constant("USE_ZARR_FORMAT", 0),
    Constant("SAVE_CSV_FILES", 0),
    Constant("RADIUS_AGENT", 10),
    Constant("VISUAL_FIELD_RESOLUTION", 1200),
    Constant("AGENT_FOV", 1),
    Constant("VISION_RANGE", 2000),
    Constant("AGENT_CONSUMPTION", 1),
    Constant("PATCHWISE_SOCIAL_EXCLUSION", 1),
    Constant("VISUAL_EXCLUSION", 0),
    Constant("TELEPORT_TO_MIDDLE", 0),
    Constant("AGENT_AGENT_COLLISION", 0),
    Constant("GHOST_WHILE_EXPLOIT", 1),
    Constant("POOLING_TIME", 0),
    Constant("POOLING_PROBABILITY", 0),
    Constant("RADIUS_RESOURCE", 15),
    Constant("REGENERATE_PATCHES", 1),
    Constant("PATCH_BORDER_OVERLAP", 1),
    Constant("MAX_RESOURCE_QUALITY", 0.25),
    Constant("MOV_EXP_VEL_MIN", 3.0),
    Constant("MOV_EXP_VEL_MAX", 3.0),
    Constant("MOV_REL_DES_VEL", 3.0),
    Constant("MOV_EXP_TH_MIN", -0.5),
    Constant("MOV_EXP_TH_MAX", 0.5),
    Constant("MOV_REL_TH_MAX", 2.0),
    Constant("CONS_STOP_RATIO", 0.175),
    Constant("DEC_TW", 0.5),
    Constant("DEC_EPSW", 2),
    Constant("DEC_GW", 0.085),
    Constant("DEC_BW", 0),
    Constant("DEC_WMAX", 1),
    Constant("DEC_TU", 0.5),
    Constant("DEC_EPSU", 1),
    Constant("DEC_GU", 0.085),
    Constant("DEC_BU", 0),
    Constant("DEC_UMAX", 1),
    Constant("DEC_SWU", 0),
    Constant("DEC_SUW", 0),
    Constant("DEC_TAU", 10),
    Constant("DEC_FN", 1),
    Constant("DEC_FR", 1),


    Constant("MIN_RESOURCE_QUALITY", 0.25),  # we fixed the max quality to negative so can control the value with MIN
    Tunable("MIN_RESOURCE_PER_PATCH", values_override=[int(sum_resources / nup) for nup in num_patches]),  # same here
    Tunable("N_RESOURCES", values_override=num_patches),
    Constant("MAX_RESOURCE_PER_PATCH", -1),  # so that the minimum value will be used as definite

]


# Creating metaprotocol and add defined criteria
mp = MetaProtocol(experiment_name=EXP_NAME, num_batches=1, parallel=True,
                  description=description_text, headless=False)

for crit in criteria_exp:
    mp.add_criterion(crit)

# Locking the overall resource units in environment
constant_runits = TunedPairRestrain("N_RESOURCES", "MIN_RESOURCE_PER_PATCH", sum_resources)
mp.add_tuned_pair(constant_runits)

# Generating temporary env files with criterion combinations. Comment this out if you want to continue simulating due
# to interruption
mp.generate_temp_env_files()


# Running the simulations
mp.run_protocols(project="MADRLForaging")
