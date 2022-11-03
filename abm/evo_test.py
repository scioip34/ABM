from abm.metarunner.evolution import EvoProtocol
import numpy as np

num_agents = 30
# Example of preparing a gaussian distribution of velocities
exp_vel_max = np.random.normal(loc=2, scale=1, size=num_agents)
exp_vel_max = [max(0.25, vel) for vel in exp_vel_max]
# for i in range(int(num_agents/2)):
#     exp_vel_max[i] = 0
print("Velocities: ", exp_vel_max)

# Example of preparing a gaussian distribution of social excitability
# Eps_w = np.random.normal(loc=1, scale=1, size=num_agents)
# Eps_w = [max(0, Ew) for Ew in Eps_w]
Eps_w = np.random.uniform(0, 7, size=num_agents)
print("Excitability: ", Eps_w)

FOV = [0.5 for i in range(num_agents)]

initial_genes = {
    "Eps_w": Eps_w,
    # "exp_vel_max": exp_vel_max,
    # "agent_fov": FOV
}

gene_mutations = {
    "Eps_w": {
        "prob": 0.2,
        "mean": 0,
        "std": 0.25,
        "min": 0,
        "max": 7
    },
    # "exp_vel_max": {
    #     "prob": 0.6,
    #     "mean": 0,
    #     "std": 0.25,
    #     "min": 0,
    #     "max": 4
    # },
    # "agent_fov": {
    #     "prob": 0.5,
    #     "mean": 0,
    #     "std": 0.1,
    #     "min": 0,
    #     "max": 1
    # }
}

evoprot = EvoProtocol(num_generations=30, gen_lifetime=3000, headless=True,
                      death_rate_limits=(1, 1), initial_genes=initial_genes, mutation_rates=gene_mutations,
                      num_populations=10)

evoprot.set_env_var("N", num_agents)
# evoprot.set_env_var("T", 3000)
evoprot.set_env_var("N_RESOURCES", 4)
evoprot.set_env_var("RADIUS_RESOURCE", 25)
evoprot.set_env_var("INIT_FRAMERATE", 300)
evoprot.set_env_var("GHOST_WHILE_EXPLOIT", 1)
evoprot.set_env_var("MIN_RESOURCE_QUALITY", 0.25)
evoprot.set_env_var("MAX_RESOURCE_QUALITY", -1)
evoprot.set_env_var("MIN_RESOURCE_PER_PATCH", 500)
evoprot.set_env_var("MAX_RESOURCE_PER_PATCH", -1)
evoprot.set_env_var("VISUAL_EXCLUSION", 0)
evoprot.set_env_var("MOV_REL_TH_MAX", 0.8)
# Saving of data
evoprot.set_env_var("USE_RAM_LOGGING", 1)
evoprot.set_env_var("USE_ZARR_FORMAT", 1)
evoprot.set_env_var("SAVE_CSV_FILES", 1)
evoprot.start_evolution()
