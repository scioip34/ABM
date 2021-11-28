from abm.simulation.sims import Simulation

# loading env variables from dotenv file
from dotenv import dotenv_values
envconf = dotenv_values(".env")

def start():
    sim = Simulation(N=int(envconf["N"]),
                     T=int(envconf["T"]),
                     v_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
                     width=int(envconf["ENV_WIDTH"]),
                     height=int(envconf["ENV_HEIGHT"]),
                     show_vis_field=bool(int(envconf["SHOW_VISUAL_FIELDS"])),
                     pooling_time=int(envconf["POOLING_TIME"]),
                     pooling_prob=float(envconf["POOLING_PROBABILITY"]),
                     agent_radius=int(envconf["RADIUS_AGENT"]),
                     N_resc=int(envconf["N_RESOURCES"]),
                     min_resc_perpatch=int(envconf["MIN_RESOURCE_PER_PATCH"]),
                     max_resc_perpatch=int(envconf["MAX_RESOURCE_PER_PATCH"]),
                     patch_radius=int(envconf["RADIUS_RESOURCE"]),
                     regenerate_patches=bool(int(envconf["REGENERATE_PATCHES"])),
                     agent_consumption=int(envconf["AGENT_CONSUMPTION"]),
                     vision_range=int(envconf["VISION_RANGE"]),
                     visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"]))
                     )
    sim.start()
