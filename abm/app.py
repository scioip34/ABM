from abm.simulation.sims import Simulation

# loading env variables from dotenv file
from dotenv import dotenv_values

envconf = dotenv_values(".env")


def start():
    sim = Simulation(N=int(envconf["N"]),
                     T=int(envconf["T"]),
                     v_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
                     agent_fov=float(envconf['AGENT_FOV']),
                     width=int(envconf["ENV_WIDTH"]),
                     height=int(envconf["ENV_HEIGHT"]),
                     show_vis_field=bool(int(envconf["SHOW_VISUAL_FIELDS"])),
                     pooling_time=int(envconf["POOLING_TIME"]),
                     pooling_prob=float(envconf["POOLING_PROBABILITY"]),
                     agent_radius=int(envconf["RADIUS_AGENT"]),
                     N_resc=int(envconf["N_RESOURCES"]),
                     min_resc_perpatch=int(envconf["MIN_RESOURCE_PER_PATCH"]),
                     max_resc_perpatch=int(envconf["MAX_RESOURCE_PER_PATCH"]),
                     min_resc_quality=float(envconf["MIN_RESOURCE_QUALITY"]),
                     max_resc_quality=float(envconf["MAX_RESOURCE_QUALITY"]),
                     patch_radius=int(envconf["RADIUS_RESOURCE"]),
                     regenerate_patches=bool(int(envconf["REGENERATE_PATCHES"])),
                     agent_consumption=int(envconf["AGENT_CONSUMPTION"]),
                     ghost_mode=bool(int(envconf["GHOST_WHILE_EXPLOIT"])),
                     patchwise_exclusion=bool(int(envconf["PATCHWISE_SOCIAL_EXCLUSION"])),
                     teleport_exploit=bool(int(envconf["TELEPORT_TO_MIDDLE"])),
                     vision_range=int(envconf["VISION_RANGE"]),
                     visual_exclusion=bool(int(envconf["VISUAL_EXCLUSION"])),
                     show_vision_range=bool(int(envconf["SHOW_VISION_RANGE"])),
                     use_ifdb_logging=bool(int(envconf["USE_IFDB_LOGGING"])),
                     save_csv_files=bool(int(envconf["SAVE_CSV_FILES"]))
                     )
    sim.start()
