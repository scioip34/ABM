from abm.simulation.sims import Simulation

# loading env variables from dotenv file
from dotenv import dotenv_values
import os
root_abm_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
envconf = dotenv_values(os.path.join(root_abm_dir, ".env"))

def start():
    sim = Simulation(N=int(float(envconf["N"])),
                     T=int(float(envconf["T"])),
                     v_field_res=int(envconf["VISUAL_FIELD_RESOLUTION"]),
                     agent_fov=float(envconf['AGENT_FOV']),
                     framerate=int(float(envconf["INIT_FRAMERATE"])),
                     with_visualization=bool(int(float(envconf["WITH_VISUALIZATION"]))),
                     width=int(float(envconf["ENV_WIDTH"])),
                     height=int(float(envconf["ENV_HEIGHT"])),
                     show_vis_field=bool(int(float(envconf["SHOW_VISUAL_FIELDS"]))),
                     pooling_time=int(float(envconf["POOLING_TIME"])),
                     pooling_prob=float(envconf["POOLING_PROBABILITY"]),
                     agent_radius=int(float(envconf["RADIUS_AGENT"])),
                     N_resc=int(float(envconf["N_RESOURCES"])),
                     min_resc_perpatch=int(float(envconf["MIN_RESOURCE_PER_PATCH"])),
                     max_resc_perpatch=int(float(envconf["MAX_RESOURCE_PER_PATCH"])),
                     min_resc_quality=float(envconf["MIN_RESOURCE_QUALITY"]),
                     max_resc_quality=float(envconf["MAX_RESOURCE_QUALITY"]),
                     patch_radius=int(float(envconf["RADIUS_RESOURCE"])),
                     regenerate_patches=bool(int(float(envconf["REGENERATE_PATCHES"]))),
                     agent_consumption=int(float(envconf["AGENT_CONSUMPTION"])),
                     ghost_mode=bool(int(float(envconf["GHOST_WHILE_EXPLOIT"]))),
                     patchwise_exclusion=bool(int(float(envconf["PATCHWISE_SOCIAL_EXCLUSION"]))),
                     teleport_exploit=bool(int(float(envconf["TELEPORT_TO_MIDDLE"]))),
                     vision_range=int(float(envconf["VISION_RANGE"])),
                     visual_exclusion=bool(int(float(envconf["VISUAL_EXCLUSION"]))),
                     show_vision_range=bool(int(float(envconf["SHOW_VISION_RANGE"]))),
                     use_ifdb_logging=bool(int(float(envconf["USE_IFDB_LOGGING"]))),
                     save_csv_files=bool(int(float(envconf["SAVE_CSV_FILES"])))
                     )
    sim.start()
