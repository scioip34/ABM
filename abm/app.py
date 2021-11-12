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
                     pooling_prob=float(envconf["POOLING_PROBABILITY"]))
    sim.start()
