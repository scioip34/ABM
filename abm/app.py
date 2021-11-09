from abm.simulation.sims import Simulation


def start():
    sim = Simulation(N=4, T=1000, v_field_res=1200, width=1000, height=600, show_vis_field=False)
    sim.start()
