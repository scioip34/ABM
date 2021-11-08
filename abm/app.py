from abm.simulation.sims import Simulation


def start():
    sim = Simulation(N=5, T=100000, v_field_res=800, width=800, height=600, show_vis_field=True)
    sim.start()
