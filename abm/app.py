from abm.simulation.sims import Simulation


def start():
    sim = Simulation(N=3, T=1000, v_field_res=800, width=800, height=600)
    sim.start()
