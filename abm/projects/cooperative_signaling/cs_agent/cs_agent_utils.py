import numpy as np


def prove_velocity(velocity, agent_state, velocity_limit=1):
    """
    Prove the velocity of the agent to be within the limits of the simulation
    """
    if agent_state == 'exploration':
        if np.abs(velocity) > velocity_limit:
            # stopping agent if too fast during exploration
            # QUESTION: can we return velocity_limit instead of 1?
            return velocity_limit
    return velocity
