from abm.agent.agent import Agent
from abm.agent import supcalc
import numpy as np


class Prey(Agent):
    """Prey class of P34ABM"""

    def __init__(self, id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                 pooling_prob, consumption, vision_range, visual_exclusion, patchwise_exclusion=True):
        super().__init__(id, radius, position, orientation, env_size, color, v_field_res, FOV, window_pad, pooling_time,
                         pooling_prob, consumption, vision_range, visual_exclusion,
                         patchwise_exclusion=patchwise_exclusion)
        self.pred_v_field = np.zeros(self.v_field_res)
        self.prey_v_field = np.zeros(self.v_field_res)
        self.phis = np.linspace(-np.pi, np.pi, self.v_field_res)

    def update(self, predators, preys):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """
        # calculate relevant projection fields (of conspecifics and predators)
        if self.visual_exclusion:
            self.pred_v_field = self.projection_field(predators, keep_distance_info=False,
                                                      non_expl_agents=preys)
            self.prey_v_field = self.projection_field(preys, keep_distance_info=False,
                                                      non_expl_agents=predators)
        else:
            self.pred_v_field = self.projection_field(predators, keep_distance_info=False)
            self.prey_v_field = self.projection_field(preys, keep_distance_info=False)

        if not self.get_mode() == "collide":
            # flocking with conspecifics
            vel, theta = supcalc.VSWRM_flocking_state_variables(self.velocity, self.phis, self.prey_v_field)
            theta *= -1

            # aversion from predators
            vel_a, theta_a = supcalc.VSWRM_flocking_state_variables(self.velocity, self.phis, self.pred_v_field)

            # # assuming superposition
            # vel = vel - vel_a
            # theta = theta - theta_a
        else:
            # COLLISION AVOIDANCE IS ACTIVE, let that guide us
            # As we don't have proximity sensor interface as with e.g. real robots we will let
            # the environment to enforce us into a collision maneuver from the simulation environment
            # so we don't change the current velocity from here.
            vel, theta = (0, 0)

        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            # self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()
        self.collected_r_before = self.collected_r
