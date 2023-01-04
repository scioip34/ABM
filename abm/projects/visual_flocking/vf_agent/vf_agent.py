import matplotlib
import numpy as np

from abm.projects.visual_flocking.vf_agent import vf_supcalc
from abm.agent.agent import Agent
from abm.contrib import colors


class VFAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # creating agent status
        # flocking or emergency
        self.agent_state = "flocking"

        # boundary conditions
        # infinite or walls
        self.boundary_cond = "walls"

        # preparing phi values for algorithm according to FOV
        self.PHI = np.arange(-np.pi, np.pi, (2*np.pi)/self.v_field_res)

        # social information: visual field projections
        self.soc_v_field_proj = np.zeros(self.v_field_res)

        # saving collision times
        self.coll_times = []

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep
        to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in
        the environment. These are not necessarily socially relevant, i.e. all
        agents.
        """
        # update agent information
        self.update_social_info(agents)
        print(np.max(self.soc_v_field))

        # update agent's state
        self.update_state()

        # perform agent's action i.e. exploration, taxis, relocation or flocking
        self.perform_action()

        # boundary conditions if applicable
        if self.boundary_cond == "walls":
            self.reflect_from_walls()
        elif self.boundary_cond == "infinite":
            self.teleport_infinite_arena()

        # updating agent visualization
        self.draw_update()


    def teleport_infinite_arena(self):
        """In case the boundary conditions are infinite (as in now reflection from walls is requested) the
        agents are teleported on a torus when reaching walls."""

        # Boundary conditions according to center of agent (simple)
        x = self.position[0] + self.radius
        y = self.position[1] + self.radius

        if x < self.boundaries_x[0]:
            self.position[0] = self.boundaries_x[1] - self.radius
        elif x > self.boundaries_x[1]:
            self.position[0] = self.boundaries_x[0] + self.radius

        if y < self.boundaries_y[0]:
            self.position[1] = self.boundaries_y[1] - self.radius
        elif y > self.boundaries_y[1]:
            self.position[1] = self.boundaries_y[0] + self.radius

    def update_social_info(self, agents):
        # calculate socially relevant projection field (e.g. according to
        # signalling agents)
        self.soc_v_field = self.calc_soc_v_proj(agents)

    def calc_soc_v_proj(self, agents):
        """
        :param agents: all agents
        """
        agents_of_interest = [ag for ag in agents if ag.id != self.id]
        visual_field = vf_supcalc.projection_field(
            fov=self.FOV,
            v_field_resolution=self.v_field_res,
            position=np.array(self.position),
            radius=self.radius,
            orientation=self.orientation,
            object_positions=[np.array(ag.position) for ag in agents_of_interest],
            boundary_cond=self.boundary_cond,
            arena_width=self.WIDTH,
            arena_height=self.HEIGHT,
            ag_id=self.id)
        # sum of all agents projections at each point in visual field
        svfield = visual_field.sum(axis=0)
        # binarize v field
        svfield[svfield > 0] = 1
        return svfield

    def update_state(self):
        # update agent state based on the decision-making process
        self.agent_state = "flocking"

    def perform_action(self):
        # we freeze agents when we move them
        if not self.is_moved_with_cursor:
            # perform the agent's action according to the current state
            if self.agent_state == "flocking":
                dv, dphi = vf_supcalc.VSWRM_flocking_state_variables(self.velocity, self.PHI, np.flip(self.soc_v_field))
                if self.id == 0:
                    print(dv, dphi)
                self.update_agent_position(dphi, dv)
            elif self.agent_state == "emergency":
                pass

    def update_agent_position(self, theta, vel):
        # updating agent's state variables according to calculated vel and
        # theta
        self.orientation += theta
        self.prove_orientation()  # bounding orientation into 0 and 2pi
        self.velocity += vel
        # self.prove_velocity()  # possibly bounding velocity of agent

        # new agent's position
        new_pos = (
            self.position[0] + self.velocity * np.cos(self.orientation),
            self.position[1] - self.velocity * np.sin(self.orientation)
        )
        # update the agent's position with constraints (reflection from the
        # walls) or with the new position
        self.position = list(new_pos)


    def change_color(self):
        """
        Changing color of agent according to the behavioral mode the agent is
        currently in.
        """
        cmap = matplotlib.cm.get_cmap('Spectral')
        rgba = np.array(cmap(self.orientation / (2 * np.pi)))
        # rescaling color for pygame
        rgba[0:3] *= 255
        self.color = rgba
