import matplotlib
import numpy as np
import pygame

from abm.projects.visual_flocking.vf_agent import vf_supcalc
from abm.agent.agent import Agent
from abm.projects.visual_flocking.vf_contrib import vf_params
from abm.contrib import colors
import importlib


class VFAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        importlib.reload(vf_supcalc)
        importlib.reload(vf_params)

        # creating agent status
        # flocking or emergency
        self.agent_state = "flocking"

        # lines to follow
        self.lines = []
        self.sensor_distance = 20  # sensor distance to follow lines
        self.sensor_size = 9
        self.line_map = np.zeros((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))

        # getting additional information from supplementary calculation functions
        self.verbose_supcalc = False
        # boundary conditions
        # infinite or walls
        self.boundary_cond = vf_params.BOUNDARY
        self.limit_movement = vf_params.LIMIT_MOVEMENT
        self.max_vel = vf_params.MAX_VEL
        self.max_th = vf_params.MAX_TH

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


    def draw_update(self):
        """
        updating the outlook of the agent according to position and orientation
        """
        # update position
        self.rect.x = self.position[0]
        self.rect.y = self.position[1]

        # change agent color according to mode
        self.change_color()

        # update surface according to new orientation
        # creating visualization surface for agent as a filled circle
        self.image = pygame.Surface([self.radius * 2, self.radius * 2])
        self.image.fill(colors.BACKGROUND)
        self.image.set_colorkey(colors.BACKGROUND)
        if self.is_moved_with_cursor:
            pygame.gfxdraw.filled_circle(
                self.image,
                self.radius,
                self.radius,
                self.radius,
                self.selected_color
            )
            pygame.gfxdraw.aacircle(self.image,
                                    self.radius,
                                    self.radius,
                                    self.radius,
                                    colors.BACKGROUND)
        else:
            pygame.gfxdraw.filled_circle(
                self.image,
                self.radius,
                self.radius,
                self.radius-1,
                self.color[0:-1]
            )
            pygame.gfxdraw.aacircle(self.image,
                                    self.radius,
                                    self.radius,
                                    self.radius-1,
                                    colors.BLACK)

            # pygame.draw.circle(
            #     self.image, self.color, (self.radius, self.radius), self.radius
            # )

        # showing agent orientation with a line towards agent orientation
        new_white = (255, 255, 254)
        pygame.draw.line(self.image, new_white, (self.radius, self.radius),
                         ((1 + np.cos(self.orientation)) * self.radius, (1 - np.sin(self.orientation)) * self.radius),
                         3)
        self.image = pygame.transform.smoothscale(self.image, [self.radius * 2, self.radius * 2])
        self.mask = pygame.mask.from_surface(self.image)


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
        try:
            dphi2 = vf_supcalc.follow_lines_local(self.position, self.radius, self.orientation, self.line_map, self.velocity,
                                                  sensor_radius=self.sensor_size, sensor_distance=self.sensor_distance)
        except:
            print("Error during line following!")

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

    def update_linemap(self):
        """Updating background line map to follow"""
        subsurface = pygame.Surface((self.WIDTH + self.window_pad, self.HEIGHT + self.window_pad))
        subsurface.fill(colors.BACKGROUND)
        subsurface.set_colorkey(colors.WHITE)
        subsurface.set_alpha(255)
        for line in self.lines:
            for pi in range(1, len(line)):
                point1 = line[pi-1]
                point2 = line[pi]
                color = colors.BLACK
                pygame.draw.line(subsurface, color, point1, point2, 1)
        from scipy.ndimage.filters import gaussian_filter
        line_map = pygame.surfarray.array3d(subsurface)
        line_map = gaussian_filter(line_map, sigma=1)
        line_map[line_map<253] = 0
        line_map = gaussian_filter(line_map, sigma=1)
        self.line_map = 1 - line_map.swapaxes(0, 1)[:, :, 0]/255


    def perform_action(self):
        # we freeze agents when we move them
        if not self.is_moved_with_cursor:
            # perform the agent's action according to the current state
            if self.agent_state == "flocking":
                if len(self.PHI) == len(self.soc_v_field):
                    if not self.verbose_supcalc:
                        dv, dphi = vf_supcalc.VSWRM_flocking_state_variables(self.velocity, self.PHI, np.flip(self.soc_v_field),
                                                                             vf_params, verbose=self.verbose_supcalc)
                        if len(self.lines) != 0:
                            dphi2 = vf_supcalc.follow_lines_local(self.position, self.radius, self.orientation, self.line_map, self.velocity,
                                                            sensor_radius=self.sensor_size, sensor_distance=self.sensor_distance)
                            dphi = dphi2
                    else:
                        self.dv, self.dphi, self.ablob, self.aedge, self.bblob, self.bedge = vf_supcalc.VSWRM_flocking_state_variables(self.velocity, self.PHI, np.flip(self.soc_v_field),
                                                                             vf_params, verbose=self.verbose_supcalc)
                        dv, dphi = self.dv, self.dphi
                else:
                    dv, dphi = 0, 0
                    print("Warning: temporary skip calculation due to mismatch in phi and vfield resolutions!")
                self.update_agent_position(dphi, dv)
            elif self.agent_state == "emergency":
                pass

    def update_agent_position(self, theta, vel):
        # updating agent's state variables according to calculated vel and
        # theta
        # maximum turning angle per timestep
        if self.limit_movement:
            theta = self.prove_turning(theta, theta_lim=self.max_th)
        self.orientation += theta
        self.prove_orientation()  # bounding orientation into 0 and 2pi

        self.velocity += vel
        if self.limit_movement:
            self.prove_velocity(velocity_limit=self.max_vel)  # possibly bounding velocity of agent

        # new agent's position
        new_pos = (
            self.position[0] + self.velocity * np.cos(self.orientation),
            self.position[1] - self.velocity * np.sin(self.orientation)
        )
        # update the agent's position with constraints (reflection from the
        # walls) or with the new position
        self.position = list(new_pos)

    def prove_turning(self, theta, theta_lim=0.2):
        t_sign = np.sign(theta)
        if t_sign == 0:
            t_sign = +1
        if np.abs(theta) > theta_lim:
            theta = theta_lim * t_sign
        return theta


    def prove_velocity(self, velocity_limit=1):
        """Restricting the absolute velocity of the agent"""
        vel_sign = np.sign(self.velocity)
        if vel_sign == 0:
            vel_sign = +1
        if np.abs(self.velocity) > velocity_limit:
            self.velocity = velocity_limit * vel_sign


    def change_color(self):
        """
        Changing color of agent according to the behavioral mode the agent is
        currently in.
        """
        cmap = matplotlib.cm.get_cmap('jet')
        rgba = np.array(cmap(self.orientation / (2 * np.pi)))
        # rescaling color for pygame
        rgba[0:3] *= 255
        self.color = rgba
