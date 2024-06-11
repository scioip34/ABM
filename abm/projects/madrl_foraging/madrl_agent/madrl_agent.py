"""
agent.py : including the main classes to create an agent. Supplementary calculations independent from class attributes
            are removed from this file.
"""
import os

import pygame
import numpy as np
import torch

from abm.contrib import colors
from abm.projects.madrl_foraging.madrl_agent import madrl_supcalc as supcalc
import matplotlib.pyplot as plt

#matplotlib.use('agg')
from abm.projects.madrl_foraging.madrl_agent.brain import DQNAgent
from abm.agent.agent import Agent
from abm.projects.madrl_foraging.madrl_contrib import madrl_learning_params as learning_params


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADRLAgent(Agent):
    """
    Agent class inhereted from Agent Class t
    Includes all private parameters of the agents and all methods necessary to move in the environment
    and to make decisions.
    """



    def __init__(self,train,**kwargs):


        # Initializing supercalss
        super().__init__(**kwargs)



        self.show_stats = True
        self.mode = "explore"

        self.train=train
        self.soc_v_field = np.zeros(self.v_field_res)
        self.search_efficiency = 0
        self.res=None
        self.reward=0
        self.closest_patch = None

        self.last_exploit_time = 1
        self.total_reloc= 0
        self.total_discov= 0
        self.new_discovery = 0
        self.brain_type = learning_params.brain_type

        #Create the policy networks
        self.policy_network = DQNAgent(state_size=self.v_field_res+ 1, action_size=3)
        if self.brain_type == "DQN" or self.brain_type == "DDQN":

            if learning_params.pretrained and learning_params.pretrained_models_dir!="":
                    print("Loading pretrained model")

                    model_path = os.path.join(learning_params.pretrained_models_dir, f"model_{self.id}.pth")

                    if train:
                        self.policy_network.load_model_train(model_path)
                    else:
                        map_location = device
                        checkpoint = torch.load(model_path, map_location)
                        try:
                            self.policy_network.current_network.load_state_dict(checkpoint['q_network_state_dict'], map_location)
                        except:
                            self.policy_network.current_network.load_state_dict(checkpoint, map_location)

            if not train :
                print("Model in evaluation mode")
                self.policy_network.current_network.eval()
                self.policy_network.epsilon_start = 0
                self.policy_network.epsilon_end = 0


    def update_decision_processes(self):
        """updating inner decision processes according to the policy network"""
        action = self.policy_network.action_tensor.item()
        if action == 0:
            self.set_mode("explore")
        elif action == 1:
            self.set_mode("exploit")
        elif action == 2:
            self.set_mode("relocate")

    def compute_reward(self):
        """
        Compute the reward of the agent: the reward is one if the agent is in exploit mode and zero otherwise
        """
        '''
        if ag.cse_w>0:

            collective_se = sum(agent.search_efficiency for agent in self.agents) / len(
                self.agents)

        reward = 0
        if ag.get_mode()=="exploit":
            reward = (ag.ise_w * ag.search_efficiency + ag.cse_w * collective_se)
        '''

        reward = 0

        if self.get_mode()=="exploit":
            reward = 1

        return reward

    def update(self, agents):
        """
        main update method of the agent. This method is called in every timestep to calculate the new state/position
        of the agent and visualize it in the environment
        :param agents: a list of all obstacle/agents coordinates as (X, Y) in the environment. These are not necessarily
                socially relevant, i.e. all agents.
        """

        self.update_decision_processes()

        if self.get_mode() == "explore":
            vel, theta = supcalc.random_walk(desired_vel=self.max_exp_vel)

        elif self.get_mode() == "exploit":
            if self.env_status == 1:
                vel, theta = (-self.velocity * self.exp_stop_ratio, 0)
            else:
                print(f"ERROR: Exploiting agent {self.id} is not on a resource patch, will relocate!")
                vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field) #,v_desired=self.max_exp_vel)

        elif self.get_mode() == "relocate":
            vel, theta = supcalc.F_reloc_LR(self.velocity, self.soc_v_field) #, v_desired=self.max_exp_vel)

        if not self.is_moved_with_cursor:  # we freeze agents when we move them
            # updating agent's state variables according to calculated vel and theta
            self.orientation += theta
            self.prove_orientation()  # bounding orientation into 0 and 2pi
            self.velocity += vel
            self.prove_velocity()  # possibly bounding velocity of agent

            # updating agent's position
            self.position[0] += self.velocity * np.cos(self.orientation)
            self.position[1] -= self.velocity * np.sin(self.orientation)

            # boundary conditions if applicable
            self.reflect_from_walls()

        # updating agent visualization
        self.draw_update()

        #TODO: Ask David why this is here
        self.collected_r_before = self.collected_r

    def get_mode(self):
        """Returning the current mode of the agent"""
        return self.mode



    def set_mode(self, mode):
        """Setting the behavioral mode of the agent. This can be:
            -explore (0)
            -exploit (1)
            -relocate (2)
        """

        self.mode = mode



    def calc_social_V_proj(self, obstacles):
        """Calculating the socially relevant visual projection field of the agent. This is calculated as the
         projection of nearby exploiting agents that are not visually excluded by other agents
         If the agents use an ideal strategy, the projection is calculated as the projection of the closest resource patch
         Args:  obstacles: a list of all agents in the environment or the nearest resource patch (if ideal strategy is used)
         """

        if self.brain_type == "ideal":
            if obstacles[0] != None:
                self.soc_v_field = self.projection_field(obstacles, keep_distance_info=True)
            else:
                self.soc_v_field = np.zeros(self.v_field_res)

            return self.soc_v_field


        # visible agents (exluding self)
        agents = [ag for ag in obstacles if supcalc.distance(self, ag) <= self.vision_range]
        # those of them that are exploiting
        exploit_agents = [ag for ag in agents if ag.id != self.id
                       and ag.get_mode() == "exploit"]


        # all other agents to calculate visual exclusions
        non_exploit_agents = [ag for ag in agents if ag not in exploit_agents]


        if self.exclude_agents_same_patch:
            # in case agents on same patch are excluded they can still cause visual exclusion for exploiting agents
            # on the same patch (i.e. they can cover agents on other patches)
            non_exploit_agents.extend([ag for ag in exploit_agents if ag.exploited_patch_id == self.exploited_patch_id])
            exploit_agents = [ag for ag in exploit_agents if ag.exploited_patch_id != self.exploited_patch_id]

        # Excluding agents that still try to exploit but can not as the patch has been emptied
        exploit_agents = [ag for ag in exploit_agents if ag.exploited_patch_id != -1]

        if self.visual_exclusion:
            self.soc_v_field = self.projection_field(exploit_agents, keep_distance_info=True,
                                                     non_expl_agents=non_exploit_agents)

        else:
            self.soc_v_field  = self.projection_field(exploit_agents, keep_distance_info=True)


        return self.soc_v_field

    def visualize_v_fields(self):
        """
        Visualizing the social visual field of the agent with a polar plot.
        """

        if self.vis_counter % 50 == 0:
            # Create a polar plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

            # Plot the social visual field
            angles = np.linspace(-np.pi, np.pi,
                                 len(self.soc_v_field))  # generates an array of evenly spaced values within the range from -pi to pi

            ax.plot(angles, self.soc_v_field, label='Exploiting Agents', color="green")

            ax.set_rticks([])  # Remove radial tick labels if not needed
            ax.set_yticklabels([])  # Remove radial tick labels if not needed
            ax.set_theta_zero_location('N')  # Set the zero angle to the top (North)
            plt.title('Social Visual Field')
            plt.legend()

            plt.close()
        self.vis_counter += 1

    def reset(self):
            """
            Resetting relevant values of the agent after each train episode.
            """
            # Reset position and orientation
            x=np.random.randint(self.window_pad - self.radius, self.WIDTH + self.window_pad - self.radius)
            y=np.random.randint(self.window_pad - self.radius, self.HEIGHT + self.window_pad - self.radius)
            #x = self.WIDTH // 2
            #y = self.HEIGHT // 2
            self.position = np.array((x,y), dtype=np.float64)
            self.orientation = np.random.uniform(0, 2 * np.pi)
            # Reset agent state variables
            self.velocity = 0
            self.collected_r = 0
            self.collected_r_before = 0
            self.exploited_patch_id = -1
            self.mode = "explore"
            self.vis_field_source_data = {}
            self.vis_counter = 0



            self.overriding_mode = None


            self.time_spent_pooling = 0
            self.env_status_before = 0
            self.env_status = 0
            self.pool_success = 0

            self.soc_v_field = np.zeros(self.v_field_res)
            self.search_efficiency = 0

            self.last_exploit_time = 1
            self.total_reloc = 0
            self.total_discov = 0
            self.policy_network.last_action=-1

            # Reset policy network
            #self.policy_network.reset()

            # Update the agent's Pygame representation
            self.rect.x = self.position[0]
            self.rect.y = self.position[1]
            self.image.fill(colors.BACKGROUND)
            pygame.draw.circle(self.image, self.color, (self.radius, self.radius), self.radius)
            pygame.draw.line(self.image, colors.BACKGROUND, (self.radius, self.radius),
                             ((1 + np.cos(self.orientation)) * self.radius,
                              (1 - np.sin(self.orientation)) * self.radius), 3)
            self.mask = pygame.mask.from_surface(self.image)

