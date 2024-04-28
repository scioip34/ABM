import random
import pygame
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from abm.monitoring import ifdb, env_saver
from abm.projects.madrl_foraging.madrl_agent.madrl_agent import MADRLAgent as Agent
from abm.contrib import colors,ifdb_params as logging_params
from abm.projects.madrl_foraging.madrl_contrib import madrl_learning_params as learning_params
from abm.simulation.sims import Simulation, notify_agent, refine_ar_overlap_group



from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADRLSimulation(Simulation):
    def __init__(self, **kwargs):
        """
        Inherited from Simulation class

        :param train: boolean, if true the simulation will be ran in training mode, if false in evaluation mode
        :param train_every: int, number of timesteps after which the agent will be trained
        :param num_episodes: int, number of training episodes, if in evaluation it is set to 1

        """
        super().__init__(**kwargs)

        self.train=learning_params.train
        self.train_every = learning_params.train_every

        self.num_episodes = learning_params.num_episodes

        seed = learning_params.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    def add_new_agent(self, id, x, y, orient, with_proove=False, behave_params=None):
        """Adding a single new agent into agent sprites"""
        agent_proven = False

        while not agent_proven:
            agent = Agent(
                    id=id,
                    radius=self.agent_radii,
                    position=(x, y),
                    orientation=orient,
                    env_size=(self.WIDTH, self.HEIGHT),
                    color=colors.BLUE,
                    v_field_res=self.v_field_res,
                    FOV=self.agent_fov,
                    window_pad=self.window_pad,
                    pooling_time=self.pooling_time,
                    pooling_prob=self.pooling_prob,
                    consumption=self.agent_consumption,
                    vision_range=self.vision_range,
                    visual_exclusion=self.visual_exclusion,
                    patchwise_exclusion=self.patchwise_exclusion,
                    behave_params=None,
                train=self.train,
                )
            if with_proove:
                if self.proove_sprite(agent):
                    self.agents.add(agent)
                    agent_proven = True
            else:
                self.agents.add(agent)
                agent_proven = True

    def agent_resource_overlap(self, agents):
        collision_group_ar = pygame.sprite.groupcollide(
            self.rescources,
            agents,
            False,
            False,
            pygame.sprite.collide_circle
        )

        # refine collision group according to point-like pooling in center of agents
        collision_group_ar = refine_ar_overlap_group(collision_group_ar)

        return collision_group_ar


    def agent2resource_interaction(self,  collided_agents):
        collision_group_ar = self.agent_resource_overlap(self.agents)

        # collecting agents that are on resource patch
        agents_on_rescs = []

        # Notifying agents about resource if pooling is successful + exploitation dynamics
        for resc, agents in collision_group_ar.items():  # looping through patches

                destroy_resc = False  # if we destroy a patch it is 1
                for agent in agents:  # looping through all agents on patches
                    self.bias_agent_towards_res_center(agent, resc)

                    # One of previous agents on patch consumed the last unit
                    if destroy_resc:
                        notify_agent(agent, -1)
                    else:
                        # Agent finished pooling on a resource patch
                        if (agent.get_mode() in ["pool", "relocate"] and agent.pool_success) \
                                or agent.pooling_time == 0:
                            # Notify about the patch
                            notify_agent(agent, 1, resc.id)
                            # Teleport agent to the middle of the patch if needed
                            if self.teleport_exploit:
                                agent.position = resc.position + resc.radius - agent.radius

                        # Agent is exploiting this patch
                        if agent.get_mode() == "exploit":

                            # continue depleting the patch
                            depl_units, destroy_resc = resc.deplete(agent.consumption)
                            agent.collected_r_before = agent.collected_r  # rolling resource memory
                            agent.collected_r += depl_units  # and increasing it's collected rescources

                            # remember the time of last exploitation
                            if destroy_resc:  # consumed unit was the last in the patch
                                # print(f"Agent {agent.id} has depleted the patch all agents must be notified that"
                                #       f"there are no more units before the next timestep, otherwise they stop"
                                #       f"exploiting with delays")
                                for agent_tob_notified in agents:
                                    # print("C notify agent NO res ", agent_tob_notified.id)
                                    notify_agent(agent_tob_notified, -1)

                    # Collect all agents on resource patches
                    agents_on_rescs.append(agent)

                # Patch is fully depleted
                if destroy_resc:
                    # we clear it from the memory and regenerate it somewhere else if needed
                    self.kill_resource(resc)

        # Notifying other agents that there is no resource patch in current position (they are not on patch)
        for agent in self.agents.sprites():
            if agent not in agents_on_rescs:  # for all the agents that are not on recourse patches
                if agent not in collided_agents:  # and are not colliding with each other currently
                    # if they finished pooling
                    if (agent.get_mode() in ["pool",
                                             "relocate"] and agent.pool_success) or agent.pooling_time == 0:
                        notify_agent(agent, -1)
                    elif agent.get_mode() == "exploit":
                        notify_agent(agent, -1)


        # Update resource patches
        self.rescources.update()

    def step(self,turned_on_vfield):
        # order the agents by id to ensure that agent 0 has priority over agent 1 and agent 1 over agent 2

        # Update internal states of the agents and their positions
        self.agents.update(self.agents)

        # Check for agent-agent collisions
        # collided_agents = self.agent2agent_interaction()
        collided_agents = []

        # Check for agent-resource interactions and update the resource patches
        self.agent2resource_interaction(collided_agents)

        for ag in self.agents:
            ag.calc_social_V_proj(self.agents)
            ag.search_efficiency = ag.collected_r / self.t if self.t != 0 else 0
        collective_se = sum(ag.search_efficiency for ag in self.agents) / len(
            self.agents)

        # Draw the updated environment and agents (and visual fields if needed)
        # deciding if vis field needs to be shown in this timestep
        self.decide_on_vis_field_visibility(turned_on_vfield)
        if self.with_visualization:
            self.draw_frame(self.stats, self.stats_pos)
            pygame.display.flip()

        return collective_se


    def initialize_environment(self):
        for ag in self.agents:
            # Check for agent-agent collisions
            # collided_agents = self.agent2agent_interaction()

            # Check for agent-resource interactions and update the resource patches
            ag_resc_overlap = self.agent_resource_overlap([ag])
            if len(ag_resc_overlap) > 0:
                ag.env_status = 1

            ag.calc_social_V_proj(self.agents)
            # Concatenate the resource signal array for the state tensor (The social visual field (1D array )+ the
            # environment status (Scalar))
            if ag.env_status == 1:
                # calculate number of resources left in the patch
                resc = list(ag_resc_overlap.keys())[0]
                ag.policy_network.state_tensor = torch.FloatTensor(
                ag.soc_v_field.tolist() +[1.0] ).unsqueeze(0).to(device) #[resc.resc_left / resc.resc_units]

            else:
                ag.policy_network.state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [0.0]).unsqueeze(0).to(device)


    def start_madqn(self):
        """Main simulation loop for training the agents with MADQN"""

        # Start time of the simulation
        start_time = datetime.now()

        # Create the agents and resources patches  in the environment
        self.create_agents()
        self.create_resources()

        # Create surface to show visual fields and local var to decide when to show visual fields
        self.stats, self.stats_pos = self.create_vis_field_graph()

        # Create a directory to save the data (models and tensorboard logs)
        save_dir = logging_params.TIMESTAMP_SAVE_DIR

        writer = SummaryWriter(save_dir)
        writer.add_text('Hyperparameters',
                        f'Gamma: {learning_params.gamma}, \n Epsilon Start: {learning_params.epsilon_start}, '
                        f'\n Epsilon End: {learning_params.epsilon_end}, \n'
                        f'Epsilon Decay: {learning_params.epsilon_decay},\n Tau: {learning_params.tau},\n Learning '
                        f'Rate: {learning_params.lr}',
                        0)

        turned_on_vfield = 0

        mode = "training" if self.train else "evaluation"
        print(f"Starting main simulation loop in MADQN in {mode} with {len(self.agents)} agents and {len(self.rescources)} resources \n "
              f"Saving data to {save_dir}!")
        for episode in range(self.num_episodes ):
            # Create a variable to indicate if the simulation is done
            done= False
            self.initialize_environment()
            collective_se_list = []
            print("Starting episode: ",episode)

            while self.t < self.T:
                # Indicate that the simulation is not done
                if self.t==self.T-1:
                    done = True

                # Agent 0 always has riority over agent 1 and agent 1 over agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed to deplete the patch, followed by agent 1, then agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed
                # to deplete the patch, followed by agent 1, then agent 2
                # If the three of them are on the same patch, and there are not enough resources agent 0 will be allowed to deplete the patch, followed by agent 1, then agent 2

                for ag in self.agents:
                    # Select an action
                    _ = ag.policy_network.select_action(ag.policy_network.state_tensor)

                collective_se = self.step(turned_on_vfield)

                collective_se_list.append(collective_se)
                # Train the agents
                for ag in self.agents:
                    if done:
                        ag.policy_network.next_state_tensor = None
                        ag.reward = collective_se
                    else:

                        # Concatenate the resource signal array for the next state tensor (The social visual field (1D array )+ the environment status (Scalar))
                        if ag.env_status == 1:

                            ag_resc_overlap = self.agent_resource_overlap([ag])
                            resc= list(ag_resc_overlap.keys())[0]

                            ag.policy_network.next_state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [1.0]).unsqueeze(0).to(device)#[resc.resc_left/resc.resc_units]

                        else:
                            ag.policy_network.next_state_tensor = torch.FloatTensor(ag.soc_v_field.tolist() + [0.0]).unsqueeze(0).to(device)

                        # Calculate the reward as a weighted sum of the individual and collective search efficiency
                        reward = ag.compute_reward()


                        if ag.policy_network.action_tensor.item() == 1:
                            ag.last_exploit_time = self.t

                    ag.policy_network.reward_tensor = torch.FloatTensor([reward]).to(device)

                    # Add the experience to the replay memory and train the agent
                    if self.train:
                        ag.policy_network.replay_memory.push(
                            ag.policy_network.state_tensor,
                            ag.policy_network.action_tensor,
                            ag.policy_network.next_state_tensor,
                            ag.policy_network.reward_tensor
                        )
                    if self.train and self.t % self.train_every == 0:
                        loss = ag.policy_network.optimize()

                        # Update the target network with soft updates
                        ag.policy_network.update_target_network()


                        if loss is not None:
                            '''
                            if math.isinf(loss):
                                print(f"Loss is infinity at timestep {self.t}!")
                            elif math.isnan(loss):
                                print(f"Loss is not a number (nan) at timestep {self.t}!")
                            elif loss < 0:
                                print(f"Loss is negative at timestep {self.t}!")
                            elif loss > 40 and ag.id == 1:
                                print(f"Loss of agent 1 is {loss} at timestep {self.t}!")
                            '''

                            writer.add_scalar(f'Agent_{ag.id}/Loss', loss, ag.policy_network.steps_done)
                        elif ag.policy_network.steps_done > ag.policy_network.batch_size:
                            print(f"Loss is None at timestep {self.t}!")

                        # Move to the next training step
                        ag.policy_network.steps_done += 1
                    ag.policy_network.state_tensor = ag.policy_network.next_state_tensor
                    ag.policy_network.last_action = ag.policy_network.action_tensor.item()
                # move to next simulation timestep (only when not paused)
                self.t += 1

                if self.save_in_ram:
                    ifdb.save_agent_data_RAM(self.agents, self.t)
                    ifdb.save_resource_data_RAM(self.rescources, self.t)


            for ag in self.agents:

                writer.add_scalar(f'Agent_{ag.id}/Individual search efficiency)', ag.search_efficiency,
                                        episode)
                writer.add_scalar('Collective search efficiency', collective_se, episode)

                ag.reset()
            for resc in self.rescources:
                self.kill_resource(resc)
            self.t=0
            print(f"Episode {episode} ended with collective search efficiency: ", collective_se)

        # Save the models
        if self.train:
            for count, ag in enumerate(self.agents):
                ag.policy_network.save_model(f'{save_dir}/model_{ag.id}.pth')
        # Close the tensorboard writer
        writer.close()
        env_saver.save_env_vars([self.env_path], "env_params.json", pop_num=None)

        if self.save_csv_files:
            if self.save_in_ifd or self.save_in_ram:

                ifdb.save_ifdb_as_csv(exp_hash=self.ifdb_hash, use_ram=self.save_in_ram, as_zar=self.use_zarr,
                                      save_extracted_vfield=False, pop_num=None)
            else:
                raise Exception("Tried to save simulation data as csv file due to env configuration, "
                                "but IFDB/RAM logging was turned off. Nothing to save! Please turn on IFDB/RAM logging"
                                " or turn off CSV saving feature.")

        # Quit the pygame environment
        pygame.quit()

        # Print the execution time
        end_time = datetime.now()
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')} Total simulation time: ",
          (end_time - start_time).total_seconds())
