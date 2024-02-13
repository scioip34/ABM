import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque
import abm.projects.madrl_foraging.madrl_contrib.madrl_learning_params as learning_params



class LSTM_DQNetwork(nn.Module):
        def __init__(self, input_size, output_size):
            super(LSTM_DQNetwork, self).__init__()
            self.lstm = nn.LSTM(input_size, 128)
            self.layer2 = nn.Linear(128, 128)

            self.layer3 = nn.Linear(128, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            x = torch.relu(self.layer2(lstm_out))
            x = self.layer3(x)
            return x

class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        # convolutional layer ?
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)


    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)
        return output


# Define experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

# Define the DQN agent with replay memory
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.id = id
        self.state_size = state_size
        self.action_size = action_size
        self.state_tensor=None
        self.next_state_tensor=None
        self.action_tensor=None
        self.reward_tensor=None
        self.agent_type = learning_params.brain_type

        self.gamma = learning_params.gamma
        self.epsilon_start = learning_params.epsilon_start
        self.epsilon_decay = learning_params.epsilon_decay

        self.epsilon_end = learning_params.epsilon_end
        self.tau=learning_params.tau
        self.steps_done=0
        self.lr = learning_params.lr
        self.batch_size = learning_params.batch_size
        self.pretrained = learning_params.pretrained
        self.brain_type = learning_params.brain_type


        # Q-network and target Q-network
        if self.brain_type=="DQN":
            self.q_network = DQNetwork(state_size, action_size)
            self.target_q_network = DQNetwork(state_size, action_size)
            self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
        #self.target_q_network.eval()  # Set target network to evaluation mode
        else:
            print("Using LSTM")
            self.q_network = LSTM_DQNetwork(state_size, action_size)
            self.target_q_network = LSTM_DQNetwork(state_size, action_size)
            self.target_q_network.load_state_dict(self.q_network.state_dict())  #

        # Optimizer
        if learning_params.optimizer=="Adam":
            print("Using Adam")
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        else:
            print("Using RMSprop")
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.lr)
        #self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        # Replay memory
        self.replay_memory = deque(maxlen=learning_params.replay_memory_capacity)
        #self.writer= SummaryWriter()



    def select_action(self, state, legal_actions):

        # Epsilon-greedy exploration
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-self.steps_done / self.epsilon_decay)

        if random.random() <= eps_threshold:
            action = random.choice(legal_actions)
            #print("Random Action", action)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.max(1).indices.view(1, 1)

                while action not in legal_actions:
                    q_values[0][action] = float('-inf')
                    action = q_values.max(1).indices.view(1, 1)


        self.action_tensor=torch.LongTensor([[action]])

        return self.action_tensor

    '''
    def sample_sequence_batch(self):
        if len(self.replay_memory) < self.sequence_length:
            return None  # Not enough experiences for a sequence

        index = random.randint(0, len(self.replay_memory) - self.sequence_length)
        sequence = list(self.replay_memory)[index:index + self.sequence_length]

        # Unpack the sequence into separate lists
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*sequence)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    '''
    def optimize(self):
        if len(self.replay_memory) < self.batch_size:
            return


        # Sample a random mini-batch from replay memory

        transitions = random.sample(self.replay_memory, self.batch_size)
        batch = Experience(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state )), dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])


        state_tensor = torch.cat(batch.state)
        action_tensor = torch.cat(batch.action)

        reward_tensor = torch.cat(batch.reward)

        #print("non_final_mask:", non_final_mask.shape)
        #print("non_final_next_states:", non_final_next_states.shape)
        #print("state_tensor:", state_tensor.shape)
        #print("action_tensor:", action_tensor.shape)
        #print("reward_tensor:", reward_tensor.shape)

        # Compute Q-values
        q_values = self.q_network(state_tensor).gather(1, action_tensor)

        #print("state_action_values:", q_values.shape)

        next_state_values = torch.zeros(self.batch_size)
        # Compute target Q-value
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).max(1)[0]

        target_q_value = reward_tensor + (self.gamma * next_state_values)
        #print("expected_state_action_values: ", target_q_value.shape)
        #print(" \n **************************** \n")

        # Compute loss and perform a gradient descent step
        #criterion that measures the mean squared error (squared L2 norm)
        #print("expected_state_action_values.unsqueeze(1) : ",target_q_value.unsqueeze(1).shape)

        loss = nn.functional.mse_loss(q_values, target_q_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        #gradient clipping
        #torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 0.1)
        self.optimizer.step()
        #self.scheduler.step()
        return loss.item()

    def update_target_network(self):
        # Update target Q-network by copying the weights from the current Q-network
        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_q_network.load_state_dict(target_net_state_dict)
        #print("target_net_state_dict:", target_net_state_dict[key].shape)
        #print("policy_net_state_dict:", policy_net_state_dict[key].shape)

    def add_to_replay_memory(self, experience):
        # Add experience to replay memory
        self.replay_memory.append(experience)

