import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque

class DQNetwork(nn.Module):
    ''''''
    def __init__(self, input_size, output_size):
        super(DQNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)

    def forward(self, state):

        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        output = self.layer3(x)
        return output

# Define experience tuple for replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# Define the DQN agent with replay memory
class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory_capacity=10000, batch_size=128,
                 gamma=0.99, epsilon_start=0.9, tau=0.005, epsilon_decay=1000, epsilon_end=0.05, lr=1e-5):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.tau=tau
        self.steps_done=0
        self.lr = lr
        self.batch_size = batch_size

        # Q-network and target Q-network

        self.q_network = DQNetwork(state_size, action_size)
        self.target_q_network = DQNetwork(state_size, action_size)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
        self.target_q_network.eval()  # Set target network to evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.9)

        # Replay memory
        self.replay_memory = deque(maxlen=replay_memory_capacity)
        #self.writer= SummaryWriter()

    def select_action(self, state):
        # Epsilon-greedy exploration
        env_status = state[len(state)-1]
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        math.exp(-1. * self.steps_done / self.epsilon_decay)
        #if all elements before the last one are 0, then the agent is in the first state
        legal_actions=[0]
        if env_status==1:
            legal_actions.append(1)
        if state[:-1].sum()!=0:
            legal_actions.append(2)

        if np.random.rand() <= eps_threshold:
            self.action = np.random.choice(legal_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_network(state_tensor)
                sorted_indices = torch.argsort(q_values, descending=True).tolist()
                for index in sorted_indices:
                    if index in legal_actions:
                        # Set self.action to the highest index in legal_actions
                        self.action = index
                        break

        return self.action

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
    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return

        # Sample a random mini-batch from replay memory
        #TODO: Change to sequence batch
        batch = random.sample(self.replay_memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_tensor = torch.FloatTensor(state_batch)
        next_state_tensor = torch.FloatTensor(next_state_batch)
        action_tensor = torch.LongTensor(action_batch)
        reward_tensor = torch.FloatTensor(reward_batch)
        done_tensor = torch.FloatTensor(done_batch)

        # Compute Q-values
        q_values = self.q_network(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1))

        # Compute target Q-value
        with torch.no_grad():
            next_q_values_target = self.target_q_network(next_state_tensor)
            max_next_q_value = torch.max(next_q_values_target, 1)[0]
            target_q_value = reward_tensor + (1 - done_tensor) * self.gamma * max_next_q_value

        # Compute loss and perform a gradient descent step
        loss = nn.functional.mse_loss(q_value, target_q_value.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def update_target_network(self):
        # Update target Q-network by copying the weights from the current Q-network
        #self.target_q_network.load_state_dict(self.q_network.state_dict())

        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_q_network.load_state_dict(target_net_state_dict)

    def add_to_replay_memory(self, experience):
        # Add experience to replay memory
        self.replay_memory.append(experience)

