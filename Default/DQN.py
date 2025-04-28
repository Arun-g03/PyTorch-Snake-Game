import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from datetime import datetime

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class DQNAgent:
    """
    DQN Network for the snake
    
    """
    def __init__(self, state_dim, action_dim, memory_size=10000, batch_size=64, gamma=0.99, lr=0.001, target_update=10, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = PrioritizedReplayMemory(memory_size, state_dim, device=device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.target_update = target_update
        self.device = device

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.loss_fn = nn.SmoothL1Loss(reduction='none')
        self.steps_done = 0
        self.saved_models = []  # List to store saved model filenames
        self.update_target_net()

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.policy_net(state).argmax().item()

    def store_transition(self, state, action, reward, next_state, done, msa):
        self.memory.push(state, action, reward, next_state, done, msa)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self, beta=0.4):
        if len(self.memory) < self.batch_size:
            return None
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, msa_batch = transitions

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            non_final_mask = (done_batch == 0).squeeze(1)
            if non_final_mask.any():
                non_final_next_states = next_state_batch[non_final_mask]
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

            expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) * (1 - done_batch) + reward_batch

        td_errors = self.loss_fn(state_action_values, expected_state_action_values)
        loss = (td_errors * weights.unsqueeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, priorities)

        return loss.item()

    def save_model(self, episode):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"model_ep{episode}_{timestamp}.pth"
        model_directory = "./models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        filepath = os.path.join(model_directory, filename)
        torch.save(self.policy_net.state_dict(), filepath)
        self.saved_models.append(filepath)
        print(f"Model saved as {filename}")
        self.manage_saved_models()

    def manage_saved_models(self):
        if len(self.saved_models) > 3:
            oldest_model = self.saved_models.pop(0)
            if os.path.exists(oldest_model):
                os.remove(oldest_model)
                print(f"Removed oldest model: {oldest_model}")

    def load_model(self, filename):
        self.policy_net.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target_net()
        print(f"Model loaded from {filename}")

class PrioritizedReplayMemory:
    def __init__(self, capacity, state_dim, alpha=0.6, device='cpu'):
        self.capacity = capacity
        self.position = 0
        self.alpha = alpha
        self.device = device

        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.msas = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

        self.priorities = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done, msa):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor([action], dtype=torch.int64, device=self.device)
        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        done = torch.as_tensor([done], dtype=torch.float32, device=self.device)
        msa = torch.as_tensor([msa], dtype=torch.float32, device=self.device)

        max_priority = self.priorities.max().item() if self.position > 0 else 1.0
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.msas[self.position] = msa
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if self.position == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=False)
        weights = (len(probabilities) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        state_batch = self.states[indices]
        action_batch = self.actions[indices]
        reward_batch = self.rewards[indices]
        next_state_batch = self.next_states[indices]
        done_batch = self.dones[indices]
        msa_batch = self.msas[indices]

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch, msa_batch), indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        batch_priorities = torch.as_tensor(batch_priorities, dtype=torch.float32, device=self.device).view(-1)
        if batch_priorities.shape != torch.Size([len(batch_indices)]):
            raise ValueError(f"Expected batch_priorities shape to be {[len(batch_indices)]}, but got {batch_priorities.shape}")
        self.priorities[batch_indices] = batch_priorities

    def __len__(self):
        return min(self.position, self.capacity)




