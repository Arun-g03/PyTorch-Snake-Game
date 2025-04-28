import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        self.shared_fc1 = nn.Linear(input_dim, 256)
        self.shared_fc2 = nn.Linear(256, 128)

        # Policy branch
        self.policy_fc = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, action_dim)
        self.policy_norm = nn.LayerNorm(64)

        # Value branch
        self.value_fc = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_norm = nn.LayerNorm(64)

    def forward(self, x):
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))

        # Policy branch
        policy_x = torch.relu(self.policy_fc(x))
        policy_x = self.policy_norm(policy_x)
        logits = self.policy_head(policy_x)

        # Value branch
        value_x = torch.relu(self.value_fc(x))
        value_x = self.value_norm(value_x)
        value = self.value_head(value_x)

        return logits, value



class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.__init__()

class PPOAgent:
    """
    PPO Network for the snake
    
    """
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, clip_epsilon=0.1, k_epochs=4, device='cpu'):
        self.device = device
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coeff = 0.05  # Start with high exploration


        self.saved_models = []

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.model(state)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(logprob)
        self.buffer.values.append(value)

        return action.item()

    def store_transition(self, reward, done):
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def optimize_model(self):
        if len(self.buffer.states) == 0:
            return  # Skip if buffer is empty

        # Convert buffers to tensors
        states = torch.cat(self.buffer.states)
        actions = torch.stack(self.buffer.actions).unsqueeze(1)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        values = torch.cat(self.buffer.values).squeeze(1).detach()

        # Compute returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Compute advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.k_epochs):
            logits, new_values = self.model(states)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            new_logprobs = dist.log_prob(actions.squeeze(1))
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(1), returns)

            # ✅ Use dynamic entropy coefficient
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        # ✅ Decay entropy coefficient slowly
        self.entropy_coeff = max(0.01, self.entropy_coeff * 0.999)

        self.buffer.clear()


    def save_model(self, episode, reward, best_reward):
        """ Save model only every 50 episodes, keep best and latest models. """
        if episode % 50 != 0:
            return best_reward  # ✅ Always return best_reward

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_directory = "./models"
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Always save the latest model
        latest_filename = "ppo_model_latest.pth"
        latest_filepath = os.path.join(model_directory, latest_filename)
        torch.save(self.model.state_dict(), latest_filepath)
        print(f"Latest model saved: {latest_filename}")

        # Save best model if new best reward
        if reward > best_reward:
            best_filename = "ppo_model_best_{best_reward}.pth"
            best_filepath = os.path.join(model_directory, best_filename)
            torch.save(self.model.state_dict(), best_filepath)
            print(f"\n\nNew best model saved: {best_filename}\n\n")
            return reward  # ✅ Update best_reward if new best

        return best_reward  # ✅ Otherwise, return unchanged best_reward




    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model loaded from {filename}")


    
