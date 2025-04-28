import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_fc = nn.Linear(input_dim, 128)
        
        self.policy_fc = nn.Linear(128, 64)
        self.value_fc = nn.Linear(128, 64)

        self.policy_head = nn.Linear(64, action_dim)
        self.value_head = nn.Linear(64, 1)

        self.policy_norm = nn.LayerNorm(64)
        self.value_norm = nn.LayerNorm(64)

    def forward(self, x):
        x = torch.relu(self.shared_fc(x))

        # Policy branch
        policy_x = torch.relu(self.policy_norm(self.policy_fc(x)))
        policy_logits = self.policy_head(policy_x)

        # Value branch
        value_x = torch.relu(self.value_norm(self.value_fc(x)))
        value = self.value_head(value_x)

        return policy_logits, value


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
        # Convert buffers to tensors
        states = torch.cat(self.buffer.states)
        actions = torch.stack(self.buffer.actions).unsqueeze(1)
        old_logprobs = torch.stack(self.buffer.logprobs).detach()
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.buffer.dones, dtype=torch.float32, device=self.device)
        values = torch.cat(self.buffer.values).squeeze(1).detach()

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


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

            loss = actor_loss + 0.5 * critic_loss - 0.05 * entropy


            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

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
            best_filename = "ppo_model_best.pth"
            best_filepath = os.path.join(model_directory, best_filename)
            torch.save(self.model.state_dict(), best_filepath)
            print(f"New best model saved: {best_filename}")
            return reward  # ✅ Update best_reward if new best

        return best_reward  # ✅ Otherwise, return unchanged best_reward




    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"Model loaded from {filename}")


    
