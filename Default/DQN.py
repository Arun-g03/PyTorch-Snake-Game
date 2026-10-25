import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config  # repo root is put on sys.path by main.py
import checkpoints


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=None):
        super().__init__()
        sizes = [input_dim, *(hidden or config.DQN_NETWORK)]
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = nn.Linear(sizes[-1], output_dim)

    def forward(self, x):
        for layer in self.hidden:
            x = torch.relu(layer(x))
        return self.out(x)


class PrioritizedReplayMemory:
    def __init__(self, capacity, state_dim, alpha=0.6, device='cpu'):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.alpha = alpha
        self.device = device

        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.priorities = torch.zeros((capacity,), dtype=torch.float32, device=device)

    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities[:self.size].max().item() if self.size > 0 else 1.0
        i = self.position
        self.states[i] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[i] = float(done)
        self.priorities[i] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        probabilities = self.priorities[:self.size] ** self.alpha
        probabilities = probabilities / probabilities.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=False)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        batch = (self.states[indices], self.actions[indices], self.rewards[indices],
                 self.next_states[indices], self.dones[indices])
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = torch.as_tensor(priorities, dtype=torch.float32, device=self.device).view(-1)

    def __len__(self):
        return self.size


class DQNAgent:
    """Double DQN with prioritised replay. Same interface as PPOAgent (see main.py)."""
    model_type = "dqn"

    def __init__(self, state_dim, action_dim, device='cpu', num_envs=1, **overrides):
        unknown = set(overrides) - set(config.DQN_HPARAMS)
        if unknown:
            raise TypeError(f"Unknown hyperparameters: {sorted(unknown)}")
        self.hparams = h = {**config.DQN_HPARAMS, **overrides}

        self.action_dim = action_dim
        self.device = device
        self.gamma = h['gamma']
        self.batch_size = h['batch_size']
        self.memory = PrioritizedReplayMemory(h['memory_size'], state_dim, device=device)

        self.model = DQN(state_dim, action_dim).to(device)  # the policy network ("model" for checkpoints)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=h['lr'], foreach=True)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, h['lr_step_size'], h['lr_gamma'])
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.env_steps = 0    # actions chosen so far (drives epsilon and beta)
        self._train_credit = 0  # environment steps not yet 'spent' on an optimisation step
        self.opt_steps = 0    # optimisation steps so far (drives target sync)
        self.update_target_net()

    @property
    def epsilon(self):
        h = self.hparams
        frac = min(1.0, self.env_steps / h['epsilon_decay_steps'])
        return h['epsilon_start'] + frac * (h['epsilon_end'] - h['epsilon_start'])

    @property
    def beta(self):
        h = self.hparams
        frac = min(1.0, self.env_steps / h['beta_frames'])
        return h['beta_start'] + frac * (1.0 - h['beta_start'])

    @torch.inference_mode()
    def act(self, state, greedy=True):
        """Best action for one `state` (greedy), or epsilon-greedy when greedy=False."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.model(state).argmax(dim=1).item()

    # Common agent interface used by main.py ---------------------------------------------------
    @torch.inference_mode()
    def select_actions(self, states):
        """Epsilon-greedy actions for a batch of states, one per environment."""
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        greedy = self.model(states).argmax(dim=1).cpu().numpy()
        explore = np.random.random(len(greedy)) < self.epsilon
        actions = np.where(explore, np.random.randint(self.action_dim, size=len(greedy)), greedy)
        self.env_steps += len(actions)
        return actions

    def observe(self, env_index, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self, next_states):
        # Learning on every environment step is the main cost; every few steps trains nearly as well
        self._train_credit += len(next_states)
        while self._train_credit >= self.hparams['train_every']:
            self._train_credit -= self.hparams['train_every']
            self.optimize_model()

    def update_target_net(self):
        self.target_net.load_state_dict(self.model.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return None
        (states, actions, rewards, next_states, dones), indices, weights = \
            self.memory.sample(self.batch_size, self.beta)

        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            # Double DQN: the policy net picks the action, the target net scores it
            best_next = self.model(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_next)
            targets = rewards + self.gamma * next_q * (1 - dones)

        td_errors = self.loss_fn(q_values, targets)
        loss = (td_errors * weights.unsqueeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams['max_grad_norm'])
        self.optimizer.step()
        self.scheduler.step()

        self.memory.update_priorities(indices, (targets - q_values).abs().detach() + 1e-5)

        self.opt_steps += 1
        if self.opt_steps % self.hparams['target_update'] == 0:
            self.update_target_net()
        return loss.item()

    def save_model(self, episode, score, best_score, force=False):
        return checkpoints.save(self.model, self.model_type, episode, score, best_score, force)

    def load_model(self, filename):
        result = checkpoints.load(self.model, filename, self.device)
        self.update_target_net()
        return result
