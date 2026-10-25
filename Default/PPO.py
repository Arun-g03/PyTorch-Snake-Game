import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config  # repo root is put on sys.path by main.py
import checkpoints

MODEL_DIR = str(config.MODEL_DIR)

# Argument validation in torch.distributions re-checks every tensor on every call; our inputs are
# always valid, so skip it (a measurable cost in the minibatch loop).
torch.distributions.Distribution.set_default_validate_args(False)

# All tunable hyperparameters live in config.py (override per run via PPOAgent(..., **overrides))
HPARAMS = dict(config.PPO_HPARAMS)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, shared1=None, shared2=None, head=None):
        super().__init__()
        shared1 = shared1 or config.PPO_NETWORK['shared1']
        shared2 = shared2 or config.PPO_NETWORK['shared2']
        head = head or config.PPO_NETWORK['head']

        # Shared feature extraction
        self.shared_fc1 = nn.Linear(input_dim, shared1)
        self.shared_fc2 = nn.Linear(shared1, shared2)

        # Policy branch
        self.policy_fc = nn.Linear(shared2, head)
        self.policy_head = nn.Linear(head, action_dim)
        self.policy_norm = nn.LayerNorm(head)

        # Value branch
        self.value_fc = nn.Linear(shared2, head)
        self.value_head = nn.Linear(head, 1)
        self.value_norm = nn.LayerNorm(head)

        # Orthogonal init: small policy head keeps the initial policy near-uniform
        for layer in (self.shared_fc1, self.shared_fc2, self.policy_fc, self.value_fc):
            nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))

        policy_x = self.policy_norm(torch.relu(self.policy_fc(x)))
        logits = self.policy_head(policy_x)

        value_x = self.value_norm(torch.relu(self.value_fc(x)))
        value = self.value_head(value_x)

        return logits, value


class PPOAgent:
    """PPO agent for the snake: GAE advantages, minibatch updates, clipped objective.

    Works on a batch of games at once. All rollout data stays on `device` as [steps, games, ...]
    tensors, so with a GPU environment nothing is copied to the CPU between updates.
    """
    model_type = "ppo"

    def __init__(self, state_dim, action_dim, device='cpu', num_envs=1, **overrides):
        unknown = set(overrides) - set(HPARAMS)
        if unknown:
            raise TypeError(f"Unknown hyperparameters: {sorted(unknown)}")
        self.hparams = {**HPARAMS, **overrides}
        h = self.hparams

        self.device = torch.device(device)
        self.num_envs = num_envs
        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=h['lr'], eps=1e-5, foreach=True)

        self.gamma = h['gamma']
        self.gae_lambda = h['gae_lambda']
        self.clip_epsilon = h['clip_epsilon']
        self.k_epochs = h['k_epochs']
        self.rollout_steps = h['rollout_steps']
        self.entropy_coeff = h['entropy_coeff']  # decays per update
        self._clear_rollout()

    def _clear_rollout(self):
        self._states, self._actions, self._logprobs = [], [], []
        self._values, self._rewards, self._dones = [], [], []

    # Common agent interface used by main.py ---------------------------------------------------
    @torch.no_grad()
    def select_actions(self, states):
        """Sample one action per game from a batch of states [N, D], recording what the update needs."""
        logits, values = self.model(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        actions = torch.multinomial(log_probs.exp(), 1)[:, 0]
        self._states.append(states)
        self._actions.append(actions)
        self._logprobs.append(log_probs.gather(1, actions[:, None])[:, 0])
        self._values.append(values[:, 0])
        return actions

    @torch.no_grad()
    def act(self, state, greedy=True):
        """Pick one action for one state without recording anything (used for watching a model play)."""
        logits, _ = self.model(torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1))
        if greedy:
            return int(logits.argmax())
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1))

    def observe(self, states, actions, rewards, next_states, dones):
        """Record the outcome of the actions chosen by the last select_actions() call."""
        self._rewards.append(rewards)
        self._dones.append(dones.float())

    def update(self, next_states):
        """Called after every step of all games; `next_states` are the games' current states."""
        if len(self._rewards) >= self.rollout_steps:
            self.optimize_model(next_states)

    @torch.no_grad()
    def _advantages(self, next_states):
        """GAE for every game at once. Returns (advantages, returns), both [T, N]."""
        values = torch.stack(self._values)
        rewards = torch.stack(self._rewards)
        dones = torch.stack(self._dones)
        next_value = self.model(next_states)[1][:, 0]  # bootstrap value after the last step

        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(next_value)
        for t in reversed(range(len(values))):
            nonterminal = 1.0 - dones[t]  # a finished episode has nothing to bootstrap from
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values[t]
        return advantages, advantages + values

    def optimize_model(self, next_states):
        """Run the PPO update on the collected rollout."""
        h = self.hparams
        advantages, returns = self._advantages(next_states)

        state_dim = self._states[0].shape[-1]
        states = torch.stack(self._states).reshape(-1, state_dim)
        actions = torch.stack(self._actions).reshape(-1)
        old_logprobs = torch.stack(self._logprobs).reshape(-1)
        advantages, returns = advantages.reshape(-1), returns.reshape(-1)
        self._clear_rollout()

        n = len(states)
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        minibatch_size = max(1, n // h['num_minibatches'])

        stop_early = False
        for _ in range(self.k_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                idx = perm[start:start + minibatch_size]

                logits, new_values = self.model(states[idx])
                dist = torch.distributions.Categorical(logits=logits)
                new_logprobs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_logprobs - old_logprobs[idx]
                ratios = torch.exp(log_ratio)
                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(new_values.squeeze(1), returns[idx])

                loss = actor_loss + h['value_coeff'] * critic_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), h['max_grad_norm'])
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratios - 1) - log_ratio).mean()
                if approx_kl > 1.5 * h['target_kl']:  # one sync per minibatch
                    stop_early = True
                    break
            if stop_early:
                break

        self.entropy_coeff = max(h['entropy_min'], self.entropy_coeff * h['entropy_decay'])
        for group in self.optimizer.param_groups:
            group['lr'] = max(h['lr_min'], group['lr'] * h['lr_decay'])

    def get_weights(self):
        """The network weights as a dict of numpy arrays (cheap to send between processes)."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()})

    def save_model(self, episode, score, best_score, force=False):
        """Save the latest model every `config.SAVE_EVERY` episodes (or when forced) and keep a single best.

        `score` is compared with `best_score`; returns the (possibly updated) best score.
        """
        return checkpoints.save(self.model, self.model_type, episode, score, best_score, force)

    def load_model(self, filename):
        """Load a checkpoint; returns (episode, best_score)."""
        return checkpoints.load(self.model, filename, self.device)
