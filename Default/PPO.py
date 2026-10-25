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


class RolloutBuffer:
    """One environment's trajectory. Plain python/numpy data, so nothing keeps an autograd graph alive."""
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.__init__()


class PPOAgent:
    """PPO agent for the snake, updating every `update_steps` environment steps with GAE.

    Acts on `num_envs` games at once: select_actions() takes one batched forward pass for all of them,
    and each environment keeps its own trajectory so advantages are computed per environment.
    """
    model_type = "ppo"

    def __init__(self, state_dim, action_dim, device='cpu', num_envs=1, **overrides):
        unknown = set(overrides) - set(HPARAMS)
        if unknown:
            raise TypeError(f"Unknown hyperparameters: {sorted(unknown)}")
        self.hparams = {**HPARAMS, **overrides}
        h = self.hparams

        self.device = device
        self.num_envs = num_envs
        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=h['lr'], eps=1e-5, foreach=True)
        self.buffers = [RolloutBuffer() for _ in range(num_envs)]

        # Acting on a handful of states is fastest as plain numpy matrix maths on a CPU copy of the
        # weights (no torch dispatch, no GPU sync); the copy is refreshed after every update.
        self._refresh_rollout_weights()
        self._rng = np.random.default_rng()

        self.gamma = h['gamma']
        self.gae_lambda = h['gae_lambda']
        self.clip_epsilon = h['clip_epsilon']
        self.k_epochs = h['k_epochs']
        self.update_steps = h['update_steps']
        self.minibatch_size = h['minibatch_size']
        self.entropy_coeff = h['entropy_coeff']  # decays per update

    def _refresh_rollout_weights(self):
        w = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.model.state_dict().items()}
        self._w = dict(
            w1=w['shared_fc1.weight'].T.copy(), b1=w['shared_fc1.bias'],
            w2=w['shared_fc2.weight'].T.copy(), b2=w['shared_fc2.bias'],
            wp=w['policy_fc.weight'].T.copy(), bp=w['policy_fc.bias'],
            gp=w['policy_norm.weight'], np_=w['policy_norm.bias'],
            wh=w['policy_head.weight'].T.copy(), bh=w['policy_head.bias'],
            wv=w['value_fc.weight'].T.copy(), bv=w['value_fc.bias'],
            gv=w['value_norm.weight'], nv=w['value_norm.bias'],
            wo=w['value_head.weight'].T.copy(), bo=w['value_head.bias'],
        )

    @staticmethod
    def _layer_norm(x, gamma, beta):
        return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * gamma + beta

    def _forward(self, states):
        """(logits [N, A], values [N]) for a batch of states - the ActorCritic forward pass in numpy."""
        w = self._w
        x = np.asarray(states, dtype=np.float32)
        x = np.maximum(x @ w['w1'] + w['b1'], 0)
        x = np.maximum(x @ w['w2'] + w['b2'], 0)
        p = self._layer_norm(np.maximum(x @ w['wp'] + w['bp'], 0), w['gp'], w['np_'])
        v = self._layer_norm(np.maximum(x @ w['wv'] + w['bv'], 0), w['gv'], w['nv'])
        return p @ w['wh'] + w['bh'], (v @ w['wo'] + w['bo'])[:, 0]

    @staticmethod
    def _softmax(logits):
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return probs / probs.sum(axis=-1, keepdims=True)

    # Common agent interface used by main.py ---------------------------------------------------
    def select_actions(self, states):
        """Sample one action per environment from a batch of states, recording what the update needs."""
        states = np.array(states, dtype=np.float32)  # copy: the caller reuses and overwrites its array
        logits, values = self._forward(states)
        probs = self._softmax(logits)
        draws = self._rng.random((len(states), 1))
        actions = np.minimum((np.cumsum(probs, axis=1) < draws).sum(axis=1), probs.shape[1] - 1)
        logprobs = np.log(probs[np.arange(len(states)), actions] + 1e-12)

        for i, buffer in enumerate(self.buffers):
            buffer.states.append(states[i])
            buffer.actions.append(int(actions[i]))
            buffer.logprobs.append(float(logprobs[i]))
            buffer.values.append(float(values[i]))
        return actions

    def act(self, state, greedy=True):
        """Pick one action for one state without recording anything (used for watching a model play)."""
        logits, _ = self._forward(np.asarray(state, dtype=np.float32)[None])
        if greedy:
            return int(logits[0].argmax())
        return int(self._rng.choice(logits.shape[1], p=self._softmax(logits[0])))

    def observe(self, env_index, state, action, reward, next_state, done):
        """Record the outcome of the action chosen for environment `env_index` by select_actions()."""
        buffer = self.buffers[env_index]
        buffer.rewards.append(reward)
        buffer.dones.append(float(done))

    def update(self, next_states):
        """Called after every batch of environment steps; `next_states` are the environments' current states."""
        if self.should_update():
            self.optimize_model(next_states)

    def should_update(self):
        return sum(len(b) for b in self.buffers) >= self.update_steps

    def optimize_model(self, next_states):
        """Run the PPO update. `next_states` (one per environment) are used to bootstrap unfinished rollouts."""
        h = self.hparams
        last_values = self._forward(np.asarray(next_states, dtype=np.float32))[1]

        all_states, all_actions, all_logprobs, all_advantages, all_returns = [], [], [], [], []
        for buffer, last_value in zip(self.buffers, last_values):
            n = len(buffer)
            if n == 0:
                continue
            rewards = np.asarray(buffer.rewards, dtype=np.float32)
            dones = np.asarray(buffer.dones, dtype=np.float32)
            values = np.asarray(buffer.values, dtype=np.float32)

            # Generalised advantage estimation; a finished episode (done) has nothing to bootstrap from
            advantages = np.zeros(n, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(n)):
                next_value = last_value if t == n - 1 else values[t + 1]
                nonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
                advantages[t] = gae

            all_states.extend(buffer.states)
            all_actions.extend(buffer.actions)
            all_logprobs.extend(buffer.logprobs)
            all_advantages.append(advantages)
            all_returns.append(advantages + values)
            buffer.clear()

        n = len(all_states)
        if n == 0:
            return

        states = torch.as_tensor(np.stack(all_states), device=self.device)
        actions = torch.as_tensor(all_actions, dtype=torch.int64, device=self.device)
        old_logprobs = torch.as_tensor(all_logprobs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.concatenate(all_advantages), device=self.device)
        returns = torch.as_tensor(np.concatenate(all_returns), device=self.device)
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stop_early = False
        for _ in range(self.k_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]

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
                    approx_kl = ((ratios - 1) - log_ratio).mean().item()
                if approx_kl > 1.5 * h['target_kl']:
                    stop_early = True
                    break
            if stop_early:
                break

        self.entropy_coeff = max(h['entropy_min'], self.entropy_coeff * h['entropy_decay'])
        for group in self.optimizer.param_groups:
            group['lr'] = max(h['lr_min'], group['lr'] * h['lr_decay'])
        self._refresh_rollout_weights()

    def get_weights(self):
        """The network weights as a dict of numpy arrays (cheap to send between processes)."""
        return {k: v.detach().cpu().numpy().copy() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in weights.items()})
        self._refresh_rollout_weights()

    def save_model(self, episode, score, best_score, force=False):
        """Save the latest model every `config.SAVE_EVERY` episodes (or when forced) and keep a single best.

        `score` is compared with `best_score`; returns the (possibly updated) best score.
        """
        return checkpoints.save(self.model, self.model_type, episode, score, best_score, force)

    def load_model(self, filename):
        """Load a checkpoint; returns (episode, best_score)."""
        result = checkpoints.load(self.model, filename, self.device)
        self._refresh_rollout_weights()
        return result
