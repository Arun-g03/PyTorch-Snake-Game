"""
Central configuration for the PyTorch snake (Default/).

Edit values here instead of hunting through the code. Nothing in this file is secret, so it is
committed to git (no .env needed).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = ROOT / "models"                           # shared by both frameworks, told apart by prefix

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
DIS_WIDTH = 600          # window / board size in pixels
DIS_HEIGHT = 600
CELL_SIZE = 20           # board is (DIS_WIDTH / CELL_SIZE) x (DIS_HEIGHT / CELL_SIZE) cells
SNAKE_SPEED = 30         # frames per second when rendering (PyTorch version)
STEP_LIMIT_FACTOR = 200  # an episode ends after this many steps per snake segment without food

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0     # also used when the snake stalls (hits the step limit)
REWARD_CLOSER = 0.1      # moved closer to the food
REWARD_FARTHER = -0.1    # moved farther from the food

# ---------------------------------------------------------------------------
# PyTorch agents (Default/)
# ---------------------------------------------------------------------------
MODEL_TYPE = "ppo"          # "ppo" or "dqn" - which agent to train / watch
PYTORCH_PREFIX = "ptorch"   # checkpoints are saved as <prefix>_<latest|best>_<MODEL_TYPE>.pth
NUM_ENVS = 5                # games simulated in parallel while training: a number, or "auto" (256 on a GPU, 16 on a CPU)
DEVICE = "auto"             # "auto" (use CUDA if available), "cuda" or "cpu"

SAVE_EVERY = 50             # episodes between "latest" checkpoints
PRINT_SECONDS = 2.0          # how often the console shows a progress line
PLOT_SECONDS = 2.0           # redraw the training plot at most this often (redraws are slow)
AVG_WINDOW = 100            # episodes in the rolling average (also used to pick the best model)
PLOT_WINDOW = 400          # episodes kept on the plot

# PPO network size. Shrink these (e.g. 64 / 32 / 16) for a light model on a Raspberry Pi; combine with
# DEVICE = "cpu" and a smaller PPO_HPARAMS["update_steps"]. Saved checkpoints only load into a
# network of the same size.
PPO_NETWORK = dict(
    shared1=256,
    shared2=128,
    head=64,
)

# --- PPO ---
PPO_HPARAMS = dict(
    gamma=0.99,             # discount factor
    gae_lambda=0.95,        # GAE bias/variance trade-off
    lr=3e-4,                # initial learning rate
    lr_min=3e-5,            # learning rate floor
    lr_decay=0.995,         # lr multiplier after every update
    clip_epsilon=0.2,       # PPO clip range
    k_epochs=10,            # passes over each rollout (cut short by target_kl)
    target_kl=0.02,         # stop the epochs early if the policy moved this far
    rollout_steps=128,      # steps each game plays between updates (batch = rollout_steps x NUM_ENVS)
    num_minibatches=8,      # minibatches per epoch, so the batch size scales with NUM_ENVS
    value_coeff=0.5,
    entropy_coeff=0.05,     # initial entropy bonus
    entropy_min=0.01,
    entropy_decay=0.99,     # multiplier after every update
    max_grad_norm=0.5,
)

# --- DQN (double DQN with prioritised replay) ---
DQN_NETWORK = (256, 128, 64)  # hidden layer sizes

DQN_HPARAMS = dict(
    gamma=0.99,
    lr=1e-3,
    train_every=4,            # environment steps per optimisation step (higher = faster, less sample-efficient)
    lr_step_size=2500,        # optimisation steps between learning rate decays
    lr_gamma=0.9,
    memory_size=50000,        # replay buffer capacity
    batch_size=64,
    max_updates_per_step=4,   # cap on optimisation steps per batch of games (keeps huge batches fast)
    target_update=250,        # optimisation steps between target network syncs
    epsilon_start=1.0,        # exploration rate, decayed linearly per environment step
    epsilon_end=0.02,
    epsilon_decay_steps=50000,
    beta_start=0.4,           # prioritised replay importance-sampling exponent...
    beta_frames=100000,       # ...annealed to 1.0 over this many steps
    max_grad_norm=1.0,
)

# --- Parallel evolution of PPO agents (menu option 4) ---
EVO_AGENTS = 4              # agents trained in parallel, one process each
EVO_GENERATIONS = 10
EVO_EPISODES = 200          # episodes each agent trains per generation
EVO_FITNESS_WINDOW = 20     # fitness = mean score over this agent's last N episodes
EVO_MUTATION_RATE = 0.1     # chance that a weight tensor is mutated
EVO_MUTATION_SCALE = 0.1    # size of the noise added to a mutated tensor
EVO_RESET_AFTER = 100       # episodes without a new best score before an agent restarts fresh (0 = never)
EVO_ENVS = 4                # games each evolving agent plays at once
