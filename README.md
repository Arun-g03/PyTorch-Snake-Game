# PyTorch Snake Game

A snake game that teaches itself to play with reinforcement learning.

- `Default/` – PyTorch PPO agent (GAE, minibatch updates) with an optional pygame view and a live matplotlib training plot.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-pytorch.txt
```

## Configuration

All settings (board size, rewards, hyperparameters, model paths and prefixes, plotting) live in [`config.py`](config.py). Edit it directly; there is no `.env` file.

## Train

```bash
cd Default
python main.py
```

Choose **1** to train a new model, **2** to continue a saved one, or **3** to watch a saved model play. Training asks whether to show the game window; it is headless by default. Press `Ctrl+C` to stop – the latest model is saved first.

Training plays `NUM_ENVS` games at once (default 8) and chooses all their moves in one batched forward pass; showing the game window (or watching a model) uses a single game. Pick the agent with `MODEL_TYPE = "ppo"` or `"dqn"` in `config.py`. Checkpoints are suffixed with the model type: `models/ptorch_latest_<type>.pth` (every 50 episodes and on exit) and `models/ptorch_best_<type>.pth` (best rolling-average reward). The load menu only lists checkpoints for the current `MODEL_TYPE`. Older checkpoints (e.g. `ptorch_ppo_latest.pth`) use the previous naming and a different network shape, so they won't be listed or load.

## How it works

- **Actions (3):** go straight, turn right, turn left – the snake can never reverse into itself.
- **State (44 values):** for 8 rays: distance to wall, distance to body, food closeness; one-hot heading and normalised length; for each move (straight / right / left): wall ahead, own body ahead and free space reachable (trap detection); and the food's position relative to the head.
- **Rewards:** +10 food, −10 death or stalling (no food for `100 × length` steps), ±0.1 for moving closer to / farther from food.

## Running on a Raspberry Pi

Use the same code with a lighter setup in `config.py`: set `DEVICE = "cpu"`, shrink `PPO_NETWORK` (e.g. 64 / 32 / 16) and lower `PPO_HPARAMS["update_steps"]`. Train headless (answer **n** to the window prompt).
