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

Choose **1** to train a new model, **2** to continue a saved one, **3** to watch a saved model play, or **4** to evolve a population of PPO agents in parallel. Training asks whether to show the game window; it is headless by default. Press `Ctrl+C` to stop – the latest model is saved first.

Training simulates `NUM_ENVS` games at once as tensors (`Default/vec_game.py`) and chooses all their moves in one batched forward pass. With `NUM_ENVS = "auto"` that is 256 games on a GPU and 16 on a CPU; on a GPU the whole loop stays on the device, so more games cost very little. Showing the game window (or watching a model) uses a single game. `python Default/test_vec_game.py` checks that the tensor game behaves exactly like the readable single-game reference in `snake_game.py`. Pick the agent with `MODEL_TYPE = "ppo"` or `"dqn"` in `config.py`. Checkpoints are suffixed with the model type: `models/ptorch_latest_<type>.pth` (every 50 episodes and on exit) and `models/ptorch_best_<type>.pth` (best rolling-average reward). The load menu only lists checkpoints for the current `MODEL_TYPE`. Older checkpoints (e.g. `ptorch_ppo_latest.pth`) use the previous naming and a different network shape, so they won't be listed or load.

### Evolution (option 4)

Several PPO agents train in separate processes. After each generation the fittest two are crossed over and mutated to form the next generation (the best agent is kept unchanged, and an agent that stalls restarts fresh). Only a new overall best is saved, to `models/ptorch_evo_ppo.pth`, which loads like any other PPO checkpoint (options 2 and 3). The population size, generations, episodes and mutation settings are the `EVO_*` values in `config.py`.

## How it works

- **Actions (3):** go straight, turn right, turn left – the snake can never reverse into itself.
- **State (44 values):** for 8 rays: distance to wall, distance to body, food closeness; one-hot heading and normalised length; for each move (straight / right / left): wall ahead, own body ahead and free space reachable (trap detection); and the food's position relative to the head.
- **Rewards:** +1 food, −1 death or stalling (no food for `100 × length` steps), ±0.01 for moving closer to / farther from food (the `REWARD_*` values in `config.py`; small rewards train about twice as fast). A checkpoint remembers the rewards it was trained with, and loading it under different ones prints a warning.

## Running on a Raspberry Pi

Use the same code with a lighter setup in `config.py`: set `DEVICE = "cpu"`, shrink `PPO_NETWORK` (e.g. 64 / 32 / 16) and lower `PPO_HPARAMS["rollout_steps"]`. Train headless (answer **n** to the window prompt).
