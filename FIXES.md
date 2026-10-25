# Suggested Fixes

> **Status: all items below have been applied.** (The `Raspberry_snake/` TensorFlow variant they mention has since been removed; its items no longer apply.) The game logic was rewritten as a `SnakeGame` class (relative actions, grid coordinates, rays for wall/body/food) in both variants. Notable changes versus the list: the single best model is now `models/ppo_model_best.pth` (judged on rolling-average reward), checkpoints store `episode` and `best_score`, and PPO updates every 2048 steps with GAE instead of once per episode.

Issues found by reading the code in `Default/` (PyTorch PPO) and `Raspberry_snake/` (TensorFlow DQN), ordered by severity.

## 🔴 Bugs

### 1. Best model filename is never formatted — `Default/PPO.py:172`
```python
best_filename = "ppo_model_best_{best_reward}.pth"   # missing f-prefix
```
Every new best overwrites the same literal file `ppo_model_best_{best_reward}.pth` (which is also tracked in git). Use `f"ppo_model_best_{reward:.0f}.pth"` and, if only one best should exist, delete the previous one.

### 2. Wall collision is detected one step late — `Default/main.py:191-194` (and `Raspberry_snake/main.py:129-131`)
The bounds check runs *before* `x1 += x1_change`, so the snake steps out of the arena and is only flagged dead on the next iteration. The check at line 216 (`is_wall_collision`) then recomputes using the *new* position, so the reward/`done` flags can disagree with `game_close`. Move the bounds check to after the position update and compute `is_wall_collision` / `is_self_collision` once, reusing them for `game_close`.

### 3. Death penalty can be skipped — `Default/snake_game.py:58-60`
`calculate_reward` only returns `-500` if `game_close` *and* (wall or self collision). Combined with bug #2, a step where `game_close` is set by the stale check but `is_wall_collision` is false falls through to the survival reward (`+2`). Simplify to `if game_close: return -500`.

### 4. Eating food and dying on the same step — `Default/snake_game.py:62`
Food reward is checked after death, fine, but food respawn/length increase in `main.py:228-231` happens even if `game_close` is true. Guard with `if not game_close`.

### 5. PPO bootstraps nothing across the rollout boundary — `Default/PPO.py:110-118`
`optimize_model()` is called once per episode, so the final `done=1` makes this valid, but `discounted_reward` is a tensor and `returns` is built via `list.insert(0, …)` (O(n²), and `torch.tensor(list_of_tensors)` is slow/warns). Build a preallocated tensor and fill it in reverse. Also compute advantages with GAE (λ≈0.95) instead of `returns - values` for lower variance.

### 6. Rollout stores tensors with graph attached — `Default/PPO.py:78-92`
`select_action` runs the model with gradients enabled and stores `logprob`/`value` tensors (still attached to the graph) in the buffer. Wrap in `torch.no_grad()` and store detached values — this leaks memory across long episodes.

### 7. `extract_episode_number` returns `None` for PPO models — `Default/main.py:20-24, 64`
PPO saves `ppo_model_latest.pth`, which doesn't match `model_ep(\d+)_`, so `episode = None` and the next `episode % 50` in `save_model` raises `TypeError`. Fall back to `0` (`return 0`), or store the episode in the checkpoint.

### 8. Un-indented method is dead code — `Raspberry_snake/agent_tf.py:55-77`
`optimize_model` is nested *inside* `update_target_net`, so `agent.optimize_model(beta)` in `main.py:161` raises `AttributeError`. De-indent it to class level. Related issues once fixed:
- `loss_fn` with `Reduction.NONE` on `(B,1)` tensors returns shape `(B,)`, but `weights` is `(B,)` and `priorities` should use `abs(td_error)`, not the Huber loss value — compute `td = target - q` separately.
- `action_batch` is `int64` shape `(B,1)`; `tf.one_hot` yields `(B,1,A)`, so the reduce_sum shape is wrong. Squeeze first.
- `target_net` is never refreshed during training (`target_update` is unused). Call `update_target_net()` every N steps.

### 9. State dimension mismatch risk — `Raspberry_snake/main.py:20`
`state_dim = 15` is hard-coded and `Default/main.py:27` uses `19`. Derive it from `len(get_state(...)[0])` instead of hard-coding, so changes to `get_state` can't silently break the network input.

### 10. Rolling-window plot lists can go out of sync — `Default/main.py:117-129`
`food_avg_data` / `reward_avg_data` are popped before the new average is appended, so the "rolling average" is the mean over a window that has been trimmed, not a true rolling mean, and lengths briefly differ from `xdata`. Use `collections.deque(maxlen=…)` for the raw data and compute the average from it.

## 🟠 Reinforcement-learning quality

- **Reward shaping is lopsided** (`snake_game.py:57-72`): `+2` per step just for surviving encourages circling forever; `+10` for approaching and `-5` for retreating is asymmetric, so oscillating towards/away from food nets positive reward. Make closer/farther symmetric (e.g. ±1) and keep food (+10…+20) and death (−10) on a similar scale. Huge rewards (`-500`, `150`) hurt value-function learning; normalise rewards or scale down.
- **Add a step limit per episode** (e.g. `100 * length`) so a looping snake terminates instead of hanging training.
- **State lacks body awareness and the 180° rule**: raycasts treat any snake cell as an obstacle but direction is encoded as sign only; consider a one-hot of the current heading plus relative (forward/left/right) actions, which removes the illegal-reverse no-op branches (`main.py:178-189`, where reversing silently keeps the old direction).
- **Raycast reports distance to wall, not distance to body, in a single channel**, and `distance` stops at the first food/body hit with no way to tell which. Add separate channels for wall / body / food.
- **Entropy coefficient** (`PPO.py:73, 149`) decays per *episode* from 0.05 to 0.01; with `clip_epsilon=0.1` and a single update per episode this is slow-learning. Consider batching several episodes (e.g. 2048 steps) per update.
- **`random.seed(seed)` each episode** (`main.py:157`) makes food spawns deterministic and repeat identically on every run, so the agent can memorise the first N food positions. Seed once, or use a seed pool for evaluation only.
- **Unused / misleading values**: `total_msa` is always `0` and `avg_msa` is printed (`main.py:164, 240`); `food_visible` gating for distance reward means no shaping when food is out of ray range.

## 🟡 Code quality

- **Duplicate imports** in `Default/PPO.py:1-9` (`torch`, `torch.nn` imported twice). `sys`, `DQN`, `DQNAgent`, `quit` unused in `Default/main.py`.
- **`from snake_game import *`** hides where `random`, `pygame`, `dis`, `clock`, etc. come from (`main.py` relies on `random` via star import). Use explicit imports.
- **`snake_game.py` opens a pygame window at import time** (`pygame.display.set_mode` at module level) even for headless training. Move initialisation into a function and use `SDL_VIDEODRIVER=dummy` when `Should_Render` is False.
- **Unused code**: `x_direction`, `y_direction`, `Direction`, `get_grid_cells`, `saved_models` in `PPOAgent`, `timestamp` in `PPO.save_model`, `target_update`/`steps_done` in the DQN agents, `import math`.
- **Spam on every episode**: `main.py:256-257` prints the whole snake body each episode; remove or gate behind a debug flag.
- **`game_loop` runs forever** with no way to stop except closing the window (and the `quit()` there is the site builtin). Add `max_episodes` / `KeyboardInterrupt` handling that saves the latest model, and use `sys.exit()`.
- **Raycast uses `[x, y] in snake_list`** (O(n) list scan per cell per ray). Use a `set` of tuples for the body.
- **Menu input** (`main.py:61`) crashes on non-numeric input (`int(input(...))`). Wrap in `try/except ValueError`.
- **Inconsistent type hints and docstrings**; add `requirements.txt` with pinned `torch`, `pygame`, `numpy`, `matplotlib` (and `tensorflow` for the Pi variant).
- **README is nearly empty**: add install, how to train (`python Default/main.py`), how to load a model, and a description of state/reward/actions.

## 🟢 Repo hygiene

- **`models/*.pth` are tracked despite `*.pth` in `.gitignore`**, and `ppo_model_latest.pth` shows as modified in `git status`. Run `git rm --cached models/*.pth` (the literal-named best file included), and keep a `models/.gitkeep` if you want the folder.
- **Committed `__pycache__/`** directories in `Default/` and `Raspberry_snake/` despite the ignore rule: `git rm -r --cached Default/__pycache__ Raspberry_snake/__pycache__`.
- Training history CSV (`training_history.csv`) is ignored by `*.csv`; fine, but document it.
- The `Raspberry_snake` folder name is misleading — it is a TensorFlow DQN variant, and it is currently broken (see #8).

## Suggested order of work

1. Fix #1, #7, #8 (crashes / broken files) — small, safe changes.
2. Fix #2–#4 (game-logic correctness) and re-train, since the reward signal changes.
3. Fix #5–#6 (PPO correctness/efficiency), then rework reward shaping.
4. Clean up repo hygiene and docs.
