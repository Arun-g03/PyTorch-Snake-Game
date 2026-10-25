"""Checkpoint naming, saving and loading shared by all agent types."""
import os

import torch

import config


def checkpoint_name(kind, model_type):
    """e.g. ptorch_latest_ppo.pth - the model type is the suffix."""
    return f"{config.PYTORCH_PREFIX}_{kind}_{model_type}.pth"


def list_checkpoints(model_type):
    """Checkpoint filenames in the model folder that belong to `model_type`."""
    if not os.path.isdir(config.MODEL_DIR):
        return []
    prefix, suffix = f"{config.PYTORCH_PREFIX}_", f"_{model_type}.pth"
    return sorted(f for f in os.listdir(config.MODEL_DIR) if f.startswith(prefix) and f.endswith(suffix))


def _write(model, filename, episode, best_score):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "episode": episode, "best_score": best_score},
               os.path.join(config.MODEL_DIR, filename))


def save(model, model_type, episode, score, best_score, force=False):
    """Write `latest` every config.SAVE_EVERY episodes (or when forced), and `best` on a new best score."""
    if episode % config.SAVE_EVERY != 0 and not force:
        return best_score

    if score > best_score:
        best_score = score
        _write(model, checkpoint_name("best", model_type), episode, best_score)
        print(f"New best model saved (score {best_score:.2f})")

    _write(model, checkpoint_name("latest", model_type), episode, best_score)
    print(f"Latest model saved: {checkpoint_name('latest', model_type)}")
    return best_score


def load(model, filename, device):
    """Load weights into `model`; returns (episode, best_score). Plain state_dicts load as episode 0."""
    checkpoint = torch.load(filename, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        episode, best_score = checkpoint["episode"], checkpoint["best_score"]
    else:
        model.load_state_dict(checkpoint)
        episode, best_score = 0, -float('inf')
    print(f"Model loaded from {filename}")
    return episode, best_score
