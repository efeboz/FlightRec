"""General utilities."""

import random
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor


def pick_device(requested: str | torch.device = "auto") -> torch.device:
    """Choose CUDA, then MPS, then CPU when *requested* is ``"auto"``."""
    if str(requested) != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def flatten_tensors(tensors: Iterable[Tensor]) -> Tensor:
    """Flatten and concatenate tensors, preserving autograd connectivity."""
    values = list(tensors)
    if not values:
        return torch.empty(0)
    return torch.cat([value.reshape(-1) for value in values])


def unflatten_vector(vector: Tensor, references: Sequence[Tensor]) -> list[Tensor]:
    """View a flat vector with the shapes of *references*."""
    result: list[Tensor] = []
    offset = 0
    for reference in references:
        size = reference.numel()
        result.append(vector[offset : offset + size].view_as(reference))
        offset += size
    if offset != vector.numel():
        raise ValueError(f"vector has {vector.numel()} elements; expected {offset}")
    return result


def move_batch(batch: object, device: torch.device | str) -> object:
    """Recursively move tensors in a nested batch to a device."""
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch(value, device) for value in batch]
    if isinstance(batch, dict):
        return {key: move_batch(value, device) for key, value in batch.items()}
    return batch
