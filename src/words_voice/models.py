import random
import numpy as np
import torch
from torch import nn


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_size, classes, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(.2))
        self.heads = nn.ModuleDict({task: nn.Linear(hidden, len(labels)) for task, labels in classes.items()})

    def forward(self, features):
        h = self.shared(features)
        return {task: head(h) for task, head in self.heads.items()}


def masked_task_loss(logits, labels):
    losses = []
    for task, values in logits.items():
        mask = labels[task] >= 0
        if mask.any():
            losses.append(nn.functional.cross_entropy(values[mask], labels[task][mask]))
    if not losses:
        return sum(v.sum() * 0 for v in logits.values())
    return torch.stack(losses).mean()


def pad_audio(waveforms):
    lengths = torch.tensor([len(y) for y in waveforms])
    values = nn.utils.rnn.pad_sequence([torch.as_tensor(y) for y in waveforms], batch_first=True)
    mask = torch.arange(values.shape[1])[None, :] < lengths[:, None]
    return values, mask, lengths
