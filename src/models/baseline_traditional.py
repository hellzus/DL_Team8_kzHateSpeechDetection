import torch
import torch.nn as nn
from typing import Any


class MLPClassifier(nn.Module):
    """
    Simple 3-layer MLP for TF-IDF features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Any:
        return self.model(x)
