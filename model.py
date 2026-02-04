"""Character-level GRU model for ASCII name classification.

This module defines:
- A fixed ASCII vocabulary with special tokens (PAD/UNK/START).
- A GRU encoder with a linear classifier head.

Intended usage:
>>> model = NameRNN(num_classes=18)
>>> logits = model(batch_tensor, lengths)
"""

from __future__ import annotations

import string
from typing import Final

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ASCII_CHARS: Final[str] = string.ascii_letters + " .,'-"

PAD_TOKEN: Final[str] = "<pad>"
UNK_TOKEN: Final[str] = "<unk>"
START_TOKEN: Final[str] = "<start>"

PAD_IDX: Final[int] = 0
UNK_IDX: Final[int] = 1
START_IDX: Final[int] = 2
BASE_OFFSET: Final[int] = 3


def build_vocab() -> tuple[str, dict[str, int], dict[int, str]]:
    """Build ASCII vocabulary and index mappings."""
    stoi: dict[str, int] = {
        PAD_TOKEN: PAD_IDX,
        UNK_TOKEN: UNK_IDX,
        START_TOKEN: START_IDX,
    }
    stoi.update({ch: idx + BASE_OFFSET for idx, ch in enumerate(ASCII_CHARS)})
    itos = {idx: token for token, idx in stoi.items()}
    return ASCII_CHARS, stoi, itos


VOCAB, STOI, ITOS = build_vocab()
VOCAB_SIZE = BASE_OFFSET + len(VOCAB)


class NameRNN(nn.Module):
    """GRU-based encoder with a linear classifier on the final hidden state."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.15,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        direction_multiplier = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * direction_multiplier, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of padded token sequences."""
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)

        if self.rnn.bidirectional:
            final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            final_hidden = hidden[-1]

        return self.classifier(final_hidden)
