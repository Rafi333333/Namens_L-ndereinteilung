"""Simple character-level RNN for ASCII name classification.

The module keeps things framework-ready:
- Convert raw ASCII names to padded tensor batches.
- Define a GRU-based encoder with a linear classifier head.
- Provide a tiny `predict` helper for inference on a single name.

Intended usage:
>>> model = NameRNN(num_classes=18)  # e.g., nationalities
>>> logits = model(batch_tensor, lengths)  # training
>>> pred = predict(model, "Müller".encode("ascii", errors="ignore").decode())

This file is intentionally self-contained; plug your own Dataset/Dataloader
and training loop around it.
"""

from __future__ import annotations

import string
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Restrict to printable ASCII; unknown chars map to the PAD token.
ASCII_CHARS = string.ascii_letters + " .,'-"
PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
BASE_OFFSET = 3  # offset applied to ASCII indices so we leave space for special tokens


def build_vocab() -> Tuple[str, dict, dict]:
    """Return vocab string and lookup maps."""
    vocab = "\0" + "\x7f" + "\x01"  # placeholders for PAD/UNK/START when inspecting
    vocab += ASCII_CHARS
    stoi = {ch: i + BASE_OFFSET for i, ch in enumerate(ASCII_CHARS)}
    stoi["<pad>"] = PAD_IDX
    stoi["<unk>"] = UNK_IDX
    stoi["<start>"] = START_IDX
    itos = {i: ch for ch, i in stoi.items()}
    return vocab, stoi, itos

VOCAB, STOI, ITOS = build_vocab()
VOCAB_SIZE = len(VOCAB) + BASE_OFFSET  # includes special tokens

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
        dir_mult = 2 if bidirectional else 1
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * dir_mult, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        tokens: LongTensor of shape (batch, seq_len)
        lengths: LongTensor of actual sequence lengths
        returns: logits of shape (batch, num_classes)
        """
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, hidden = self.rnn(packed)

        if self.rnn.bidirectional:
            # hidden shape: (num_layers*2, batch, hidden_size)
            hidden_cat = torch.cat(
                (hidden[-2], hidden[-1]), dim=-1
            )  # last layer forward/backward
            final = hidden_cat
        else:
            final = hidden[-1]  # last layer, forward direction

        logits = self.classifier(final)
        return logits