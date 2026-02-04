from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import PAD_IDX, START_IDX, STOI, UNK_IDX, NameRNN


def to_ascii(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_name.strip().split())


def encode_name(name: str) -> list[int]:
    tokens = [START_IDX]
    tokens.extend(STOI.get(ch, UNK_IDX) for ch in to_ascii(name))
    return tokens if len(tokens) > 1 else [START_IDX, UNK_IDX]


class NameDataset(Dataset):
    def __init__(self, samples: list[tuple[list[int], int]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


def load_samples(data_dir: Path) -> tuple[list[tuple[list[int], int]], list[str]]:
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Keine Label-Dateien gefunden in: {data_dir}")

    labels = [file.stem for file in files]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    samples: list[tuple[list[int], int]] = []
    for file in files:
        label = file.stem
        label_idx = label_to_idx[label]
        for line in file.read_text(encoding="utf-8").splitlines():
            name = line.strip()
            if not name:
                continue
            samples.append((encode_name(name), label_idx))

    if not samples:
        raise ValueError(f"Keine Namen in den Dateien unter: {data_dir}")
    return samples, labels


def split_samples(
    samples: list[tuple[list[int], int]], val_ratio: float, seed: int
) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    if 0 < val_size < len(shuffled):
        return shuffled[val_size:], shuffled[:val_size]
    return shuffled, []


def collate_batch(
    batch: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(tokens) for tokens, _ in batch], dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    tokens = torch.full((len(batch), max_len), PAD_IDX, dtype=torch.long)
    for row, (seq, _) in enumerate(batch):
        tokens[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return tokens, lengths, labels


def run_epoch(
    model: NameRNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for tokens, lengths, labels in loader:
        tokens = tokens.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(tokens, lengths)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += batch_size

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_correct / total_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRU for name origin classification.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/names"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--save-path", type=Path, default=Path("name_rnn.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, labels = load_samples(args.data_dir)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    train_loader = DataLoader(
        NameDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        NameDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    model = NameRNN(
        num_classes=len(labels),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(
        f"Samples: train={len(train_samples)}"
        + (f", val={len(val_samples)}" if val_samples else ", val=0 (aus)")
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        if val_loader.dataset:
            with torch.inference_mode():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "model_config": {
            "num_classes": len(labels),
            "embed_dim": args.embed_dim,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
        },
    }
    torch.save(checkpoint, args.save_path)
    print(f"Modell gespeichert: {args.save_path}")


if __name__ == "__main__":
    main()
