from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloder import Datas, Langs
from model import PAD_IDX, START_IDX, STOI, UNK_IDX, NameRNN


def encode_name(name: str) -> list[int]:
    tokens = [START_IDX]
    tokens.extend(STOI.get(ch, UNK_IDX) for ch in name)
    return tokens if len(tokens) > 1 else [START_IDX, UNK_IDX]


def load_samples() -> tuple[list[tuple[list[int], int]], list[str]]:
    labels = [Path(lang).stem for lang in Langs]
    samples: list[tuple[list[int], int]] = []
    for name, onehot in Datas:
        if 1 not in onehot:
            continue
        samples.append((encode_name(name), onehot.index(1)))

    if not samples:
        raise ValueError("Keine Namen in Datas gefunden.")
    return samples, labels


def split_samples(
    samples: list[tuple[list[int], int]], val_ratio: float, seed: int
) -> tuple[list[tuple[list[int], int]], list[tuple[list[int], int]]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_size = int(len(shuffled) * val_ratio)
    val_size = max(0, min(val_size, max(0, len(shuffled) - 1)))
    return shuffled[val_size:], shuffled[:val_size]


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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=Path("name_rnn.pt"))
    parser.add_argument("--plot-path", type=Path, default=Path("training_plot.png"))
    return parser.parse_args()


def plot_history(history: dict[str, list[float]], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plot uebersprungen: matplotlib nicht installiert.")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    if history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    if history["val_acc"]:
        axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    print(f"Plot gespeichert: {plot_path}")


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, labels = load_samples()
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)

    train_loader = DataLoader(
        train_samples,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = None
    if val_samples:
        val_loader = DataLoader(
            val_samples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )

    model = NameRNN(num_classes=len(labels)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Device: {device}")
    print(
        f"Samples: train={len(train_samples)}"
        + (f", val={len(val_samples)}" if val_samples else ", val=0 (aus)")
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        if val_loader is not None:
            with torch.inference_mode():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
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
        "model_config": {"num_classes": len(labels)},
    }
    torch.save(checkpoint, args.save_path)
    print(f"Modell gespeichert: {args.save_path}")
    plot_history(history, args.plot_path)


if __name__ == "__main__":
    main()
