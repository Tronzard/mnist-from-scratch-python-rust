"""Command-line trainer. All metrics come from actual runs, not placeholders."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from .dataset import load_split
from .network import MLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MNIST MLP using NumPy backpropagation")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/python.json"))
    args = parser.parse_args()
    
    if min(args.samples, args.val_size, args.hidden, args.epochs, args.batch_size) <= 0 or args.lr <= 0:
        parser.error("Sample counts, sizes, epochs and learning rate must be positive")

    x_all, y_all = load_split(args.data, "train")
    x_test, y_test = load_split(args.data, "test")
    if args.samples + args.val_size > len(x_all):
        parser.error("Training samples + held-out validation samples must be <= 60,000")
    x_train, y_train = x_all[:args.samples], y_all[:args.samples]
    x_val, y_val = x_all[-args.val_size:], y_all[-args.val_size:]
    rng = np.random.default_rng(args.seed)
    model = MLP(hidden_dim=args.hidden, seed=args.seed)
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(args.samples)
        weighted_loss = 0.0
        for begin in range(0, args.samples, args.batch_size):
            indices = order[begin:begin + args.batch_size]
            weighted_loss += model.train_batch(x_train[indices], y_train[indices], args.lr) * len(indices)
        val_acc = model.accuracy(x_val, y_val)
        print(f"epoch {epoch}/{args.epochs} | loss={weighted_loss/args.samples:.4f} | val_acc={val_acc:.2%}", flush=True)
    elapsed = perf_counter() - started
    test_acc = model.accuracy(x_test, y_test)
    result = {
        "language": "Python", "implementation": "NumPy vectorized (CPU)",
        "samples": args.samples, "validation_samples": args.val_size,
        "hidden": args.hidden, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "seed": args.seed,
        "train_seconds": round(elapsed, 4), "val_accuracy": round(val_acc, 6),
        "test_accuracy": round(test_acc, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"test_acc={test_acc:.2%} | training={elapsed:.2f}s | saved {args.output}")


if __name__ == "__main__":
    main()
