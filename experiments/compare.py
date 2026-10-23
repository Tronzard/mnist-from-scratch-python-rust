"""Print an honest comparison of two saved measurements."""
import json
from pathlib import Path


def main():
    paths = [Path("results/python.json"), Path("results/rust.json")]
    data = []
    for path in paths:
        if not path.exists():
            print(f"Missing {path}; run both training programs first")
            return
        data.append(json.loads(path.read_text(encoding="utf-8")))
    parameters = ("samples", "validation_samples", "hidden", "epochs", "batch_size", "learning_rate")
    if any(data[0][key] != data[1][key] for key in parameters):
        print("WARNING: hyperparameters/data split differ; speed/accuracy are not apples-to-apples.")
    print("\n| Implementation | Training (s) | Validation acc. | Test acc. |")
    print("|---|---:|---:|---:|")
    for run in data:
        print(f"| {run['implementation']} | {run['train_seconds']:.2f} | {run['val_accuracy']:.2%} | {run['test_accuracy']:.2%} |")
    print("\nTraining time excludes data loading and final test evaluation. CPU, BLAS, optimization, and build mode matter.")


if __name__ == "__main__":
    main()
