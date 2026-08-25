from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from chomikgrad import Linear, ReLU, SGD, Sequential, Tensor, cross_entropy, no_grad


def accuracy(
    model: Sequential,
    features: np.ndarray,
    labels: np.ndarray,
    compiler: str,
) -> float:
    with no_grad():
        predictions = model(Tensor(features)).numpy(compiler=compiler).argmax(axis=1)
    return float((predictions == labels).mean())


def train(epochs: int, batch_size: int, seed: int, compiler: str) -> float:
    digits = load_digits()
    features = (digits.data / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )

    rng = np.random.default_rng(seed)
    model = Sequential(
        Linear(64, 48, rng=rng),
        ReLU(),
        Linear(48, 10, rng=rng),
    )
    optimizer = SGD(model.parameters(), lr=0.12)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            batch_x, batch_y = train_x[indexes], train_y[indexes]
            optimizer.zero_grad()
            loss = cross_entropy(model(Tensor(batch_x)), batch_y)
            loss.backward()
            optimizer.step(compiler=compiler)

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            with no_grad():
                train_loss = float(
                    cross_entropy(model(Tensor(train_x)), train_y).item(
                        compiler=compiler
                    )
                )
            test_accuracy = accuracy(model, test_x, test_y, compiler)
            print(
                f"epoch={epoch:02d} "
                f"loss={train_loss:.4f} "
                f"test_accuracy={test_accuracy:.3f}"
            )

    return accuracy(model, test_x, test_y, compiler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny MLP on sklearn digits")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--compiler", choices=("cpu", "mlx"), default="cpu")
    arguments = parser.parse_args()
    final_accuracy = train(
        arguments.epochs,
        arguments.batch_size,
        arguments.seed,
        arguments.compiler,
    )
    if final_accuracy < 0.90:
        raise SystemExit(f"expected at least 90% accuracy, got {final_accuracy:.1%}")
