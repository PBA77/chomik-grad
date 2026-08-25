from __future__ import annotations

import argparse

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from chomikgrad import (
    LayerNorm,
    Linear,
    Module,
    Parameter,
    SGD,
    Tensor,
    TransformerEncoderBlock,
    cross_entropy,
    no_grad,
)


class DigitsTransformer(Module):
    def __init__(self, rng: np.random.Generator) -> None:
        self.embedding = Linear(8, 32, rng=rng)
        self.position = Parameter(
            rng.normal(0.0, 0.02, size=(8, 32)).astype(np.float32)
        )
        self.blocks = [
            TransformerEncoderBlock(32, 4, 64, rng=rng),
            TransformerEncoderBlock(32, 4, 64, rng=rng),
        ]
        self.norm = LayerNorm(32)
        self.classifier = Linear(32, 10, rng=rng)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (8, 8):
            raise ValueError("expected digits shaped (batch, 8, 8)")
        encoded = self.embedding(inputs) + self.position
        for block in self.blocks:
            encoded = block(encoded)
        pooled = self.norm(encoded).mean(axis=1)
        return self.classifier(pooled)


def accuracy(
    model: DigitsTransformer,
    features: np.ndarray,
    labels: np.ndarray,
    compiler: str,
) -> float:
    with no_grad():
        inputs = Tensor(features.reshape(-1, 8, 8))
        predictions = model(inputs).numpy(compiler=compiler).argmax(axis=1)
    return float((predictions == labels).mean())


def train(
    epochs: int,
    batch_size: int,
    seed: int,
    compiler: str,
    learning_rate: float,
) -> float:
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
    model = DigitsTransformer(rng)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            inputs = Tensor(train_x[indexes].reshape(-1, 8, 8))
            optimizer.zero_grad()
            loss = cross_entropy(model(inputs), train_y[indexes])
            loss.backward()
            optimizer.step(compiler=compiler)

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            with no_grad():
                train_inputs = Tensor(train_x.reshape(-1, 8, 8))
                train_loss = float(
                    cross_entropy(model(train_inputs), train_y).item(
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
    parser = argparse.ArgumentParser(
        description="Train a tiny Vision Transformer on sklearn digits"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--compiler", choices=("cpu", "mlx"), default="cpu")
    arguments = parser.parse_args()
    final_accuracy = train(
        arguments.epochs,
        arguments.batch_size,
        arguments.seed,
        arguments.compiler,
        arguments.learning_rate,
    )
    if final_accuracy < 0.75:
        raise SystemExit(f"expected at least 75% accuracy, got {final_accuracy:.1%}")
