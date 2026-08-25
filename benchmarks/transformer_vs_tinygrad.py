from __future__ import annotations

import argparse
import time
from typing import Iterator, List, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from chomikgrad import SGD, Tensor, cross_entropy, no_grad
from examples.train_digits_transformer import DigitsTransformer


def load_data(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    features = (digits.data / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)
    return train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=seed,
        stratify=labels,
    )


def benchmark_chomik(
    compiler: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    model = DigitsTransformer(rng)
    optimizer = SGD(model.parameters(), lr=0.03)

    started = time.perf_counter()
    for _ in range(epochs):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), batch_size):
            indexes = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = cross_entropy(
                model(
                    Tensor(
                        train_x[indexes].reshape(-1, 8, 8),
                        copy=False,
                    )
                ),
                train_y[indexes],
            )
            loss.backward()
            optimizer.step(compiler=compiler)
    model.parameters()[0].numpy(compiler=compiler)
    elapsed = time.perf_counter() - started

    with no_grad():
        predictions = (
            model(Tensor(test_x.reshape(-1, 8, 8), copy=False))
            .numpy(compiler=compiler)
            .argmax(axis=1)
        )
    return elapsed, float((predictions == test_y).mean())


class TinyLinear:
    def __init__(self, values: Iterator[np.ndarray], parameters: List[object]) -> None:
        from tinygrad import Tensor as TinyTensor

        self.weight = TinyTensor(next(values))
        self.bias = TinyTensor(next(values))
        parameters.extend((self.weight, self.bias))

    def __call__(self, inputs: object) -> object:
        return inputs @ self.weight.transpose() + self.bias  # type: ignore[operator,no-any-return]


class TinyLayerNorm:
    def __init__(self, values: Iterator[np.ndarray], parameters: List[object]) -> None:
        from tinygrad import Tensor as TinyTensor

        self.weight = TinyTensor(next(values))
        self.bias = TinyTensor(next(values))
        parameters.extend((self.weight, self.bias))

    def __call__(self, inputs: object) -> object:
        mean = inputs.mean(axis=-1, keepdim=True)  # type: ignore[attr-defined]
        centered = inputs - mean  # type: ignore[operator]
        variance = (centered * centered).mean(axis=-1, keepdim=True)
        return centered / (variance + 1e-5).sqrt() * self.weight + self.bias


class TinyAttention:
    def __init__(self, values: Iterator[np.ndarray], parameters: List[object]) -> None:
        self.query = TinyLinear(values, parameters)
        self.key = TinyLinear(values, parameters)
        self.value = TinyLinear(values, parameters)
        self.output = TinyLinear(values, parameters)

    @staticmethod
    def split_heads(inputs: object) -> object:
        batch, tokens, _ = inputs.shape  # type: ignore[attr-defined]
        return inputs.reshape(batch, tokens, 4, 8).permute(0, 2, 1, 3)  # type: ignore[attr-defined,no-any-return]

    def __call__(self, inputs: object) -> object:
        query = self.split_heads(self.query(inputs))
        key = self.split_heads(self.key(inputs))
        value = self.split_heads(self.value(inputs))
        scores = (query @ key.transpose(-2, -1)) / np.sqrt(8)  # type: ignore[operator,attr-defined]
        attended = scores.softmax(axis=-1) @ value
        batch, _, tokens, _ = attended.shape
        merged = attended.permute(0, 2, 1, 3).reshape(batch, tokens, 32)
        return self.output(merged)


class TinyBlock:
    def __init__(self, values: Iterator[np.ndarray], parameters: List[object]) -> None:
        self.norm1 = TinyLayerNorm(values, parameters)
        self.attention = TinyAttention(values, parameters)
        self.norm2 = TinyLayerNorm(values, parameters)
        self.linear1 = TinyLinear(values, parameters)
        self.linear2 = TinyLinear(values, parameters)

    def __call__(self, inputs: object) -> object:
        attended = inputs + self.attention(self.norm1(inputs))  # type: ignore[operator]
        return attended + self.linear2(self.linear1(self.norm2(attended)).relu())


class TinyDigitsTransformer:
    def __init__(self, arrays: List[np.ndarray]) -> None:
        from tinygrad import Tensor as TinyTensor

        values = iter(arrays)
        self.parameters: List[object] = []
        self.embedding = TinyLinear(values, self.parameters)
        self.position = TinyTensor(next(values))
        self.parameters.append(self.position)
        self.blocks = [
            TinyBlock(values, self.parameters),
            TinyBlock(values, self.parameters),
        ]
        self.norm = TinyLayerNorm(values, self.parameters)
        self.classifier = TinyLinear(values, self.parameters)
        try:
            next(values)
        except StopIteration:
            return
        raise RuntimeError("unused reference parameter")

    def __call__(self, inputs: object) -> object:
        encoded = self.embedding(inputs) + self.position
        for block in self.blocks:
            encoded = block(encoded)
        return self.classifier(self.norm(encoded).mean(axis=1))


def benchmark_tinygrad(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> Tuple[float, float]:
    try:
        from tinygrad import Tensor as TinyTensor, TinyJit
        from tinygrad.helpers import Context
        from tinygrad.nn.optim import SGD as TinySGD
    except ImportError as error:
        raise SystemExit("install tinygrad to run this benchmark") from error

    rng = np.random.default_rng(seed)
    reference = DigitsTransformer(rng)
    arrays = [parameter.numpy().copy() for parameter in reference.parameters()]
    model = TinyDigitsTransformer(arrays)
    optimizer = TinySGD(model.parameters, lr=0.03)

    def train_step(inputs: object, targets: object) -> object:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = -(logits.log_softmax(axis=1) * targets).sum() / inputs.shape[0]
        loss.backward()
        optimizer.step()
        return loss

    step = TinyJit(train_step)
    tail_step = TinyJit(train_step)

    started = time.perf_counter()
    with Context(TRAINING=1):
        for _ in range(epochs):
            order = rng.permutation(len(train_x))
            for start in range(0, len(order), batch_size):
                indexes = order[start : start + batch_size]
                one_hot = np.zeros((len(indexes), 10), dtype=np.float32)
                one_hot[np.arange(len(indexes)), train_y[indexes]] = 1
                selected_step = step if len(indexes) == batch_size else tail_step
                selected_step(
                    TinyTensor(train_x[indexes].reshape(-1, 8, 8)),
                    TinyTensor(one_hot),
                )
    model.parameters[0].numpy()
    elapsed = time.perf_counter() - started

    predictions = (
        model(TinyTensor(test_x.reshape(-1, 8, 8))).argmax(axis=1).numpy()
    )
    return elapsed, float((predictions == test_y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare identical digits Transformers in chomik-grad and tinygrad"
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=("chomik-cpu", "chomik-mlx", "tinygrad"),
        default=("chomik-cpu", "chomik-mlx", "tinygrad"),
    )
    arguments = parser.parse_args()
    train_x, test_x, train_y, test_y = load_data(arguments.seed)

    for framework in arguments.frameworks:
        durations = []
        accuracies = []
        for trial in range(1, arguments.trials + 1):
            if framework == "tinygrad":
                elapsed, accuracy = benchmark_tinygrad(
                    train_x,
                    test_x,
                    train_y,
                    test_y,
                    arguments.seed,
                    arguments.epochs,
                    arguments.batch_size,
                )
            else:
                elapsed, accuracy = benchmark_chomik(
                    framework.removeprefix("chomik-"),
                    train_x,
                    test_x,
                    train_y,
                    test_y,
                    arguments.seed,
                    arguments.epochs,
                    arguments.batch_size,
                )
            durations.append(elapsed)
            accuracies.append(accuracy)
            print(
                f"framework={framework} trial={trial} "
                f"seconds={elapsed:.6f} accuracy={accuracy:.6f}"
            )
        print(
            f"framework={framework} median_seconds={np.median(durations):.6f} "
            f"accuracy={np.median(accuracies):.6f}"
        )


if __name__ == "__main__":
    main()
