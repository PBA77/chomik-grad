from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data, requires_grad=True)


class Module:
    def __call__(self, *args: object, **kwargs: object) -> Tensor:
        return self.forward(*args, **kwargs)

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise NotImplementedError

    def parameters(self) -> List[Parameter]:
        found: List[Parameter] = []
        seen = set()

        def collect(value: object) -> None:
            if isinstance(value, Parameter):
                if id(value) not in seen:
                    seen.add(id(value))
                    found.append(value)
            elif isinstance(value, Module):
                for child in value.__dict__.values():
                    collect(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect(child)

        for attribute in self.__dict__.values():
            collect(attribute)
        return found

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        rng: Union[np.random.Generator, None] = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature counts must be positive")
        generator = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)
        weights = generator.normal(
            0.0, scale, size=(out_features, in_features)
        ).astype(np.float32)
        self.weight = Parameter(weights)
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) if bias else None

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs @ self.weight.T
        return output + self.bias if self.bias is not None else output


class ReLU(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        self.layers = list(layers)

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output


class LayerNorm(Module):
    def __init__(self, features: int, eps: float = 1e-5) -> None:
        if features <= 0:
            raise ValueError("feature count must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.features = features
        self.eps = float(eps)
        self.weight = Parameter(np.ones(features, dtype=np.float32))
        self.bias = Parameter(np.zeros(features, dtype=np.float32))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.features:
            raise ValueError(
                f"expected final dimension {self.features}, got {inputs.shape[-1]}"
            )
        return inputs.layer_norm(self.weight, self.bias, self.eps)


class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        features: int,
        heads: int,
        *,
        rng: Union[np.random.Generator, None] = None,
    ) -> None:
        if features <= 0 or heads <= 0 or features % heads:
            raise ValueError("features must be positive and divisible by heads")
        self.features = features
        self.heads = heads
        self.head_features = features // heads
        generator = rng or np.random.default_rng()
        self.query = Linear(features, features, rng=generator)
        self.key = Linear(features, features, rng=generator)
        self.value = Linear(features, features, rng=generator)
        self.output = Linear(features, features, rng=generator)

    def _split_heads(self, inputs: Tensor) -> Tensor:
        batch, tokens, _ = inputs.shape
        return inputs.reshape(
            batch, tokens, self.heads, self.head_features
        ).permute(0, 2, 1, 3)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.features:
            raise ValueError(
                f"expected inputs shaped (batch, tokens, {self.features})"
            )
        query = self._split_heads(self.query(inputs))
        key = self._split_heads(self.key(inputs))
        value = self._split_heads(self.value(inputs))
        scores = (query @ key.transpose(-2, -1)) / np.sqrt(self.head_features)
        attended = scores.softmax(axis=-1) @ value
        batch, _, tokens, _ = attended.shape
        merged = attended.permute(0, 2, 1, 3).reshape(
            batch, tokens, self.features
        )
        return self.output(merged)


class TransformerEncoderBlock(Module):
    def __init__(
        self,
        features: int,
        heads: int,
        hidden_features: int,
        *,
        rng: Union[np.random.Generator, None] = None,
    ) -> None:
        if hidden_features <= 0:
            raise ValueError("hidden feature count must be positive")
        generator = rng or np.random.default_rng()
        self.norm1 = LayerNorm(features)
        self.attention = MultiHeadSelfAttention(features, heads, rng=generator)
        self.norm2 = LayerNorm(features)
        self.linear1 = Linear(features, hidden_features, rng=generator)
        self.linear2 = Linear(hidden_features, features, rng=generator)

    def forward(self, inputs: Tensor) -> Tensor:
        attended = inputs + self.attention(self.norm1(inputs))
        return attended + self.linear2(self.linear1(self.norm2(attended)).relu())


def cross_entropy(logits: Tensor, targets: Sequence[int]) -> Tensor:
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects logits shaped (batch, classes)")
    labels = np.asarray(targets)
    if labels.shape != (logits.shape[0],):
        raise ValueError(f"expected targets shaped ({logits.shape[0]},)")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("targets must contain integer class indexes")
    if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
        raise ValueError("target class index is out of range")

    one_hot = np.zeros(logits.shape, dtype=logits.dtype)
    one_hot[np.arange(logits.shape[0]), labels] = 1
    log_probabilities = logits.log_softmax(axis=1)
    return -(log_probabilities * Tensor(one_hot, copy=False)).sum() / logits.shape[0]


class CrossEntropyLoss(Module):
    def forward(self, logits: Tensor, targets: Sequence[int]) -> Tensor:
        return cross_entropy(logits, targets)
