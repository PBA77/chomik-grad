from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .tensor import Tensor


@dataclass(frozen=True)
class LlamaConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> "LlamaConfig":
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        missing = [name for name in required if name not in config]
        if missing:
            raise ValueError(f"missing Llama config fields: {', '.join(missing)}")
        result = cls(
            vocab_size=int(config["vocab_size"]),
            hidden_size=int(config["hidden_size"]),
            intermediate_size=int(config["intermediate_size"]),
            num_hidden_layers=int(config["num_hidden_layers"]),
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            max_position_embeddings=int(config["max_position_embeddings"]),
            rms_norm_eps=float(config.get("rms_norm_eps", 1e-5)),
            rope_theta=float(config.get("rope_theta", 10_000.0)),
            bos_token_id=int(config.get("bos_token_id", 1)),
            eos_token_id=int(config.get("eos_token_id", 2)),
        )
        result.validate()
        return result

    @property
    def head_size(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def query_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def validate(self) -> None:
        integer_fields = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("Llama dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if self.head_size % 2:
            raise ValueError("RoPE requires an even attention head size")


def required_weight_shapes(config: LlamaConfig) -> Dict[str, Tuple[int, ...]]:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    kv_hidden = config.num_key_value_heads * config.head_size
    shapes: Dict[str, Tuple[int, ...]] = {
        "model.embed_tokens.weight": (config.vocab_size, hidden),
        "model.norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        shapes.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.self_attn.q_proj.weight": (hidden, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_hidden, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_hidden, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, hidden),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return shapes


def _rope_rotations(
    sequence_length: int,
    head_size: int,
    theta: float,
    dtype: np.dtype,
    *,
    offset: int = 0,
) -> np.ndarray:
    half = head_size // 2
    positions = np.arange(offset, offset + sequence_length, dtype=np.float32)
    inverse_frequencies = 1.0 / (
        theta ** (np.arange(0, head_size, 2, dtype=np.float32) / head_size)
    )
    angles = positions[:, None] * inverse_frequencies[None, :]
    cosine = np.cos(angles)
    sine = np.sin(angles)
    rotations = np.zeros(
        (1, 1, sequence_length, head_size, head_size), dtype=dtype
    )
    indexes = np.arange(half)
    rotations[0, 0, :, indexes, indexes] = cosine.T
    rotations[0, 0, :, indexes + half, indexes + half] = cosine.T
    # Tensors use row-vector matmul, so columns encode the rotated outputs.
    rotations[0, 0, :, indexes, indexes + half] = sine.T
    rotations[0, 0, :, indexes + half, indexes] = -sine.T
    return rotations


def token_one_hot(
    token_ids: Sequence[int],
    sequence_length: int,
    vocab_size: int,
    *,
    fill_token_id: int,
    dtype: np.dtype,
) -> np.ndarray:
    if len(token_ids) > sequence_length:
        raise ValueError("token sequence exceeds the fixed generation context")
    if fill_token_id < 0 or fill_token_id >= vocab_size:
        raise ValueError("fill token is outside the vocabulary")
    values = np.full(sequence_length, fill_token_id, dtype=np.int64)
    values[: len(token_ids)] = token_ids
    if np.any(values < 0) or np.any(values >= vocab_size):
        raise ValueError("token id is outside the vocabulary")
    result = np.zeros((1, sequence_length, vocab_size), dtype=dtype)
    result[0, np.arange(sequence_length), values] = 1
    return result


def position_selector(
    position: int, sequence_length: int, *, dtype: np.dtype
) -> np.ndarray:
    if position < 0 or position >= sequence_length:
        raise ValueError("selected position is outside the generation context")
    result = np.zeros((1, sequence_length), dtype=dtype)
    result[0, position] = 1
    return result


def load_mlx_safetensors(path: Path) -> Dict[str, Tensor]:
    """Load safetensors directly as native MLX leaves, preserving BF16."""
    try:
        import ml_dtypes
        import mlx.core as mx
    except ImportError as error:
        raise ImportError(
            "loading BF16 Llama weights requires `chomik-grad[llm,mlx]`"
        ) from error

    native = mx.load(str(path))
    if not isinstance(native, dict):
        raise ValueError("expected a named safetensors weight file")
    dtype_map = {
        mx.bfloat16: np.dtype(ml_dtypes.bfloat16),
        mx.float16: np.dtype(np.float16),
        mx.float32: np.dtype(np.float32),
    }
    weights: Dict[str, Tensor] = {}
    for name, value in native.items():
        try:
            dtype = dtype_map[value.dtype]
        except KeyError as error:
            raise TypeError(
                f"unsupported MLX weight dtype for {name}: {value.dtype}"
            ) from error
        weights[name] = Tensor.from_native(
            value,
            backend="mlx",
            shape=tuple(value.shape),
            dtype=dtype,
        )
    return weights


class LlamaForCausalLM:
    """A Llama decoder expressed entirely with Chomik's five-operation IR."""

    def __init__(
        self,
        config: LlamaConfig,
        weights: Mapping[str, Tensor],
        *,
        sequence_length: int,
        dtype: np.dtype,
        cache_length: Optional[int] = None,
    ) -> None:
        config.validate()
        if sequence_length <= 0 or sequence_length > config.max_position_embeddings:
            raise ValueError("invalid fixed generation context length")
        expected = required_weight_shapes(config)
        missing = [name for name in expected if name not in weights]
        if missing:
            raise ValueError(f"missing model weight: {missing[0]}")
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ValueError(
                    f"weight {name} has shape {weights[name].shape}, expected {shape}"
                )

        self.config = config
        self.weights = dict(weights)
        self.sequence_length = sequence_length
        self.dtype = np.dtype(dtype)
        self.cache_length = cache_length
        if cache_length is not None:
            if cache_length < sequence_length:
                raise ValueError("KV cache cannot be shorter than the prompt")
            placement = np.zeros((sequence_length, cache_length), dtype=self.dtype)
            placement[np.arange(sequence_length), np.arange(sequence_length)] = 1
            self.cache_placement = Tensor(placement)
        else:
            self.cache_placement = None
        mask = np.triu(
            np.full(
                (sequence_length, sequence_length),
                -1e4,
                dtype=self.dtype,
            ),
            k=1,
        ).reshape(1, 1, sequence_length, sequence_length)
        self.causal_mask = Tensor(mask)
        self.rotations = Tensor(
            _rope_rotations(
                sequence_length,
                config.head_size,
                config.rope_theta,
                self.dtype,
            )
        )
        self.kv_repeat = Tensor.ones(
            (1, 1, config.query_groups, 1, 1), dtype=self.dtype
        )

    @property
    def parameter_count(self) -> int:
        return sum(int(np.prod(weight.shape)) for weight in self.weights.values())

    @staticmethod
    def _linear(inputs: Tensor, weight: Tensor) -> Tensor:
        return inputs @ weight.T

    def _rms_norm(self, inputs: Tensor, weight: Tensor) -> Tensor:
        variance = (inputs * inputs).mean(axis=-1, keepdims=True)
        return inputs / (variance + self.config.rms_norm_eps).sqrt() * weight

    @staticmethod
    def _silu(inputs: Tensor) -> Tensor:
        return inputs / (1 + (-inputs).exp())

    def _apply_rope(self, inputs: Tensor) -> Tensor:
        batch, heads, tokens, head_size = inputs.shape
        return (
            inputs.reshape(batch, heads, tokens, 1, head_size)
            @ self.rotations
        ).reshape(batch, heads, tokens, head_size)

    def _repeat_kv(self, inputs: Tensor) -> Tensor:
        batch, kv_heads, tokens, head_size = inputs.shape
        return (
            inputs.reshape(batch, kv_heads, 1, tokens, head_size)
            * self.kv_repeat
        ).reshape(batch, self.config.num_attention_heads, tokens, head_size)

    def _attention(
        self, inputs: Tensor, layer: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        config = self.config
        prefix = f"model.layers.{layer}.self_attn"
        batch, tokens, _ = inputs.shape
        query = self._linear(inputs, self.weights[f"{prefix}.q_proj.weight"])
        key = self._linear(inputs, self.weights[f"{prefix}.k_proj.weight"])
        value = self._linear(inputs, self.weights[f"{prefix}.v_proj.weight"])
        query = query.reshape(
            batch, tokens, config.num_attention_heads, config.head_size
        ).permute(0, 2, 1, 3)
        key = key.reshape(
            batch, tokens, config.num_key_value_heads, config.head_size
        ).permute(0, 2, 1, 3)
        value = value.reshape(
            batch, tokens, config.num_key_value_heads, config.head_size
        ).permute(0, 2, 1, 3)
        query = self._apply_rope(query)
        key = self._apply_rope(key)
        cached_key, cached_value = key, value
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)
        scores = (
            query @ key.transpose(-2, -1)
        ) / np.sqrt(config.head_size) + self.causal_mask
        attended = scores.softmax(axis=-1) @ value
        merged = attended.permute(0, 2, 1, 3).reshape(
            batch, tokens, config.hidden_size
        )
        output = self._linear(merged, self.weights[f"{prefix}.o_proj.weight"])
        return output, cached_key, cached_value

    def _mlp(self, inputs: Tensor, layer: int) -> Tensor:
        prefix = f"model.layers.{layer}.mlp"
        gate = self._silu(
            self._linear(inputs, self.weights[f"{prefix}.gate_proj.weight"])
        )
        up = self._linear(inputs, self.weights[f"{prefix}.up_proj.weight"])
        return self._linear(
            gate * up,
            self.weights[f"{prefix}.down_proj.weight"],
        )

    def _forward(
        self,
        token_matrix: Tensor,
        selector: Tensor,
        *,
        collect_cache: bool,
    ) -> Tuple[Tensor, ...]:
        if token_matrix.shape != (
            1,
            self.sequence_length,
            self.config.vocab_size,
        ):
            raise ValueError("token matrix has the wrong fixed-context shape")
        if selector.shape != (1, self.sequence_length):
            raise ValueError("position selector has the wrong shape")

        hidden = token_matrix @ self.weights["model.embed_tokens.weight"]
        caches = []
        for layer in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{layer}"
            normalized = self._rms_norm(
                hidden, self.weights[f"{prefix}.input_layernorm.weight"]
            )
            attended, key, value = self._attention(normalized, layer)
            hidden = hidden + attended
            if collect_cache:
                if self.cache_placement is None:
                    raise RuntimeError("prefill requires a configured KV cache length")
                key = (
                    key.permute(0, 1, 3, 2) @ self.cache_placement
                ).permute(0, 1, 3, 2)
                value = (
                    value.permute(0, 1, 3, 2) @ self.cache_placement
                ).permute(0, 1, 3, 2)
                caches.extend((key, value))
            normalized = self._rms_norm(
                hidden, self.weights[f"{prefix}.post_attention_layernorm.weight"]
            )
            hidden = hidden + self._mlp(normalized, layer)
        hidden = self._rms_norm(hidden, self.weights["model.norm.weight"])
        selected = (selector @ hidden).reshape(1, self.config.hidden_size)
        logits = self._linear(selected, self.weights["lm_head.weight"])
        return (logits, *caches)

    def __call__(self, token_matrix: Tensor, selector: Tensor) -> Tensor:
        return self._forward(token_matrix, selector, collect_cache=False)[0]

    def prefill(self, token_matrix: Tensor, selector: Tensor) -> Tuple[Tensor, ...]:
        return self._forward(token_matrix, selector, collect_cache=True)


class LlamaDecoderStep(LlamaForCausalLM):
    """One cached autoregressive step, still using only the five IR ops."""

    def __init__(
        self,
        config: LlamaConfig,
        weights: Mapping[str, Tensor],
        *,
        cache_length: int,
        dtype: np.dtype,
    ) -> None:
        super().__init__(
            config,
            weights,
            sequence_length=1,
            dtype=dtype,
        )
        if cache_length <= 0 or cache_length > config.max_position_embeddings:
            raise ValueError("invalid KV cache length")
        self.cache_length = cache_length

    def __call__(
        self,
        token_matrix: Tensor,
        rotation: Tensor,
        attention_mask: Tensor,
        write_mask: Tensor,
        caches: Sequence[Tuple[Tensor, Tensor]],
    ) -> Tuple[Tensor, ...]:
        config = self.config
        if token_matrix.shape != (1, 1, config.vocab_size):
            raise ValueError("decode token matrix must contain exactly one token")
        if rotation.shape != (1, 1, 1, config.head_size, config.head_size):
            raise ValueError("decode RoPE rotation has the wrong shape")
        if attention_mask.shape != (1, 1, 1, self.cache_length):
            raise ValueError("decode attention mask has the wrong shape")
        if write_mask.shape != (1, 1, self.cache_length, 1):
            raise ValueError("decode cache write mask has the wrong shape")
        if len(caches) != config.num_hidden_layers:
            raise ValueError("decode requires one KV cache pair per layer")

        hidden = token_matrix @ self.weights["model.embed_tokens.weight"]
        updated_caches = []
        for layer, (cached_key, cached_value) in enumerate(caches):
            expected = (
                1,
                config.num_key_value_heads,
                self.cache_length,
                config.head_size,
            )
            if cached_key.shape != expected or cached_value.shape != expected:
                raise ValueError("KV cache has the wrong shape")
            prefix = f"model.layers.{layer}"
            normalized = self._rms_norm(
                hidden, self.weights[f"{prefix}.input_layernorm.weight"]
            )
            attention_prefix = f"{prefix}.self_attn"
            query = self._linear(
                normalized, self.weights[f"{attention_prefix}.q_proj.weight"]
            ).reshape(1, config.num_attention_heads, 1, config.head_size)
            key = self._linear(
                normalized, self.weights[f"{attention_prefix}.k_proj.weight"]
            ).reshape(1, config.num_key_value_heads, 1, config.head_size)
            value = self._linear(
                normalized, self.weights[f"{attention_prefix}.v_proj.weight"]
            ).reshape(1, config.num_key_value_heads, 1, config.head_size)
            query = (
                query.reshape(1, config.num_attention_heads, 1, 1, config.head_size)
                @ rotation
            ).reshape(1, config.num_attention_heads, 1, config.head_size)
            key = (
                key.reshape(1, config.num_key_value_heads, 1, 1, config.head_size)
                @ rotation
            ).reshape(1, config.num_key_value_heads, 1, config.head_size)
            cached_key = cached_key * (1 - write_mask) + key * write_mask
            cached_value = cached_value * (1 - write_mask) + value * write_mask
            updated_caches.append((cached_key, cached_value))
            repeated_key = self._repeat_kv(cached_key)
            repeated_value = self._repeat_kv(cached_value)
            scores = (
                query @ repeated_key.transpose(-2, -1)
            ) / np.sqrt(config.head_size) + attention_mask
            attended = scores.softmax(axis=-1) @ repeated_value
            merged = attended.permute(0, 2, 1, 3).reshape(
                1, 1, config.hidden_size
            )
            hidden = hidden + self._linear(
                merged, self.weights[f"{attention_prefix}.o_proj.weight"]
            )
            normalized = self._rms_norm(
                hidden, self.weights[f"{prefix}.post_attention_layernorm.weight"]
            )
            hidden = hidden + self._mlp(normalized, layer)

        hidden = self._rms_norm(hidden, self.weights["model.norm.weight"])
        logits = self._linear(
            hidden.reshape(1, config.hidden_size),
            self.weights["lm_head.weight"],
        )
        flat_caches = [tensor for pair in updated_caches for tensor in pair]
        return (logits, *flat_caches)
