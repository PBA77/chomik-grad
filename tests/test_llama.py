import unittest
from pathlib import Path

import numpy as np

from chomikgrad import NumpyDeviceAdapter, Tensor, realize
from chomikgrad.llama import (
    LlamaConfig,
    LlamaDecoderBlock,
    LlamaDecoderStep,
    LlamaForCausalLM,
    _rope_rotations,
    load_safetensors,
    position_selector,
    required_weight_shapes,
    token_one_hot,
)


class FakeWeightDevice(NumpyDeviceAdapter):
    name = "fake"

    def load_safetensors(self, path):
        return {"weight": np.array([[1.0, 2.0]], dtype=np.float32)}


class LlamaHelpersTests(unittest.TestCase):
    def test_token_one_hot_pads_fixed_context(self) -> None:
        value = token_one_hot(
            [1, 3], 4, 5, fill_token_id=2, dtype=np.float32
        )
        np.testing.assert_array_equal(value.argmax(axis=-1), [[1, 3, 2, 2]])

    def test_position_selector(self) -> None:
        value = position_selector(2, 4, dtype=np.float32)
        np.testing.assert_array_equal(value, [[0, 0, 1, 0]])

    def test_rope_rotation_matches_llama_rotate_half(self) -> None:
        value = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8)
        rotation = _rope_rotations(1, 8, 10_000.0, np.dtype(np.float32))
        actual = value.reshape(1, 1, 1, 1, 8) @ rotation
        np.testing.assert_allclose(actual.reshape(value.shape), value)

        rotation = _rope_rotations(2, 8, 10_000.0, np.dtype(np.float32))
        angle = 1 / (10_000.0 ** (np.arange(0, 8, 2) / 8))
        cosine, sine = np.cos(angle), np.sin(angle)
        current = value.reshape(8)
        expected = np.concatenate(
            [
                current[:4] * cosine - current[4:] * sine,
                current[4:] * cosine + current[:4] * sine,
            ]
        )
        actual = value.reshape(1, 1, 1, 1, 8) @ rotation[:, :, 1:2]
        np.testing.assert_allclose(actual.reshape(8), expected, rtol=1e-6, atol=1e-6)

    def test_config_rejects_invalid_grouped_query_attention(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by KV heads"):
            LlamaConfig(
                vocab_size=10,
                hidden_size=24,
                intermediate_size=48,
                num_hidden_layers=1,
                num_attention_heads=6,
                num_key_value_heads=4,
                max_position_embeddings=16,
            ).validate()

    def test_native_tensor_metadata(self) -> None:
        marker = object()
        value = Tensor.from_native(
            marker,
            backend="test",
            shape=(2, 3),
            dtype=np.float32,
        )
        self.assertEqual(value.shape, (2, 3))
        self.assertEqual(value.dtype, np.dtype(np.float32))
        self.assertIs(value._node.native_values["test"], marker)

    def test_weight_loading_uses_device_adapter(self) -> None:
        weights = load_safetensors(Path("unused.safetensors"), FakeWeightDevice())

        self.assertEqual(weights["weight"].shape, (1, 2))
        self.assertEqual(weights["weight"].dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(
            weights["weight"]._node.native_values["fake"], [[1.0, 2.0]]
        )

    def test_cached_decode_matches_full_causal_forward(self) -> None:
        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
        rng = np.random.default_rng(4)
        weights = {}
        for name, shape in required_weight_shapes(config).items():
            value = (
                np.ones(shape, dtype=np.float32)
                if "norm.weight" in name
                else rng.normal(0, 0.05, shape).astype(np.float32)
            )
            weights[name] = Tensor(value)

        prompt = [1, 3, 4]
        cache_length = 5
        prefill = LlamaForCausalLM(
            config,
            weights,
            sequence_length=len(prompt),
            cache_length=cache_length,
            dtype=np.float32,
        )
        prefill_outputs = prefill.prefill(
            Tensor(
                token_one_hot(
                    prompt,
                    len(prompt),
                    config.vocab_size,
                    fill_token_id=config.eos_token_id,
                    dtype=np.float32,
                )
            ),
            Tensor(
                position_selector(
                    len(prompt) - 1, len(prompt), dtype=np.float32
                )
            ),
        )
        native = realize(*prefill_outputs)
        caches = [
            (Tensor(native[index]), Tensor(native[index + 1]))
            for index in range(1, len(native), 2)
        ]

        next_input = 6
        position = len(prompt)
        attention_mask = np.full(
            (1, 1, 1, cache_length), -1e4, dtype=np.float32
        )
        attention_mask[..., : position + 1] = 0
        write_mask = np.zeros((1, 1, cache_length, 1), dtype=np.float32)
        write_mask[:, :, position, :] = 1
        decoder = LlamaDecoderStep(
            config, weights, cache_length=cache_length, dtype=np.float32
        )
        decoded = realize(
            *decoder(
                Tensor(
                    token_one_hot(
                        [next_input],
                        1,
                        config.vocab_size,
                        fill_token_id=config.eos_token_id,
                        dtype=np.float32,
                    )
                ),
                Tensor(
                    _rope_rotations(
                        1,
                        config.head_size,
                        config.rope_theta,
                        np.dtype(np.float32),
                        offset=position,
                    )
                ),
                Tensor(attention_mask),
                Tensor(write_mask),
                caches,
            )
        )[0]
        all_tokens = prompt + [next_input]
        full = LlamaForCausalLM(
            config,
            weights,
            sequence_length=len(all_tokens),
            dtype=np.float32,
        )
        expected = full(
            Tensor(
                token_one_hot(
                    all_tokens,
                    len(all_tokens),
                    config.vocab_size,
                    fill_token_id=config.eos_token_id,
                    dtype=np.float32,
                )
            ),
            Tensor(
                position_selector(
                    len(all_tokens) - 1,
                    len(all_tokens),
                    dtype=np.float32,
                )
            ),
        ).numpy()
        np.testing.assert_allclose(decoded, expected, rtol=2e-5, atol=2e-6)

    def test_block_decode_matches_each_full_causal_position(self) -> None:
        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
        rng = np.random.default_rng(14)
        weights = {
            name: Tensor(
                np.ones(shape, dtype=np.float32)
                if "norm.weight" in name
                else rng.normal(0, 0.05, shape).astype(np.float32)
            )
            for name, shape in required_weight_shapes(config).items()
        }
        prompt = [1, 3, 4]
        proposed = [6, 2]
        cache_length = len(prompt) + len(proposed)
        prefill = LlamaForCausalLM(
            config,
            weights,
            sequence_length=len(prompt),
            cache_length=cache_length,
            dtype=np.float32,
        )
        prefill_native = realize(
            *prefill.prefill(
                Tensor(np.asarray([prompt], dtype=np.int32), copy=False),
                Tensor(position_selector(2, 3, dtype=np.float32)),
            )
        )
        caches = [
            (Tensor(prefill_native[index]), Tensor(prefill_native[index + 1]))
            for index in range(1, len(prefill_native), 2)
        ]
        placement = np.zeros((2, cache_length), dtype=np.float32)
        placement[np.arange(2), np.arange(3, 5)] = 1
        write_mask = placement.sum(axis=0).reshape(1, 1, cache_length, 1)
        attention_mask = np.full(
            (1, 1, 2, cache_length), -1e4, dtype=np.float32
        )
        attention_mask[0, 0, 0, :4] = 0
        attention_mask[0, 0, 1, :5] = 0
        block = LlamaDecoderBlock(
            config,
            weights,
            block_length=2,
            cache_length=cache_length,
            dtype=np.float32,
        )
        actual = realize(
            *block(
                Tensor(np.asarray([proposed], dtype=np.int32), copy=False),
                Tensor(
                    _rope_rotations(
                        2,
                        config.head_size,
                        config.rope_theta,
                        np.dtype(np.float32),
                        offset=3,
                    )
                ),
                Tensor(attention_mask),
                Tensor(placement),
                Tensor(write_mask),
                caches,
            )
        )

        for index in range(2):
            tokens = prompt + proposed[: index + 1]
            full = LlamaForCausalLM(
                config,
                weights,
                sequence_length=len(tokens),
                dtype=np.float32,
            )
            expected = full(
                Tensor(np.asarray([tokens], dtype=np.int32), copy=False),
                Tensor(
                    position_selector(
                        len(tokens) - 1, len(tokens), dtype=np.float32
                    )
                ),
            ).numpy()
            np.testing.assert_allclose(
                actual[0][:, index, :], expected, rtol=2e-5, atol=2e-6
            )

        full_prefill = LlamaForCausalLM(
            config,
            weights,
            sequence_length=cache_length,
            cache_length=cache_length,
            dtype=np.float32,
        )
        complete = realize(
            *full_prefill.prefill(
                Tensor(
                    np.asarray([prompt + proposed], dtype=np.int32), copy=False
                ),
                Tensor(
                    position_selector(
                        cache_length - 1, cache_length, dtype=np.float32
                    )
                ),
            )
        )
        for block_cache, full_cache in zip(actual[1:], complete[1:]):
            np.testing.assert_allclose(
                block_cache, full_cache, rtol=2e-5, atol=2e-6
            )


if __name__ == "__main__":
    unittest.main()
