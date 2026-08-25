import importlib.util
import platform
import unittest

import numpy as np

from chomikgrad import Tensor, compile_graph, no_grad
from chomikgrad.llama import (
    LlamaConfig,
    LlamaForCausalLM,
    position_selector,
    required_weight_shapes,
)


COREML_AVAILABLE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("coremltools") is not None
)


@unittest.skipUnless(COREML_AVAILABLE, "Core ML requires Apple silicon")
class CoreMLBackendTests(unittest.TestCase):
    def test_six_ir_operations_match_cpu_with_dynamic_inputs(self) -> None:
        source = Tensor(
            np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], np.float16),
            copy=False,
        )
        weight = Tensor(np.arange(12, dtype=np.float16).reshape(3, 4))
        table = Tensor(np.arange(20, dtype=np.float16).reshape(5, 4))
        indices = Tensor(np.array([[3, 1]], np.int32), copy=False)
        projected = (
            (source + np.float16(1))
            .relu()
            .sum(axis=0, keepdims=True)
            .reshape(3, 1)
            .T
            @ weight
        )
        gathered = table.gather(indices).permute(0, 2, 1)
        cpu = compile_graph(projected, gathered, compiler="cpu")()
        program = compile_graph(
            projected,
            gathered,
            compiler="coreml",
            dynamic_inputs=(source, indices),
        )

        actual = program()

        np.testing.assert_allclose(actual[0], cpu[0], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(actual[1], cpu[1], rtol=2e-3, atol=2e-3)
        self.assertEqual(program.inputs, (source._node, indices._node))
        self.assertEqual(program.device.name, "coreml")
        self.assertEqual(program.device.argmax(actual[0]), int(actual[0].argmax()))
        self.assertIn("segments = 1", program.source)

    def test_small_llama_transformer_matches_portable_graph(self) -> None:
        config = LlamaConfig(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
        rng = np.random.default_rng(31)
        weights = {
            name: Tensor(
                (
                    np.ones(shape)
                    if "norm.weight" in name
                    else rng.normal(0, 0.05, shape)
                ).astype(np.float16)
            )
            for name, shape in required_weight_shapes(config).items()
        }
        model = LlamaForCausalLM(
            config, weights, sequence_length=3, dtype=np.float16
        )
        tokens = Tensor(np.array([[1, 3, 4]], np.int32), copy=False)
        selector = Tensor(
            position_selector(2, 3, dtype=np.float16), copy=False
        )
        with no_grad():
            output = model(tokens, selector)
        expected = output.numpy(compiler="cpu")

        program = compile_graph(
            output, compiler="coreml", dynamic_inputs=(tokens,)
        )

        np.testing.assert_allclose(
            program()[0], expected, rtol=3e-3, atol=3e-3
        )
        plan = program.compute_plan_summary()
        self.assertGreater(plan["neural_engine_supported_operations"], 0)

    def test_prefill_sized_static_linear_prefers_neural_engine(self) -> None:
        rng = np.random.default_rng(37)
        source = Tensor(
            rng.normal(size=(1, 32, 768)).astype(np.float16), copy=False
        )
        weight = Tensor(
            rng.normal(0, 0.01, size=(768, 768)).astype(np.float16),
            copy=False,
        )
        output = source @ weight.T
        program = compile_graph(
            output, compiler="coreml", dynamic_inputs=(source,)
        )

        program.run(synchronize=True)
        plan = program.compute_plan_summary()

        preferred = plan["preferred"]
        self.assertGreater(preferred.get("MLNeuralEngineComputeDevice", 0), 0)
        self.assertEqual(plan["compute_units"], "CPU_AND_NE")


if __name__ == "__main__":
    unittest.main()
