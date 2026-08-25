import importlib.util
import unittest

import numpy as np

from chomikgrad import (
    Op,
    Parameter,
    SGD,
    Tensor,
    compile_graph,
    cross_entropy,
    get_compiler,
)
from chomikgrad.llama import (
    LlamaConfig,
    LlamaForCausalLM,
    position_selector,
    required_weight_shapes,
)


MLX_INSTALLED = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(MLX_INSTALLED, "optional MLX dependency is not installed")
class MLXBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        import mlx.core as mx

        if not mx.metal.is_available():
            self.skipTest("Metal GPU is not available")
        self.mx = mx

    def test_six_ir_operations_match_cpu_on_gpu(self) -> None:
        left = Tensor(
            [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0]], dtype=np.float32
        )
        right = Tensor(
            [[2.0, 1.0], [-1.0, 0.5], [3.0, -2.0]], dtype=np.float32
        )
        result = (
            ((left @ right).relu() + 0.25)
            .permute(1, 0)
            .reshape(1, 4)
            .sum(axis=1)
        )

        self.mx.reset_peak_memory()
        program = compile_graph(result, compiler="mlx")
        gpu_result = program()[0]
        cpu_result = result.numpy(compiler="cpu")

        np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-6)
        self.assertIn("mx.matmul", program.source)
        self.assertIn("mx.transpose", program.source)
        self.assertGreater(self.mx.get_peak_memory(), 0)
        self.assertEqual(len(Op), 6)

        table = Tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        indices = Tensor(np.array([[2, 0]], dtype=np.int32), copy=False)
        gathered = table.gather(indices)
        gather_program = compile_graph(gathered, compiler="mlx")
        self.assertIn("mx.take", gather_program.source)
        np.testing.assert_allclose(
            gather_program()[0], [[[5.0, 6.0], [1.0, 2.0]]]
        )

    def test_lazy_autograd_graph_matches_cpu(self) -> None:
        raw = np.array([[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
        labels = np.array([0, 2], dtype=np.int64)
        logits = Tensor(raw, requires_grad=True)
        loss = cross_entropy(logits, labels)
        loss.backward()

        cpu_loss = loss.numpy(compiler="cpu")
        gpu_loss = loss.numpy(compiler="mlx")
        cpu_gradient = logits.grad.numpy(compiler="cpu")
        gpu_gradient = logits.grad.numpy(compiler="mlx")
        np.testing.assert_allclose(gpu_loss, cpu_loss, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            gpu_gradient, cpu_gradient, rtol=1e-5, atol=1e-6
        )

    def test_native_program_can_defer_synchronization(self) -> None:
        result = Tensor([1.0, -2.0], dtype=np.float32).relu() + 0.5
        program = compile_graph(result, compiler="mlx")
        native_result = program.run(synchronize=False)[0]
        self.mx.eval(native_result)
        np.testing.assert_allclose(np.array(native_result), [1.5, 0.5])

    def test_native_program_can_replace_a_dynamic_input(self) -> None:
        source = Tensor([1.0, 2.0], dtype=np.float32)
        program = compile_graph(
            source * 2.0,
            compiler="mlx",
            dynamic_inputs=(source,),
        )
        replacement = self.mx.array([3.0, 4.0])
        native_result = program.run(
            {source._node: replacement}, synchronize=True
        )[0]

        np.testing.assert_allclose(np.array(native_result), [6.0, 8.0])
        np.testing.assert_allclose(program()[0], [2.0, 4.0])
        self.assertEqual(program.inputs, (source._node,))
        self.assertIs(program.device, get_compiler("mlx").device)
        self.assertEqual(program.device.argmax(native_result), 1)
        np.testing.assert_array_equal(
            program.device.argmax_last_axis(native_result), 1
        )

        compatibility_result = program.run_native(
            input_values={source._node: replacement}
        )[0]
        np.testing.assert_allclose(np.array(compatibility_result), [6.0, 8.0])

    def test_llama_lowerings_match_portable_cpu_graph(self) -> None:
        config = LlamaConfig(
            vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=8,
        )
        rng = np.random.default_rng(23)
        weights = {
            name: Tensor(
                np.ones(shape, dtype=np.float32)
                if "norm.weight" in name
                else rng.normal(0, 0.05, shape).astype(np.float32)
            )
            for name, shape in required_weight_shapes(config).items()
        }
        model = LlamaForCausalLM(
            config, weights, sequence_length=3, dtype=np.float32
        )
        output = model(
            Tensor(np.array([[1, 3, 4]], dtype=np.int32), copy=False),
            Tensor(position_selector(2, 3, dtype=np.float32), copy=False),
        )

        program = compile_graph(output, compiler="mlx")
        self.assertIn("mx.take", program.source)
        self.assertIn("mx.fast.rms_norm", program.source)
        self.assertIn("mx.fast.rope", program.source)
        self.assertIn("mx.fast.scaled_dot_product_attention", program.source)
        self.assertNotIn("mx.sqrt", program.source)
        np.testing.assert_allclose(
            program()[0],
            output.numpy(compiler="cpu"),
            rtol=2e-4,
            atol=2e-5,
        )

    def test_copy_false_observes_mutations_across_mlx_calls(self) -> None:
        source = np.array([1.0, 2.0], dtype=np.float32)
        tensor = Tensor(source, copy=False)
        np.testing.assert_allclose(tensor.numpy(compiler="mlx"), [1.0, 2.0])

        source[0] = 9.0

        np.testing.assert_allclose(tensor.numpy(compiler="mlx"), [9.0, 2.0])

    def test_batched_matmul_matches_cpu(self) -> None:
        rng = np.random.default_rng(19)
        left = Tensor(rng.normal(size=(2, 4, 3, 5)).astype(np.float32))
        right = Tensor(rng.normal(size=(2, 4, 5, 6)).astype(np.float32))
        result = left @ right
        np.testing.assert_allclose(
            result.numpy(compiler="mlx"),
            result.numpy(compiler="cpu"),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_sgd_keeps_parameters_on_gpu_and_cpu_can_read_them(self) -> None:
        parameter = Parameter(np.array([[1.0], [2.0]], dtype=np.float32))
        inputs = Tensor(np.array([[3.0, 4.0]], dtype=np.float32))
        loss = (inputs @ parameter).sum()
        loss.backward()

        SGD([parameter], lr=0.1).step(compiler="mlx")

        self.assertIsNone(parameter._node.value)
        self.assertIn("mlx", parameter._node.native_values)
        np.testing.assert_allclose(
            parameter.numpy(compiler="cpu"),
            [[0.7], [1.6]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_sgd_can_switch_from_mlx_to_cpu_without_manual_readback(self) -> None:
        parameter = Parameter(np.array([[1.0], [2.0]], dtype=np.float32))
        inputs = Tensor(np.array([[3.0, 4.0]], dtype=np.float32))
        optimizer = SGD([parameter], lr=0.1)

        first_loss = (inputs @ parameter).sum()
        first_loss.backward()
        optimizer.step(compiler="mlx")

        optimizer.zero_grad()
        second_loss = (inputs @ parameter).sum()
        second_loss.backward()
        optimizer.step(compiler="cpu")

        np.testing.assert_allclose(
            parameter.numpy(compiler="cpu"),
            [[0.4], [1.2]],
            rtol=1e-6,
            atol=1e-6,
        )

    def test_compiled_graph_is_cached_by_structure(self) -> None:
        compiler = get_compiler("mlx")
        first = Tensor([1.0, 2.0], dtype=np.float32).relu().softmax()
        first.numpy(compiler="mlx")
        after_first = compiler.cache_size

        second = Tensor([3.0, 4.0], dtype=np.float32).relu().softmax()
        second.numpy(compiler="mlx")
        self.assertEqual(compiler.cache_size, after_first)

    def test_compiled_cache_preserves_leaf_aliasing(self) -> None:
        compiler = get_compiler("mlx")
        left = Tensor([1.0, 2.0], dtype=np.float32)
        right = Tensor([10.0, 20.0], dtype=np.float32)

        shared = left + left
        distinct = left + right
        shared_result = shared.numpy(compiler="mlx")
        after_shared = compiler.cache_size
        distinct_result = distinct.numpy(compiler="mlx")

        self.assertEqual(compiler.cache_size, after_shared + 1)
        np.testing.assert_allclose(shared_result, [2.0, 4.0])
        np.testing.assert_allclose(distinct_result, [11.0, 22.0])


if __name__ == "__main__":
    unittest.main()
