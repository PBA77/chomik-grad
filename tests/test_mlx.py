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


MLX_INSTALLED = importlib.util.find_spec("mlx") is not None


@unittest.skipUnless(MLX_INSTALLED, "optional MLX dependency is not installed")
class MLXBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        import mlx.core as mx

        if not mx.metal.is_available():
            self.skipTest("Metal GPU is not available")
        self.mx = mx

    def test_all_five_ir_operations_match_cpu_on_gpu(self) -> None:
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
        self.assertEqual(len(Op), 5)

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


if __name__ == "__main__":
    unittest.main()
