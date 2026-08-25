import importlib.util
import unittest

import numpy as np

from chomikgrad import Parameter, SGD, Tensor, compile_graph, cross_entropy


CUPY_INSTALLED = importlib.util.find_spec("cupy") is not None


@unittest.skipUnless(CUPY_INSTALLED, "optional CuPy dependency is not installed")
class CUDABackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import cupy as cp

        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise unittest.SkipTest("CUDA GPU is not available")
        except Exception as error:
            raise unittest.SkipTest(f"CUDA runtime is not available: {error}")
        cls.cp = cp

    def test_six_ir_operations_and_gather_match_cpu(self) -> None:
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
        program = compile_graph(result, compiler="cuda")
        np.testing.assert_allclose(
            program()[0], result.numpy(compiler="cpu"), rtol=1e-5, atol=1e-6
        )
        self.assertIn("cp.matmul", program.source)
        self.assertIn("cp.transpose", program.source)

        table = Tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        indices = Tensor(np.array([[2, 0]], dtype=np.int32), copy=False)
        gathered = table.gather(indices)
        gather_program = compile_graph(gathered, compiler="cuda")
        self.assertIn("cp.take", gather_program.source)
        np.testing.assert_allclose(
            gather_program()[0], [[[5.0, 6.0], [1.0, 2.0]]]
        )

    def test_autograd_and_batched_matmul_match_cpu(self) -> None:
        raw = np.array([[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
        logits = Tensor(raw, requires_grad=True)
        loss = cross_entropy(logits, np.array([0, 2], dtype=np.int64))
        loss.backward()
        np.testing.assert_allclose(
            loss.numpy(compiler="cuda"),
            loss.numpy(compiler="cpu"),
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            logits.grad.numpy(compiler="cuda"),
            logits.grad.numpy(compiler="cpu"),
            rtol=1e-5,
            atol=1e-6,
        )

        rng = np.random.default_rng(19)
        left = Tensor(rng.normal(size=(2, 4, 3, 5)).astype(np.float32))
        right = Tensor(rng.normal(size=(2, 4, 5, 6)).astype(np.float32))
        result = left @ right
        np.testing.assert_allclose(
            result.numpy(compiler="cuda"),
            result.numpy(compiler="cpu"),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_dynamic_input_and_deferred_synchronization(self) -> None:
        source = Tensor([1.0, 2.0], dtype=np.float32)
        program = compile_graph(
            source * 2.0, compiler="cuda", dynamic_inputs=(source,)
        )
        replacement = self.cp.asarray([3.0, 4.0], dtype=self.cp.float32)
        native = program.run({source._node: replacement}, synchronize=False)[0]
        program.device.synchronize()
        np.testing.assert_allclose(self.cp.asnumpy(native), [6.0, 8.0])
        self.assertEqual(program.inputs, (source._node,))
        self.assertEqual(program.device.argmax(native), 1)

    def test_copy_false_observes_host_mutation(self) -> None:
        source = np.array([1.0, 2.0], dtype=np.float32)
        tensor = Tensor(source, copy=False)
        np.testing.assert_allclose(tensor.numpy(compiler="cuda"), [1.0, 2.0])
        source[0] = 9.0
        np.testing.assert_allclose(tensor.numpy(compiler="cuda"), [9.0, 2.0])

    def test_sgd_keeps_parameters_on_cuda_and_cpu_can_read_them(self) -> None:
        parameter = Parameter(np.array([[1.0], [2.0]], dtype=np.float32))
        inputs = Tensor(np.array([[3.0, 4.0]], dtype=np.float32))
        loss = (inputs @ parameter).sum()
        loss.backward()
        SGD([parameter], lr=0.1).step(compiler="cuda")

        self.assertIsNone(parameter._node.value)
        self.assertIn("cuda", parameter._node.native_values)
        np.testing.assert_allclose(
            parameter.numpy(compiler="cpu"),
            [[0.7], [1.6]],
            rtol=1e-6,
            atol=1e-6,
        )
