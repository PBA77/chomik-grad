import importlib.util
import unittest

import numpy as np

from chomikgrad import (
    Parameter,
    SGD,
    Tensor,
    compile_graph,
    compile_train_step,
    cross_entropy,
)


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
        self.assertIn("fused_elementwise", program.source)
        self.assertIn("    del ", program.source)

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

        projection = Tensor(rng.normal(size=(2, 4, 5)).astype(np.float32))
        weight = Tensor(rng.normal(size=(5, 7)).astype(np.float32))
        projected = projection @ weight
        program = compile_graph(projected, compiler="cuda")
        self.assertIn("cp.reshape", program.source)
        np.testing.assert_allclose(
            program()[0],
            projected.numpy(compiler="cpu"),
            rtol=1e-5,
            atol=1e-6,
        )

        singleton_batch = Tensor(
            rng.normal(size=(1, 3, 5)).astype(np.float32),
            requires_grad=True,
        )
        shared_weight = Tensor(
            rng.normal(size=(5, 7)).astype(np.float32),
            requires_grad=True,
        )
        (singleton_batch @ shared_weight).sum().backward()
        gradient_program = compile_graph(shared_weight.grad, compiler="cuda")
        self.assertNotIn("cp.sum", gradient_program.source)
        self.assertIn("cp.matmul(cp.reshape", gradient_program.source)
        expected_gradient = singleton_batch._node.numpy_value()[0].T @ np.ones(
            (3, 7), dtype=np.float32
        )
        np.testing.assert_allclose(
            gradient_program()[0],
            expected_gradient,
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

    def test_float32_layer_norm_lowering_matches_portable_graph(self) -> None:
        rng = np.random.default_rng(23)
        source = Tensor(rng.normal(size=(3, 4, 64)).astype(np.float32))
        weight = Tensor(rng.normal(size=64).astype(np.float32))
        bias = Tensor(rng.normal(size=64).astype(np.float32))
        result = source.layer_norm(weight, bias)
        expected = result.numpy(compiler="cpu")

        program = compile_graph(result, compiler="cuda")
        self.assertIn("layer_norm", program.source)
        np.testing.assert_allclose(program()[0], expected, rtol=1e-5, atol=1e-5)

    def test_fused_softmax_backward_matches_portable_graph(self) -> None:
        rng = np.random.default_rng(29)
        source = Tensor(
            rng.normal(size=(3, 5, 64)).astype(np.float32),
            requires_grad=True,
        )
        upstream = Tensor(rng.normal(size=source.shape).astype(np.float32))
        (source.softmax(axis=-1) * upstream).sum().backward()
        expected = source.grad.numpy(compiler="cpu")

        program = compile_graph(source.grad, compiler="cuda")
        self.assertIn(" = softmax(", program.source)
        self.assertIn("softmax_backward", program.source)
        np.testing.assert_allclose(
            program()[0], expected, rtol=1e-5, atol=1e-6
        )

        log_source = Tensor(
            rng.normal(size=(4, 7)).astype(np.float32),
            requires_grad=True,
        )
        log_upstream = Tensor(
            rng.normal(size=log_source.shape).astype(np.float32)
        )
        (log_source.log_softmax(axis=-1) * log_upstream).sum().backward()
        log_program = compile_graph(log_source.grad, compiler="cuda")
        self.assertIn(" = softmax(", log_program.source)
        self.assertIn("log_softmax_backward", log_program.source)
        np.testing.assert_allclose(
            log_program()[0],
            log_source.grad.numpy(compiler="cpu"),
            rtol=2e-5,
            atol=2e-6,
        )

        non_last = Tensor(
            rng.normal(size=(2, 3, 4)).astype(np.float32),
            requires_grad=True,
        )
        non_last_upstream = Tensor(
            rng.normal(size=non_last.shape).astype(np.float32)
        )
        (non_last.softmax(axis=1) * non_last_upstream).sum().backward()
        np.testing.assert_allclose(
            non_last.grad.numpy(compiler="cuda"),
            non_last.grad.numpy(compiler="cpu"),
            rtol=2e-5,
            atol=2e-6,
        )

        empty = Tensor(np.empty((0, 4), dtype=np.float32)).softmax(axis=-1)
        self.assertEqual(empty.numpy(compiler="cuda").shape, (0, 4))

    def test_fused_layer_norm_backward_matches_portable_graph(self) -> None:
        rng = np.random.default_rng(31)
        source = Tensor(
            rng.normal(size=(3, 5, 64)).astype(np.float32),
            requires_grad=True,
        )
        weight = Tensor(
            rng.normal(size=64).astype(np.float32),
            requires_grad=True,
        )
        bias = Tensor(
            rng.normal(size=64).astype(np.float32),
            requires_grad=True,
        )
        upstream = Tensor(rng.normal(size=source.shape).astype(np.float32))
        (source.layer_norm(weight, bias) * upstream).sum().backward()
        expected = tuple(
            gradient.numpy(compiler="cpu")
            for gradient in (source.grad, weight.grad, bias.grad)
        )

        program = compile_graph(
            source.grad,
            weight.grad,
            bias.grad,
            compiler="cuda",
        )
        self.assertEqual(program.source.count("layer_norm_backward("), 1)
        for actual, reference in zip(program(), expected):
            np.testing.assert_allclose(
                actual, reference, rtol=2e-5, atol=2e-5
            )

    def test_copy_false_observes_host_mutation(self) -> None:
        source = np.array([1.0, 2.0], dtype=np.float32)
        tensor = Tensor(source, copy=False)
        np.testing.assert_allclose(tensor.numpy(compiler="cuda"), [1.0, 2.0])
        source[0] = 9.0
        np.testing.assert_allclose(tensor.numpy(compiler="cuda"), [9.0, 2.0])

    def test_sgd_keeps_parameters_on_cuda_and_cpu_can_read_them(self) -> None:
        parameter = Parameter(np.array([[1.0], [2.0]], dtype=np.float32))
        inputs = Tensor(np.array([[3.0, 4.0]], dtype=np.float32))
        old_graph = parameter * 2.0
        storage_before = compile_graph(parameter, compiler="cuda").run(
            synchronize=True
        )[0]
        loss = (inputs @ parameter).sum()
        loss.backward()
        SGD([parameter], lr=0.1).step(compiler="cuda")

        self.assertIsNone(parameter._node.value)
        self.assertIn("cuda", parameter._node.native_values)
        storage_after = parameter._node.native_values["cuda"]
        self.assertNotEqual(storage_before.data.ptr, storage_after.data.ptr)
        np.testing.assert_allclose(
            parameter.numpy(compiler="cpu"),
            [[0.7], [1.6]],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            old_graph.numpy(compiler="cuda"), [[2.0], [4.0]]
        )

    def test_inplace_sgd_reuses_cuda_parameter_storage(self) -> None:
        parameter = Parameter(np.array([[1.0], [2.0]], dtype=np.float32))
        inputs = Tensor(np.array([[3.0, 4.0]], dtype=np.float32))
        old_graph = parameter * 2.0
        storage_before = compile_graph(parameter, compiler="cuda").run(
            synchronize=True
        )[0]
        loss = (inputs @ parameter).sum()
        loss.backward()
        SGD([parameter], lr=0.1, inplace=True).step(compiler="cuda")

        storage_after = parameter._node.native_values["cuda"]
        self.assertEqual(storage_before.data.ptr, storage_after.data.ptr)
        np.testing.assert_allclose(
            parameter.numpy(compiler="cpu"),
            [[0.7], [1.6]],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            old_graph.numpy(compiler="cuda"), [[1.4], [3.2]]
        )

    def test_sgd_fuses_updates_across_multiple_groups(self) -> None:
        parameters = [
            Parameter(np.array([float(index)], dtype=np.float32))
            for index in range(18)
        ]
        loss = parameters[0].sum()
        for parameter in parameters[1:]:
            loss = loss + parameter.sum()
        loss.backward()

        SGD(parameters, lr=0.25).step(compiler="cuda")

        for index, parameter in enumerate(parameters):
            np.testing.assert_allclose(
                parameter.numpy(compiler="cuda"),
                [index - 0.25],
                rtol=1e-6,
                atol=1e-6,
            )

    def test_compiled_train_step_reuses_backward_graph(self) -> None:
        initial = np.array(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]], dtype=np.float32
        )
        reference = Parameter(initial.copy())
        compiled = Parameter(initial.copy())
        reference_optimizer = SGD([reference], lr=0.05)
        compiled_optimizer = SGD([compiled], lr=0.05)
        captures = 0

        def loss(parameter: Tensor, inputs: Tensor, targets: Tensor) -> Tensor:
            logits = inputs @ parameter
            return -(logits.log_softmax(axis=1) * targets).sum() / inputs.shape[0]

        def compiled_loss(inputs: Tensor, targets: Tensor) -> Tensor:
            nonlocal captures
            captures += 1
            return loss(compiled, inputs, targets)

        step = compile_train_step(
            compiled_loss,
            compiled_optimizer,
            Tensor.zeros((4, 2)),
            Tensor.zeros((4, 3)),
            compiler="cuda",
            return_loss=True,
        )
        batches = (
            (
                np.array(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
                    dtype=np.float32,
                ),
                np.eye(3, dtype=np.float32)[[0, 1, 2, 1]],
            ),
            (
                np.array(
                    [[0.5, 1.0], [1.5, -0.5], [-0.5, 1.0], [2.0, 1.0]],
                    dtype=np.float32,
                ),
                np.eye(3, dtype=np.float32)[[2, 0, 1, 2]],
            ),
        )
        for inputs, targets in batches:
            reference_optimizer.zero_grad()
            reference_loss = loss(
                reference,
                Tensor(inputs, copy=False),
                Tensor(targets, copy=False),
            )
            reference_loss.backward()
            reference_optimizer.step(compiler="cuda")
            compiled_result = step(inputs, targets)
            np.testing.assert_allclose(
                compiled_result.numpy(compiler="cuda"),
                reference_loss.numpy(compiler="cuda"),
                rtol=2e-5,
                atol=2e-6,
            )

        self.assertEqual(captures, 1)
        np.testing.assert_allclose(
            compiled.numpy(compiler="cuda"),
            reference.numpy(compiler="cuda"),
            rtol=2e-5,
            atol=2e-6,
        )
