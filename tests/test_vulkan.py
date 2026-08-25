import importlib.util
import unittest

import numpy as np

from chomikgrad import (
    Parameter,
    SGD,
    Tensor,
    compile_graph,
    cross_entropy,
    get_compiler,
)


WGPU_INSTALLED = importlib.util.find_spec("wgpu") is not None


@unittest.skipUnless(WGPU_INSTALLED, "optional wgpu dependency is not installed")
class VulkanBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.compiler = get_compiler("vulkan")
        except Exception as error:
            raise unittest.SkipTest(f"Vulkan runtime is not available: {error}")
        if str(cls.compiler.adapter_info["backend_type"]).lower() != "vulkan":
            raise AssertionError("wgpu did not select its Vulkan backend")

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
        program = compile_graph(result, compiler="vulkan")
        np.testing.assert_allclose(
            program()[0], result.numpy(compiler="cpu"), rtol=1e-5, atol=1e-6
        )
        self.assertIn("matmul", program.source)
        self.assertIn("permute", program.source)
        self.assertIn("    del ", program.source)

        table = Tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32
        )
        indices = Tensor(np.array([[2, 0]], dtype=np.int64), copy=False)
        np.testing.assert_allclose(
            table.gather(indices).numpy(compiler="vulkan"),
            [[[5.0, 6.0], [1.0, 2.0]]],
        )

    def test_autograd_broadcast_and_batched_matmul_match_cpu(self) -> None:
        raw = np.array(
            [[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32
        )
        logits = Tensor(raw, requires_grad=True)
        loss = cross_entropy(logits, np.array([0, 2], dtype=np.int64))
        loss.backward()
        np.testing.assert_allclose(
            loss.numpy(compiler="vulkan"),
            loss.numpy(compiler="cpu"),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            logits.grad.numpy(compiler="vulkan"),
            logits.grad.numpy(compiler="cpu"),
            rtol=3e-5,
            atol=3e-6,
        )

        rng = np.random.default_rng(37)
        left = Tensor(rng.normal(size=(2, 1, 3, 5)).astype(np.float32))
        right = Tensor(rng.normal(size=(1, 4, 5, 6)).astype(np.float32))
        result = left @ right
        np.testing.assert_allclose(
            result.numpy(compiler="vulkan"),
            result.numpy(compiler="cpu"),
            rtol=3e-5,
            atol=3e-6,
        )

        source = Tensor(
            rng.normal(size=(2, 3, 5)).astype(np.float32),
            requires_grad=True,
        )
        weight = Tensor(
            rng.normal(size=(5, 7)).astype(np.float32), requires_grad=True
        )
        (source @ weight).sum().backward()
        np.testing.assert_allclose(
            weight.grad.numpy(compiler="vulkan"),
            weight.grad.numpy(compiler="cpu"),
            rtol=3e-5,
            atol=3e-6,
        )

    def test_dynamic_input_stays_on_vulkan(self) -> None:
        source = Tensor([1.0, 2.0], dtype=np.float32)
        program = compile_graph(
            source * 2.0, compiler="vulkan", dynamic_inputs=(source,)
        )
        replacement = program.device.array(
            np.array([3.0, 4.0], dtype=np.float32)
        )
        native = program.run(
            {source._node: replacement}, synchronize=False
        )[0]
        program.device.synchronize()
        np.testing.assert_allclose(program.device.to_numpy(native), [6.0, 8.0])
        self.assertEqual(program.inputs, (source._node,))
        self.assertEqual(program.device.argmax(native), 1)

    def test_sgd_keeps_parameters_on_vulkan(self) -> None:
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                parameter = Parameter(
                    np.array([[1.0], [2.0]], dtype=np.float32)
                )
                inputs = Tensor(
                    np.array([[3.0, 4.0]], dtype=np.float32)
                )
                old_graph = parameter * 2.0
                storage_before = compile_graph(
                    parameter, compiler="vulkan"
                ).run(synchronize=True)[0]
                (inputs @ parameter).sum().backward()
                SGD([parameter], lr=0.1, inplace=inplace).step(
                    compiler="vulkan"
                )

                storage_after = parameter._node.native_values["vulkan"]
                if inplace:
                    self.assertIs(
                        storage_before.buffer,
                        storage_after.buffer,
                    )
                    expected_old_graph = [[1.4], [3.2]]
                else:
                    self.assertIsNot(
                        storage_before.buffer,
                        storage_after.buffer,
                    )
                    expected_old_graph = [[2.0], [4.0]]
                np.testing.assert_allclose(
                    parameter.numpy(compiler="cpu"),
                    [[0.7], [1.6]],
                    rtol=1e-6,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    old_graph.numpy(compiler="vulkan"), expected_old_graph
                )


if __name__ == "__main__":
    unittest.main()
