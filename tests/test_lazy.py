import unittest

import numpy as np

from chomikgrad import (
    Compiler,
    NumpyCompiler,
    Op,
    Tensor,
    compile_graph,
    register_compiler,
)


class RecordingCompiler(Compiler):
    def __init__(self) -> None:
        self.compilations = 0

    def compile(self, outputs, dynamic_inputs=()):
        self.compilations += 1
        return NumpyCompiler().compile(outputs, dynamic_inputs)


class LazyExecutionTests(unittest.TestCase):
    def test_graph_is_lazy_and_compiler_is_pluggable(self) -> None:
        recorder = RecordingCompiler()
        register_compiler("test-recording", recorder, replace=True)

        left = Tensor([1.0, 2.0], dtype=np.float32)
        result = (left + 3.0).relu()
        self.assertEqual(recorder.compilations, 0)

        np.testing.assert_allclose(
            result.numpy(compiler="test-recording"), [4.0, 5.0]
        )
        self.assertEqual(recorder.compilations, 1)

    def test_cpu_compiler_generates_straight_line_numpy_code(self) -> None:
        left = Tensor([[1.0, 2.0]], dtype=np.float32)
        right = Tensor([[3.0], [4.0]], dtype=np.float32)
        output = left @ right

        program = compile_graph(output)
        self.assertIn("np.matmul", program.source)
        self.assertNotIn("ELEMENTWISE", program.source)
        np.testing.assert_allclose(program(), (np.array([[11.0]]),))

    def test_program_run_can_rebind_inputs_without_recompiling(self) -> None:
        source = Tensor([1.0, 2.0], dtype=np.float32)
        program = compile_graph(
            source * 2.0, dynamic_inputs=(source,)
        )

        outputs = program.run(
            {source._node: program.device.array([3.0, 4.0])},
            synchronize=False,
        )

        np.testing.assert_allclose(outputs[0], [6.0, 8.0])
        np.testing.assert_allclose(program()[0], [2.0, 4.0])
        self.assertEqual(program.device.name, "cpu")
        self.assertEqual(program.device.argmax(outputs[0]), 1)

    def test_instruction_set_has_only_six_operations(self) -> None:
        self.assertEqual(len(Op), 6)

    def test_gather_forward_and_repeated_index_gradient(self) -> None:
        table = Tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
            requires_grad=True,
        )
        indices = Tensor(np.array([[2, 0, 2]], dtype=np.int32), copy=False)
        selected = table.gather(indices)
        loss = selected.sum()
        loss.backward()

        program = compile_graph(selected)
        self.assertIn("np.take", program.source)
        np.testing.assert_allclose(
            program()[0],
            [[[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]]],
        )
        np.testing.assert_allclose(
            table.grad.numpy(),
            [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]],
        )

    def test_existing_lazy_graph_keeps_pre_assignment_value(self) -> None:
        value = Tensor([2.0], dtype=np.float32)
        old_graph = value * 3.0
        value.assign([5.0])

        np.testing.assert_allclose(old_graph.numpy(), [6.0])
        np.testing.assert_allclose((value * 3.0).numpy(), [15.0])

    def test_copy_false_explicitly_shares_numpy_storage(self) -> None:
        source = np.array([1.0, 2.0], dtype=np.float32)
        shared = Tensor(source, copy=False)
        owned = Tensor(source)

        source[0] = 9.0

        np.testing.assert_allclose(shared.numpy(), [9.0, 2.0])
        np.testing.assert_allclose(owned.numpy(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
