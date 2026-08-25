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

    def compile(self, outputs):
        self.compilations += 1
        return NumpyCompiler().compile(outputs)


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

    def test_instruction_set_has_only_five_operations(self) -> None:
        self.assertEqual(len(Op), 5)

    def test_existing_lazy_graph_keeps_pre_assignment_value(self) -> None:
        value = Tensor([2.0], dtype=np.float32)
        old_graph = value * 3.0
        value.assign([5.0])

        np.testing.assert_allclose(old_graph.numpy(), [6.0])
        np.testing.assert_allclose((value * 3.0).numpy(), [15.0])


if __name__ == "__main__":
    unittest.main()
