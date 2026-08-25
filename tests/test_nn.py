import unittest

import numpy as np

from chomikgrad import (
    Linear,
    ReLU,
    SGD,
    Sequential,
    Tensor,
    compile_train_step,
    cross_entropy,
)


class NeuralNetworkTests(unittest.TestCase):
    def test_compiled_train_step_requires_backend_support(self) -> None:
        parameter = Linear(1, 1).weight
        optimizer = SGD([parameter])
        with self.assertRaisesRegex(
            RuntimeError, "does not support compiled training steps"
        ):
            compile_train_step(
                lambda inputs: (inputs @ parameter).sum(),
                optimizer,
                Tensor.zeros((1, 1)),
                compiler="cpu",
            )

    def test_inplace_sgd_requires_backend_support(self) -> None:
        parameter = Linear(1, 1).weight
        parameter.grad = Tensor(np.ones(parameter.shape, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "does not support in-place"):
            SGD([parameter], inplace=True).step(compiler="cpu")

    def test_optimizer_learns_small_multiclass_problem(self) -> None:
        rng = np.random.default_rng(7)
        features = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -0.5],
                [2.0, -1.0],
                [1.5, -0.5],
                [0.0, 2.0],
                [0.5, 1.5],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        model = Sequential(Linear(2, 8, rng=rng), ReLU(), Linear(8, 3, rng=rng))
        optimizer = SGD(model.parameters(), lr=0.08)

        initial = cross_entropy(model(Tensor(features)), labels).item()
        for _ in range(80):
            optimizer.zero_grad()
            loss = cross_entropy(model(Tensor(features)), labels)
            loss.backward()
            optimizer.step()

        logits = model(Tensor(features)).numpy()
        final = cross_entropy(Tensor(logits), labels).item()
        self.assertLess(final, initial * 0.25)
        np.testing.assert_array_equal(logits.argmax(axis=1), labels)


if __name__ == "__main__":
    unittest.main()
