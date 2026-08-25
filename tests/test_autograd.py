import unittest

import numpy as np

from chomikgrad import Tensor, cross_entropy


class AutogradTests(unittest.TestCase):
    def test_sqrt_value_and_gradient(self) -> None:
        value = Tensor(np.array([4.0], dtype=np.float32), requires_grad=True)
        result = value.sqrt().sum()
        result.backward()
        np.testing.assert_allclose(result.numpy(), 2.0)
        np.testing.assert_allclose(value.grad.numpy(), [0.25])

    def test_broadcast_gradient_is_reduced_to_input_shape(self) -> None:
        inputs = Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            requires_grad=True,
            dtype=np.float32,
        )
        bias = Tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=np.float32)

        loss = ((inputs + bias) * (inputs + bias)).mean()
        loss.backward()

        expected_inputs = 2 * (
            np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            + np.array([1, 2, 3], dtype=np.float32)
        ) / 6
        np.testing.assert_allclose(inputs.grad.numpy(), expected_inputs, rtol=1e-6)
        np.testing.assert_allclose(
            bias.grad.numpy(), expected_inputs.sum(axis=0), rtol=1e-6
        )

    def test_matmul_gradient(self) -> None:
        left_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        right_data = np.array([[2.0, 1.0], [0.5, 3.0]], dtype=np.float32)
        left = Tensor(left_data, requires_grad=True)
        right = Tensor(right_data, requires_grad=True)

        (left @ right).sum().backward()

        upstream = np.ones((2, 2), dtype=np.float32)
        np.testing.assert_allclose(left.grad.numpy(), upstream @ right_data.T)
        np.testing.assert_allclose(right.grad.numpy(), left_data.T @ upstream)

    def test_batched_matmul_gradient_with_broadcast(self) -> None:
        left_data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
        right_data = np.arange(20, dtype=np.float32).reshape(4, 5) / 10
        left = Tensor(left_data, requires_grad=True)
        right = Tensor(right_data, requires_grad=True)

        (left @ right).sum().backward()

        upstream = np.ones((2, 3, 5), dtype=np.float32)
        expected_left = upstream @ right_data.T
        expected_right = (left_data.transpose(0, 2, 1) @ upstream).sum(axis=0)
        np.testing.assert_allclose(left.grad.numpy(), expected_left, rtol=1e-6)
        np.testing.assert_allclose(right.grad.numpy(), expected_right, rtol=1e-6)

    def test_max_splits_gradient_across_equal_values(self) -> None:
        values = Tensor([[1.0, 3.0, 3.0]], requires_grad=True, dtype=np.float32)
        values.max().backward()
        np.testing.assert_allclose(values.grad.numpy(), [[0.0, 0.5, 0.5]])

    def test_cross_entropy_value_and_gradient(self) -> None:
        raw = np.array([[2.0, -1.0, 0.5], [0.0, 1.0, -0.5]], dtype=np.float32)
        labels = np.array([0, 2], dtype=np.int64)
        logits = Tensor(raw, requires_grad=True)

        loss = cross_entropy(logits, labels)
        loss.backward()

        shifted = raw - raw.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        expected_gradient = probabilities.copy()
        expected_gradient[np.arange(2), labels] -= 1
        expected_gradient /= 2
        expected_loss = -np.log(probabilities[np.arange(2), labels]).mean()
        self.assertAlmostEqual(float(loss.item()), float(expected_loss), places=6)
        np.testing.assert_allclose(
            logits.grad.numpy(), expected_gradient, rtol=1e-6, atol=1e-7
        )

    def test_softmax_is_stable_and_differentiable(self) -> None:
        raw = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
        weights = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
        values = Tensor(raw, requires_grad=True)

        probabilities = values.softmax(axis=1)
        (probabilities * Tensor(weights)).sum().backward()

        shifted = raw - raw.max(axis=1, keepdims=True)
        expected = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        expected_gradient = expected * (
            weights - (expected * weights).sum(axis=1, keepdims=True)
        )
        np.testing.assert_allclose(probabilities.numpy(), expected, rtol=1e-6)
        np.testing.assert_allclose(
            values.grad.numpy(), expected_gradient, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_allclose(
            values.log_softmax(axis=1).exp().numpy(), expected, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
